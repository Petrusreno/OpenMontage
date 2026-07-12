# React Composite Perf Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ffmpeg-native single-pass `colorkey`+`overlay` fast path (new default `engine="ffmpeg"`) to `green_screen_composite`, ~6×+ faster than the legacy per-frame PIL path, preserving all four layouts, speaker audio, and honest failure.

**Architecture:** Split `execute` into a dispatcher + `_execute_pil` (the current body, moved verbatim) + `_execute_ffmpeg` (new). A pure `_layout_filtergraph` string builder produces the validated per-layout graph; `_ffmpeg_color` maps `#RRGGBB`→`0xRRGGBB`. The PIL path and its helpers are byte-unchanged.

**Tech Stack:** Python 3, ffmpeg (`colorkey`, `overlay`, `crop`, `pad`, `scale`), numpy/PIL (PIL path only), pytest.

## Global Constraints

- Tool file: `tools/video/green_screen_composite.py`. Test file (new): `tests/tools/test_green_screen_composite.py`.
- Run tests ONLY via `.venv/bin/python -m pytest` (bare `python3` lacks pytest + is RTK-proxied). Capture the exit code separately — `... > /tmp/log 2>&1; echo "exit: $?"` — NEVER trust `| tail` (it masks pytest's exit code; this bit us twice earlier this batch).
- `engine="ffmpeg"` is the DEFAULT. `engine="pil"` must produce byte-identical output to the pre-change tool (the PIL body moves verbatim — do not "improve" it).
- Fast path: NO silent PIL fallback. ffmpeg failure or missing/empty output → `ToolResult(success=False, error=<ffmpeg stderr>)`.
- Keyer is `colorkey` (RGB, solid `#0E172A`), NOT green `chromakey`. Defaults: `key_similarity=0.10`, `key_blend=0.08`.
- Audio (fast path): `original_audio_path` given → mux from it (`-map 1:a:0`, validate exists); else speaker passthrough `-map 0:a?` (the `?` makes a silent speaker safe). `-c:a aac -b:a 192k`.
- Output dims = background dims. `target_fps = min(speaker_fps, bg_fps)` (≤0 → 15.0). `duration = min(speaker, bg)`, capped via `-t`.
- Boundary-validate before ffmpeg work: `engine` in enum; `key_similarity`/`key_blend` numeric in `[0,1]`; `bg_color_hex` 6 hex digits. Non-numeric/out-of-range → `success=False`, never a traceback.
- Anti-fabrication: every metric measured on real output; `frame_count = round(duration*target_fps)` on the fast path; `has_audio` reflects whether an audio stream was actually mapped.
- Validated filtergraphs (copy exactly), inputs `[0:v]`=speaker `[1:v]`=background, `ck = colorkey={ff_color}:{sim}:{blend}`, `W,H`=out dims, `hw=W//2`, `S=bg_shift_up`, `sc=speaker_scale`:
  - full_behind: `[1:v]scale=W:H[bg];[0:v]scale=W:H,{ck}[fg];[bg][fg]overlay=0:0[v]`
  - news_anchor: `[1:v]scale=W:H,crop=W:H-S:0:S,pad=W:H:0:0:black[bg];[0:v]scale=iw*sc:ih*sc,{ck}[fg];[bg][fg]overlay=(W-w)/2:H-h[v]`
  - pip: `[1:v]scale=W:H[bg];[0:v]scale=W*0.30:H*0.30,{ck}[fg];[bg][fg]overlay=W-w-20:H-h-20[v]`
  - split: `color=c=black:s=WxH[base];[0:v]scale=hw:H,{ck}[l];[1:v]scale=hw:H[r];[base][l]overlay=0:0[t];[t][r]overlay=hw:0[v]`

---

### Task 1: Pure helpers — `_ffmpeg_color` and `_layout_filtergraph`

**Files:**
- Modify: `tools/video/green_screen_composite.py`
- Test: `tests/tools/test_green_screen_composite.py` (create)

**Interfaces:**
- Produces: `_ffmpeg_color(self, bg_color_hex: str) -> str` (`"#0E172A"`→`"0x0E172A"`; raises `ValueError` on non-6-hex-digit input). `_layout_filtergraph(self, layout: str, out_w: int, out_h: int, speaker_scale: float, bg_shift_up: int, ff_color: str, sim: float, blend: float) -> str` (raises `ValueError` on unknown layout).

- [ ] **Step 1: Write failing tests**

```python
import pytest
from tools.video.green_screen_composite import GreenScreenComposite

def test_ffmpeg_color_maps_hex():
    t = GreenScreenComposite()
    assert t._ffmpeg_color("#0E172A") == "0x0E172A"
    assert t._ffmpeg_color("0E172A") == "0x0E172A"
    for bad in ["#0E172", "#0E172AA", "#GG1234", "navy"]:
        with pytest.raises(ValueError):
            t._ffmpeg_color(bad)

def test_layout_filtergraph_full_behind():
    g = GreenScreenComposite()._layout_filtergraph("full_behind", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert g == "[1:v]scale=1920:1080[bg];[0:v]scale=1920:1080,colorkey=0x0E172A:0.1:0.08[fg];[bg][fg]overlay=0:0[v]"

def test_layout_filtergraph_news_anchor_shift_and_scale():
    g = GreenScreenComposite()._layout_filtergraph("news_anchor", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert "crop=1920:780:0:300,pad=1920:1080:0:0:black[bg]" in g
    assert "scale=iw*0.65:ih*0.65,colorkey=0x0E172A:0.1:0.08[fg]" in g
    assert g.endswith("[bg][fg]overlay=(W-w)/2:H-h[v]")

def test_layout_filtergraph_pip_and_split():
    t = GreenScreenComposite()
    p = t._layout_filtergraph("pip", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert "scale=1920*0.30:1080*0.30,colorkey=0x0E172A:0.1:0.08[fg]" in p
    assert p.endswith("[bg][fg]overlay=W-w-20:H-h-20[v]")
    s = t._layout_filtergraph("split", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert "color=c=black:s=1920x1080[base]" in s
    assert "[0:v]scale=960:1080,colorkey=0x0E172A:0.1:0.08[l]" in s
    assert s.endswith("[base][l]overlay=0:0[t];[t][r]overlay=960:0[v]")

def test_layout_filtergraph_unknown_raises():
    with pytest.raises(ValueError):
        GreenScreenComposite()._layout_filtergraph("bogus", 1920, 1080, 0.65, 300, "0x0E172A", 0.1, 0.08)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py -q > /tmp/t1.log 2>&1; echo "exit: $?"`
Expected: FAIL (AttributeError: no `_ffmpeg_color`/`_layout_filtergraph`).

- [ ] **Step 3: Implement the helpers**

Add to `GreenScreenComposite`. Format floats so `0.10`→`0.1` (use `float`'s `repr`, i.e. plain f-string — `f"{0.10}"` is `"0.1"`, `f"{0.08}"` is `"0.08"`; matches the test strings). Scale strings for pip/news use literal `*0.30`/`*sc` so ffmpeg evaluates them.

```python
def _ffmpeg_color(self, bg_color_hex: str) -> str:
    h = bg_color_hex.lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"bg_color_hex must be 6 hex digits, got {bg_color_hex!r}")
    return f"0x{h.upper()}"

def _layout_filtergraph(self, layout, out_w, out_h, speaker_scale, bg_shift_up, ff_color, sim, blend):
    W, H, hw, S, sc = out_w, out_h, out_w // 2, bg_shift_up, speaker_scale
    ck = f"colorkey={ff_color}:{sim}:{blend}"
    if layout == "full_behind":
        return f"[1:v]scale={W}:{H}[bg];[0:v]scale={W}:{H},{ck}[fg];[bg][fg]overlay=0:0[v]"
    if layout == "news_anchor":
        return (f"[1:v]scale={W}:{H},crop={W}:{H - S}:0:{S},pad={W}:{H}:0:0:black[bg];"
                f"[0:v]scale=iw*{sc}:ih*{sc},{ck}[fg];[bg][fg]overlay=(W-w)/2:H-h[v]")
    if layout == "pip":
        return (f"[1:v]scale={W}:{H}[bg];[0:v]scale={W}*0.30:{H}*0.30,{ck}[fg];"
                f"[bg][fg]overlay=W-w-20:H-h-20[v]")
    if layout == "split":
        return (f"color=c=black:s={W}x{H}[base];[0:v]scale={hw}:{H},{ck}[l];[1:v]scale={hw}:{H}[r];"
                f"[base][l]overlay=0:0[t];[t][r]overlay={hw}:0[v]")
    raise ValueError(f"Unknown layout: {layout}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py -q > /tmp/t1.log 2>&1; echo "exit: $?"`
Expected: PASS (exit 0). Confirm by reading the log, not `| tail`.

- [ ] **Step 5: Commit**

```bash
git add tools/video/green_screen_composite.py tests/tools/test_green_screen_composite.py
git commit -m "feat: green_screen_composite pure fast-path helpers (color + layout filtergraph)"
```

---

### Task 2: Split `execute` into dispatcher + `_execute_pil` (behavior-preserving)

**Files:**
- Modify: `tools/video/green_screen_composite.py`
- Test: `tests/tools/test_green_screen_composite.py`

**Interfaces:**
- Consumes: helpers from Task 1.
- Produces: `execute` validates + dispatches on `engine`; `_execute_pil(self, speaker_path, background_path, output_path, *, original_audio_path, layout, speaker_scale, bg_shift_up, bg_color_hex) -> ToolResult` (the current body, verbatim, with `data` gaining `"engine": "pil"`). Schema gains `engine`/`key_similarity`/`key_blend`; `idempotency_key_fields` gains them.

- [ ] **Step 1: Write failing tests**

```python
import shutil
import numpy as np
import subprocess as sp
from pathlib import Path
from PIL import Image

def _make_speaker(path, sr_audio=True, w=320, h=180, dur=2, fps=15):
    # white disc on solid #0E172A, optional tone
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
           "-i", f"color=c=0x0E172A:s={w}x{h}:d={dur}:r={fps}"]
    if sr_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    cmd += ["-vf", "drawbox=x='(w-80)/2':y=40:w=80:h=100:color=white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if sr_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    sp.run(cmd, check=True, capture_output=True)

def _make_bg(path, w=320, h=180, dur=2):
    sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", f"testsrc2=s={w}x{h}:d={dur}:r=30", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)

def _has_audio(path):
    r = sp.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())

def test_execute_rejects_bad_engine(tmp_path):
    sp_path = tmp_path / "s.mp4"; bg = tmp_path / "b.mp4"
    sp_path.write_bytes(b"x"); bg.write_bytes(b"x")
    r = GreenScreenComposite().execute({"speaker_path": str(sp_path), "background_path": str(bg),
                                        "output_path": str(tmp_path / "o.mp4"), "engine": "nope"})
    assert not r.success and "engine" in (r.error or "").lower()

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_pil_engine_still_works(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
                                        "output_path": str(out), "layout": "full_behind", "engine": "pil"})
    assert r.success, r.error
    assert r.data["engine"] == "pil" and out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py -q > /tmp/t2.log 2>&1; echo "exit: $?"`
Expected: FAIL (`engine` not validated; `data["engine"]` KeyError).

- [ ] **Step 3: Implement dispatcher + move PIL body**

In `execute`: keep the existing input reads, add `engine = inputs.get("engine", "ffmpeg")`, `key_similarity = inputs.get("key_similarity", 0.10)`, `key_blend = inputs.get("key_blend", 0.08)`. Validate BEFORE the filesystem checks:
```python
if engine not in ("ffmpeg", "pil"):
    return ToolResult(success=False, error=f"engine must be 'ffmpeg' or 'pil', got {engine!r}")
try:
    key_similarity = float(key_similarity); key_blend = float(key_blend)
except (TypeError, ValueError):
    return ToolResult(success=False, error="key_similarity/key_blend must be numeric.")
if not (0.0 <= key_similarity <= 1.0) or not (0.0 <= key_blend <= 1.0):
    return ToolResult(success=False, error="key_similarity/key_blend must be in [0, 1].")
```
Keep the speaker/background/audio existence checks. Then dispatch:
```python
if engine == "pil":
    return self._execute_pil(speaker_path, background_path, output_path,
        original_audio_path=original_audio_path, layout=layout, speaker_scale=speaker_scale,
        bg_shift_up=bg_shift_up, bg_color_hex=bg_color_hex)
return self._execute_ffmpeg(speaker_path, background_path, output_path,
    original_audio_path=original_audio_path, layout=layout, speaker_scale=speaker_scale,
    bg_shift_up=bg_shift_up, bg_color_hex=bg_color_hex,
    key_similarity=key_similarity, key_blend=key_blend)
```
Move the current `start = time.time()` … `finally: self._cleanup_temp(temp_dir)` block verbatim into `_execute_pil(self, speaker_path, background_path, output_path, *, original_audio_path, layout, speaker_scale, bg_shift_up, bg_color_hex)`, adding `"engine": "pil"` to its `data`. Add a stub `_execute_ffmpeg(...) -> ToolResult` returning `ToolResult(success=False, error="not implemented")` (Task 3 fills it). Add the three schema properties and extend `idempotency_key_fields` with `"engine", "key_similarity", "key_blend"`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py -q > /tmp/t2.log 2>&1; echo "exit: $?"`
Expected: PASS (exit 0). Read the log to confirm.

- [ ] **Step 5: Commit**

```bash
git add tools/video/green_screen_composite.py tests/tools/test_green_screen_composite.py
git commit -m "refactor: green_screen_composite engine dispatch + _execute_pil (behavior-preserving)"
```

---

### Task 3: Implement `_execute_ffmpeg` (the fast path)

**Files:**
- Modify: `tools/video/green_screen_composite.py`
- Test: `tests/tools/test_green_screen_composite.py`

**Interfaces:**
- Consumes: `_probe_video`, `_ffmpeg_color`, `_layout_filtergraph`, `run_command` (from `BaseTool`).
- Produces: `_execute_ffmpeg(...)` fully implemented — one guarded ffmpeg call, honest failure, `data` with `"engine": "ffmpeg"`.

- [ ] **Step 1: Write failing tests** (append to the test file; reuse `_make_speaker`/`_make_bg`/`_has_audio`)

```python
def _sample_top_navy_pct(mp4, tmp_path):
    png = tmp_path / "_frame.png"
    sp.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(mp4), "-frames:v", "1", str(png)],
           check=True, capture_output=True)
    a = np.asarray(Image.open(png).convert("RGB")).astype(float)
    navy = np.array([0x0E, 0x17, 0x2A], dtype=float)
    top = a[0:30, :, :].reshape(-1, 3)          # strip above the disc
    d = np.sqrt(((top - navy) ** 2).sum(1))
    return float((d < 40).mean()) * 100

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_keying_reveals_background(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind"})   # default engine=ffmpeg
    assert r.success, r.error
    assert r.data["engine"] == "ffmpeg"
    assert _sample_top_navy_pct(out, tmp_path) < 5.0         # bg shows through (validated 0.0%)

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.parametrize("layout", ["full_behind", "news_anchor", "pip", "split"])
def test_ffmpeg_all_layouts(tmp_path, layout):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b, w=320, h=180)
    out = tmp_path / f"o_{layout}.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": layout})
    assert r.success, r.error
    assert out.exists() and out.stat().st_size > 0
    assert r.data["dimensions"] == "320x180"
    assert r.data["frame_count"] >= 1

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_speaker_audio_passthrough(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=True); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind"})
    assert r.success, r.error
    assert r.data["has_audio"] is True and _has_audio(out)

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_silent_speaker_no_audio_ok(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind"})
    assert r.success, r.error                                 # ? map => safe
    assert r.data["has_audio"] is False and not _has_audio(out)

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_faster_than_pil(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False, dur=3); _make_bg(b, dur=3)
    base = {"speaker_path": str(s), "background_path": str(b), "layout": "full_behind"}
    import time
    t0 = time.time(); r1 = GreenScreenComposite().execute({**base, "output_path": str(tmp_path/"f.mp4"), "engine": "ffmpeg"}); ff = time.time()-t0
    t0 = time.time(); r2 = GreenScreenComposite().execute({**base, "output_path": str(tmp_path/"p.mp4"), "engine": "pil"}); pil = time.time()-t0
    assert r1.success and r2.success
    assert ff < pil                                           # direction of the win, not a brittle multiplier

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_bad_background_fails_cleanly(tmp_path):
    s = tmp_path / "s.mp4"; _make_speaker(s, sr_audio=False)
    bad = tmp_path / "b.mp4"; bad.write_bytes(b"not a video")
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(bad),
        "output_path": str(tmp_path / "o.mp4"), "layout": "full_behind"})
    assert not r.success and r.error                          # no exception

def test_execute_rejects_bad_key_params(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"; s.write_bytes(b"x"); b.write_bytes(b"x")
    base = {"speaker_path": str(s), "background_path": str(b), "output_path": str(tmp_path / "o.mp4")}
    assert not GreenScreenComposite().execute({**base, "key_similarity": "x"}).success
    assert not GreenScreenComposite().execute({**base, "key_blend": 1.5}).success
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py -q > /tmp/t3.log 2>&1; echo "exit: $?"`
Expected: FAIL (`_execute_ffmpeg` stub returns not-implemented).

- [ ] **Step 3: Implement `_execute_ffmpeg`**

```python
def _execute_ffmpeg(self, speaker_path, background_path, output_path, *, original_audio_path,
                    layout, speaker_scale, bg_shift_up, bg_color_hex, key_similarity, key_blend):
    start = time.time()
    try:
        ff_color = self._ffmpeg_color(bg_color_hex)
    except ValueError as e:
        return ToolResult(success=False, error=str(e))

    speaker_info = self._probe_video(speaker_path)
    bg_info = self._probe_video(background_path)
    if not speaker_info or not bg_info:
        return ToolResult(success=False, error="Failed to probe one or both input videos")

    target_fps = min(speaker_info["fps"], bg_info["fps"])
    if target_fps <= 0:
        target_fps = 15.0
    out_w, out_h = bg_info["width"], bg_info["height"]
    duration = min(speaker_info["duration"], bg_info["duration"])

    try:
        fg = self._layout_filtergraph(layout, out_w, out_h, speaker_scale, bg_shift_up,
                                      ff_color, key_similarity, key_blend)
    except ValueError as e:
        return ToolResult(success=False, error=str(e))

    cmd = ["ffmpeg", "-y", "-i", str(speaker_path), "-i", str(background_path)]
    has_audio = False
    if original_audio_path:
        cmd += ["-i", str(original_audio_path)]
        audio_map = ["-map", "2:a:0"]
        has_audio = True
    else:
        audio_map = ["-map", "0:a?"]
        has_audio = bool(speaker_info.get("has_audio"))   # see note below
    cmd += ["-filter_complex", fg, "-map", "[v]", *audio_map,
            "-t", f"{duration:.3f}", "-r", str(target_fps),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(output_path)]
    try:
        self.run_command(cmd, timeout=1200)
    except Exception as e:
        return ToolResult(success=False, error=f"FFmpeg fast-path failed: {e}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        return ToolResult(success=False, error="Output video was not created")

    return ToolResult(success=True, data={
        "output": str(output_path), "layout": layout, "fps": target_fps,
        "frame_count": round(duration * target_fps), "duration": round(duration, 2),
        "dimensions": f"{out_w}x{out_h}", "speaker_scale": speaker_scale,
        "engine": "ffmpeg", "has_audio": has_audio,
    }, artifacts=[str(output_path)], duration_seconds=round(time.time() - start, 2))
```

`_probe_video` must report whether the video has an audio stream so `has_audio` is truthful when
`-map 0:a?` maps nothing. Extend `_probe_video`'s returned dict with `"has_audio"`: after finding the
video stream, set `has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))`
and include it. (This is additive — the PIL path ignores the new key.) Note the audio-map index: with
`original_audio_path` present it is the **third** input (`2:a:0`), matching the three `-i` above.

- [ ] **Step 4: Run to verify pass (full file)**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py -q > /tmp/t3.log 2>&1; echo "exit: $?"`
Expected: PASS (exit 0). Read the log; confirm the ffmpeg-gated tests ran (not skipped).

- [ ] **Step 5: Commit**

```bash
git add tools/video/green_screen_composite.py tests/tools/test_green_screen_composite.py
git commit -m "feat: green_screen_composite ffmpeg fast path (colorkey+overlay, single pass)"
```

---

### Task 4: Docstring + full-suite regression check

**Files:**
- Modify: `tools/video/green_screen_composite.py` (module docstring only)

- [ ] **Step 1: Update the module docstring**

Amend the top docstring to state both engines: ffmpeg single-pass `colorkey`+`overlay` (default, fast) and the legacy PIL per-frame path (`engine="pil"`). Note the keyer is `colorkey` on the solid `bg_color_hex`. One short paragraph — no code.

- [ ] **Step 2: Run the tool's tests + a broad regression sweep**

Run: `.venv/bin/python -m pytest tests/tools/test_green_screen_composite.py tests/tools/test_beat_sync.py tests/contracts/ -q > /tmp/t4.log 2>&1; echo "exit: $?"`
Expected: PASS (exit 0). If the registry-contract test enumerates tools, confirm `green_screen_composite` still loads. Read the log fully.

- [ ] **Step 3: Commit**

```bash
git add tools/video/green_screen_composite.py
git commit -m "docs: green_screen_composite dual-engine docstring"
```

---

## Self-Review

- **Spec coverage:** engine param + default (T2), 4 layouts (T1 graphs, T3 e2e), keying honesty anchor (T3), audio passthrough + override + silent-safe (T3), boundary validation (T2/T3), no-silent-fallback (T3 bad-bg), PIL byte-preservation (T2 moves verbatim + regression test), schema/idempotency additions (T2). Covered.
- **Placeholders:** none — every step has concrete code/commands.
- **Type consistency:** `_layout_filtergraph`/`_ffmpeg_color`/`_execute_ffmpeg`/`_execute_pil` signatures match across tasks; `data["engine"]` set in both engines; `_probe_video` `has_audio` addition is additive.
- **Known risk to watch in review:** the float-formatting in `_layout_filtergraph` must render `0.10`→`"0.1"` to match Task 1's asserted strings; if a reviewer changes the sim/blend defaults, the asserted graph strings in Task 1 must change with them.
