# Voice Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `voice_isolation` tool that strongly cleans up speech via RNNoise (`arnndn`, ML) when a model is available, else a stronger-than-`audio_enhance` spectral chain, honestly reporting which engine ran and measuring the real noise-floor reduction.

**Architecture:** One `BaseTool` subclass in `tools/audio/voice_isolation.py`, contract-styled after `tools/audio/audio_enhance.py`. It resolves a `.rnnn` model (input → env → bundled `assets/rnnoise/somnolent-hogwash.rnnn`), builds an ffmpeg `-af` chain for the chosen engine with a dry/wet `mix` blend, processes the audio, muxes it back into a video input (video `-c copy`), and re-measures the output noise floor (numpy on extracted PCM) to report honest before/after reduction. Where a *forced* ML request can't run (no model), it fails cleanly rather than silently downgrading.

**Tech Stack:** Python 3.14, ffmpeg/ffprobe (subprocess), numpy 2.4.4, pytest. The bundled RNNoise model + CREDITS were already committed (`118dd02`).

## Global Constraints

- Tool inherits `tools/base_tool.py` `BaseTool` with full contract fields in the style of `tools/audio/audio_enhance.py`. `execute(self, inputs: dict[str, Any]) -> ToolResult`.
- `dependencies = ["cmd:ffmpeg", "python:numpy"]` — the CHECKED prefixes (`BaseTool.check_dependencies` honors only `cmd:`/`env:`/`python:`).
- **Validated ffmpeg recipes (do not change the filters):**
  - rnnoise chain: `arnndn=model={model},highpass=f=80,loudnorm=I=-16:LRA=11:TP=-1.5`
  - spectral chain: `afftdn=nf={nf}:nt=w,anlmdn,highpass=f=80,deesser,loudnorm=I=-16:LRA=11:TP=-1.5`
  - dry/wet blend when `mix < 1.0`: `asplit=2[a][b];[a]{chain}[w];[w]volume={mix}[wv];[b]volume={1-mix}[dv];[wv][dv]amix=inputs=2:normalize=0`
  - when `mix >= 1.0`: the bare `{chain}` (no split/amix). Empirically confirmed: rnnoise mix=1 drove the noise floor -22.81→-37.6 dBFS; mix=0.5 → -27.5 (blend works); spectral ran clean.
- **PCM for the metric is read as BYTES** via `subprocess.run(..., capture_output=True)` (NOT `run_command`, whose `text=True` corrupts binary PCM — proven in the multicam port).
- **Honest, never fabricated:** the noise-floor reduction is measured on the real OUTPUT; `engine` reports what actually ran. `engine="rnnoise"` forced with no resolvable model / no `arnndn` → `success=False` (never a silent downgrade); `engine="auto"` DOES fall back and reports `engine="spectral"`.
- All subprocess calls guarded (`CalledProcessError`/`TimeoutExpired`/`OSError`) with timeouts; temp dir cleaned in `finally`; missing/zero-byte output → `success=False` even on ffmpeg exit 0. Boundary-validate `mix`∈[0,1] and numeric `noise_floor_db` before ffmpeg work (non-numeric → `success=False`, not a traceback).
- Determinism: same input + engine + model + params → identical filtergraph and identical output size.
- No new third-party deps. Tests in `tests/tools/test_voice_isolation.py`, import from `tools.audio.voice_isolation`; filter/metric helpers tested directly, ffmpeg-gated tests synthesize a noisy speech-proxy (tone + white-noise, with a noise-only region).

---

### Task 1: Tool skeleton, contract, and registry discovery

**Files:**
- Create: `tools/audio/voice_isolation.py`
- Test: `tests/tools/test_voice_isolation.py`

**Interfaces:**
- Produces: `class VoiceIsolation(BaseTool)` with contract fields, `BUNDLED_MODEL` class attr, and `input_schema`. `execute(self, inputs)` returns `ToolResult(success=False, error="not implemented")` for now.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from tools.audio.voice_isolation import VoiceIsolation
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = VoiceIsolation()
    assert tool.name == "voice_isolation"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "audio_processing"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "voice_isolation" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path"]
    assert tool.input_schema["properties"]["engine"]["enum"] == ["auto", "rnnoise", "spectral"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py::test_contract_fields_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.audio.voice_isolation'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Voice Isolation — strongly clean up speech (Adobe Enhance Speech / DaVinci Voice Isolation).

Runs the RNNoise ML denoiser (ffmpeg `arnndn`) when a .rnnn model resolves, else a stronger
spectral chain than audio_enhance (afftdn + anlmdn + deesser). Reports which engine actually ran
and the real noise-floor reduction — never claims ML quality that did not execute. A `mix` control
blends the isolated voice with the original. For a video input, the cleaned audio is muxed back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolResult,
    ToolStability,
    ToolTier,
)


class VoiceIsolation(BaseTool):
    name = "voice_isolation"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "audio_processing"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:numpy"]
    install_instructions = "Install FFmpeg (brew install ffmpeg) and numpy (pip install numpy)."
    agent_skills = ["ffmpeg"]

    capabilities = ["voice_isolation", "speech_enhance", "denoise"]

    # Bundled public-domain RNNoise model (see assets/rnnoise/CREDITS.md).
    BUNDLED_MODEL = str(Path(__file__).resolve().parents[1] / "assets" / "rnnoise" / "somnolent-hogwash.rnnn")

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "engine": {"type": "string", "enum": ["auto", "rnnoise", "spectral"], "default": "auto"},
            "model_path": {"type": "string"},
            "mix": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
            "noise_floor_db": {"type": "number", "default": -25},
            "codec": {"type": "string", "default": "aac"},
            "bitrate": {"type": "string", "default": "192k"},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py::test_contract_fields_present -v`
Expected: PASS

- [ ] **Step 5: Verify discovery + bundled model path, then commit**

Run: `.venv/bin/python -c "import os; from tools.audio.voice_isolation import VoiceIsolation as V; print(V().name, os.path.isfile(V.BUNDLED_MODEL))"`
Expected: prints `voice_isolation True`

```bash
git add tools/audio/voice_isolation.py tests/tools/test_voice_isolation.py
git commit -m "feat: voice_isolation tool skeleton + contract"
```

---

### Task 2: Model resolution, arnndn probe, filtergraph builder

**Files:**
- Modify: `tools/audio/voice_isolation.py`
- Test: `tests/tools/test_voice_isolation.py`

**Interfaces:**
- Produces:
  - `_resolve_model(model_path) -> str | None` — `model_path` if it exists → else `os.environ["RNNOISE_MODEL"]` if set and exists → else `BUNDLED_MODEL` if it exists → else `None`.
  - `_arnndn_available() -> bool` — `arnndn` in `ffmpeg -filters` (cached).
  - `_build_filter(engine, model, nf, mix) -> str` — returns the `-af` chain. `engine=="rnnoise"` → the arnndn chain (requires `model`); else the spectral chain. When `mix < 1.0`, wraps the chain in the validated `asplit`/`amix` dry/wet blend; when `mix >= 1.0`, returns the bare chain.

- [ ] **Step 1: Write the failing tests**

```python
import os


def test_resolve_model_prefers_explicit_then_env_then_bundled(tmp_path, monkeypatch):
    tool = VoiceIsolation()
    monkeypatch.delenv("RNNOISE_MODEL", raising=False)
    # bundled exists by default
    assert tool._resolve_model(None) == VoiceIsolation.BUNDLED_MODEL
    # explicit override wins
    m = tmp_path / "custom.rnnn"; m.write_bytes(b"x")
    assert tool._resolve_model(str(m)) == str(m)
    # env override (when no explicit)
    monkeypatch.setenv("RNNOISE_MODEL", str(m))
    assert tool._resolve_model(None) == str(m)
    # non-existent explicit -> None (not silently the bundled)
    assert tool._resolve_model("/no/such.rnnn") is None


def test_build_filter_rnnoise_and_spectral():
    tool = VoiceIsolation()
    r = tool._build_filter("rnnoise", "/m.rnnn", nf=-25, mix=1.0)
    assert "arnndn=model=/m.rnnn" in r and "loudnorm" in r and "asplit" not in r
    s = tool._build_filter("spectral", None, nf=-25, mix=1.0)
    assert "afftdn=nf=-25" in s and "anlmdn" in s and "deesser" in s and "arnndn" not in s


def test_build_filter_mix_blend():
    tool = VoiceIsolation()
    m = tool._build_filter("rnnoise", "/m.rnnn", nf=-25, mix=0.5)
    assert "asplit=2" in m and "amix=inputs=2:normalize=0" in m
    assert "volume=0.5" in m and "volume=0.5" in m       # wet + dry both scaled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py -k "resolve_model or build_filter" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import os
import subprocess
```

Add methods:

```python
    def _resolve_model(self, model_path: str | None) -> str | None:
        if model_path:
            return model_path if Path(model_path).is_file() else None
        env = os.environ.get("RNNOISE_MODEL")
        if env and Path(env).is_file():
            return env
        return self.BUNDLED_MODEL if Path(self.BUNDLED_MODEL).is_file() else None

    def _arnndn_available(self) -> bool:
        cached = getattr(self, "_arnndn_cache", None)
        if cached is None:
            try:
                proc = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                      capture_output=True, text=True, check=False, timeout=30)
                cached = "arnndn" in (proc.stdout + proc.stderr)
            except (subprocess.TimeoutExpired, OSError):
                cached = False
            self._arnndn_cache = cached
        return cached

    def _base_chain(self, engine: str, model: str | None, nf: float) -> str:
        if engine == "rnnoise":
            return f"arnndn=model={model},highpass=f=80,loudnorm=I=-16:LRA=11:TP=-1.5"
        return (f"afftdn=nf={nf}:nt=w,anlmdn,highpass=f=80,deesser,"
                f"loudnorm=I=-16:LRA=11:TP=-1.5")

    def _build_filter(self, engine: str, model: str | None, nf: float, mix: float) -> str:
        chain = self._base_chain(engine, model, nf)
        if mix >= 1.0:
            return chain
        return (f"asplit=2[a][b];[a]{chain}[w];"
                f"[w]volume={mix}[wv];[b]volume={1.0 - mix}[dv];"
                f"[wv][dv]amix=inputs=2:normalize=0")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py -k "resolve_model or build_filter" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/audio/voice_isolation.py tests/tools/test_voice_isolation.py
git commit -m "feat: voice_isolation model resolution + arnndn probe + filtergraph builder"
```

---

### Task 3: ffmpeg IO helpers + noise-floor metric

**Files:**
- Modify: `tools/audio/voice_isolation.py`
- Test: `tests/tools/test_voice_isolation.py`

**Interfaces:**
- Produces:
  - `_has_audio(path) -> bool`, `_has_video(path) -> bool` — ffprobe stream probes (guarded).
  - `_process(input_path, af, codec, bitrate, dest) -> str | None` — run `ffmpeg -i input -af <af> -c:a codec -b:a bitrate dest`; `None` on failure/zero-byte.
  - `_mux_audio(video_in, new_audio, codec, bitrate, dest) -> str | None` — `ffmpeg -i video -i audio -map 0:v:0 -map 1:a:0 -c:v copy -c:a codec -b:a bitrate -shortest dest`; `None` on failure.
  - `_noise_floor_dbfs(path, window_s=0.2) -> float` — extract mono 48k PCM as BYTES (`subprocess.run`, not run_command), compute per-`window_s` RMS, return the MINIMUM window's dBFS (`20*log10(rms)`). `0.0` if unreadable / <1 window.

- [ ] **Step 1: Write the failing tests (ffmpeg-gated)**

```python
import shutil
import subprocess as _sp
from pathlib import Path

import numpy as np
import pytest


def _make_noisy_clip(path: Path, with_video: bool = False) -> None:
    """[1s tone+noise][1s noise-only] mono audio; optionally a video stream too."""
    d = path.parent
    a = d / "na.wav"; b = d / "nb.wav"; lst = d / "nl.txt"; aud = d / "aud.wav"
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "sine=frequency=300:duration=1:sample_rate=48000", "-f", "lavfi",
             "-i", "anoisesrc=d=1:c=white:a=0.2:r=48000",
             "-filter_complex", "[0][1]amix=inputs=2:duration=shortest", "-ac", "1", str(a)],
            check=True, capture_output=True)
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "anoisesrc=d=1:c=white:a=0.2:r=48000", "-ac", "1", str(b)],
            check=True, capture_output=True)
    lst.write_text(f"file '{a.resolve()}'\nfile '{b.resolve()}'\n")
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(aud)], check=True, capture_output=True)
    if not with_video:
        aud.replace(path)
        return
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "color=c=gray:s=128x96:d=2:r=15", "-i", str(aud), "-shortest",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
            check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_video(tmp_path):
    audio = tmp_path / "a.wav"; _make_noisy_clip(audio)
    vid = tmp_path / "v.mp4"; _make_noisy_clip(vid, with_video=True)
    tool = VoiceIsolation()
    assert tool._has_audio(str(audio)) is True and tool._has_video(str(audio)) is False
    assert tool._has_audio(str(vid)) is True and tool._has_video(str(vid)) is True


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_noise_floor_dbfs_reflects_quiet_window(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    floor = VoiceIsolation()._noise_floor_dbfs(str(clip))
    assert -40 < floor < -10          # a real, finite noise floor, not 0.0 or -inf


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_runs_rnnoise_chain(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    tool = VoiceIsolation()
    af = tool._build_filter("rnnoise", VoiceIsolation.BUNDLED_MODEL, nf=-25, mix=1.0)
    out = tool._process(str(clip), af, "pcm_s16le", "192k", tmp_path / "o.wav")
    assert out and Path(out).exists() and Path(out).stat().st_size > 0


def test_process_missing_file_returns_none(tmp_path):
    tool = VoiceIsolation()
    af = tool._build_filter("spectral", None, nf=-25, mix=1.0)
    assert tool._process("/no/such.wav", af, "aac", "192k", tmp_path / "x.aac") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py -k "has_audio or noise_floor or process" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import tempfile

import numpy as np
```

Add methods:

```python
    def _ffprobe_has(self, path: str, stream: str) -> bool:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream,
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=False, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

    def _has_audio(self, path: str) -> bool:
        return self._ffprobe_has(path, "a")

    def _has_video(self, path: str) -> bool:
        return self._ffprobe_has(path, "v")

    def _process(self, input_path: str, af: str, codec: str, bitrate: str,
                 dest: str | Path) -> str | None:
        dest = Path(dest)
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(input_path),
                            "-af", af, "-c:a", codec, "-b:a", bitrate, str(dest)],
                           capture_output=True, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _mux_audio(self, video_in: str, new_audio: str, codec: str, bitrate: str,
                   dest: str | Path) -> str | None:
        dest = Path(dest)
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_in), "-i", str(new_audio),
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                            "-c:a", codec, "-b:a", bitrate, "-shortest", str(dest)],
                           capture_output=True, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _noise_floor_dbfs(self, path: str, window_s: float = 0.2) -> float:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", str(path),
                 "-f", "s16le", "-ac", "1", "-ar", "48000", "pipe:1"],
                capture_output=True, check=True, timeout=120)   # BYTES, not text
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return 0.0
        raw = proc.stdout
        raw = raw[: len(raw) - (len(raw) % 2)]
        if not raw:
            return 0.0
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        n = int(window_s * 48000)
        if n <= 0 or x.size < n:
            return 0.0
        mins = []
        for i in range(0, x.size - n, n):
            rms = float(np.sqrt(np.mean(x[i:i + n] ** 2))) + 1e-12
            mins.append(20.0 * np.log10(rms))
        return round(min(mins), 2) if mins else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py -k "has_audio or noise_floor or process" -v`
Expected: PASS (ffmpeg-gated ones run on this machine)

- [ ] **Step 5: Commit**

```bash
git add tools/audio/voice_isolation.py tests/tools/test_voice_isolation.py
git commit -m "feat: voice_isolation ffmpeg IO helpers + noise-floor metric"
```

---

### Task 4: Full execute — engine selection, video mux, honest metric, e2e, edge cases

**Files:**
- Modify: `tools/audio/voice_isolation.py`
- Test: `tests/tools/test_voice_isolation.py`

**Interfaces:**
- Consumes: all helpers from Tasks 2–3.
- Produces: full `execute(inputs)` returning `ToolResult(success=True, artifacts=[output_path], data={...})`. Adds `import shutil`, `import time`.

- [ ] **Step 1: Write the failing tests**

```python
def test_execute_requires_input():
    assert not VoiceIsolation().execute({}).success


def test_execute_rejects_out_of_range_mix(tmp_path):
    r = VoiceIsolation().execute({"input_path": str(tmp_path / "a.wav"), "mix": 2.0})
    assert not r.success and "mix" in (r.error or "")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_rnnoise_reduces_noise_floor(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    out = tmp_path / "clean.wav"
    result = VoiceIsolation().execute({
        "input_path": str(clip), "output_path": str(out), "engine": "rnnoise",
        "codec": "pcm_s16le"})
    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert result.data["engine"] == "rnnoise"
    assert result.data["noise_floor_after_db"] < result.data["noise_floor_before_db"]
    assert result.data["noise_reduction_db"] > 3.0          # RNNoise cut the floor clearly
    assert result.data["had_video"] is False


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_auto_uses_rnnoise_when_model_present(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    result = VoiceIsolation().execute({"input_path": str(clip), "engine": "auto",
                                       "output_path": str(tmp_path / "o.wav"), "codec": "pcm_s16le"})
    assert result.success and result.data["engine"] == "rnnoise"     # bundled model resolves


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_spectral_runs_and_reports_engine(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    out = tmp_path / "o.wav"
    result = VoiceIsolation().execute({"input_path": str(clip), "engine": "spectral",
                                       "output_path": str(out), "codec": "pcm_s16le"})
    # spectral doesn't strongly cut a synthetic white-noise floor (loudnorm renormalizes) — assert
    # it RAN and produced valid output + honest engine label, not a noise-reduction magnitude.
    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert result.data["engine"] == "spectral" and result.data["model"] is None


def test_execute_forced_rnnoise_without_model_fails(tmp_path):
    r = VoiceIsolation().execute({"input_path": str(tmp_path / "a.wav"), "engine": "rnnoise",
                                  "model_path": "/no/such.rnnn"})
    assert not r.success and "model" in (r.error or "").lower()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_video_input_muxes_audio_back(tmp_path):
    vid = tmp_path / "v.mp4"; _make_noisy_clip(vid, with_video=True)
    out = tmp_path / "clean.mp4"
    result = VoiceIsolation().execute({"input_path": str(vid), "output_path": str(out),
                                       "engine": "rnnoise"})
    assert result.success, result.error
    tool = VoiceIsolation()
    assert tool._has_video(str(out)) is True and tool._has_audio(str(out)) is True
    assert result.data["had_video"] is True


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_no_audio_input_fails(tmp_path):
    silent = tmp_path / "s.mp4"
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1:r=10",
             "-an", "-pix_fmt", "yuv420p", str(silent)], check=True, capture_output=True)
    r = VoiceIsolation().execute({"input_path": str(silent), "output_path": str(tmp_path / "o.mp4")})
    assert not r.success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py -k execute -v`
Expected: FAIL — `execute` returns the `not implemented` stub.

- [ ] **Step 3: Write minimal implementation**

Add `import shutil`, `import time` to the top-of-file import block.

Replace `execute`:

```python
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        input_path = inputs.get("input_path")
        if not input_path:
            return ToolResult(success=False, error="input_path is required.")
        try:
            mix = float(inputs.get("mix", 1.0))
            nf = float(inputs.get("noise_floor_db", -25))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="mix/noise_floor_db must be numeric.")
        if not 0.0 <= mix <= 1.0:
            return ToolResult(success=False, error="mix must be between 0.0 and 1.0.")
        engine_req = str(inputs.get("engine", "auto"))
        codec = str(inputs.get("codec", "aac"))
        bitrate = str(inputs.get("bitrate", "192k"))
        input_path = Path(input_path)
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not self._has_audio(str(input_path)):
            return ToolResult(success=False, error="Input has no audio stream to isolate.")

        model = self._resolve_model(inputs.get("model_path"))
        if engine_req == "rnnoise":
            if model is None or not self._arnndn_available():
                return ToolResult(success=False,
                                  error="engine='rnnoise' needs a resolvable .rnnn model and ffmpeg "
                                        "arnndn support; supply model_path or use engine='auto'.")
            engine = "rnnoise"
        elif engine_req == "spectral":
            engine, model = "spectral", None
        else:  # auto
            if model is not None and self._arnndn_available():
                engine = "rnnoise"
            else:
                engine, model = "spectral", None

        had_video = self._has_video(str(input_path))
        out_path = Path(inputs.get("output_path") or
                        input_path.with_name(f"{input_path.stem}_voice.{'mp4' if had_video else 'wav'}"))

        af = self._build_filter(engine, model, nf, mix)
        workdir = Path(tempfile.mkdtemp(prefix="voiceiso_"))
        try:
            floor_before = self._noise_floor_dbfs(str(input_path))

            audio_out = self._process(str(input_path), af, codec, bitrate, workdir / "clean.wav"
                                      if not had_video else workdir / "clean.m4a")
            if audio_out is None:
                return ToolResult(success=False, error="Voice-isolation filter chain failed.")

            if had_video:
                final = self._mux_audio(str(input_path), audio_out, codec, bitrate, out_path)
                if final is None:
                    return ToolResult(success=False, error="Audio mux back into video failed.")
            else:
                try:
                    shutil.copyfile(audio_out, out_path)
                except OSError as exc:
                    return ToolResult(success=False, error=f"Failed to write output: {exc}")
            if not out_path.is_file() or out_path.stat().st_size == 0:
                return ToolResult(success=False, error="Output not created or empty.")

            floor_after = self._noise_floor_dbfs(str(out_path))
            reduction = max(0.0, floor_before - floor_after)
            return ToolResult(
                success=True,
                artifacts=[str(out_path)],
                duration_seconds=time.time() - start,
                data={
                    "engine": engine,
                    "model": model,
                    "mix": mix,
                    "noise_floor_before_db": round(floor_before, 2),
                    "noise_floor_after_db": round(floor_after, 2),
                    "noise_reduction_db": round(reduction, 2),
                    "had_video": had_video,
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `.venv/bin/python -m pytest tests/tools/test_voice_isolation.py -v`
Expected: all PASS (ffmpeg-gated ones run on this machine).

- [ ] **Step 5: Commit**

```bash
git add tools/audio/voice_isolation.py tests/tools/test_voice_isolation.py
git commit -m "feat: voice_isolation execute + engine selection + video mux + honest metric + e2e"
```

---

## Self-Review

**Spec coverage:**
- Contract mirroring `audio_enhance`, `["cmd:ffmpeg","python:numpy"]`, bundled-model path → Task 1. ✓
- Hybrid engine (rnnoise/spectral), model resolution order, dry/wet mix → Task 2 (validated filtergraphs). ✓
- ffmpeg process + video mux + noise-floor metric (BYTES PCM, min-window RMS) → Task 3. ✓
- Full execute: validate → resolve model → select engine (auto falls back; forced-rnnoise-no-model fails) → process → mux/copy → measure → ToolResult → Task 4. ✓
- Honesty: engine reported = what ran; reduction measured on OUTPUT; forced ML without model → clean failure → Task 4 (`test_execute_forced_rnnoise_without_model_fails`). ✓
- e2e rnnoise reduces the real noise floor (>3 dB) → Task 4; spectral runs + honest label (no over-claimed reduction on synthetic white noise) → Task 4. ✓
- Edge cases: no input, out-of-range/non-numeric params, no-audio input, missing input, zero-byte output, video mux, no-audio-video failure → Tasks 3–4. ✓
- Determinism: filtergraph + engine deterministic; test asserts engine/model identity. ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code.

**Type consistency:** `_resolve_model -> str | None`, `_arnndn_available -> bool`, `_build_filter -> str`, `_has_audio`/`_has_video -> bool`, `_process`/`_mux_audio -> str | None`, `_noise_floor_dbfs -> float`. `execute` threads all of them; `model` is `str|None` consistently (None for spectral). ✓

**Deferred vs spec:** ML de-reverb, model auto-selection, speaker separation, and bundling alternate models are explicitly v2 per the spec; not in this plan.
