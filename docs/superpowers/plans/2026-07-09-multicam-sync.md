# Multicam Auto-Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `multicam_sync` tool that synchronizes ≥2 clips by audio cross-correlation and emits a JSON offsets report (per clip: source, offset_seconds, confidence).

**Architecture:** One `BaseTool` subclass in `tools/audio/multicam_sync.py`, contract-styled after `tools/analysis/audio_energy.py`. It extracts low-rate mono PCM from each clip via ffmpeg (`-f s16le -ac 1 -ar R`), cross-correlates each clip against a pivot with pure-numpy FFT to get lag + confidence, re-baselines onto a common timeline (auto = earliest-start clip as reference so offsets ≥ 0, or an explicit `reference_index`), and writes a deterministic JSON report. No rendering.

**Tech Stack:** Python 3.14, ffmpeg/ffprobe (subprocess via `BaseTool.run_command`), numpy 2.4.4 (FFT cross-correlation; scipy is NOT available), pytest.

## Global Constraints

- Tool inherits `tools/base_tool.py` `BaseTool` with full contract fields in the style of `tools/analysis/audio_energy.py`. `execute(self, inputs: dict[str, Any]) -> ToolResult`.
- `dependencies = ["cmd:ffmpeg", "python:numpy"]` — the CHECKED prefixes (`BaseTool.check_dependencies` honors only `cmd:`/`env:`/`python:`; `binary:` is silently unchecked). Precedent: `tools/video/green_screen_composite.py`.
- Cross-correlation is pure numpy `np.fft.rfft`/`irfft` — do NOT import scipy (not installed).
- **Never fabricate:** a clip with no audio / failed extraction goes into `skipped` with a reason and `offset_seconds: null` — never an invented number. If < 2 clips remain correlatable → `success=False`. All subprocess calls guarded so a missing binary / bad file returns `success=False`, never crashes.
- **Sign convention is locked by test, not prose:** the core test builds a synthetic pair with a KNOWN earlier-starting clip and asserts the recovered offsets (earliest = reference = 0, the later clip's offset = the known positive delay). If the sign inverts, that test fails.
- Report JSON written with `sort_keys=True` → identical inputs produce byte-identical output (determinism).
- No new third-party deps. Tests in `tests/tools/test_multicam_sync.py`, import from `tools.audio.multicam_sync`; correlation tests build numpy arrays directly (no media), ffmpeg-gated tests synthesize tiny clips.

---

### Task 1: Tool skeleton, contract, and registry discovery

**Files:**
- Create: `tools/audio/multicam_sync.py`
- Test: `tests/tools/test_multicam_sync.py`

**Interfaces:**
- Produces: `class MulticamSync(BaseTool)` with contract fields and `input_schema`. `execute(self, inputs)` returns `ToolResult(success=False, error="not implemented")` for now.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_multicam_sync.py
from __future__ import annotations

from tools.audio.multicam_sync import MulticamSync
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = MulticamSync()
    assert tool.name == "multicam_sync"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "analysis"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "multicam_sync" in tool.capabilities
    assert tool.input_schema["required"] == ["clips"]
    assert tool.input_schema["properties"]["clips"]["minItems"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py::test_contract_fields_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.audio.multicam_sync'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/audio/multicam_sync.py
"""Multicam auto-sync — align clips by audio cross-correlation.

Extracts low-rate mono PCM from each clip, cross-correlates each against a
pivot with a pure-numpy FFT to recover a lag + confidence, re-baselines onto a
common timeline (auto: earliest-start clip = reference, offsets >= 0; or an
explicit reference_index), and emits a JSON offsets report. No rendering.
"""

from __future__ import annotations

from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolResult,
    ToolStability,
    ToolTier,
)


class MulticamSync(BaseTool):
    name = "multicam_sync"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "ffmpeg+numpy"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:numpy"]
    install_instructions = "Install FFmpeg (brew install ffmpeg) and numpy (pip install numpy)."
    agent_skills = ["ffmpeg"]

    capabilities = ["multicam_sync", "audio_sync", "waveform_alignment"]

    input_schema = {
        "type": "object",
        "required": ["clips"],
        "properties": {
            "clips": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "reference_index": {"type": "integer", "minimum": 0},
            "sample_rate": {"type": "integer", "default": 8000, "minimum": 1000},
            "window_seconds": {"type": "number", "default": 60, "minimum": 1},
            "min_confidence": {"type": "number", "default": 0.1, "minimum": 0, "maximum": 1},
            "output_path": {"type": "string"},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py::test_contract_fields_present -v`
Expected: PASS

- [ ] **Step 5: Verify discovery, then commit**

Run: `.venv/bin/python -c "import tools.audio.multicam_sync as m; print(m.MulticamSync().name)"`
Expected: prints `multicam_sync`

```bash
git add tools/audio/multicam_sync.py tests/tools/test_multicam_sync.py
git commit -m "feat: multicam_sync tool skeleton + contract"
```

---

### Task 2: FFT cross-correlation core (`_xcorr_lag`)

**Files:**
- Modify: `tools/audio/multicam_sync.py`
- Test: `tests/tools/test_multicam_sync.py`

**Interfaces:**
- Produces: `MulticamSync._xcorr_lag(ref, other, sample_rate) -> tuple[float, float]` — returns `(lag_seconds, confidence)`. `lag_seconds` is the offset at which `other` aligns to `ref`: for `other` = `ref` delayed by D seconds (D seconds of silence prepended), the returned lag is `+D`. `confidence` = normalized correlation peak in `[0, 1]` (1.0 for identical, ~0 for uncorrelated). Both inputs are 1-D float32 numpy arrays.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np


def _click_signal(sample_rate: int, dur: float = 1.0, at: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * 440 * t) * np.exp(-((t - at) ** 2) / 0.001)).astype(np.float32)


def test_xcorr_recovers_known_delay():
    sr = 8000
    a = _click_signal(sr)
    delay = 0.4
    b = np.concatenate([np.zeros(int(delay * sr), np.float32), a])  # b = a delayed by 0.4s
    lag, conf = MulticamSync()._xcorr_lag(a, b, sr)
    assert abs(lag - delay) < 1.5 / sr        # within ~1 sample
    assert conf > 0.5


def test_xcorr_confidence_low_for_uncorrelated():
    sr = 8000
    a = _click_signal(sr)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(a)).astype(np.float32)
    _, conf = MulticamSync()._xcorr_lag(a, noise, sr)
    assert conf < 0.3


def test_xcorr_zero_lag_for_identical():
    sr = 8000
    a = _click_signal(sr)
    lag, conf = MulticamSync()._xcorr_lag(a, a, sr)
    assert abs(lag) < 1.5 / sr
    assert conf > 0.99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k xcorr -v`
Expected: FAIL with `AttributeError: ... '_xcorr_lag'`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import numpy as np
```

Add method to the class (verified empirically to recover +0.4s for a 0.4s-delayed copy):

```python
    def _xcorr_lag(self, ref: np.ndarray, other: np.ndarray, sample_rate: int) -> tuple[float, float]:
        """FFT cross-correlation. Returns (lag_seconds, confidence in [0,1]).

        lag is the shift at which `other` aligns to `ref`: if `other` is `ref`
        delayed by D seconds, lag == +D.
        """
        ref = np.asarray(ref, dtype=np.float64)
        other = np.asarray(other, dtype=np.float64)
        if ref.size == 0 or other.size == 0:
            return 0.0, 0.0
        n = 1 << int(np.ceil(np.log2(ref.size + other.size)))
        fa = np.fft.rfft(ref, n)
        fb = np.fft.rfft(other, n)
        cc = np.fft.irfft(fb * np.conj(fa), n)
        # Reassemble into full correlation with zero-lag centered.
        cc = np.concatenate([cc[-(ref.size - 1):], cc[:other.size]])
        lags = np.arange(-(ref.size - 1), other.size)
        peak_idx = int(np.argmax(cc))
        lag_samples = int(lags[peak_idx])
        energy = float(np.sqrt(np.sum(ref ** 2) * np.sum(other ** 2)))
        confidence = 0.0 if energy == 0.0 else float(max(0.0, cc[peak_idx] / energy))
        return lag_samples / sample_rate, min(1.0, confidence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k xcorr -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/audio/multicam_sync.py tests/tools/test_multicam_sync.py
git commit -m "feat: multicam_sync FFT cross-correlation core"
```

---

### Task 3: PCM extraction and audio probe

**Files:**
- Modify: `tools/audio/multicam_sync.py`
- Test: `tests/tools/test_multicam_sync.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `MulticamSync._has_audio(path) -> bool` — ffprobe for an audio stream; `False` on missing binary / error.
  - `MulticamSync._extract_samples(path, sample_rate, window_seconds) -> np.ndarray | None` — mono float32 samples in `[-1, 1]` via ffmpeg `-f s16le -ac 1 -ar <rate>`, bounded to the first `window_seconds`; `None` (never a fabricated array) if the clip has no audio or ffmpeg fails/produces empty output.

- [ ] **Step 1: Write the failing tests (ffmpeg-gated)**

```python
import shutil
import subprocess as _sp
from pathlib import Path

import pytest


def _make_tone_clip(path: Path, sr: int = 8000, dur: float = 1.0) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={dur}:sample_rate={sr}",
        str(path),
    ], check=True, capture_output=True)


def _make_silent_video_no_audio(path: Path) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=160x120:rate=15:duration=1",
        "-an", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_extract_samples_returns_float_array(tmp_path):
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, sr=8000, dur=1.0)
    samples = MulticamSync()._extract_samples(str(clip), 8000, 60)
    assert samples is not None
    assert samples.dtype == np.float32
    assert 7000 < samples.size <= 8000          # ~1s at 8kHz
    assert float(np.max(np.abs(samples))) <= 1.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_extract_samples_window_bounds_length(tmp_path):
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, sr=8000, dur=3.0)
    samples = MulticamSync()._extract_samples(str(clip), 8000, 1.0)   # 1s window of a 3s clip
    assert samples is not None and samples.size <= 8000 + 10


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_true_false(tmp_path):
    tone = tmp_path / "tone.wav"
    _make_tone_clip(tone)
    silent = tmp_path / "silent.mp4"
    _make_silent_video_no_audio(silent)
    tool = MulticamSync()
    assert tool._has_audio(str(tone)) is True
    assert tool._has_audio(str(silent)) is False


def test_extract_samples_missing_file_returns_none():
    assert MulticamSync()._extract_samples("/no/such/file.wav", 8000, 60) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k "extract or has_audio" -v`
Expected: FAIL with `AttributeError: ... '_extract_samples'` / `'_has_audio'`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import subprocess
```

Add methods:

```python
    def _has_audio(self, path: str) -> bool:
        try:
            proc = self.run_command([
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
            ])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

    def _extract_samples(self, path: str, sample_rate: int, window_seconds: float):
        if not self._has_audio(path):
            return None
        try:
            proc = self.run_command([
                "ffmpeg", "-v", "quiet", "-t", f"{float(window_seconds):.3f}",
                "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
            ])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        raw = proc.stdout
        if isinstance(raw, str):
            raw = raw.encode("latin-1", "ignore")
        if not raw:
            return None
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
```

Note: `run_command` captures text by default. To get raw bytes for PCM, the extraction must capture binary. If `run_command` forces text mode and corrupts the PCM bytes, call `subprocess.run(cmd, capture_output=True)` directly here (bytes stdout) instead of `run_command`, still wrapped in the same `try/except`. Verify which the base class does before relying on it (see Step 3b).

- [ ] **Step 3b: Verify PCM byte integrity**

Run:
```
.venv/bin/python -c "
import subprocess, numpy as np, tempfile, os
from tools.audio.multicam_sync import MulticamSync
d=tempfile.mkdtemp(); c=os.path.join(d,'t.wav')
subprocess.run(['ffmpeg','-y','-f','lavfi','-i','sine=frequency=440:duration=1:sample_rate=8000',c],check=True,capture_output=True)
s=MulticamSync()._extract_samples(c,8000,60)
print('size',None if s is None else s.size,'max',None if s is None else float(np.max(np.abs(s))))
"
```
Expected: `size ~8000 max <=1.0`. If size is 0/None or max is garbage, switch `_extract_samples` to `subprocess.run(cmd, capture_output=True)` (bytes) instead of `run_command`, then re-run.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k "extract or has_audio" -v`
Expected: PASS (ffmpeg-gated ones run on this machine)

- [ ] **Step 5: Commit**

```bash
git add tools/audio/multicam_sync.py tests/tools/test_multicam_sync.py
git commit -m "feat: multicam_sync PCM extraction + audio probe"
```

---

### Task 4: Offset re-baseline and report assembly

**Files:**
- Modify: `tools/audio/multicam_sync.py`
- Test: `tests/tools/test_multicam_sync.py`

**Interfaces:**
- Consumes: `_xcorr_lag` from Task 2.
- Produces:
  - `_pairwise_offsets(samples_list, sample_rate) -> list[tuple[float, float]]` — for each clip, `(lag_vs_pivot, confidence)` where the pivot is `samples_list[0]` (pivot returns `(0.0, 1.0)`). `lag_vs_pivot` is `_xcorr_lag(pivot, clip)`.
  - `_rebaseline(pairwise, reference_index=None) -> tuple[int, list[float]]` — convert pivot-relative lags into common-timeline offsets. `start_i = -lag_i` (a clip whose content is delayed vs the pivot started earlier, so its t=0 sits earlier). If `reference_index is None` (auto): reference = the clip with the smallest `start_i` (earliest real start); return offsets `start_i - min_start` (all ≥ 0). If `reference_index` given: reference = that clip; offsets = `start_i - start_ref` (the reference is 0; others may be negative).
  - `_to_report(clips, reference_index, offsets, confidences, skipped, params) -> dict` — assemble the report dict.

- [ ] **Step 1: Write the failing tests**

```python
def test_rebaseline_auto_picks_earliest_and_offsets_nonneg():
    tool = MulticamSync()
    # pivot=clip0. lag_i = _xcorr_lag(pivot, clip_i): clip1 delayed +0.4 vs pivot,
    # clip2 delayed -0.2 vs pivot (i.e. clip2 started later than pivot).
    pairwise = [(0.0, 1.0), (0.4, 0.9), (-0.2, 0.8)]
    ref, offsets = tool._rebaseline(pairwise, reference_index=None)
    # start_i = -lag_i => [0.0, -0.4, 0.2]; earliest = clip1 (start -0.4) => ref=1
    assert ref == 1
    assert offsets == pytest.approx([0.4, 0.0, 0.6])
    assert min(offsets) == pytest.approx(0.0)
    assert all(o >= -1e-9 for o in offsets)


def test_rebaseline_explicit_reference_is_zero_others_relative():
    tool = MulticamSync()
    pairwise = [(0.0, 1.0), (0.4, 0.9), (-0.2, 0.8)]   # start_i = [0.0, -0.4, 0.2]
    ref, offsets = tool._rebaseline(pairwise, reference_index=2)
    assert ref == 2
    assert offsets[2] == pytest.approx(0.0)
    # relative spacing preserved: start_i - start_2  => [-0.2, -0.6, 0.0]
    assert offsets == pytest.approx([-0.2, -0.6, 0.0])


def test_to_report_shape():
    tool = MulticamSync()
    report = tool._to_report(
        clips=["a.mp4", "b.mp4"], reference_index=0,
        offsets=[0.0, 0.4], confidences=[1.0, 0.9], skipped=[],
        params={"sample_rate": 8000, "window_seconds": 60, "min_confidence": 0.1},
    )
    assert report["reference_index"] == 0
    assert report["reference_source"] == "a.mp4"
    assert [o["index"] for o in report["offsets"]] == [0, 1]
    assert report["offsets"][1]["offset_seconds"] == pytest.approx(0.4)
    assert report["offsets"][1]["low_confidence"] is False
    assert report["sample_rate"] == 8000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k "rebaseline or to_report" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
    def _pairwise_offsets(self, samples_list: list, sample_rate: int) -> list[tuple[float, float]]:
        pivot = samples_list[0]
        out = [(0.0, 1.0)]
        for s in samples_list[1:]:
            out.append(self._xcorr_lag(pivot, s, sample_rate))
        return out

    def _rebaseline(self, pairwise: list[tuple[float, float]],
                    reference_index: int | None = None) -> tuple[int, list[float]]:
        starts = [-lag for (lag, _conf) in pairwise]
        if reference_index is None:
            min_start = min(starts)
            ref = starts.index(min_start)
            offsets = [s - min_start for s in starts]
        else:
            ref = reference_index
            base = starts[ref]
            offsets = [s - base for s in starts]
        return ref, offsets

    def _to_report(self, clips: list[str], reference_index: int, offsets: list[float],
                   confidences: list[float], skipped: list[dict], params: dict) -> dict:
        min_conf = float(params.get("min_confidence", 0.1))
        entries = []
        for i, src in enumerate(clips):
            if any(sk["index"] == i for sk in skipped):
                entries.append({"index": i, "source": src, "offset_seconds": None,
                                "confidence": None, "low_confidence": True})
            else:
                conf = round(float(confidences[i]), 4)
                entries.append({"index": i, "source": src,
                                "offset_seconds": round(float(offsets[i]), 4),
                                "confidence": conf, "low_confidence": conf < min_conf})
        usable = [e["confidence"] for e in entries if e["confidence"] is not None]
        return {
            "version": "1.0",
            "reference_index": reference_index,
            "reference_source": clips[reference_index],
            "sample_rate": int(params.get("sample_rate", 8000)),
            "window_seconds": float(params.get("window_seconds", 60)),
            "offsets": entries,
            "skipped": skipped,
            "max_confidence": round(max(usable), 4) if usable else 0.0,
            "min_confidence_observed": round(min(usable), 4) if usable else 0.0,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k "rebaseline or to_report" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/audio/multicam_sync.py tests/tools/test_multicam_sync.py
git commit -m "feat: multicam_sync offset rebaseline + report assembly"
```

---

### Task 5: Full execute — orchestration, skipped clips, edge cases, e2e

**Files:**
- Modify: `tools/audio/multicam_sync.py`
- Test: `tests/tools/test_multicam_sync.py`

**Interfaces:**
- Consumes: `_extract_samples`, `_pairwise_offsets`, `_rebaseline`, `_to_report`.
- Produces: full `execute(inputs)` returning `ToolResult(success=True, artifacts=[report_json], data={...})`.

Note on skipped clips: when a clip is skipped, correlation runs only over the non-skipped clips. Build a `usable` index list; run `_pairwise_offsets`/`_rebaseline` over usable clips; then map offsets/confidences back to original clip indices (skipped clips get null). The pivot is the first USABLE clip. When `reference_index` is given, it must refer to a usable clip (validate).

- [ ] **Step 1: Write the failing tests**

```python
import json


def _write_report_and_load(tool, inputs):
    result = tool.execute(inputs)
    return result


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_end_to_end_recovers_offset(tmp_path):
    sr = 8000
    early = tmp_path / "early.wav"
    late = tmp_path / "late.wav"
    _make_tone_clip(early, sr=sr, dur=2.0)
    # `late` = 0.5s silence + the same tone => it started 0.5s EARLIER in real time?
    # No: prepending silence means its shared content occurs 0.5s later, i.e. `late`
    # started recording 0.5s BEFORE `early`. So `late` is the earliest => reference,
    # and `early`'s offset should be +0.5.
    _sp.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono", "-t", "0.5",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration=2:sample_rate={sr}",
        "-filter_complex", "[0][1]concat=n=2:v=0:a=1", str(late),
    ], check=True, capture_output=True)
    out = tmp_path / "sync.json"
    result = MulticamSync().execute({
        "clips": [str(early), str(late)], "sample_rate": sr,
        "window_seconds": 60, "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists()
    report = json.loads(out.read_text())
    # `late` is the earliest-start clip => reference (offset 0).
    ref_src = report["reference_source"]
    assert ref_src == str(late)
    early_entry = next(o for o in report["offsets"] if o["source"] == str(early))
    assert early_entry["offset_seconds"] == pytest.approx(0.5, abs=0.05)
    assert early_entry["confidence"] > 0.3


def test_execute_requires_two_clips():
    result = MulticamSync().execute({"clips": ["only_one.wav"]})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_skips_no_audio_clip(tmp_path):
    a = tmp_path / "a.wav"; b = tmp_path / "b.wav"
    _make_tone_clip(a); _make_tone_clip(b)
    silent = tmp_path / "silent.mp4"
    _make_silent_video_no_audio(silent)
    out = tmp_path / "s.json"
    result = MulticamSync().execute({
        "clips": [str(a), str(b), str(silent)], "output_path": str(out)})
    assert result.success, result.error
    report = json.loads(out.read_text())
    assert any(sk["source"] == str(silent) for sk in report["skipped"])
    silent_entry = next(o for o in report["offsets"] if o["source"] == str(silent))
    assert silent_entry["offset_seconds"] is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_fails_when_fewer_than_two_usable(tmp_path):
    a = tmp_path / "a.wav"; _make_tone_clip(a)
    s1 = tmp_path / "s1.mp4"; s2 = tmp_path / "s2.mp4"
    _make_silent_video_no_audio(s1); _make_silent_video_no_audio(s2)
    result = MulticamSync().execute({"clips": [str(a), str(s1), str(s2)],
                                     "output_path": str(tmp_path / "x.json")})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_is_deterministic(tmp_path):
    a = tmp_path / "a.wav"; b = tmp_path / "b.wav"
    _make_tone_clip(a, dur=1.5); _make_tone_clip(b, dur=1.5)
    o1 = tmp_path / "o1.json"; o2 = tmp_path / "o2.json"
    MulticamSync().execute({"clips": [str(a), str(b)], "output_path": str(o1)})
    MulticamSync().execute({"clips": [str(a), str(b)], "output_path": str(o2)})
    assert o1.read_text() == o2.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -k execute -v`
Expected: FAIL — `execute` returns the `not implemented` stub.

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import json
import time
from pathlib import Path
```

Replace the placeholder `execute`:

```python
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        clips = inputs.get("clips") or []
        if len(clips) < 2:
            return ToolResult(success=False, error="multicam_sync needs at least 2 clips.")

        sample_rate = int(inputs.get("sample_rate", 8000))
        window_seconds = float(inputs.get("window_seconds", 60))
        params = {
            "sample_rate": sample_rate,
            "window_seconds": window_seconds,
            "min_confidence": float(inputs.get("min_confidence", 0.1)),
        }

        # Extract samples; record skips honestly.
        samples_by_index: dict[int, Any] = {}
        skipped: list[dict] = []
        for i, clip in enumerate(clips):
            s = self._extract_samples(clip, sample_rate, window_seconds)
            if s is None or s.size == 0:
                skipped.append({"index": i, "source": clip, "reason": "no audio or extraction failed"})
            else:
                samples_by_index[i] = s

        usable = sorted(samples_by_index.keys())
        if len(usable) < 2:
            return ToolResult(success=False,
                              error="Fewer than 2 clips have usable audio; cannot synchronize.")

        # Optional explicit reference must be a usable clip.
        req_ref = inputs.get("reference_index")
        if req_ref is not None and req_ref not in usable:
            return ToolResult(success=False,
                              error=f"reference_index {req_ref} is not a usable (audio-bearing) clip.")

        usable_samples = [samples_by_index[i] for i in usable]
        pairwise = self._pairwise_offsets(usable_samples, sample_rate)
        local_ref = None if req_ref is None else usable.index(req_ref)
        ref_local, offsets_local = self._rebaseline(pairwise, local_ref)

        # Map usable-local results back to original clip indices.
        offsets_full = [0.0] * len(clips)
        conf_full = [0.0] * len(clips)
        for local_i, orig_i in enumerate(usable):
            offsets_full[orig_i] = offsets_local[local_i]
            conf_full[orig_i] = pairwise[local_i][1]
        reference_index = usable[ref_local]

        report = self._to_report(clips, reference_index, offsets_full, conf_full, skipped, params)

        out_path = Path(inputs.get("output_path") or
                        Path(clips[reference_index]).with_suffix(".sync.json"))
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        usable_confs = [conf_full[i] for i in usable]
        return ToolResult(
            success=True,
            artifacts=[str(out_path)],
            duration_seconds=time.time() - start,
            data={
                "reference_index": reference_index,
                "reference_source": clips[reference_index],
                "sample_rate": sample_rate,
                "window_seconds": window_seconds,
                "offsets": report["offsets"],
                "skipped": skipped,
                "max_confidence": round(max(usable_confs), 4),
                "min_confidence_observed": round(min(usable_confs), 4),
            },
        )
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `.venv/bin/python -m pytest tests/tools/test_multicam_sync.py -v`
Expected: all PASS (ffmpeg-gated ones run on this machine).

- [ ] **Step 5: Commit**

```bash
git add tools/audio/multicam_sync.py tests/tools/test_multicam_sync.py
git commit -m "feat: multicam_sync execute orchestration + edge cases + e2e"
```

---

## Self-Review

**Spec coverage:**
- Contract mirroring `audio_energy`, `["cmd:ffmpeg","python:numpy"]` → Task 1. ✓
- FFT cross-correlation, pure numpy, confidence → Task 2 (+ sign/magnitude locked by `test_xcorr_recovers_known_delay`). ✓
- PCM extraction (`-f s16le -ac 1 -ar`), window bound, `_has_audio` → Task 3 (+ byte-integrity check Step 3b). ✓
- Auto earliest-start reference (offsets ≥ 0) and explicit `reference_index` (ref=0, others may be negative) → Task 4 (both locked by tests). ✓
- Report shape, null for skipped, low_confidence flag, determinism (`sort_keys`) → Tasks 4–5. ✓
- Orchestration: extract → skip honestly → correlate usable → rebaseline → map back → write → ToolResult → Task 5. ✓
- Edge cases: <2 clips, <2 usable, no-audio skip, explicit ref must be usable, never-crash subprocess guards → Tasks 3 & 5. ✓
- e2e recovers a known real delay → Task 5 (`test_execute_end_to_end_recovers_offset`). ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code. Step 3b is a real verification step with a fallback instruction, not a placeholder.

**Type consistency:** `_xcorr_lag -> (float, float)` (Tasks 2,4); `_extract_samples -> np.ndarray | None` (Tasks 3,5); `_pairwise_offsets -> list[(float,float)]` and `_rebaseline -> (int, list[float])` (Tasks 4,5); `_to_report -> dict` consumed by `execute`. `reference_index` local/original mapping handled explicitly in Task 5. ✓

**Deferred vs spec:** Rendering aligned clips and drift correction are explicitly v2 per the spec; not in this plan by design.
