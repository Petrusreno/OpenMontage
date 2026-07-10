# Color Match Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `color_match` tool that matches a target clip's color to a reference (clip or still) via per-channel Reinhard mean+std transfer, applied to the whole clip with ffmpeg `lutrgb`.

**Architecture:** One `BaseTool` subclass in `tools/enhancement/color_match.py`, contract-styled after `tools/enhancement/color_grade.py`. It extracts one frame from the target and one from the reference (ffmpeg), computes per-channel mean/std (PIL+numpy), builds a per-channel clamped affine transform (Reinhard, folded with an `intensity` blend), and applies it to the entire target clip via a single ffmpeg `lutrgb` pass. It re-measures the output frame to report an honest before/after color-delta metric. No new schema, no cv2.

**Tech Stack:** Python 3.14, ffmpeg/ffprobe (subprocess), numpy 2.4.4, PIL 12.2.0, pytest.

## Global Constraints

- Tool inherits `tools/base_tool.py` `BaseTool` with full contract fields in the style of `tools/enhancement/color_grade.py`. `execute(self, inputs: dict[str, Any]) -> ToolResult`.
- `dependencies = ["cmd:ffmpeg", "python:numpy", "python:PIL"]` — the CHECKED prefixes (`BaseTool.check_dependencies` honors only `cmd:`/`env:`/`python:`).
- The transform is per-channel affine `out = gain_c*v + offset_c`, clamped to [0,255] inside the ffmpeg `lutrgb` expression. `intensity` is folded into gain/offset: raw Reinhard `g = s_r/s_t` (guarded), then `gain = 1 + intensity*(g-1)`, `offset = intensity*(m_r - g*m_t)`.
- **Honest guards, never silent, never fabricated:** a flat target channel (`s_t < EPS`) → that channel's raw gain forced to 1.0 (mean-only shift) with a `clamp_notes` entry; a raw gain above `GAIN_MAX` → clamped with a `clamp_notes` entry. `EPS = 1.0`, `GAIN_MAX = 3.0`. Frame-extraction failure → `success=False` naming which side failed (never proceed with invented stats). Missing/zero-byte ffmpeg output → failure even on exit 0.
- All subprocess calls guarded (`CalledProcessError`/`TimeoutExpired`/`OSError`) so a missing binary / bad file returns `success=False`, never a traceback. Boundary-validate inputs before ffmpeg work.
- Determinism: same inputs → identical gains/offsets and identical output size.
- No new third-party deps. Tests in `tests/tools/test_color_match.py`, import from `tools.enhancement.color_match`; math tests use numpy arrays directly, ffmpeg-gated tests synthesize tiny solid-color clips.

---

### Task 1: Tool skeleton, contract, and registry discovery

**Files:**
- Create: `tools/enhancement/color_match.py`
- Test: `tests/tools/test_color_match.py`

**Interfaces:**
- Produces: `class ColorMatch(BaseTool)` with contract fields and `input_schema`. `execute(self, inputs)` returns `ToolResult(success=False, error="not implemented")` for now.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_color_match.py
from __future__ import annotations

from tools.enhancement.color_match import ColorMatch
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = ColorMatch()
    assert tool.name == "color_match"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "enhancement"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "python:PIL" in tool.dependencies
    assert "color_match" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path", "reference_path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py::test_contract_fields_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.enhancement.color_match'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/enhancement/color_match.py
"""Color Match — match a target clip's color to a reference (clip or still).

Extracts one frame from the target and one from the reference, computes a
per-channel Reinhard mean+std transfer (folded with an intensity blend), and
applies it to the whole target clip via a single ffmpeg `lutrgb` pass. Reports
an honest before/after per-channel color-delta metric. Complements color_grade
(subjective looks) with reference-driven matching.
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


class ColorMatch(BaseTool):
    name = "color_match"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "enhancement"
    provider = "ffmpeg+numpy"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:numpy", "python:PIL"]
    install_instructions = "Install FFmpeg (brew install ffmpeg) and pip install numpy pillow."
    agent_skills = ["ffmpeg"]

    capabilities = ["color_match", "color_transfer", "reference_grade"]

    input_schema = {
        "type": "object",
        "required": ["input_path", "reference_path"],
        "properties": {
            "input_path": {"type": "string"},
            "reference_path": {"type": "string"},
            "output_path": {"type": "string"},
            "reference_time": {"type": "number", "minimum": 0},
            "input_time": {"type": "number", "minimum": 0},
            "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
            "codec": {"type": "string", "default": "libx264"},
            "crf": {"type": "integer", "default": 20},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py::test_contract_fields_present -v`
Expected: PASS

- [ ] **Step 5: Verify discovery, then commit**

Run: `.venv/bin/python -c "import tools.enhancement.color_match as m; print(m.ColorMatch().name)"`
Expected: prints `color_match`

```bash
git add tools/enhancement/color_match.py tests/tools/test_color_match.py
git commit -m "feat: color_match tool skeleton + contract"
```

---

### Task 2: Transform math — `_channel_affine` + `_lutrgb_expr`

**Files:**
- Modify: `tools/enhancement/color_match.py`
- Test: `tests/tools/test_color_match.py`

**Interfaces:**
- Produces:
  - `ColorMatch.EPS = 1.0`, `ColorMatch.GAIN_MAX = 3.0` (class attributes).
  - `_channel_affine(m_t, s_t, m_r, s_r, intensity) -> tuple[list[float], list[float], list[dict]]` —
    `m_t,s_t,m_r,s_r` are length-3 sequences (per-channel mean/std, 0–255). Returns
    `(gains[3], offsets[3], clamp_notes)` where `out = gain*v + offset` folds intensity, and
    `clamp_notes` is a list of `{channel, reason}` for any EPS/GAIN_MAX clamp fired.
  - `_lutrgb_expr(gains, offsets) -> str` — an ffmpeg `lutrgb=r=...:g=...:b=...` string with each
    channel `clip(gain*val{+/-}offset,0,255)`.

- [ ] **Step 1: Write the failing tests**

```python
import math


def test_channel_affine_full_intensity_matches_mean():
    tool = ColorMatch()
    m_t, s_t = [50.0, 100.0, 150.0], [20.0, 20.0, 20.0]
    m_r, s_r = [150.0, 80.0, 40.0], [40.0, 10.0, 20.0]
    gains, offsets, notes = tool._channel_affine(m_t, s_t, m_r, s_r, intensity=1.0)
    for c in range(3):
        # matched mean == reference mean
        assert math.isclose(gains[c] * m_t[c] + offsets[c], m_r[c], rel_tol=1e-6)
    assert notes == []


def test_channel_affine_zero_intensity_is_identity():
    tool = ColorMatch()
    gains, offsets, notes = tool._channel_affine(
        [50, 100, 150], [20, 20, 20], [150, 80, 40], [40, 10, 20], intensity=0.0)
    assert gains == [1.0, 1.0, 1.0]
    assert offsets == [0.0, 0.0, 0.0]


def test_channel_affine_half_intensity_when_std_equal():
    tool = ColorMatch()
    # s_t == s_r => raw gain 1.0; half intensity => matched mean = midpoint
    gains, offsets, _ = tool._channel_affine(
        [50.0], [20.0], [150.0], [20.0], intensity=0.5)
    assert math.isclose(gains[0] * 50.0 + offsets[0], 100.0, rel_tol=1e-6)  # (50+150)/2


def test_channel_affine_flat_channel_forces_gain_one():
    tool = ColorMatch()
    gains, offsets, notes = tool._channel_affine(
        [50.0], [0.0], [150.0], [30.0], intensity=1.0)   # s_t == 0 (flat)
    assert gains[0] == 1.0                    # gain not exploded
    assert math.isclose(gains[0] * 50.0 + offsets[0], 150.0, rel_tol=1e-6)  # mean still matched
    assert any(n["reason"] == "flat_channel" for n in notes)


def test_channel_affine_gain_clamped_to_max():
    tool = ColorMatch()
    # tiny s_t would give gain 100; clamp to GAIN_MAX 3.0
    gains, offsets, notes = tool._channel_affine(
        [50.0], [1.0], [150.0], [100.0], intensity=1.0)
    assert gains[0] == 3.0
    assert any(n["reason"] == "gain_clamped" for n in notes)


def test_lutrgb_expr_format():
    tool = ColorMatch()
    expr = tool._lutrgb_expr([1.0, 1.0, 1.0], [128.0, -16.0, -111.0])
    assert expr.startswith("lutrgb=")
    assert "r='clip(1.000000*val+128.000000,0,255)'" in expr
    assert "g='clip(1.000000*val-16.000000,0,255)'" in expr
    assert "b='clip(1.000000*val-111.000000,0,255)'" in expr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py -k "affine or lutrgb" -v`
Expected: FAIL with `AttributeError: ... '_channel_affine'`

- [ ] **Step 3: Write minimal implementation**

Add class attributes and methods to `ColorMatch`:

```python
    EPS = 1.0
    GAIN_MAX = 3.0

    def _channel_affine(self, m_t, s_t, m_r, s_r, intensity: float):
        gains: list[float] = []
        offsets: list[float] = []
        notes: list[dict] = []
        channel_names = ["r", "g", "b"]
        for c in range(3):
            mt, st_, mr, sr = float(m_t[c]), float(s_t[c]), float(m_r[c]), float(s_r[c])
            if st_ < self.EPS:
                g = 1.0
                notes.append({"channel": channel_names[c], "reason": "flat_channel"})
            else:
                g = sr / st_
                if g > self.GAIN_MAX:
                    g = self.GAIN_MAX
                    notes.append({"channel": channel_names[c], "reason": "gain_clamped"})
                elif g < 0.0:
                    g = 0.0
            gain = 1.0 + intensity * (g - 1.0)
            offset = intensity * (mr - g * mt)
            gains.append(gain)
            offsets.append(offset)
        return gains, offsets, notes

    def _lutrgb_expr(self, gains: list[float], offsets: list[float]) -> str:
        channels = ["r", "g", "b"]
        parts = [
            f"{channels[c]}='clip({gains[c]:.6f}*val{offsets[c]:+.6f},0,255)'"
            for c in range(3)
        ]
        return "lutrgb=" + ":".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py -k "affine or lutrgb" -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/enhancement/color_match.py tests/tools/test_color_match.py
git commit -m "feat: color_match Reinhard per-channel transform + lutrgb expr"
```

---

### Task 3: Frame extraction and per-channel stats

**Files:**
- Modify: `tools/enhancement/color_match.py`
- Test: `tests/tools/test_color_match.py`

**Interfaces:**
- Produces:
  - `_midpoint(path) -> float` — clip duration / 2 via ffprobe; `0.0` on failure / still image.
  - `_extract_frame(path, at_seconds, dest) -> Path | None` — one PNG frame via ffmpeg
    `-ss <t> -i <path> -frames:v 1 <dest>`; `None` on failure / missing binary.
  - `_frame_stats(png_path) -> tuple[list[float], list[float]]` — `(mean_rgb[3], std_rgb[3])` via
    PIL `convert("RGB")` + numpy.
  - `_has_audio(path) -> bool` — ffprobe for an audio stream (guarded; `False` on error).

- [ ] **Step 1: Write the failing tests**

```python
import shutil
import subprocess as _sp
from pathlib import Path

import numpy as np
import pytest


def _make_solid_clip(path: Path, hexcolor: str, dur: float = 1.0) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={hexcolor}:s=48x48:d={dur}:r=10", str(path),
    ], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_extract_frame_and_stats(tmp_path):
    clip = tmp_path / "blue.mp4"
    _make_solid_clip(clip, "0x3060A0")            # R=0x30=48, G=0x60=96, B=0xA0=160
    tool = ColorMatch()
    frame = tool._extract_frame(str(clip), 0.0, tmp_path / "f.png")
    assert frame is not None and Path(frame).exists()
    mean, std = tool._frame_stats(frame)
    assert abs(mean[0] - 48) < 3 and abs(mean[1] - 96) < 3 and abs(mean[2] - 160) < 3
    assert max(std) < 2.0                          # solid color => near-zero variance


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_midpoint_of_two_second_clip(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_solid_clip(clip, "0x808080", dur=2.0)
    assert abs(ColorMatch()._midpoint(str(clip)) - 1.0) < 0.2


def test_extract_frame_missing_file_returns_none(tmp_path):
    assert ColorMatch()._extract_frame("/no/such.mp4", 0.0, tmp_path / "x.png") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py -k "extract or stats or midpoint" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
```

Add methods:

```python
    def _midpoint(self, path: str) -> float:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "json", str(path)], capture_output=True, text=True, check=True)
            return float(json.loads(proc.stdout)["format"]["duration"]) / 2.0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError,
                ValueError, KeyError):
            return 0.0

    def _extract_frame(self, path: str, at_seconds: float, dest) -> Path | None:
        dest = Path(dest)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "quiet", "-ss", f"{float(at_seconds):.3f}",
                 "-i", str(path), "-frames:v", "1", str(dest)],
                capture_output=True, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        if not dest.is_file() or dest.stat().st_size == 0:
            return None
        return dest

    def _frame_stats(self, png_path) -> tuple[list[float], list[float]]:
        arr = np.asarray(Image.open(png_path).convert("RGB")).reshape(-1, 3).astype(np.float64)
        return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

    def _has_audio(self, path: str) -> bool:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py -k "extract or stats or midpoint" -v`
Expected: PASS (ffmpeg-gated ones run on this machine)

- [ ] **Step 5: Commit**

```bash
git add tools/enhancement/color_match.py tests/tools/test_color_match.py
git commit -m "feat: color_match frame extraction + per-channel stats"
```

---

### Task 4: Full execute — orchestration, apply, before/after metric, edge cases, e2e

**Files:**
- Modify: `tools/enhancement/color_match.py`
- Test: `tests/tools/test_color_match.py`

**Interfaces:**
- Consumes: `_midpoint`, `_extract_frame`, `_frame_stats`, `_channel_affine`, `_lutrgb_expr`, `_has_audio`.
- Produces: full `execute(inputs)` returning `ToolResult(success=True, artifacts=[output_path], data={...})`. Adds `_mean_delta(a, b) -> float` (mean absolute per-channel difference).

- [ ] **Step 1: Write the failing tests**

```python
def test_mean_delta():
    tool = ColorMatch()
    assert abs(tool._mean_delta([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])) < 1e-9
    assert abs(tool._mean_delta([0.0, 0.0, 0.0], [3.0, 6.0, 9.0]) - 6.0) < 1e-9


def test_execute_requires_both_paths(tmp_path):
    result = ColorMatch().execute({"input_path": str(tmp_path / "a.mp4")})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_matches_target_toward_reference(tmp_path):
    target = tmp_path / "target.mp4"
    reference = tmp_path / "ref.mp4"
    _make_solid_clip(target, "0x3060A0")          # bluish
    _make_solid_clip(reference, "0xB05030")       # reddish
    out = tmp_path / "matched.mp4"
    result = ColorMatch().execute({
        "input_path": str(target), "reference_path": str(reference),
        "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert result.data["improved"] is True
    assert result.data["mean_delta_after"] < result.data["mean_delta_before"]
    # after-match target mean is close to the reference mean
    for c in range(3):
        assert abs(result.data["target_mean_after"][c] - result.data["reference_mean"][c]) < 8


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_intensity_zero_is_near_identity(tmp_path):
    target = tmp_path / "t.mp4"; reference = tmp_path / "r.mp4"
    _make_solid_clip(target, "0x3060A0"); _make_solid_clip(reference, "0xB05030")
    out = tmp_path / "o.mp4"
    result = ColorMatch().execute({
        "input_path": str(target), "reference_path": str(reference),
        "output_path": str(out), "intensity": 0.0})
    assert result.success, result.error
    assert result.data["gains"] == [1.0, 1.0, 1.0]
    assert result.data["offsets"] == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_missing_reference_fails(tmp_path):
    target = tmp_path / "t.mp4"; _make_solid_clip(target, "0x3060A0")
    result = ColorMatch().execute({
        "input_path": str(target), "reference_path": str(tmp_path / "nope.mp4"),
        "output_path": str(tmp_path / "o.mp4")})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_is_deterministic(tmp_path):
    target = tmp_path / "t.mp4"; reference = tmp_path / "r.mp4"
    _make_solid_clip(target, "0x3060A0"); _make_solid_clip(reference, "0xB05030")
    r1 = ColorMatch().execute({"input_path": str(target), "reference_path": str(reference),
                               "output_path": str(tmp_path / "o1.mp4")})
    r2 = ColorMatch().execute({"input_path": str(target), "reference_path": str(reference),
                               "output_path": str(tmp_path / "o2.mp4")})
    assert r1.data["gains"] == r2.data["gains"]
    assert r1.data["offsets"] == r2.data["offsets"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py -k "execute or mean_delta" -v`
Expected: FAIL — `execute` returns the `not implemented` stub.

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import tempfile
import time
```

Replace the placeholder `execute` and add the helper:

```python
    def _mean_delta(self, a, b) -> float:
        return float(sum(abs(float(a[c]) - float(b[c])) for c in range(3)) / 3.0)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        input_path = inputs.get("input_path")
        reference_path = inputs.get("reference_path")
        if not input_path or not reference_path:
            return ToolResult(success=False, error="input_path and reference_path are required.")
        input_path = Path(input_path)
        reference_path = Path(reference_path)
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Target not found: {input_path}")
        if not reference_path.is_file():
            return ToolResult(success=False, error=f"Reference not found: {reference_path}")

        intensity = float(inputs.get("intensity", 1.0))
        codec = str(inputs.get("codec", "libx264"))
        crf = int(inputs.get("crf", 20))
        in_t = inputs.get("input_time")
        ref_t = inputs.get("reference_time")
        out_path = Path(inputs.get("output_path") or
                        input_path.with_name(f"{input_path.stem}_matched.mp4"))

        workdir = Path(tempfile.mkdtemp(prefix="colormatch_"))
        try:
            t_at = float(in_t) if in_t is not None else self._midpoint(str(input_path))
            r_at = float(ref_t) if ref_t is not None else self._midpoint(str(reference_path))
            t_frame = self._extract_frame(str(input_path), t_at, workdir / "t.png")
            if t_frame is None:
                return ToolResult(success=False, error="Failed to extract a frame from the target.")
            r_frame = self._extract_frame(str(reference_path), r_at, workdir / "r.png")
            if r_frame is None:
                return ToolResult(success=False, error="Failed to extract a frame from the reference.")

            m_t, s_t = self._frame_stats(t_frame)
            m_r, s_r = self._frame_stats(r_frame)
            gains, offsets, clamp_notes = self._channel_affine(m_t, s_t, m_r, s_r, intensity)
            vf = self._lutrgb_expr(gains, offsets)

            cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vf", vf,
                   "-c:v", codec, "-crf", str(crf)]
            cmd += ["-c:a", "copy"] if self._has_audio(str(input_path)) else ["-an"]
            cmd += [str(out_path)]
            try:
                self.run_command(cmd, timeout=600)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                return ToolResult(success=False, error=f"color match render failed: {exc}")
            if not out_path.is_file() or out_path.stat().st_size == 0:
                return ToolResult(success=False, error="Output not created or empty.")

            # Honest before/after metric: re-measure the OUTPUT frame.
            out_frame = self._extract_frame(str(out_path), t_at, workdir / "o.png")
            m_after = self._frame_stats(out_frame)[0] if out_frame else m_t
            delta_before = self._mean_delta(m_t, m_r)
            delta_after = self._mean_delta(m_after, m_r)

            return ToolResult(
                success=True,
                artifacts=[str(out_path)],
                duration_seconds=time.time() - start,
                data={
                    "gains": [round(g, 6) for g in gains],
                    "offsets": [round(o, 6) for o in offsets],
                    "reference_mean": [round(x, 3) for x in m_r],
                    "target_mean_before": [round(x, 3) for x in m_t],
                    "target_mean_after": [round(x, 3) for x in m_after],
                    "mean_delta_before": round(delta_before, 3),
                    "mean_delta_after": round(delta_after, 3),
                    "improved": delta_after < delta_before,
                    "clamp_notes": clamp_notes,
                    "intensity": intensity,
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
```

Add `import shutil` to the top-of-file import block (with `tempfile`/`time`).

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `.venv/bin/python -m pytest tests/tools/test_color_match.py -v`
Expected: all PASS (ffmpeg-gated ones run on this machine).

- [ ] **Step 5: Commit**

```bash
git add tools/enhancement/color_match.py tests/tools/test_color_match.py
git commit -m "feat: color_match execute orchestration + before/after metric + e2e"
```

---

## Self-Review

**Spec coverage:**
- Contract mirroring `color_grade`, `["cmd:ffmpeg","python:numpy","python:PIL"]` → Task 1. ✓
- Reinhard per-channel mean+std transfer, intensity fold, EPS/GAIN_MAX guards → Task 2 (locked by 6 tests incl. mean-match, identity, flat-channel, gain-clamp). ✓
- lutrgb expression (clamped affine, `{:+.6f}` sign format) → Task 2. ✓
- Frame extraction (midpoint default, `-ss`/`-frames:v 1`), PIL+numpy stats, `_has_audio` → Task 3. ✓
- Full execute: extract both frames → stats → affine → lutrgb apply → re-measure → ToolResult; audio copy vs `-an`; temp cleanup in `finally` → Task 4. ✓
- Honest before/after metric (`mean_delta_before/after`, `improved`) → Task 4 (+ e2e asserts real convergence). ✓
- Edge cases: missing paths, extraction failure per side, zero-byte output, intensity=0 identity, missing reference, determinism → Tasks 3–4. ✓
- ToolResult `data` fields (gains/offsets/means/deltas/improved/clamp_notes/intensity) → Task 4. ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code.

**Type consistency:** `_channel_affine -> (list[float], list[float], list[dict])` (Tasks 2,4); `_lutrgb_expr(gains, offsets) -> str` (Tasks 2,4); `_frame_stats -> (list[float], list[float])` and `_extract_frame -> Path | None` and `_midpoint -> float` (Tasks 3,4); `_mean_delta(a, b) -> float` (Task 4). `execute` threads all of them. ✓

**Deferred vs spec:** Lab/decorrelated transfer, histogram matching, `.cube` export, skin-tone protection are explicitly v2 per the spec; not in this plan by design.
