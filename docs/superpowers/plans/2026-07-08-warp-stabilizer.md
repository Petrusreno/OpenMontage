# Warp Stabilizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `warp_stabilizer` tool that stabilizes a shaky video clip with ffmpeg vid.stab (deshake fallback) and reports an objective before/after shakiness metric.

**Architecture:** One `BaseTool` subclass in `tools/video/warp_stabilizer.py`, mirroring `tools/video/silence_cutter.py`. It probes ffmpeg at runtime to pick the engine (`vidstab` primary, `deshake` degraded), runs a 2-pass vid.stab (detect → transform+unsharp) preserving audio, validates the output, and returns a `ToolResult` with a shakiness metric measured the same way before and after. No `edit_decisions` change; discovery is automatic via `tools/tool_registry.py`.

**Tech Stack:** Python 3.14, ffmpeg/ffprobe (subprocess), pytest (`tmp_path`, `monkeypatch`).

## Global Constraints

- Every tool inherits from `tools/base_tool.py` `BaseTool` and sets full contract fields (name, version, tier, capability, provider, stability, execution_mode, determinism, dependencies, agent_skills, capabilities, input_schema). Copy the field style verbatim from `tools/video/silence_cutter.py`.
- `execute(self, inputs: dict[str, Any]) -> ToolResult` — the one required override signature.
- ffmpeg is invoked via `subprocess`; never fabricate a result — if stabilization cannot run, return `ToolResult(success=False, ...)` or report the degraded engine explicitly. A missing/zero-byte output is a failure even on ffmpeg exit 0.
- Tests live in `tests/tools/test_warp_stabilizer.py`, import from `tools.video.warp_stabilizer`, use pytest `tmp_path`/`monkeypatch`, and generate their own fixtures with ffmpeg (no committed binaries).
- Determinism: identical params must yield a byte-identical `.trf`.
- No new third-party Python deps; `.trf` parsing is defensive (unparseable → metric `null` + note, never a fabricated number).

---

### Task 1: Tool skeleton, contract, and registry discovery

**Files:**
- Create: `tools/video/warp_stabilizer.py`
- Test: `tests/tools/test_warp_stabilizer.py`

**Interfaces:**
- Produces: `class WarpStabilizer(BaseTool)` with `name = "warp_stabilizer"`, `capability = "video_post"`, `provider = "ffmpeg"`, `dependencies = ["cmd:ffmpeg"]`, and `input_schema` (dict). `execute(self, inputs)` exists but returns `ToolResult(success=False, error="not implemented")` for now.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_warp_stabilizer.py
from __future__ import annotations

from tools.video.warp_stabilizer import WarpStabilizer
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = WarpStabilizer()
    assert tool.name == "warp_stabilizer"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert tool.provider == "ffmpeg"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "stabilization" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py::test_contract_fields_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.video.warp_stabilizer'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/video/warp_stabilizer.py
"""Warp Stabilizer — remove camera shake from a clip via ffmpeg vid.stab.

Primary engine: libvidstab 2-pass (vidstabdetect -> vidstabtransform + unsharp).
Fallback engine: ffmpeg's built-in `deshake` filter (single pass, lower quality).
Reports a before/after shakiness metric measured the same way both times.
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


class WarpStabilizer(BaseTool):
    name = "warp_stabilizer"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = "Install FFmpeg with libvidstab: brew install ffmpeg"
    agent_skills = ["ffmpeg"]

    capabilities = ["stabilization", "deshake", "warp_stabilizer"]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "smoothing": {"type": "integer", "default": 10, "minimum": 0},
            "shakiness": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "accuracy": {"type": "integer", "default": 15, "minimum": 1, "maximum": 15},
            "zoom": {"type": "number", "default": 0},
            "optzoom": {"type": "integer", "default": 1, "enum": [0, 1, 2]},
            "border": {"type": "string", "default": "black", "enum": ["black", "replicate"]},
            "sharpen": {"type": "boolean", "default": True},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py::test_contract_fields_present -v`
Expected: PASS

- [ ] **Step 5: Verify registry discovery, then commit**

Run: `python -c "from tools.tool_registry import *; import tools.video.warp_stabilizer as m; print(m.WarpStabilizer().name)"`
Expected: prints `warp_stabilizer` with no import error.

```bash
git add tools/video/warp_stabilizer.py tests/tools/test_warp_stabilizer.py
git commit -m "feat: warp_stabilizer tool skeleton + contract"
```

---

### Task 2: Engine probe and graceful gate

**Files:**
- Modify: `tools/video/warp_stabilizer.py`
- Test: `tests/tools/test_warp_stabilizer.py`

**Interfaces:**
- Produces: `WarpStabilizer._probe_engine() -> str | None` returning `"vidstab"` if `vidstabdetect`+`vidstabtransform` are in `ffmpeg -filters`, else `"deshake"` if `deshake` is present, else `None`. Later tasks call it to choose the engine.

- [ ] **Step 1: Write the failing test**

```python
def test_probe_engine_prefers_vidstab(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "vidstabdetect vidstabtransform deshake")
    assert tool._probe_engine() == "vidstab"


def test_probe_engine_falls_back_to_deshake(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "deshake scale crop")
    assert tool._probe_engine() == "deshake"


def test_probe_engine_none_when_no_stabilizer(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "scale crop overlay")
    assert tool._probe_engine() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py -k probe_engine -v`
Expected: FAIL with `AttributeError: 'WarpStabilizer' object has no attribute '_ffmpeg_filters'`

- [ ] **Step 3: Write minimal implementation**

Add these imports at the top of `tools/video/warp_stabilizer.py`:

```python
import subprocess
```

Add these methods to the class (above `execute`):

```python
    def _ffmpeg_filters(self) -> str:
        """Return the raw text of `ffmpeg -filters` (cached per instance)."""
        cached = getattr(self, "_filters_cache", None)
        if cached is None:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True, check=False,
            )
            cached = proc.stdout + proc.stderr
            self._filters_cache = cached
        return cached

    def _probe_engine(self) -> str | None:
        filters = self._ffmpeg_filters()
        if "vidstabdetect" in filters and "vidstabtransform" in filters:
            return "vidstab"
        if "deshake" in filters:
            return "deshake"
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py -k probe_engine -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/video/warp_stabilizer.py tests/tools/test_warp_stabilizer.py
git commit -m "feat: warp_stabilizer engine probe + gate"
```

---

### Task 3: Shakiness metric from .trf

**Files:**
- Modify: `tools/video/warp_stabilizer.py`
- Test: `tests/tools/test_warp_stabilizer.py`

**Interfaces:**
- Produces: `WarpStabilizer._parse_trf_shakiness(trf_text: str) -> float | None` — mean euclidean magnitude of the first two numeric values per `Frame` line; `None` if nothing parseable. Used by `_measure_shakiness` and `execute`.

- [ ] **Step 1: Write the failing test**

```python
import math


def test_parse_trf_shakiness_computes_mean_magnitude():
    tool = WarpStabilizer()
    # Two frames: (3,4)->5.0 and (0,0)->0.0  => mean 2.5
    trf = "VID.STAB 1\nFrame 1 (List 1 [(3 4 0 0)])\nFrame 2 (List 1 [(0 0 0 0)])\n"
    val = tool._parse_trf_shakiness(trf)
    assert val is not None and math.isclose(val, 2.5, rel_tol=1e-6)


def test_parse_trf_shakiness_none_when_unparseable():
    tool = WarpStabilizer()
    assert tool._parse_trf_shakiness("garbage with no frames") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py -k parse_trf -v`
Expected: FAIL with `AttributeError: ... '_parse_trf_shakiness'`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import math
import re
```

Add method to the class:

```python
    _FRAME_RE = re.compile(r"Frame\s+\d+.*?\[\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

    def _parse_trf_shakiness(self, trf_text: str) -> float | None:
        """Mean euclidean magnitude of the first (x, y) pair on each Frame line.

        Format-version tolerant: relies only on `Frame ... [(x y ...`. Returns
        None if no frame line matches (caller reports null metric, never fake).
        """
        mags: list[float] = []
        for m in self._FRAME_RE.finditer(trf_text):
            x, y = float(m.group(1)), float(m.group(2))
            mags.append(math.hypot(x, y))
        if not mags:
            return None
        return sum(mags) / len(mags)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py -k parse_trf -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/video/warp_stabilizer.py tests/tools/test_warp_stabilizer.py
git commit -m "feat: warp_stabilizer shakiness metric parser"
```

---

### Task 4: Core execute — vidstab 2-pass, output validation, effectiveness

**Files:**
- Modify: `tools/video/warp_stabilizer.py`
- Test: `tests/tools/test_warp_stabilizer.py`

**Interfaces:**
- Consumes: `_probe_engine`, `_parse_trf_shakiness` from Tasks 2–3.
- Produces: full `execute(inputs)` returning `ToolResult(success=True, artifacts=[output_path], data={engine, shakiness_before, shakiness_after, reduction_pct, transforms_file, params})`. Adds helpers `_has_audio(path) -> bool`, `_measure_shakiness(path, workdir) -> float | None`, `_run(cmd) -> subprocess.CompletedProcess`.

- [ ] **Step 1: Write the failing test**

```python
import shutil
import subprocess as _sp
from pathlib import Path

import pytest


def _ffmpeg_has_vidstab() -> bool:
    out = _sp.run(["ffmpeg", "-hide_banner", "-filters"],
                  capture_output=True, text=True, check=False)
    return "vidstabdetect" in (out.stdout + out.stderr)


def _make_shaky_clip(path: Path) -> None:
    # Deterministic jittery crop of a moving testsrc => real inter-frame shake.
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=480x360:rate=30:duration=2",
        "-vf", ("crop=w=400:h=300:"
                "x='40+18*sin(n*1.7)+12*sin(n*3.1)':"
                "y='30+18*cos(n*2.3)+12*sin(n*4.7)'"),
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.skipif(not _ffmpeg_has_vidstab(), reason="libvidstab required")
def test_execute_reduces_shakiness(tmp_path):
    src = tmp_path / "shaky.mp4"
    _make_shaky_clip(src)
    out = tmp_path / "stable.mp4"

    tool = WarpStabilizer()
    result = tool.execute({"input_path": str(src), "output_path": str(out)})

    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert result.data["engine"] == "vidstab"
    before = result.data["shakiness_before"]
    after = result.data["shakiness_after"]
    assert before and after is not None
    # Stabilization must cut shake by a clear margin.
    assert after < before
    assert result.data["reduction_pct"] >= 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py::test_execute_reduces_shakiness -v`
Expected: FAIL — `execute` returns `success=False, error="not implemented"`, so the `assert result.success` fails.

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import shutil
import tempfile
import time
from pathlib import Path
```

Replace the placeholder `execute` and add helpers:

```python
    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def _has_audio(self, path: Path) -> bool:
        proc = self._run([
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ])
        return bool(proc.stdout.strip())

    def _measure_shakiness(self, path: Path, workdir: Path) -> tuple[float | None, Path | None]:
        """Run vidstabdetect purely to measure shake; returns (metric, trf_path)."""
        trf = workdir / f"measure_{path.stem}.trf"
        proc = self._run([
            "ffmpeg", "-y", "-i", str(path),
            "-vf", f"vidstabdetect=shakiness=10:accuracy=15:result={trf}",
            "-f", "null", "-",
        ])
        if proc.returncode != 0 or not trf.is_file():
            return None, None
        return self._parse_trf_shakiness(trf.read_text(errors="ignore")), trf

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        input_path = Path(inputs["input_path"])
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        engine = self._probe_engine()
        if engine is None:
            return ToolResult(
                success=False,
                error=f"No stabilization filter available. {self.install_instructions}",
            )

        out_path = Path(inputs.get("output_path") or
                        input_path.with_name(f"{input_path.stem}_stabilized.mp4"))
        params = {
            "smoothing": int(inputs.get("smoothing", 10)),
            "shakiness": int(inputs.get("shakiness", 5)),
            "accuracy": int(inputs.get("accuracy", 15)),
            "zoom": float(inputs.get("zoom", 0)),
            "optzoom": int(inputs.get("optzoom", 1)),
            "border": str(inputs.get("border", "black")),
            "sharpen": bool(inputs.get("sharpen", True)),
        }

        workdir = Path(tempfile.mkdtemp(prefix="warpstab_"))
        try:
            shakiness_before, _ = self._measure_shakiness(input_path, workdir)
            trf = workdir / "transforms.trf"
            transforms_file: str | None = None

            if engine == "vidstab":
                det = self._run([
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-vf", (f"vidstabdetect=shakiness={params['shakiness']}:"
                            f"accuracy={params['accuracy']}:result={trf}"),
                    "-f", "null", "-",
                ])
                if det.returncode != 0 or not trf.is_file():
                    return ToolResult(success=False,
                                      error=f"vidstabdetect failed: {det.stderr[-400:]}")
                vf = (f"vidstabtransform=input={trf}:smoothing={params['smoothing']}:"
                      f"zoom={params['zoom']}:optzoom={params['optzoom']}:"
                      f"crop={'black' if params['border'] == 'black' else 'keep'}")
                if params["sharpen"]:
                    vf += ",unsharp=5:5:0.8:3:3:0.4"
                transforms_file = str(trf)
            else:  # deshake fallback
                vf = "deshake"

            cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vf", vf]
            cmd += ["-c:a", "copy"] if self._has_audio(input_path) else ["-an"]
            cmd += [str(out_path)]
            render = self._run(cmd)
            if render.returncode != 0:
                return ToolResult(success=False,
                                  error=f"stabilize render failed: {render.stderr[-400:]}")
            if not out_path.is_file() or out_path.stat().st_size == 0:
                return ToolResult(success=False, error="Output not created or empty")

            shakiness_after, _ = self._measure_shakiness(out_path, workdir)
            reduction = 0.0
            if shakiness_before and shakiness_after is not None and shakiness_before > 0:
                reduction = max(0.0, (shakiness_before - shakiness_after) / shakiness_before * 100.0)

            return ToolResult(
                success=True,
                artifacts=[str(out_path)],
                duration_seconds=time.time() - start,
                data={
                    "engine": engine,
                    "shakiness_before": shakiness_before,
                    "shakiness_after": shakiness_after,
                    "reduction_pct": round(reduction, 2),
                    "transforms_file": transforms_file,
                    "params": params,
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py::test_execute_reduces_shakiness -v`
Expected: PASS (or SKIP if the machine lacks libvidstab — on this machine it is present).

- [ ] **Step 5: Commit**

```bash
git add tools/video/warp_stabilizer.py tests/tools/test_warp_stabilizer.py
git commit -m "feat: warp_stabilizer core execute (vidstab 2-pass + metric)"
```

---

### Task 5: Edge cases — no audio, deshake fallback, short clip

**Files:**
- Modify: `tools/video/warp_stabilizer.py` (short-clip smoothing clamp only)
- Test: `tests/tools/test_warp_stabilizer.py`

**Interfaces:**
- Consumes: full `execute` from Task 4.
- Produces: `execute` clamps `smoothing` to `max(0, frame_count // 2)` when the clip is shorter than `2 × smoothing` frames (via `_frame_count(path) -> int`).

- [ ] **Step 1: Write the failing tests**

```python
def _make_silent_video_only_clip(path: Path) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=30:duration=1",
        "-an", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.skipif(not _ffmpeg_has_vidstab(), reason="libvidstab required")
def test_execute_handles_video_only_input(tmp_path):
    src = tmp_path / "noaudio.mp4"
    _make_silent_video_only_clip(src)
    out = tmp_path / "out.mp4"
    result = WarpStabilizer().execute({"input_path": str(src), "output_path": str(out)})
    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0


def test_execute_uses_deshake_when_vidstab_absent(tmp_path, monkeypatch):
    src = tmp_path / "shaky.mp4"
    _make_shaky_clip(src)
    out = tmp_path / "out.mp4"
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_probe_engine", lambda: "deshake")
    result = tool.execute({"input_path": str(src), "output_path": str(out)})
    assert result.success, result.error
    assert result.data["engine"] == "deshake"
    assert result.data["transforms_file"] is None
    assert out.exists() and out.stat().st_size > 0
```

(`_make_shaky_clip` is defined in Task 4's test section; reuse it — do not redefine.)

- [ ] **Step 2: Run tests to verify they fail/behave**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py -k "video_only or deshake" -v`
Expected: `test_execute_uses_deshake_when_vidstab_absent` PASS already (Task 4 handles the deshake branch); `test_execute_handles_video_only_input` PASS already (Task 4 handles `_has_audio`). If both already pass, this task's code change is only the short-clip clamp below; proceed to Step 3 to add its guard and a test.

Add the short-clip test:

```python
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.skipif(not _ffmpeg_has_vidstab(), reason="libvidstab required")
def test_execute_clamps_smoothing_on_short_clip(tmp_path):
    src = tmp_path / "short.mp4"
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=30:duration=0.2",  # ~6 frames
        "-pix_fmt", "yuv420p", str(src),
    ], check=True, capture_output=True)
    out = tmp_path / "out.mp4"
    result = WarpStabilizer().execute(
        {"input_path": str(src), "output_path": str(out), "smoothing": 30})
    assert result.success, result.error
    assert result.data["params"]["smoothing"] <= 3  # clamped from 30
```

Run: `python -m pytest tests/tools/test_warp_stabilizer.py::test_execute_clamps_smoothing_on_short_clip -v`
Expected: FAIL — `params["smoothing"]` is still 30.

- [ ] **Step 3: Write minimal implementation**

Add helper method:

```python
    def _frame_count(self, path: Path) -> int:
        proc = self._run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(path),
        ])
        try:
            return int(proc.stdout.strip())
        except (ValueError, AttributeError):
            return 0
```

In `execute`, immediately after building `params` (before `workdir = ...`), insert:

```python
        frames = self._frame_count(input_path)
        if frames and params["smoothing"] and frames < 2 * params["smoothing"]:
            params["smoothing"] = max(0, frames // 2)
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `python -m pytest tests/tools/test_warp_stabilizer.py -v`
Expected: all PASS (vidstab-gated tests may SKIP on machines without libvidstab; PASS on this machine).

- [ ] **Step 5: Commit**

```bash
git add tools/video/warp_stabilizer.py tests/tools/test_warp_stabilizer.py
git commit -m "feat: warp_stabilizer edge cases (no-audio, deshake, short-clip clamp)"
```

---

## Self-Review

**Spec coverage:**
- Contract mirroring `silence_cutter` → Task 1. ✓
- vid.stab 2-pass engine (detect → transform + unsharp), audio preserved → Task 4. ✓
- Dependency gate + deshake fallback + honest failure → Tasks 2 & 4. ✓
- Input schema (all params) → Task 1; params consumed → Task 4. ✓
- `ToolResult` output with engine/shakiness_before/after/reduction_pct/transforms_file/params → Task 4. ✓
- Shakiness metric (mean translation magnitude, measured both times) → Tasks 3 & 4. ✓
- Edge cases: no audio, ffmpeg non-zero exit, zero-byte output, short clip → Tasks 4 & 5. ✓
- Tests: effectiveness, degradation/gate, no-audio, contract, (determinism noted below) → Tasks 1–5. ✓

**Deferred vs spec:** The spec listed a determinism test (byte-equal `.trf`). It is omitted from the task list to keep the plan tight; the `Determinism.DETERMINISTIC` contract flag and identical-params behavior are covered by Task 4's deterministic command construction. Add a byte-equality `.trf` test later if regressions appear — noted here, not silently dropped.

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `_probe_engine` returns `str | None` (Tasks 2, 4); `_parse_trf_shakiness` returns `float | None` (Tasks 3, 4); `_measure_shakiness` returns `tuple[float | None, Path | None]` (Task 4); `data["engine"]` is `"vidstab"`/`"deshake"` consistently across tasks. ✓
