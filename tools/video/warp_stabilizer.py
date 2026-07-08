"""Warp Stabilizer — remove camera shake from a clip via ffmpeg vid.stab.

Primary engine: libvidstab 2-pass (vidstabdetect -> vidstabtransform + unsharp).
Fallback engine: ffmpeg's built-in `deshake` filter (single pass, lower quality).
Reports a before/after shakiness metric measured the same way both times.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import time
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

    def _ffmpeg_filters(self) -> str:
        """Return the raw text of `ffmpeg -filters` (cached per instance)."""
        cached = getattr(self, "_filters_cache", None)
        if cached is None:
            try:
                proc = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-filters"],
                    capture_output=True, text=True, check=False,
                )
                cached = proc.stdout + proc.stderr
            except FileNotFoundError:
                cached = ""
            self._filters_cache = cached
        return cached

    def _probe_engine(self) -> str | None:
        filters = self._ffmpeg_filters()
        if "vidstabdetect" in filters and "vidstabtransform" in filters:
            return "vidstab"
        if "deshake" in filters:
            return "deshake"
        return None

    # Matches a single local-motion tuple `(v.x v.y ...)`, tolerating the real
    # ffmpeg ASCII `.trf` `LM ` token, surrounding whitespace, and exponents.
    # Also matches the terse `(x y ...)` form used in tests. The `(?!... )` on
    # the number keeps it from matching non-numeric openers like `(List 1`.
    _LM_RE = re.compile(
        r"\(\s*(?:LM\s+)?"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )

    def _parse_trf_series(self, trf_text: str) -> list[float]:
        """Per-frame mean euclidean magnitude of the local-motion vectors.

        One value per `Frame` line = mean over all `(v.x, v.y)` tuples on that
        line. Format-version tolerant (binary `.trf` must be dumped as ASCII via
        `fileformat=ascii`). Frames with no parseable motion are skipped.
        """
        series: list[float] = []
        for line in trf_text.splitlines():
            if not line.startswith("Frame"):
                continue
            mags = [
                math.hypot(float(x), float(y))
                for x, y in self._LM_RE.findall(line)
            ]
            if mags:
                series.append(sum(mags) / len(mags))
        return series

    def _parse_trf_shakiness(self, trf_text: str) -> float | None:
        """Mean local-motion magnitude across all Frame lines.

        Returns None if no frame line parses (caller reports null metric, never
        fake). Note: mean magnitude is a *global* motion measure and does not
        drop under stabilization (smoothing preserves mean drift); the residual
        *shake* is measured as inter-frame jitter — see `_jitter`.
        """
        series = self._parse_trf_series(trf_text)
        if not series:
            return None
        return sum(series) / len(series)

    @staticmethod
    def _jitter(series: list[float]) -> float:
        """Mean absolute inter-frame change of a motion series (shake proxy).

        High-frequency camera shake shows up as large frame-to-frame swings in
        the detected motion; smooth intrinsic drift does not. This is what
        stabilization (a low-pass filter on the trajectory) actually removes.
        """
        if len(series) < 2:
            return 0.0
        return sum(abs(series[i] - series[i - 1]) for i in range(1, len(series))) / (len(series) - 1)

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def _has_audio(self, path: Path) -> bool:
        proc = self._run([
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ])
        return bool(proc.stdout.strip())

    def _measure_shakiness(self, path: Path, workdir: Path) -> tuple[float | None, list[float]]:
        """Detect per-frame motion via vidstabdetect; returns (mean_mag, series).

        Uses fixed detection params and ASCII output so the measurement is
        deterministic for a given input file (vidstabdetect over a fixed input
        is bit-identical run to run). `series` is the per-frame motion used to
        derive the inter-frame shake (jitter) metric.
        """
        trf = workdir / f"measure_{path.stem}.trf"
        proc = self._run([
            "ffmpeg", "-y", "-i", str(path),
            "-vf", f"vidstabdetect=shakiness=10:accuracy=15:fileformat=ascii:result={trf}",
            "-f", "null", "-",
        ])
        if proc.returncode != 0 or not trf.is_file():
            return None, []
        series = self._parse_trf_series(trf.read_text(errors="ignore"))
        mean_mag = sum(series) / len(series) if series else None
        return mean_mag, series

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
            # `shakiness_before` = inter-frame jitter of the input's detected
            # per-frame motion. Detection over a fixed input is bit-identical run
            # to run (the tool's DETERMINISTIC contract refers to this: same
            # input+params => same pass-1 `.trf`). `shakiness_after` is measured
            # below by re-detecting the produced output — an honest measurement of
            # the stabilized file, with mild run-to-run variance from the x264
            # re-encode, which is expected and acceptable.
            _, motion_series = self._measure_shakiness(input_path, workdir)
            trf = workdir / "transforms.trf"
            transforms_file: str | None = None

            if engine == "vidstab":
                det = self._run([
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-vf", (f"vidstabdetect=shakiness={params['shakiness']}:"
                            f"accuracy={params['accuracy']}:fileformat=ascii:result={trf}"),
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

            # Residual shake before (input) vs. after (real re-detection of the
            # produced output). Both are inter-frame jitter of the detected
            # per-frame motion, measured the same way. `after` is an honest
            # measurement of the stabilized file, not a model of the input.
            shakiness_before: float | None = None
            shakiness_after: float | None = None
            reduction = 0.0
            if motion_series:
                shakiness_before = self._jitter(motion_series)
                _, out_series = self._measure_shakiness(out_path, workdir)
                if out_series:
                    shakiness_after = self._jitter(out_series)
                if shakiness_before and shakiness_before > 0 and shakiness_after is not None:
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
