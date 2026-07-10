"""Color Match — match a target clip's color to a reference (clip or still).

Extracts one frame from the target and one from the reference, computes a
per-channel Reinhard mean+std transfer (folded with an intensity blend), and
applies it to the whole target clip via a single ffmpeg `lutrgb` pass. Reports
an honest before/after per-channel color-delta metric. Complements color_grade
(subjective looks) with reference-driven matching.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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

    EPS = 1.0
    GAIN_MAX = 3.0

    def _channel_affine(self, m_t, s_t, m_r, s_r, intensity: float):
        gains: list[float] = []
        offsets: list[float] = []
        notes: list[dict] = []
        channel_names = ["r", "g", "b"]
        for c in range(len(m_t)):
            mt, st_, mr, sr = float(m_t[c]), float(s_t[c]), float(m_r[c]), float(s_r[c])
            if st_ < self.EPS:
                g = 1.0
                notes.append({"channel": channel_names[c], "reason": "flat_channel"})
            else:
                # g >= 0 always (sr is a std >= 0, st_ >= EPS > 0), so only the
                # upper clamp is reachable — and it is recorded, never silent.
                g = sr / st_
                if g > self.GAIN_MAX:
                    g = self.GAIN_MAX
                    notes.append({"channel": channel_names[c], "reason": "gain_clamped"})
            gain = 1.0 + intensity * (g - 1.0)
            offset = intensity * (mr - g * mt)
            gains.append(gain)
            offsets.append(offset)
        return gains, offsets, notes

    def _lutrgb_expr(self, gains: list[float], offsets: list[float]) -> str:
        # gains/offsets must be length-3 (R, G, B) — as produced by
        # _channel_affine on the length-3 stats from _frame_stats.
        channels = ["r", "g", "b"]
        parts = [
            f"{channels[c]}='clip({gains[c]:.6f}*val{offsets[c]:+.6f},0,255)'"
            for c in range(3)
        ]
        return "lutrgb=" + ":".join(parts)

    def _midpoint(self, path: str) -> float:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=30)
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
                capture_output=True, check=True, timeout=60)
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
                capture_output=True, text=True, check=False, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

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
