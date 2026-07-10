"""Color Match — match a target clip's color to a reference (clip or still).

Extracts one frame from the target and one from the reference, computes a
per-channel Reinhard mean+std transfer (folded with an intensity blend), and
applies it to the whole target clip via a single ffmpeg `lutrgb` pass. Reports
an honest before/after per-channel color-delta metric. Complements color_grade
(subjective looks) with reference-driven matching.
"""

from __future__ import annotations

import json
import subprocess
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

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
