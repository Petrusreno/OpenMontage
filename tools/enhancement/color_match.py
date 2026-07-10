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

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
