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
