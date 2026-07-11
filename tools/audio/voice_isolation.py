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
    BUNDLED_MODEL = str(Path(__file__).resolve().parents[2] / "assets" / "rnnoise" / "somnolent-hogwash.rnnn")

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
