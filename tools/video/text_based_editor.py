"""Text-based editor — remove filler/selected words from a clip.

Emits an `edit_decisions` artifact whose cuts are the KEPT spans (everything
except removed words), and optionally renders the cut MP4. Word timestamps are
supplied by the caller or obtained from the Transcriber tool. Filler detection
is rule-based, deterministic, and language-specific; ambiguous meaningful words
are never removed unless the caller asks explicitly.
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


class TextBasedEditor(BaseTool):
    name = "text_based_editor"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "whisperx+ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = (
        "Install FFmpeg (brew install ffmpeg). For the self-transcription path, "
        "also install faster-whisper (pip install faster-whisper), or pass word_timestamps."
    )
    agent_skills = ["ffmpeg", "whisperx"]

    capabilities = ["text_based_editing", "filler_removal", "transcript_cut"]

    input_schema = {
        "type": "object",
        "required": [],
        "properties": {
            "input_path": {"type": "string"},
            "word_timestamps": {"type": "array"},
            "source": {"type": "string"},
            "language": {"type": "string", "default": "pt", "enum": ["pt", "en"]},
            "remove_fillers": {"type": "boolean", "default": True},
            "remove_repetitions": {"type": "boolean", "default": True},
            "filler_lexicon": {"type": "array"},
            "remove_words": {"type": "array"},
            "remove_word_indices": {"type": "array"},
            "remove_ranges": {"type": "array"},
            "padding_seconds": {"type": "number", "default": 0.08, "minimum": 0.0},
            "min_gap_seconds": {"type": "number", "default": 0.05, "minimum": 0.0},
            "repetition_max_gap": {"type": "number", "default": 0.6, "minimum": 0.0},
            "render": {"type": "boolean", "default": False},
            "output_path": {"type": "string"},
            "render_path": {"type": "string"},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
