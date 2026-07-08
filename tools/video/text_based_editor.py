"""Text-based editor — remove filler/selected words from a clip.

Emits an `edit_decisions` artifact whose cuts are the KEPT spans (everything
except removed words), and optionally renders the cut MP4. Word timestamps are
supplied by the caller or obtained from the Transcriber tool. Filler detection
is rule-based, deterministic, and language-specific; ambiguous meaningful words
are never removed unless the caller asks explicitly.
"""

from __future__ import annotations

import re
import unicodedata
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

    _DEFAULT_LEXICONS = {
        "pt": {"ãã", "ã", "ahn", "ãhn", "hum", "hmm", "eh", "ehh"},
        "en": {"um", "uh", "uhm", "hmm", "er", "err", "ah", "mm", "mhm"},
    }

    @staticmethod
    def _normalize(word: str) -> str:
        text = unicodedata.normalize("NFC", word or "")
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE).lower()

    def _lexicon_for(self, language: str, filler_lexicon: list[str] | None) -> set[str]:
        base = set(self._DEFAULT_LEXICONS.get(language, self._DEFAULT_LEXICONS["pt"]))
        if filler_lexicon:
            base |= {self._normalize(w) for w in filler_lexicon}
        return {self._normalize(w) for w in base if self._normalize(w)}

    @staticmethod
    def _valid_time(w: dict) -> bool:
        return isinstance(w.get("start"), (int, float)) and isinstance(w.get("end"), (int, float))

    def _detect_fillers(self, words: list[dict], lexicon: set[str]) -> list[dict]:
        spans = []
        for w in words:
            if self._valid_time(w) and self._normalize(w.get("word", "")) in lexicon:
                spans.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": w.get("word", "").strip(), "reason": "filler"})
        return spans

    def _detect_repetitions(self, words: list[dict], max_gap: float) -> list[dict]:
        spans = []
        valid = [w for w in words if self._valid_time(w)]
        i = 0
        while i < len(valid):
            j = i
            while (j + 1 < len(valid)
                   and self._normalize(valid[j + 1].get("word", "")) == self._normalize(valid[i].get("word", ""))
                   and self._normalize(valid[i].get("word", "")) != ""
                   and float(valid[j + 1]["start"]) - float(valid[j]["end"]) <= max_gap):
                j += 1
            if j > i:  # run of length >= 2: remove all but the last (index j)
                for k in range(i, j):
                    spans.append({"start": float(valid[k]["start"]), "end": float(valid[k]["end"]),
                                  "word": valid[k].get("word", "").strip(), "reason": "repetition"})
            i = j + 1
        return spans

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
