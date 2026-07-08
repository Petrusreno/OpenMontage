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

    def _match_literal_words(self, words: list[dict], remove_words: list[str]) -> list[dict]:
        targets = {self._normalize(w) for w in (remove_words or []) if self._normalize(w)}
        spans = []
        for w in words:
            if self._valid_time(w) and self._normalize(w["word"]) in targets:
                spans.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": w["word"].strip(), "reason": "literal"})
        return spans

    def _indices_and_ranges_to_spans(self, words: list[dict], indices: list[int] | None,
                                     ranges: list[dict] | None) -> list[dict]:
        spans = []
        for i in (indices or []):
            if isinstance(i, int) and 0 <= i < len(words) and self._valid_time(words[i]):
                w = words[i]
                spans.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": w["word"].strip(), "reason": "index"})
        for r in (ranges or []):
            s, e = r.get("start_seconds"), r.get("end_seconds")
            if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                spans.append({"start": float(s), "end": float(e), "word": None, "reason": "range"})
        return spans

    def _merge_spans(self, spans: list[dict]) -> list[dict]:
        ordered = sorted(spans, key=lambda s: (float(s["start"]), float(s["end"])))
        merged: list[dict] = []
        for s in ordered:
            start, end = float(s["start"]), float(s["end"])
            if merged and start <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], end)
            else:
                merged.append({"start": start, "end": end})
        return merged

    def _keep_segments(self, merged: list[dict], duration: float,
                       padding: float, min_gap: float) -> list[dict]:
        segments = []
        cursor = 0.0
        for rem in merged:
            keep_end = float(rem["start"]) + padding
            if keep_end > cursor:
                segments.append({"start": cursor, "end": min(keep_end, duration)})
            cursor = max(cursor, float(rem["end"]) - padding)
        if cursor < duration:
            segments.append({"start": cursor, "end": duration})

        merged_keeps: list[dict] = []
        for seg in segments:
            if seg["end"] - seg["start"] < 0.01:
                continue
            if merged_keeps and seg["start"] - merged_keeps[-1]["end"] < min_gap:
                merged_keeps[-1]["end"] = seg["end"]
            else:
                merged_keeps.append({"start": round(seg["start"], 3), "end": round(seg["end"], 3)})
        return merged_keeps

    def _to_edit_decisions(self, keeps: list[dict], source: str,
                           removed: list[dict], meta: dict) -> dict:
        cuts = [
            {"id": f"cut_{i:04d}", "source": source,
             "in_seconds": round(float(k["start"]), 3), "out_seconds": round(float(k["end"]), 3)}
            for i, k in enumerate(keeps)
        ]
        removed_seconds = round(sum(float(r["end"]) - float(r["start"]) for r in removed), 3)
        kept_seconds = round(sum(c["out_seconds"] - c["in_seconds"] for c in cuts), 3)
        return {
            "version": "1.0",
            "cuts": cuts,
            "metadata": {
                "tool": "text_based_editor",
                "removed_count": len(removed),
                "removed_seconds": removed_seconds,
                "kept_seconds": kept_seconds,
                **meta,
            },
        }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
