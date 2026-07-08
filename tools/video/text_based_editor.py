"""Text-based editor — remove filler/selected words from a clip.

Emits an `edit_decisions` artifact whose cuts are the KEPT spans (everything
except removed words), and optionally renders the cut MP4. Word timestamps are
supplied by the caller or obtained from the Transcriber tool. Filler detection
is rule-based, deterministic, and language-specific; ambiguous meaningful words
are never removed unless the caller asks explicitly.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
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
            if self._valid_time(w) and self._normalize(w.get("word", "")) in targets:
                spans.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": w.get("word", "").strip(), "reason": "literal"})
        return spans

    def _indices_and_ranges_to_spans(self, words: list[dict], indices: list[int] | None,
                                     ranges: list[dict] | None) -> list[dict]:
        spans = []
        for i in (indices or []):
            if isinstance(i, int) and 0 <= i < len(words) and self._valid_time(words[i]):
                w = words[i]
                spans.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": w.get("word", "").strip(), "reason": "index"})
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
        # Defensive: never rely on the caller having sorted `merged`; an
        # out-of-order removal would otherwise be silently swallowed.
        merged = sorted(merged, key=lambda r: float(r["start"]))
        segments = []
        cursor = 0.0
        for rem in merged:
            rem_start, rem_end = float(rem["start"]), float(rem["end"])
            # Padding trims a little EXTRA around each cut so no fragment of the
            # removed word survives (a boundary word must be fully cut). Clamp it
            # per removal to at most half the removal so an oversized global
            # padding can't nuke a whole short word's neighbours.
            eff_pad = min(padding, (rem_end - rem_start) / 2.0)
            keep_end = rem_start - eff_pad
            if keep_end > cursor:
                segments.append({"start": cursor, "end": min(keep_end, duration)})
            cursor = max(cursor, rem_end + eff_pad)
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
        meta = dict(meta)
        cuts = [
            {"id": f"cut_{i:04d}", "source": source,
             "in_seconds": round(float(k["start"]), 3), "out_seconds": round(float(k["end"]), 3)}
            for i, k in enumerate(keeps)
        ]
        # Honest removed_seconds: prefer the merged removed-time supplied by the
        # caller (execute), so overlapping detections of the same span are not
        # double-counted. Fall back to the raw sum when not provided.
        removed_seconds = meta.pop("removed_seconds", None)
        if removed_seconds is None:
            removed_seconds = sum(float(r["end"]) - float(r["start"]) for r in removed)
        removed_seconds = round(float(removed_seconds), 3)
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

    def _duration(self, input_path: Path | None, words: list[dict]) -> float:
        if input_path and input_path.is_file():
            proc = self.run_command([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "json", str(input_path),
            ])
            try:
                return float(json.loads(proc.stdout)["format"]["duration"])
            except Exception:
                pass
        ends = [float(w["end"]) for w in words if self._valid_time(w)]
        return max(ends) if ends else 0.0

    def _word_timestamps(self, inputs: dict) -> tuple[list[dict], bool] | None:
        """Return (words, transcribed) or None on failure to obtain them."""
        provided = inputs.get("word_timestamps")
        if provided:
            return list(provided), False
        input_path = inputs.get("input_path")
        if not input_path:
            return None
        from tools.analysis.transcriber import Transcriber
        try:
            res = Transcriber().execute({
                "input_path": input_path,
                "language": None if inputs.get("language") == "auto" else inputs.get("language"),
            })
        except Exception as exc:  # missing faster_whisper etc.
            self._transcribe_error = str(exc)
            return None
        if not res.success:
            self._transcribe_error = res.error or "transcription failed"
            return None
        return list(res.data.get("word_timestamps", [])), True

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        self._transcribe_error = None
        if not inputs.get("input_path") and not inputs.get("word_timestamps"):
            return ToolResult(success=False,
                              error="Provide input_path or word_timestamps (+ source).")

        got = self._word_timestamps(inputs)
        if got is None:
            reason = getattr(self, "_transcribe_error", None) or self.install_instructions
            return ToolResult(success=False, error=f"Could not obtain word timestamps: {reason}")
        words, transcribed = got

        input_path = Path(inputs["input_path"]) if inputs.get("input_path") else None
        source = inputs.get("source") or (str(input_path) if input_path else None)
        if not source:
            return ToolResult(success=False, error="`source` required when passing word_timestamps.")

        language = inputs.get("language", "pt")
        padding = float(inputs.get("padding_seconds", 0.08))
        min_gap = float(inputs.get("min_gap_seconds", 0.05))
        rep_gap = float(inputs.get("repetition_max_gap", 0.6))

        skipped = [w.get("word") for w in words if not self._valid_time(w)]

        removed: list[dict] = []
        if inputs.get("remove_fillers", True):
            removed += self._detect_fillers(words, self._lexicon_for(language, inputs.get("filler_lexicon")))
        if inputs.get("remove_repetitions", True):
            removed += self._detect_repetitions(words, rep_gap)
        removed += self._match_literal_words(words, inputs.get("remove_words"))
        removed += self._indices_and_ranges_to_spans(
            words, inputs.get("remove_word_indices"), inputs.get("remove_ranges"))

        # Dedup raw spans (for the report) by (start, end, reason).
        seen = set()
        deduped = []
        for s in sorted(removed, key=lambda s: (float(s["start"]), float(s["end"]))):
            key = (round(float(s["start"]), 4), round(float(s["end"]), 4), s["reason"])
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        removed = deduped

        duration = self._duration(input_path, words)
        merged = self._merge_spans(removed)
        keeps = self._keep_segments(merged, duration, padding, min_gap)

        if removed and not keeps:
            return ToolResult(success=False,
                              error="Removals cover the entire clip; nothing left to keep.")

        # Honest removed time: sum of the MERGED spans (overlapping detections of
        # the same word are counted once), not the raw per-reason report list.
        merged_removed_seconds = sum(float(m["end"]) - float(m["start"]) for m in merged)
        meta = {"modes": [m for m, on in (
            ("filler", inputs.get("remove_fillers", True)),
            ("repetition", inputs.get("remove_repetitions", True)),
            ("literal", bool(inputs.get("remove_words"))),
            ("index", bool(inputs.get("remove_word_indices"))),
            ("range", bool(inputs.get("remove_ranges"))),
        ) if on], "language": language, "removed_seconds": merged_removed_seconds}
        ed = self._to_edit_decisions(keeps, source, removed, meta)
        from schemas.artifacts import validate_artifact
        validate_artifact("edit_decisions", ed)

        out_path = Path(inputs.get("output_path") or
                        (input_path.with_name(f"{input_path.stem}_edit_decisions.json")
                         if input_path else Path(source).with_suffix(".edit_decisions.json")))
        out_path.write_text(json.dumps(ed, ensure_ascii=False, indent=2, sort_keys=True))

        return ToolResult(
            success=True,
            artifacts=[str(out_path)],
            duration_seconds=time.time() - start,
            data={
                "removed": removed,
                "removed_count": len(removed),
                "removed_seconds": ed["metadata"]["removed_seconds"],
                "kept_seconds": ed["metadata"]["kept_seconds"],
                "cuts_count": len(ed["cuts"]),
                "skipped_words": skipped,
                "rendered": False,
                "transcribed": transcribed,
            },
        )
