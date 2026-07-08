# Text-Based Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `text_based_editor` tool that removes filler words / caller-specified words from a clip by emitting an `edit_decisions` artifact of kept spans (optionally rendering the cut MP4).

**Architecture:** One `BaseTool` subclass in `tools/video/text_based_editor.py`, contract-styled after `tools/video/silence_cutter.py`. It obtains word-level timestamps (supplied by the caller, or via the existing `Transcriber` tool), builds removal spans from a language-specific filler lexicon + repetition heuristic + explicit caller input, computes the complementary kept segments (reusing silence_cutter's padding/gap-merge logic), and emits a schema-validated `edit_decisions` artifact. `render=true` also produces a cut MP4 via cut+concat.

**Tech Stack:** Python 3.14, ffmpeg/ffprobe (subprocess via `BaseTool.run_command`), `faster_whisper` (only for the self-transcription path), `jsonschema` (via `schemas/artifacts.validate_artifact`), pytest.

## Global Constraints

- Tool inherits `tools/base_tool.py` `BaseTool` with full contract fields in the style of `tools/video/silence_cutter.py`. `execute(self, inputs: dict[str, Any]) -> ToolResult`.
- **Content-fidelity (hard):** the default filler lexicon is language-specific and contains ONLY high-confidence disfluencies. `pt` = `{"ãã","ã","ahn","ãhn","hum","hmm","eh","ehh"}` (NO `um`/`uh` — `um` is a real PT word). `en` = `{"um","uh","uhm","hmm","er","err","ah","mm","mhm"}`. Ambiguous words (`então`,`é`,`né`,`tipo`,`aí`,`assim`,`like`,`so`) are removed ONLY when the caller passes them in `filler_lexicon` or `remove_words`.
- Every removed span is reported as `{word, start, end, reason}` with `reason ∈ {"filler","repetition","literal","index","range"}`. Never silently remove.
- Emitted `edit_decisions` MUST pass `validate_artifact("edit_decisions", data)` before writing/returning. Shape: `{"version":"1.0","cuts":[{"id","source","in_seconds","out_seconds"}],"metadata":{...}}`.
- Never fabricate: words missing numeric `start`/`end` are skipped (reported in `data["skipped_words"]`), not assigned invented timestamps. If removals cover the whole clip (no kept spans) → `success=False`, no artifact written.
- No new third-party Python deps. Tests in `tests/tools/test_text_based_editor.py`, import from `tools.video.text_based_editor`; tests pass `word_timestamps` directly so they never require a real transcription model (except the ffmpeg-gated render test).
- Determinism: identical inputs → byte-identical `edit_decisions` JSON.

---

### Task 1: Tool skeleton, contract, and registry discovery

**Files:**
- Create: `tools/video/text_based_editor.py`
- Test: `tests/tools/test_text_based_editor.py`

**Interfaces:**
- Produces: `class TextBasedEditor(BaseTool)` with contract fields and `input_schema`. `execute(self, inputs)` returns `ToolResult(success=False, error="not implemented")` for now.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_text_based_editor.py
from __future__ import annotations

from tools.video.text_based_editor import TextBasedEditor
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = TextBasedEditor()
    assert tool.name == "text_based_editor"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "text_based_editing" in tool.capabilities
    assert "input_path" in tool.input_schema["properties"]
    assert "word_timestamps" in tool.input_schema["properties"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py::test_contract_fields_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.video.text_based_editor'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/video/text_based_editor.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py::test_contract_fields_present -v`
Expected: PASS

- [ ] **Step 5: Verify discovery, then commit**

Run: `.venv/bin/python -c "import tools.video.text_based_editor as m; print(m.TextBasedEditor().name)"`
Expected: prints `text_based_editor`

```bash
git add tools/video/text_based_editor.py tests/tools/test_text_based_editor.py
git commit -m "feat: text_based_editor tool skeleton + contract"
```

---

### Task 2: Word normalization, filler + repetition detection

**Files:**
- Modify: `tools/video/text_based_editor.py`
- Test: `tests/tools/test_text_based_editor.py`

**Interfaces:**
- Produces:
  - `TextBasedEditor._normalize(word: str) -> str` — lowercased, punctuation/whitespace stripped, accents preserved.
  - `_DEFAULT_LEXICONS: dict[str, set[str]]` — normalized `pt`/`en` disfluency sets.
  - `_lexicon_for(language, filler_lexicon) -> set[str]` — default set for language, plus any caller additions (normalized).
  - `_detect_fillers(words, lexicon) -> list[dict]` — spans `{start,end,word,reason:"filler"}`.
  - `_detect_repetitions(words, max_gap) -> list[dict]` — spans `{...,reason:"repetition"}` for all-but-last of a consecutive equal-normalized run within `max_gap`.
  - A span is `{"start":float,"end":float,"word":str|None,"reason":str}`. Words with non-numeric `start`/`end` are ignored by detectors.

- [ ] **Step 1: Write the failing tests**

```python
def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_detect_fillers_pt_flags_only_disfluencies():
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.3), _w(" gato", 0.3, 0.7), _w("ãã", 0.7, 1.0)]
    lex = tool._lexicon_for("pt", None)
    spans = tool._detect_fillers(words, lex)
    got = sorted((round(s["start"], 2), s["word"]) for s in spans)
    assert got == [(0.0, "hum"), (0.7, "ãã")]
    assert all(s["reason"] == "filler" for s in spans)


def test_content_fidelity_default_keeps_ambiguous_words():
    tool = TextBasedEditor()
    words = [_w("então", 0.0, 0.4), _w("é", 0.4, 0.6), _w("né", 0.6, 0.9)]
    lex = tool._lexicon_for("pt", None)
    assert tool._detect_fillers(words, lex) == []          # default removes none
    lex2 = tool._lexicon_for("pt", ["então", "é", "né"])   # opt-in
    assert len(tool._detect_fillers(words, lex2)) == 3


def test_detect_repetitions_removes_all_but_last():
    tool = TextBasedEditor()
    words = [_w("o", 0.0, 0.2), _w("o", 0.25, 0.45), _w("gato", 0.5, 0.9)]
    spans = tool._detect_repetitions(words, 0.6)
    assert len(spans) == 1
    assert round(spans[0]["start"], 2) == 0.0 and spans[0]["reason"] == "repetition"


def test_detect_repetitions_respects_gap():
    tool = TextBasedEditor()
    words = [_w("sim", 0.0, 0.2), _w("sim", 2.0, 2.2)]   # 1.8s apart > max_gap
    assert tool._detect_repetitions(words, 0.6) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k "fillers or fidelity or repetition" -v`
Expected: FAIL with `AttributeError: ... '_lexicon_for'` / `'_detect_fillers'`

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import re
```

Add class attributes and methods:

```python
    _DEFAULT_LEXICONS = {
        "pt": {"ãã", "ã", "ahn", "ãhn", "hum", "hmm", "eh", "ehh"},
        "en": {"um", "uh", "uhm", "hmm", "er", "err", "ah", "mm", "mhm"},
    }

    @staticmethod
    def _normalize(word: str) -> str:
        return re.sub(r"[^\w]", "", (word or ""), flags=re.UNICODE).lower()

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
            if self._valid_time(w) and self._normalize(w["word"]) in lexicon:
                spans.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": w["word"].strip(), "reason": "filler"})
        return spans

    def _detect_repetitions(self, words: list[dict], max_gap: float) -> list[dict]:
        spans = []
        run_start = 0
        valid = [w for w in words if self._valid_time(w)]
        i = 0
        while i < len(valid):
            j = i
            while (j + 1 < len(valid)
                   and self._normalize(valid[j + 1]["word"]) == self._normalize(valid[i]["word"])
                   and self._normalize(valid[i]["word"]) != ""
                   and float(valid[j + 1]["start"]) - float(valid[j]["end"]) <= max_gap):
                j += 1
            if j > i:  # run of length >= 2: remove all but the last (index j)
                for k in range(i, j):
                    spans.append({"start": float(valid[k]["start"]), "end": float(valid[k]["end"]),
                                  "word": valid[k]["word"].strip(), "reason": "repetition"})
            i = j + 1
        return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k "fillers or fidelity or repetition" -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/video/text_based_editor.py tests/tools/test_text_based_editor.py
git commit -m "feat: text_based_editor filler + repetition detection"
```

---

### Task 3: Explicit removals and span merging

**Files:**
- Modify: `tools/video/text_based_editor.py`
- Test: `tests/tools/test_text_based_editor.py`

**Interfaces:**
- Consumes: `_normalize`, `_valid_time` from Task 2.
- Produces:
  - `_match_literal_words(words, remove_words) -> list[dict]` — spans `reason:"literal"` for every word whose normalized text is in the normalized `remove_words` set.
  - `_indices_and_ranges_to_spans(words, indices, ranges) -> list[dict]` — spans `reason:"index"` for valid `remove_word_indices`; spans `reason:"range"` (`word=None`) for each `{start_seconds,end_seconds}` in `remove_ranges`.
  - `_merge_spans(spans) -> list[dict]` — sorted, overlapping/adjacent-merged `{start,end}` list (used only for keep computation; the per-word report keeps the raw spans).

- [ ] **Step 1: Write the failing tests**

```python
def test_match_literal_words_all_occurrences():
    tool = TextBasedEditor()
    words = [_w("Tá", 0.0, 0.2), _w("bom", 0.2, 0.5), _w("tá!", 0.5, 0.7)]
    spans = tool._match_literal_words(words, ["ta"])
    assert len(spans) == 2 and all(s["reason"] == "literal" for s in spans)


def test_indices_and_ranges_to_spans():
    tool = TextBasedEditor()
    words = [_w("a", 0.0, 0.2), _w("b", 0.2, 0.4), _w("c", 0.4, 0.6)]
    spans = tool._indices_and_ranges_to_spans(
        words, indices=[1], ranges=[{"start_seconds": 1.0, "end_seconds": 1.5}])
    reasons = sorted(s["reason"] for s in spans)
    assert reasons == ["index", "range"]
    idx = next(s for s in spans if s["reason"] == "index")
    assert round(idx["start"], 2) == 0.2


def test_merge_spans_combines_overlaps():
    tool = TextBasedEditor()
    spans = [{"start": 0.0, "end": 0.3}, {"start": 0.25, "end": 0.5}, {"start": 1.0, "end": 1.2}]
    merged = tool._merge_spans(spans)
    assert merged == [{"start": 0.0, "end": 0.5}, {"start": 1.0, "end": 1.2}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k "literal or indices or merge_spans" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k "literal or indices or merge_spans" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/video/text_based_editor.py tests/tools/test_text_based_editor.py
git commit -m "feat: text_based_editor explicit removals + span merge"
```

---

### Task 4: Keep-segment computation and edit_decisions emission

**Files:**
- Modify: `tools/video/text_based_editor.py`
- Test: `tests/tools/test_text_based_editor.py`

**Interfaces:**
- Consumes: `_merge_spans` from Task 3.
- Produces:
  - `_keep_segments(merged, duration, padding, min_gap) -> list[dict]` — complement of `merged` over `[0, duration]`, padding kept spans `padding` into removals, tiny segments (<0.01s) dropped, gaps < `min_gap` merged. (Adapted from `silence_cutter._compute_speech_segments`.)
  - `_to_edit_decisions(keeps, source, removed, meta) -> dict` — a `validate_artifact("edit_decisions", ...)`-valid dict.

- [ ] **Step 1: Write the failing tests**

```python
from schemas.artifacts import validate_artifact


def test_keep_segments_is_complement_with_padding():
    tool = TextBasedEditor()
    merged = [{"start": 1.0, "end": 2.0}]           # remove 1..2 of a 3s clip
    keeps = tool._keep_segments(merged, duration=3.0, padding=0.0, min_gap=0.05)
    assert keeps == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]


def test_keep_segments_identity_when_no_removals():
    tool = TextBasedEditor()
    keeps = tool._keep_segments([], duration=5.0, padding=0.0, min_gap=0.05)
    assert keeps == [{"start": 0.0, "end": 5.0}]


def test_to_edit_decisions_validates_against_schema():
    tool = TextBasedEditor()
    keeps = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    ed = tool._to_edit_decisions(keeps, source="clip.mp4", removed=[], meta={"modes": ["filler"]})
    validate_artifact("edit_decisions", ed)          # raises on failure
    assert ed["version"] == "1.0"
    assert [c["in_seconds"] for c in ed["cuts"]] == [0.0, 2.0]
    assert all(c["source"] == "clip.mp4" for c in ed["cuts"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k "keep_segments or to_edit_decisions" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k "keep_segments or to_edit_decisions" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/video/text_based_editor.py tests/tools/test_text_based_editor.py
git commit -m "feat: text_based_editor keep segments + edit_decisions emission"
```

---

### Task 5: Full execute — orchestration, transcription, artifact write, edge cases

**Files:**
- Modify: `tools/video/text_based_editor.py`
- Test: `tests/tools/test_text_based_editor.py`

**Interfaces:**
- Consumes: all detectors/helpers from Tasks 2–4.
- Produces: full `execute(inputs)` returning `ToolResult(success=True, artifacts=[edit_decisions_path], data={...})`. Adds `_duration(input_path, words) -> float`, `_word_timestamps(inputs) -> tuple[list[dict], bool] | None`.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path


def test_execute_emits_edit_decisions_from_word_timestamps(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.4), _w("olá", 0.4, 0.8), _w("mundo", 0.8, 1.2)]
    out = tmp_path / "ed.json"
    result = tool.execute({
        "word_timestamps": words, "source": "clip.mp4", "language": "pt",
        "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists()
    ed = json.loads(out.read_text())
    validate_artifact("edit_decisions", ed)
    # "hum" (0.0-0.4) removed → first kept cut starts at/after 0.4
    assert ed["cuts"][0]["in_seconds"] >= 0.4 - 1e-9
    assert result.data["removed_count"] == 1
    assert result.data["removed"][0]["reason"] == "filler"


def test_execute_identity_when_nothing_removed(tmp_path):
    tool = TextBasedEditor()
    words = [_w("olá", 0.0, 0.5), _w("mundo", 0.5, 1.0)]
    out = tmp_path / "ed.json"
    result = tool.execute({"word_timestamps": words, "source": "c.mp4",
                           "remove_fillers": False, "output_path": str(out)})
    assert result.success
    ed = json.loads(out.read_text())
    assert len(ed["cuts"]) == 1
    assert ed["cuts"][0]["in_seconds"] == 0.0
    assert result.data["removed_count"] == 0


def test_execute_fails_when_everything_removed(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 1.0)]
    out = tmp_path / "ed.json"
    result = tool.execute({"word_timestamps": words, "source": "c.mp4",
                           "output_path": str(out)})
    assert not result.success
    assert not out.exists()


def test_execute_requires_input_or_words():
    tool = TextBasedEditor()
    result = tool.execute({})
    assert not result.success and "input_path" in (result.error or "")


def test_execute_is_deterministic(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.4), _w("olá", 0.4, 0.8), _w("olá", 0.85, 1.2), _w("fim", 1.3, 1.6)]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    tool.execute({"word_timestamps": words, "source": "c.mp4", "output_path": str(a)})
    tool.execute({"word_timestamps": words, "source": "c.mp4", "output_path": str(b)})
    assert a.read_text() == b.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k execute -v`
Expected: FAIL — `execute` returns the `not implemented` stub, so `result.success` is False for the success-path tests.

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import json
import time
from pathlib import Path
```

Replace the placeholder `execute` and add helpers:

```python
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

        meta = {"modes": [m for m, on in (
            ("filler", inputs.get("remove_fillers", True)),
            ("repetition", inputs.get("remove_repetitions", True)),
            ("literal", bool(inputs.get("remove_words"))),
            ("index", bool(inputs.get("remove_word_indices"))),
            ("range", bool(inputs.get("remove_ranges"))),
        ) if on], "language": language}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -k execute -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/video/text_based_editor.py tests/tools/test_text_based_editor.py
git commit -m "feat: text_based_editor execute orchestration + edge cases"
```

---

### Task 6: Optional render (cut + concat MP4)

**Files:**
- Modify: `tools/video/text_based_editor.py`
- Test: `tests/tools/test_text_based_editor.py`

**Interfaces:**
- Consumes: `keeps` computed in `execute`; `run_command` from `BaseTool`.
- Produces: `_render_cuts(input_path, keeps, render_path) -> str | None` (returns the written path, or None on failure). `execute` calls it when `inputs["render"]` is true and `input_path` is set, adds `render_path` to `artifacts`, sets `data["rendered"]=True`.

- [ ] **Step 1: Write the failing test**

```python
import shutil
import subprocess as _sp


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_render_produces_shorter_clip(tmp_path):
    src = tmp_path / "src.mp4"
    # 3s clip with an audio track so -c:a copy path is exercised.
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-pix_fmt", "yuv420p", "-shortest", str(src),
    ], check=True, capture_output=True)
    # Remove the middle second by explicit range.
    words = [_w("a", 0.0, 0.9), _w("b", 1.0, 2.0), _w("c", 2.1, 3.0)]
    out = tmp_path / "ed.json"
    render = tmp_path / "cut.mp4"
    result = TextBasedEditor().execute({
        "input_path": str(src), "word_timestamps": words, "source": str(src),
        "remove_fillers": False, "remove_ranges": [{"start_seconds": 1.0, "end_seconds": 2.0}],
        "output_path": str(out), "render": True, "render_path": str(render),
    })
    assert result.success, result.error
    assert render.exists() and render.stat().st_size > 0
    assert result.data["rendered"] is True
    # Rendered duration should be < source duration.
    def _dur(p):
        r = _sp.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        return float(r.stdout.strip())
    assert _dur(render) < _dur(src) - 0.3
```

Import guard at the top of the test file if not present already:

```python
import pytest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py::test_execute_render_produces_shorter_clip -v`
Expected: FAIL — `execute` ignores `render`, so `result.data["rendered"]` is False and `render` is not created.

- [ ] **Step 3: Write minimal implementation**

Add at top of file:

```python
import tempfile
import shutil
```

Add the render helper:

```python
    def _render_cuts(self, input_path: Path, keeps: list[dict], render_path: Path) -> str | None:
        if not keeps:
            return None
        workdir = Path(tempfile.mkdtemp(prefix="tbe_render_"))
        try:
            seg_files = []
            for i, k in enumerate(keeps):
                seg = workdir / f"seg_{i:04d}.mp4"
                self.run_command([
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-ss", f"{float(k['start']):.3f}", "-to", f"{float(k['end']):.3f}",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "192k",
                    "-force_key_frames", f"{float(k['start']):.3f}", str(seg),
                ], timeout=300)
                if seg.is_file() and seg.stat().st_size > 0:
                    seg_files.append(seg)
            if not seg_files:
                return None
            list_path = workdir / "concat.txt"
            list_path.write_text("".join(f"file '{sf.resolve()}'\n" for sf in seg_files))
            self.run_command([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
                "-c", "copy", str(render_path),
            ], timeout=300)
            if render_path.is_file() and render_path.stat().st_size > 0:
                return str(render_path)
            return None
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
```

In `execute`, replace the final `return ToolResult(...)` block so it renders when requested. Insert BEFORE building the `data` dict:

```python
        rendered_path = None
        if inputs.get("render") and input_path is not None:
            default_render = input_path.with_name(f"{input_path.stem}_cut.mp4")
            rendered_path = self._render_cuts(
                input_path, keeps, Path(inputs.get("render_path") or default_render))
            if rendered_path is None:
                return ToolResult(success=False, error="Render failed (cut/concat produced no output).")
```

Then change the return to include the render artifact and flag:

```python
        artifacts = [str(out_path)]
        if rendered_path:
            artifacts.append(rendered_path)
        return ToolResult(
            success=True,
            artifacts=artifacts,
            duration_seconds=time.time() - start,
            data={
                "removed": removed,
                "removed_count": len(removed),
                "removed_seconds": ed["metadata"]["removed_seconds"],
                "kept_seconds": ed["metadata"]["kept_seconds"],
                "cuts_count": len(ed["cuts"]),
                "skipped_words": skipped,
                "rendered": rendered_path is not None,
                "transcribed": transcribed,
            },
        )
```

(Delete the old `return ToolResult(...)` block from Task 5 so there is exactly one return at the end.)

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `.venv/bin/python -m pytest tests/tools/test_text_based_editor.py -v`
Expected: all PASS (the render test runs when ffmpeg is present).

- [ ] **Step 5: Commit**

```bash
git add tools/video/text_based_editor.py tests/tools/test_text_based_editor.py
git commit -m "feat: text_based_editor optional cut+concat render"
```

---

## Self-Review

**Spec coverage:**
- Contract mirroring `silence_cutter` → Task 1. ✓
- Language-specific default lexicon + content-fidelity guard → Task 2 (+ locked by `test_content_fidelity_default_keeps_ambiguous_words`). ✓
- Filler + repetition detection → Task 2; explicit words/indices/ranges → Task 3. ✓
- Span merge + keep-segment complement (padding/gap-merge, reused from silence_cutter) → Tasks 3–4. ✓
- edit_decisions emission validated via `validate_artifact` → Task 4 (+ Task 5 writes it). ✓
- Obtain word_timestamps from caller or Transcriber; faster_whisper-missing → clean failure → Task 5 (`_word_timestamps`). ✓
- Skipped words (missing timestamps) reported, never fabricated → Task 5. ✓
- Identity (no removals) and everything-removed cases → Task 5 tests. ✓
- Determinism (byte-identical JSON via `sort_keys`) → Task 5 test. ✓
- Optional render (cut+concat, audio preserved, temp cleaned) → Task 6. ✓
- ToolResult `data` fields (removed/removed_count/removed_seconds/kept_seconds/cuts_count/rendered/transcribed) → Tasks 5–6. ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code. ✓

**Type consistency:** span dict `{start:float,end:float,word:str|None,reason:str}` consistent across `_detect_fillers`/`_detect_repetitions`/`_match_literal_words`/`_indices_and_ranges_to_spans`; `_merge_spans` consumes those and returns `{start,end}`; `_keep_segments` returns `{start,end}`; `_to_edit_decisions` consumes keeps+removed; `execute` threads all of them and `_word_timestamps` returns `tuple[list[dict], bool] | None`. `data["removed_seconds"]`/`kept_seconds` sourced from `ed["metadata"]` in both Task 5 and Task 6 returns. ✓

**Deferred vs spec:** Snapping cut points to silence and transcript-diff editing are explicitly v2 per the spec; not in this plan by design. Noted, not silently dropped.
