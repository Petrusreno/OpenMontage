# Text-Based Editor — Design Spec

> Sub-project 3 of 4 in **Premiere automation/render features → OpenMontage**.
> Date: 2026-07-08 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-text-based-editor`

## Context

Porting Premiere's **Text-Based Editing** (edit video by editing its transcript; one-click
"delete all filler words") into OpenMontage natively. This is behavior reverse-engineering, not
binary decompilation.

Evidence gathered 2026-07-08:
- **Word-level timestamps already exist.** `tools/analysis/transcriber.py` (provider `whisperx`,
  faster_whisper backend) emits `data["word_timestamps"] = [{word, start, end}, ...]` and
  `segments[].words`. `tools/analysis/whisperx_remote.py` emits the same. This is the exact input
  text-based editing needs.
- **The keep/cut/concat primitive already exists.** `tools/video/silence_cutter.py` computes
  "keep" segments as the complement of "remove" segments (with padding + tiny-gap merge) and
  renders by cutting each keep and concatenating. Text-based editing is the *same operation* with
  removal spans sourced from **words** instead of silence.
- **The output artifact already fits.** `schemas/artifacts/edit_decisions.schema.json` has
  `cuts[]` of `{id, source, in_seconds, out_seconds}` — one kept span → one cut. A validation
  helper exists: `schemas/artifacts.validate_artifact("edit_decisions", data)`.

Decisions locked during brainstorming:
- **v1 removal modes:** (a) automatic PT-BR filler-word removal, (b) explicit words/ranges.
  Transcript-diff editing (edit the text, diff against original) is deferred to v2.
- **Output:** always emit an `edit_decisions` artifact; optionally also render a cut MP4.
- **Filler detection is rule-based and deterministic** (no LLM in v1).

## Goal

A new tool `text_based_editor` that, given a video/audio clip (or precomputed word timestamps)
and a removal specification, emits an `edit_decisions` artifact whose `cuts[]` are the kept spans
(everything except the removed words), and optionally renders the cut MP4. Every removed span is
reported with the word text and the reason it was removed (traceable, never silent).

Non-goals (YAGNI): transcript-diff editing (v2), multi-speaker/diarization-aware editing,
snapping cut points to detected silence (v2 refinement), sentence-level semantic editing.

## Content-fidelity guard (hard requirement)

Deleting a meaningful word changes the author's content. Per the user's content-fidelity rule,
the tool must not silently remove ambiguous words that are often legitimate.

- **Default filler lexicon = high-confidence disfluencies only, and is language-specific**
  (selected by `language`), because a hesitation token in one language can be a real word in
  another:
  - `pt`: `ãã`, `ã`, `ahn`, `ãhn`, `hum`, `hmm`, `eh`, `ehh`, `ã-ã`. (Deliberately **excludes**
    `um`/`uh` — in PT-BR `um` is the article/numeral "one/a", a real word.)
  - `en`: `um`, `uh`, `uhm`, `hmm`, `er`, `err`, `ah`, `mm`, `mhm`.
- Ambiguous words that are frequently meaningful — `então`, `é`, `né`, `tipo`, `aí`, `assim`,
  `sabe`, `cara` (pt); `like`, `so`, `you know`, `actually` (en) — are **NOT** in the default
  lexicon. They are removed only if the caller explicitly lists them in `filler_lexicon` or
  `remove_words`.
- Repeated-word stutters (e.g. `é é`, `the the`) are handled by the repetition rule, not the
  lexicon, so the last (intended) occurrence is always kept.
- The report lists every removed span with `{word, start, end, reason}` so the caller can audit
  exactly what was cut before rendering.

## Architecture

One `BaseTool` subclass, `tools/video/text_based_editor.py`, mirroring the contract style of
`tools/video/silence_cutter.py`.

- **Contract:** `name="text_based_editor"`, `version="0.1.0"`, `tier=ToolTier.CORE`,
  `capability="video_post"`, `provider="whisperx+ffmpeg"`, `stability=EXPERIMENTAL`,
  `execution_mode=SYNC`, `determinism=Determinism.DETERMINISTIC`,
  `dependencies=["cmd:ffmpeg"]` (hard), `agent_skills=["ffmpeg","whisperx"]`,
  `capabilities=["text_based_editing","filler_removal","transcript_cut"]`.
- **Transcription dependency is conditional:** if `word_timestamps` are supplied, no transcription
  runs. If only `input_path` is given, the tool runs the existing `Transcriber` tool
  (`tools/analysis/transcriber.py`) to obtain word timestamps — which requires `faster_whisper`.
  That requirement is checked at runtime and, if missing, returns a clean
  `ToolResult(success=False, error=...)` telling the caller to install it or pass
  `word_timestamps`. Never crash.

### Data flow

```
input_path ──(if no word_timestamps)──► Transcriber ──► word_timestamps[{word,start,end}]
word_timestamps + removal spec ──► build remove_spans (filler + repetition + explicit)
remove_spans ──► merge overlaps ──► keep_segments = complement over [0, duration] (+padding, +gap-merge)
keep_segments ──► edit_decisions{cuts[]} ──validate("edit_decisions")──► write JSON
                └─(if render=true)─► cut each keep + concat (reuse silence_cutter render) ─► MP4
```

### Helper units (each independently testable)

- `_detect_fillers(words, lexicon) -> list[span]` — words whose normalized text is in the lexicon.
- `_detect_repetitions(words, max_gap_s) -> list[span]` — consecutive normalized-identical words
  within `max_gap_s`; removes the earlier duplicate(s), keeps the last occurrence.
- `_match_literal_words(words, remove_words) -> list[span]` — all occurrences of caller-listed
  words (normalized, case/punctuation-insensitive).
- `_indices_and_ranges_to_spans(words, remove_word_indices, remove_ranges) -> list[span]` —
  explicit word indices and explicit time ranges `{start_seconds,end_seconds}`.
- `_merge_spans(spans) -> list[span]` — sort + merge overlapping/adjacent removal spans.
- `_keep_segments(remove_spans, duration, padding, min_gap) -> list[{start,end}]` — complement
  with padding and tiny-gap merge (adapted from `silence_cutter._compute_speech_segments`).
- `_to_edit_decisions(keep_segments, source, removed, mode) -> dict` — build the schema-valid
  artifact.
- `_render_cuts(input_path, keep_segments, render_path) -> None` — cut+concat (reuse the proven
  `silence_cutter` render path).

Each `span` is `{start: float, end: float, word: str|None, reason: str}` where `reason ∈
{"filler","repetition","literal","index","range"}`.

## Input schema

```jsonc
{
  "type": "object",
  "required": [],                       // must supply input_path OR (word_timestamps AND source)
  "properties": {
    "input_path":        { "type": "string", "description": "Video/audio to edit and (if needed) transcribe." },
    "word_timestamps":   { "type": "array", "description": "Precomputed [{word,start,end}]; skips transcription." },
    "source":            { "type": "string", "description": "Value for cuts[].source; defaults to input_path." },
    "language":          { "type": "string", "default": "pt", "enum": ["pt", "en"] },
    "remove_fillers":    { "type": "boolean", "default": true },
    "remove_repetitions":{ "type": "boolean", "default": true },
    "filler_lexicon":    { "type": "array", "description": "Overrides/extends the default lexicon for `language`." },
    "remove_words":      { "type": "array", "description": "Literal words; all occurrences removed." },
    "remove_word_indices":{ "type": "array", "description": "Integer indices into word_timestamps." },
    "remove_ranges":     { "type": "array", "description": "Explicit time ranges [{start_seconds,end_seconds}]." },
    "padding_seconds":   { "type": "number", "default": 0.08, "minimum": 0.0 },
    "min_gap_seconds":   { "type": "number", "default": 0.05, "minimum": 0.0 },
    "repetition_max_gap":{ "type": "number", "default": 0.6, "minimum": 0.0 },
    "render":            { "type": "boolean", "default": false },
    "output_path":       { "type": "string", "description": "Where to write the edit_decisions JSON." },
    "render_path":       { "type": "string", "description": "Where to write the cut MP4 (render=true)." }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[edit_decisions_json_path]` (+ `[render_path]` when `render=true`)
- `data`:
  - `removed`: `list[{word, start, end, reason}]` — every removed span
  - `removed_count`: int
  - `removed_seconds`: float
  - `kept_seconds`: float
  - `cuts_count`: int
  - `rendered`: bool
  - `transcribed`: bool  (true if the tool ran the transcriber itself)
- `duration_seconds`: wall time

The emitted `edit_decisions` dict is `{"version":"1.0", "cuts":[...], "metadata":{...}}` and MUST
pass `validate_artifact("edit_decisions", data)` before it is written or returned.

## Error / edge-case handling

- **Neither `input_path` nor (`word_timestamps` + `source`) provided:** `success=False`, clear error.
- **`input_path` given, no `word_timestamps`, faster_whisper missing:** `success=False` instructing
  to install it or pass `word_timestamps`. No crash.
- **Word entries with missing/None `start`/`end`** (faster_whisper occasionally emits these): skip
  that word for removal (cannot cut what cannot be located) and add it to a `data["skipped_words"]`
  note; never fabricate a timestamp.
- **Nothing to remove:** emit an identity `edit_decisions` with a single full-length cut
  `[0, duration]`, `success=True`, `removed_count=0`.
- **Removals cover the entire clip:** `keep_segments` empty → `success=False` with a clear message
  (refuse to emit an empty edit).
- **ffmpeg non-zero / missing or zero-byte render output** (render path): `success=False`.
- Inputs validated at the boundary against `input_schema` before any ffmpeg/transcription work.

## Testing

`tests/tools/test_text_based_editor.py` (pytest, `tmp_path`, `monkeypatch`; import from
`tools.video.text_based_editor`). Tests that need word data pass `word_timestamps` directly so
they do not depend on a real transcription model.

1. **Filler detection:** synthetic words incl. `hum`, `ãã`, and a real word → only the disfluencies
   are flagged, each with `reason="filler"`.
2. **Content-fidelity guard:** words incl. `então`, `é`, `né` with default lexicon → NONE removed;
   after adding them to `filler_lexicon` → removed. (Locks the hard requirement.)
3. **Repetition:** `["o","o","gato"]` close together → the first `o` flagged `reason="repetition"`,
   the second `o` kept.
4. **Literal words / indices / ranges:** each explicit mode removes exactly the intended spans.
5. **Keep-segment computation:** given removal spans over a known duration, the complementary keeps
   (with padding + gap-merge) are correct and cover `duration - removed`.
6. **edit_decisions validity:** emitted artifact passes `validate_artifact("edit_decisions", ...)`;
   `cuts[]` are ordered, non-overlapping, and reference `source`.
7. **Identity case:** no removals → exactly one cut `[0, duration]`, `success=True`.
8. **Everything removed:** `success=False`, no artifact written.
9. **Determinism:** same inputs → byte-identical edit_decisions JSON.
10. **Render (ffmpeg-gated):** `render=true` on a real short clip with known word timestamps
    produces an MP4 shorter than the input, audio preserved, and the JSON alongside it.

## Integration touchpoints

- Discoverable via `tools/tool_registry.py` (drop-in, no wiring).
- Emits the canonical `edit_decisions` artifact → any pipeline's edit/compose stage can consume it;
  `render=true` gives a standalone MP4 for quick inspection.
- No schema changes (uses existing `edit_decisions.schema.json`). Reuses `Transcriber` and the
  `silence_cutter` render approach rather than reimplementing them.

## Open decisions (defaults chosen, override at implementation)

- Default `padding_seconds=0.08` and `min_gap_seconds=0.05` match `silence_cutter` conventions.
- `repetition_max_gap=0.6s` — consecutive identical words farther apart than this are treated as
  intentional repetition, not a stutter; tune once tested on real speech.
- Snapping cut points to the nearest silence for smoother audio is deferred to v2.
