# Multicam Auto-Sync — Design Spec

> Sub-project 4 of 4 in **Premiere automation/render features → OpenMontage**.
> Date: 2026-07-09 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-multicam-sync`

## Context

Porting Premiere's **multicam auto-sync by audio waveform** (synchronize several angles/recorders
by cross-correlating their audio) into OpenMontage natively. Behavior reverse-engineering, not
binary decompilation.

Evidence gathered 2026-07-09:
- **numpy 2.4.4 is available; scipy is NOT.** So cross-correlation must be pure-numpy (FFT via
  `np.fft`), not `scipy.signal.fftconvolve`.
- **No raw-PCM extraction to reuse.** `tools/analysis/audio_energy.py` uses ffmpeg's ebur128
  loudness meter, not raw samples. We extract mono PCM ourselves via ffmpeg
  (`-f s16le -ac 1 -ar <rate>`) + `np.frombuffer`.
- **Dependency prefixes:** `BaseTool.check_dependencies` only honors `cmd:` / `env:` / `python:`.
  `audio_energy` declares `binary:ffmpeg`, which is silently NOT checked (a latent bug in 4 tools,
  out of scope here). The correct, checked form — with precedent in
  `green_screen_composite.py` (`["cmd:ffmpeg", "python:numpy", "python:PIL"]`) — is what we use.

Decisions locked during brainstorming:
- **Output:** an offsets **report** artifact (JSON), no rendering. A downstream stage (or the
  editor) applies the offsets. `render` is out of scope for v1.
- **Reference selection:** auto — anchor on the clip that **starts earliest** (so every other
  clip's offset is ≥ 0, which is intuitive). Caller may override with `reference_index`.

## Goal

A new tool `multicam_sync` that, given ≥2 clips recording the same event (each with an audio
track), computes each clip's time **offset** relative to a common timeline by audio
cross-correlation, and emits a JSON report: per clip `{source, offset_seconds, confidence}` plus
which clip is the reference (offset 0). Every offset carries a confidence so the caller knows when
a match is trustworthy vs. a guess. Nothing is fabricated: a clip whose audio can't be located or
correlated is reported with `offset_seconds: null` and a reason, never an invented number.

Non-goals (YAGNI): rendering/aligning the clips (a downstream concern), drift/clock-skew
correction, multi-track/embedded-timecode sync, >~1h clips at full resolution, video-based
(visual flash) sync.

## Sign convention — locked by test, not by prose

The easy thing to get wrong in cross-correlation is the **sign** of the lag (which clip is ahead).
This spec deliberately does NOT assert the sign in prose. Instead:

- `offset_seconds[i]` is defined operationally as *"how many seconds clip i's t=0 must be shifted
  later on the common timeline so its audio aligns with the reference"*, with the earliest-starting
  clip as reference (offset 0) and all others ≥ 0.
- The implementation's sign convention MUST be pinned by a test that builds a synthetic pair with a
  **known delay** (a shared signal, one copy prepended with N seconds of silence) and asserts the
  recovered offset equals that known delay within tolerance. If the sign is inverted, that test
  fails. This is the honest, verifiable anchor.

## Architecture

One `BaseTool` subclass, `tools/audio/multicam_sync.py`, contract-styled after
`tools/analysis/audio_energy.py`.

- **Contract:** `name="multicam_sync"`, `version="0.1.0"`, `tier=ToolTier.CORE`,
  `capability="analysis"`, `provider="ffmpeg+numpy"`, `stability=EXPERIMENTAL`,
  `execution_mode=SYNC`, `determinism=Determinism.DETERMINISTIC`,
  `dependencies=["cmd:ffmpeg", "python:numpy"]`, `agent_skills=["ffmpeg"]`,
  `capabilities=["multicam_sync","audio_sync","waveform_alignment"]`.

### Data flow

```
clips[] ──► for each: extract mono PCM @ sample_rate (ffmpeg -f s16le -ac 1 -ar R), bounded to
            the first `window_seconds` ──► np.frombuffer -> float32 samples (normalized)
reference pick ──► correlate every clip against a common pivot (clip 0) via FFT cross-correlation
                    ──► raw lag_i (seconds) + normalized peak (confidence_i)
re-baseline ──► reference = the clip whose audio starts earliest (min real-world start)
                 ──► offset_i = start_i - min_start  (reference offset = 0, others ≥ 0)
emit ──► JSON report {version, reference_index, reference_source, sample_rate, window_seconds,
                       offsets:[{index, source, offset_seconds, confidence}], skipped:[...]}
```

### Helper units (each independently testable)

- `_extract_samples(path, sample_rate, window_seconds) -> np.ndarray | None` — mono float32
  samples via ffmpeg s16le; `None` (not a fabricated array) if the clip has no audio or ffmpeg
  fails. Bounded to `window_seconds` to keep FFT tractable and to avoid silent unbounded compute.
- `_has_audio(path) -> bool` — ffprobe for an audio stream (mirrors the pattern used in #3).
- `_xcorr_lag(a, b, sample_rate) -> tuple[float, float]` — FFT cross-correlation of two sample
  arrays; returns `(lag_seconds, confidence)` where confidence is the peak normalized by
  `sqrt(energy_a * energy_b)` (∈ [0, 1], 0 if either is silent). Pure numpy `np.fft.rfft`/`irfft`.
- `_pairwise_offsets(samples_list, sample_rate) -> list[tuple[float, float]]` — lag+confidence of
  each clip vs. the pivot (clip 0), pivot = `(0.0, 1.0)`.
- `_rebaseline(pairwise, reference_index=None) -> tuple[int, list[float]]` — convert pivot-relative
  lags into common-timeline offsets. If `reference_index is None` (auto): pick the earliest-starting
  clip as reference and shift so all offsets are ≥ 0. If `reference_index` is given: set that clip's
  offset to exactly 0 and express the others relative to it (offsets may be negative). Returns
  `(reference_index, offsets)`.
- `_to_report(clips, reference_index, offsets, confidences, skipped, params) -> dict` — assemble
  the JSON report.

## Input schema

```jsonc
{
  "type": "object",
  "required": ["clips"],
  "properties": {
    "clips":           { "type": "array", "items": { "type": "string" },
                         "minItems": 2, "description": "Paths to the clips to synchronize." },
    "reference_index": { "type": "integer", "minimum": 0,
                         "description": "Override: force this clip (0-based) as the offset-0 anchor. Other offsets become relative to it and MAY be negative (a clip that started earlier). Without this, auto mode anchors the earliest-starting clip so all offsets are ≥ 0." },
    "sample_rate":     { "type": "integer", "default": 8000, "minimum": 1000,
                         "description": "Mono resample rate for correlation. Lower = faster, coarser." },
    "window_seconds":  { "type": "number", "default": 60, "minimum": 1,
                         "description": "Correlate only the first N seconds of each clip (bounds compute)." },
    "min_confidence":  { "type": "number", "default": 0.1, "minimum": 0, "maximum": 1,
                         "description": "Offsets below this confidence are flagged low_confidence in the report." },
    "output_path":     { "type": "string", "description": "Where to write the sync report JSON." }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[sync_report_json_path]`
- `data`:
  - `reference_index`: int
  - `reference_source`: str
  - `sample_rate`: int
  - `window_seconds`: number
  - `offsets`: `list[{index, source, offset_seconds, confidence, low_confidence: bool}]`
    (`offset_seconds` is `null` for a skipped clip)
  - `skipped`: `list[{index, source, reason}]` — clips with no audio / extraction failure
  - `max_confidence`, `min_confidence_observed`: numbers (quick health read)
- `duration_seconds`: wall time

The report JSON is written with `sort_keys=True` for byte-stable determinism.

## Error / edge-case handling

- **< 2 clips** (after schema check): `success=False`, clear error.
- **A clip has no audio / ffmpeg extraction fails:** it goes into `skipped` with a reason and
  `offset_seconds: null`; it does NOT abort the whole run as long as ≥ 2 clips remain correlatable.
- **Fewer than 2 correlatable clips remain:** `success=False` (can't sync a single clip).
- **`reference_index` out of range or points at a skipped clip:** `success=False`, clear error
  (don't silently fall back to auto — the caller asked for a specific anchor).
- **All-silent / no correlation peak** for a clip: `confidence=0`, offset still computed from the
  (degenerate) peak but flagged `low_confidence`; never dropped silently — the low number IS the
  signal that the sync is untrustworthy.
- **ffmpeg / ffprobe binary missing:** guarded so the tool returns `success=False` (never crashes),
  consistent with the honest-gate principle used across these ports.
- Inputs validated at the boundary against `input_schema` before any ffmpeg work.

## Testing

`tests/tools/test_multicam_sync.py` (pytest; import from `tools.audio.multicam_sync`). Correlation
tests build synthetic sample arrays directly (numpy) so they do not need real media; the
ffmpeg-gated tests generate tiny clips.

1. **Sign + magnitude (core, locks the convention):** build a base signal; clip A = signal,
   clip B = `N` seconds of silence + signal. `_xcorr_lag`/full pipeline recovers B's offset ≈ `N`
   within tolerance (one sample period). If the sign is inverted, this fails.
2. **Confidence is discriminative:** two correlated signals → confidence near 1; a signal vs.
   independent random noise → confidence near 0.
3. **Reference = earliest start:** three synthetic clips with known relative starts → the earliest
   is chosen as reference (offset 0) and all other offsets are ≥ 0 and match the known deltas.
4. **`reference_index` override:** forcing a non-earliest clip as reference sets that clip's offset
   to exactly 0; the other offsets are relative to it and MAY be negative (a clip that started
   earlier gets a negative offset). Assert the forced reference is 0 and the relative spacing
   between clips matches the auto-mode result shifted by the forced reference's auto offset. (Auto
   mode's ≥ 0 guarantee applies ONLY to auto mode, not to an explicit override.)
5. **No-audio clip skipped, not fabricated:** a clip reported in `skipped` with `offset_seconds`
   absent/null; run still succeeds if ≥ 2 clips remain.
6. **< 2 correlatable clips → failure:** `success=False`, no fabricated offsets.
7. **Determinism:** same inputs → byte-identical report JSON.
8. **Report shape:** emitted JSON has the documented keys; offsets ordered by clip index.
9. **ffmpeg-gated end-to-end:** two tiny real clips built from the same synthetic audio, one
   delayed by a known amount, produce a recovered offset within tolerance and a written JSON.

## Integration touchpoints

- Discoverable via `tools/tool_registry.py` (drop-in, no wiring).
- Emits a self-contained sync-report artifact; a downstream align/compose step (or the editor)
  consumes `offsets` to place each clip on the timeline. No `edit_decisions` change — multicam sync
  produces alignment metadata, not cuts.
- Reuses ffmpeg (already a hard dep) and numpy (already installed); no new third-party dependency.

## Open decisions (defaults chosen, override at implementation)

- Default `sample_rate=8000` Hz mono and `window_seconds=60` bound compute while staying accurate
  to well under a frame for speech/transient content; tune once tested on real footage.
- `min_confidence=0.1` for the `low_confidence` flag is a starting threshold; calibrate on real
  multi-recorder audio.
- Rendering aligned clips and drift correction are explicitly deferred (not in v1).
