# Beat Sync — Design Spec

> Automation port (CapCut "auto beat" / beat-synced editing → OpenMontage).
> Sub-project 3 of the second batch (morph-cut ✓ · voice-isolation ✓ · beat-sync · react perf-refinement).
> Date: 2026-07-11 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-beat-sync`

## Context

Porting CapCut's **auto beat-sync** editing: detect the beats in a music bed and align cut points to
them. Behavior reverse-engineering. High creative value for viral reels/montages; genuinely absent
from the tool inventory.

Evidence gathered 2026-07-11 (empirical, before speccing):
- **No beat/onset/tempo tool exists** (grep over `tools/` found only substring false positives).
- **librosa, aubio, and scipy are all ABSENT** — so beat detection is pure numpy (`np.fft`) + ffmpeg.
- **Naive autocorrelation tempo estimation hits the classic octave error** (measured: a 120-BPM click
  track was detected as 60 BPM). So the design does NOT rely on autocorrelation for tempo.
- **The robust, validated approach:** a spectral-flux onset envelope + adaptive-threshold peak-picking
  recovers the beat TIMES directly (measured: 11/12 clicks of a 120-BPM track at ~0.49 s spacing), and
  **BPM is derived from the median inter-beat interval** (measured: 120.2 BPM — no octave error).
  Snapping a set of cut points to the nearest detected beat also verified
  (`[0.42,1.13,2.55,4.28] → [0.488,0.975,2.485,4.481]`).

Decision locked during brainstorming: v1 output = the beat grid + BPM + (when `cut_seconds` given) each
cut snapped to the nearest beat. Emitting a beat-aligned `edit_decisions` artifact is deferred to v2.

## Goal

A new tool `beat_sync` that detects the beat times in an audio or video's audio track, reports them and
an estimated BPM, and — when the caller supplies `cut_seconds` — returns each cut snapped to its nearest
beat (with the applied offset). It never fabricates: the beats are the measured onset peaks, BPM is the
measured median inter-beat interval, and if the audio is too short / silent to find ≥2 beats it says so
rather than inventing a grid.

Non-goals (YAGNI): downbeat/bar detection, time-signature inference, tempo-curve tracking (variable
tempo), genre-tuned onset models, emitting `edit_decisions` (v2), actually re-rendering a beat-cut video
(a downstream compose step consumes the snapped cuts).

## The algorithm (validated, pure numpy)

1. Extract mono PCM at `sample_rate` (default 22050 Hz) as BYTES via ffmpeg (`-f s16le`), never
   `run_command` text mode.
2. **Onset envelope (spectral flux):** frame the signal (`win=1024`, `hop=512`, Hann window); per frame
   compute `|rfft|`, take the half-wave-rectified positive difference from the previous frame's
   magnitude, sum → one flux value per frame. Z-normalise the envelope. Frame rate `fps = sample_rate/hop`.
3. **Beat times (peak-pick):** local maxima above `mean + k·std` (default `k=1.0`), enforcing a minimum
   gap (`min_gap_s`, default 0.15 s) so a single onset doesn't double-trigger. Peak frame → `time = frame/fps`.
4. **BPM:** `60 / median(diff(beat_times))` — median inter-beat interval, robust to the octave error and
   to missed/extra onsets. `0.0` if < 2 beats.
5. **Snap:** for each `cut_seconds[i]`, the snapped time is the nearest beat; report `{original, snapped,
   offset}`.

## Architecture

One `BaseTool` subclass, `tools/audio/beat_sync.py`, contract-styled after `tools/analysis/audio_energy.py`.

- **Contract:** `name="beat_sync"`, `version="0.1.0"`, `tier=ToolTier.CORE`, `capability="analysis"`,
  `provider="ffmpeg+numpy"`, `stability=EXPERIMENTAL`, `execution_mode=SYNC`,
  `determinism=Determinism.DETERMINISTIC`, `dependencies=["cmd:ffmpeg", "python:numpy"]`,
  `agent_skills=["ffmpeg"]`, `capabilities=["beat_sync","beat_detection","onset_detection"]`.

### Helper units (each independently testable)

- `_has_audio(path) -> bool` — ffprobe audio-stream probe (guarded).
- `_extract_samples(path, sample_rate) -> np.ndarray | None` — mono float32 PCM as BYTES; `None` on
  no-audio / failure (odd-byte trim before `frombuffer`).
- `_onset_envelope(x, sample_rate, win, hop) -> tuple[np.ndarray, float]` — z-normalised spectral-flux
  envelope + `fps`. Pure numpy.
- `_pick_beats(envelope, fps, k, min_gap_s) -> list[float]` — beat times from adaptive-threshold
  peak-picking. Pure numpy.
- `_estimate_bpm(beats) -> float` — `60 / median(inter-beat-interval)`, `0.0` if < 2 beats.
- `_snap(cut_seconds, beats) -> list[dict]` — each cut → `{original, snapped, offset}` (nearest beat).
  `[]` and a note if no beats.

## Input schema

```jsonc
{
  "type": "object",
  "required": ["input_path"],
  "properties": {
    "input_path":   { "type": "string", "description": "Audio or video whose audio holds the beat." },
    "cut_seconds":  { "type": "array", "items": { "type": "number", "minimum": 0 },
                      "description": "Optional cut points to snap to the nearest detected beat." },
    "output_path":  { "type": "string", "description": "Where to write the beat-grid JSON report." },
    "sample_rate":  { "type": "integer", "default": 22050, "minimum": 8000 },
    "sensitivity":  { "type": "number", "default": 1.0, "minimum": 0.0,
                      "description": "Peak threshold in std-devs above the mean (k); lower = more beats." },
    "min_gap_seconds": { "type": "number", "default": 0.15, "minimum": 0.02,
                         "description": "Minimum spacing between detected beats (debounce)." }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[beat_report_json_path]`
- `data`:
  - `bpm`: float (median-IBI estimate)
  - `beat_count`: int
  - `beats`: `list[float]` (detected beat times, seconds)
  - `snapped_cuts`: `list[{original, snapped, offset}]` (empty if no `cut_seconds`)
  - `sample_rate`, `sensitivity`, `min_gap_seconds`
- `duration_seconds`: wall time

The report JSON is written with `sort_keys=True` for byte-stable determinism.

## Error / edge-case handling

- **Missing `input_path` / no audio stream:** `success=False`, clear error.
- **Fewer than 2 beats detected** (too short / silent / featureless audio): `success=False` with a message
  (can't report a BPM or a meaningful grid); no fabricated tempo. (A caller wanting raw onsets can lower
  `sensitivity`.)
- **`cut_seconds` given but no beats:** covered by the < 2 beats failure above (snapping needs a grid).
- **Boundary-validate** `sample_rate`/`sensitivity`/`min_gap_seconds` numeric + in range, and each
  `cut_seconds` numeric, before ffmpeg work; non-numeric → `success=False`, not a traceback.
- **ffmpeg extraction failure / empty PCM:** `success=False`.
- All subprocess calls guarded (`CalledProcessError`/`TimeoutExpired`/`OSError`) with timeouts; temp is
  in-memory (no temp dir needed beyond the report write).
- Determinism: same input + params → identical beats, BPM, snapped cuts, and byte-identical report JSON.

## Testing

`tests/tools/test_beat_sync.py` (pytest; import from `tools.audio.beat_sync`). Signal helpers tested on
synthetic numpy arrays; ffmpeg-gated tests synthesize a click track at a known BPM.

1. **`_onset_envelope`:** on a synthetic array with periodic impulses, the envelope has peaks at the
   impulse frames (higher there than between).
2. **`_pick_beats`:** on a synthetic envelope with impulses every N frames, returns times at ~N/fps
   spacing, count ≈ the number of impulses (± the boundary one).
3. **`_estimate_bpm`:** beats spaced 0.5 s → 120.0; < 2 beats → 0.0; robust to one missing beat (median).
4. **`_snap`:** cut points snap to the nearest beat with the correct `offset`; monotonic beats preserved.
5. **e2e recovers a known BPM (honesty anchor, ffmpeg-gated):** a synthesized 120-BPM click track →
   `abs(bpm - 120) < 6` (within tolerance of the median-IBI estimate), `beat_count` ≈ 12, the JSON exists.
   Threshold from the validated measurement (120.2 observed), NOT loosened.
6. **e2e snaps cut points to the beat (ffmpeg-gated):** the same track + `cut_seconds=[0.42, 2.55]` →
   `snapped_cuts` each within half a beat-period of the original, landing on a detected beat.
7. **Video input:** a video with a beat-carrying audio track → detection runs on the audio; `beats` found.
8. **Too-short/silent → failure:** a 0.2 s silent clip → `success=False` (no fabricated BPM).
9. **Determinism:** same input → identical `bpm`/`beats` and byte-identical report JSON.
10. **Boundary validation:** non-numeric `sensitivity` → `success=False`, no traceback.

## Integration touchpoints

- Discoverable via `tools/tool_registry.py` (drop-in, no wiring).
- A montage / reels pipeline detects the music's beats and snaps its clip-cut timestamps to them before
  compose; the `snapped_cuts` feed a downstream cut/compose step. No schema change; reuses ffmpeg (hard
  dep) + numpy (installed). No new third-party dependency (deliberately avoids librosa/scipy).

## Open decisions (defaults chosen, override at implementation)

- `sample_rate=22050`, `win=1024`, `hop=512`, `sensitivity(k)=1.0`, `min_gap_seconds=0.15` — the values
  validated on the click track; tune on real music (dense mixes may want higher `k`).
- Median-IBI BPM (not autocorrelation) is the deliberate anti-octave-error choice.
- Downbeat/bar detection, variable-tempo tracking, and `edit_decisions` emission are explicitly v2.
