# Morph Cut — Design Spec

> Automation port 5 of N (Premiere/DaVinci features → OpenMontage). Sub-project 1 of the second
> batch (morph-cut · voice-isolation · beat-sync · react perf-refinement).
> Date: 2026-07-10 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-morph-cut`

## Context

Porting Premiere's **Morph Cut** (DaVinci's **Smooth Cut**): smooth a hard jump cut with optical-flow
interpolation so the join looks continuous instead of a jarring jump. Behavior reverse-engineering.

This **completes the cut-editing cluster** we already built: `silence_cutter` and `text_based_editor`
*create* jump cuts (removing silence/filler words); `morph_cut` *smooths* them.

Evidence gathered 2026-07-10 (empirical, before speccing):
- **No morph/optical-flow tool exists.** `grep` over `tools/` for `minterpolate|morph|optical.flow|rife`
  found no real usage. ffmpeg `minterpolate` (motion-compensated interpolation) IS available.
- **`minterpolate=mci` genuinely morphs — within a displacement limit.** Measured by tracking a moving
  box across the junction of a synthetic A→B cut:
  - displacement 8–40 px (a talking head whose head barely moves between the removed segments) →
    **4+ intermediate flow positions = a real morph**.
  - displacement ~80 px (a big framing/content change) → **only the two endpoints = a hard cut**; motion
    estimation finds no correspondence and degrades to the original jump.
- **This matches Premiere's own documented limitation** ("Morph Cut works best with a stationary
  subject against a static background"). Where the flow can't bridge the change, it degrades to the
  hard cut — **no warping artifact**, just no smoothing. That graceful degradation is a feature, and
  the tool reports it honestly per cut.

Decision locked during brainstorming:
- **v1 input:** explicit `cut_seconds` (the caller/pipeline knows where the joins are). Auto-detection
  via the existing `scene_detect` is deferred to v2.

## Goal

A new tool `morph_cut` that, given a rendered video and a list of cut timestamps, replaces a short
window around each cut with an optical-flow-interpolated morph, producing a smoother output. It reports
per cut whether the morph actually smoothed the join (an honest effectiveness metric measured on the
output), and never fabricates smoothing: where the flow can't bridge the change, the original cut is
kept and reported as `smoothed: false`.

Non-goals (YAGNI): auto-detecting cut points (v2 via scene_detect), face-aware/subject-locked flow,
RIFE/ML interpolation, cross-shot dissolve fallback, morphing arbitrary transitions (only hard cuts).

## The mechanism (validated)

For each cut at time `T` with transition duration `D`:
1. The window `[T - D/2, T + D/2]` of the video contains the tail of shot A and the head of shot B with
   the hard jump at its center.
2. `minterpolate=fps=<morph_fps>:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1` over that window
   generates flow-warped in-between frames across the jump. The window keeps the **same wall-clock
   duration `D`** (minterpolate raises frame count, not duration), so audio stays in sync.
3. The morphed window is spliced back in place of the original window.

For multiple cuts, the video is split at all window boundaries, each window is morphed, and the pieces
are concatenated in order (the proven `silence_cutter` cut+concat pattern). Audio is taken from the
original (duration-preserving morph → sync holds).

### Honest effectiveness metric

The junction's **max consecutive-frame mean-absolute-difference (MAD)** within the transition window:
- A hard jump = one large frame-to-frame MAD spike.
- A successful morph spreads that change over several small steps → the max MAD drops.

`smoothed[i] = max_MAD_after < max_MAD_before * SMOOTH_RATIO` (e.g. ratio 0.9). Measured by sampling the
transition-region frames of the OUTPUT, not modeled. A cut that didn't smooth (large change) reports
`smoothed: false` with both numbers — never a fabricated success.

## Architecture

One `BaseTool` subclass, `tools/video/morph_cut.py`, contract-styled after `tools/video/silence_cutter.py`.

- **Contract:** `name="morph_cut"`, `version="0.1.0"`, `tier=ToolTier.CORE`, `capability="video_post"`,
  `provider="ffmpeg"`, `stability=EXPERIMENTAL`, `execution_mode=SYNC`,
  `determinism=Determinism.DETERMINISTIC`, `dependencies=["cmd:ffmpeg", "python:numpy", "python:PIL"]`,
  `agent_skills=["ffmpeg"]`, `capabilities=["morph_cut","smooth_cut","optical_flow_transition"]`.

### Helper units (each independently testable)

- `_duration(path) -> float` — clip duration via ffprobe (guarded → 0.0).
- `_has_audio(path) -> bool` — ffprobe audio-stream probe (guarded).
- `_plan_windows(cut_seconds, duration, D) -> tuple[list[dict], list[dict]]` — from sorted, de-duplicated
  cut times, produce the ordered list of segments (`kept` passthrough spans and `morph` windows) that
  tile `[0, duration]`. Clamps a window that runs past the start/end; drops/merges cuts whose windows
  overlap (recorded in notes). Pure function, no ffmpeg.
- `_morph_window(input_path, start, end, morph_fps, dest) -> str | None` — extract `[start,end]`, run the
  validated `minterpolate` filter, write `dest`; `None` on failure.
- `_extract_segment(input_path, start, end, dest) -> str | None` — plain passthrough cut (re-encoded for
  clean concat), `None` on failure.
- `_concat(parts, dest) -> str | None` — concat-demuxer join.
- `_junction_max_mad(video_path, at_seconds, radius) -> float` — max consecutive-frame MAD in a small
  window around `at_seconds` (PIL+numpy), the effectiveness probe.

## Input schema

```jsonc
{
  "type": "object",
  "required": ["input_path", "cut_seconds"],
  "properties": {
    "input_path":         { "type": "string" },
    "cut_seconds":        { "type": "array", "items": { "type": "number", "minimum": 0 }, "minItems": 1,
                            "description": "Timestamps (in the rendered video) of the hard cuts to smooth." },
    "output_path":        { "type": "string" },
    "transition_duration":{ "type": "number", "default": 0.2, "minimum": 0.04,
                            "description": "Total morph window duration per cut (seconds)." },
    "morph_fps":          { "type": "integer", "default": 60, "minimum": 30,
                            "description": "Interpolated frame rate inside the morph window; higher = smoother." },
    "codec":              { "type": "string", "default": "libx264" },
    "crf":                { "type": "integer", "default": 18 }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[output_path]`
- `data`:
  - `cuts_requested`: int, `cuts_processed`: int
  - `per_cut`: `list[{ time, max_mad_before, max_mad_after, smoothed: bool }]`
  - `smoothed_count`: int (how many joins were actually smoothed)
  - `skipped_cuts`: `list[{ time, reason }]` — out-of-range or overlapping-window cuts, dropped honestly
  - `transition_duration`, `morph_fps`
- `duration_seconds`: wall time

## Error / edge-case handling

- **Missing `input_path` / empty `cut_seconds`:** `success=False`, clear error.
- **Boundary-validate** `transition_duration`/`morph_fps`/`crf` numeric + in range before ffmpeg work;
  non-numeric → `success=False` (not a traceback).
- **Cut ≤ D/2 from start or end:** clamp the window to the available span; if the clampable window is
  below a minimum (< 2 frames), skip that cut into `skipped_cuts` with a reason (don't morph a
  degenerate window).
- **Overlapping windows** (two cuts closer than `D`): keep the first, skip the later into `skipped_cuts`
  (recorded), so windows never overlap in the splice plan.
- **minterpolate degrades to a hard cut** (large change): the output still renders; the effectiveness
  metric reports `smoothed: false` for that cut. This is expected, not a failure — the run still
  succeeds as long as the video was produced.
- **ffmpeg non-zero / missing or zero-byte output / concat failure:** `success=False` even on exit 0.
- **No audio in input:** morph video-only, output without an audio track (`-an`); with audio, mux the
  original audio back (duration is preserved, so sync holds).
- All subprocess calls guarded (`CalledProcessError`/`TimeoutExpired`/`OSError`) with timeouts; temp dir
  cleaned in `finally`.
- Determinism: same input + cuts + params → identical splice plan and identical output size.

## Testing

`tests/tools/test_morph_cut.py` (pytest; import from `tools.video.morph_cut`). Plan/metric helpers tested
on synthetic data; ffmpeg-gated tests synthesize tiny clips with a controllable jump.

1. **`_plan_windows` tiles the timeline:** given cuts and duration, the returned segments cover
   `[0, duration]` with morph windows centered on each cut and passthrough spans between — no gaps, no
   overlaps; the ordered concat of their durations equals the input duration.
2. **Window clamp at boundaries:** a cut at 0.05s with D=0.2 clamps the window to the available head; a
   degenerate (<2-frame) window is skipped into `skipped_cuts`.
3. **Overlap handling:** two cuts 0.1s apart with D=0.2 → the second is skipped, recorded with a reason.
4. **`_junction_max_mad`:** on a synthetic hard jump it returns a high value; on a smooth pan a low value.
5. **e2e smooths a small-displacement jump (honesty anchor, ffmpeg-gated):** build A→B where B is A
   shifted a few px (a realistic micro-jump); after `morph_cut`, the junction `max_mad_after <
   max_mad_before` and `per_cut[0].smoothed is True`; the MP4 exists and its duration ≈ the input's.
   Threshold measured on the fixture, NOT loosened to pass.
6. **e2e degrades honestly on a large jump (ffmpeg-gated):** an A→B with a big content change → run still
   `success=True`, output exists, and `per_cut[0].smoothed is False` (reported, not faked).
7. **No-audio input:** morphs a video-only clip without error.
8. **Determinism:** same inputs → identical `per_cut` plan and identical output size.
9. **Empty/oob inputs:** empty `cut_seconds` → failure; non-numeric `transition_duration` → failure, no
   traceback.

## Integration touchpoints

- Discoverable via `tools/tool_registry.py` (drop-in, no wiring).
- Natural downstream of `silence_cutter` / `text_based_editor`: run their jump-cut output through
  `morph_cut` with the join timestamps to soften the cuts. No schema change; reuses the `silence_cutter`
  cut+concat pattern and ffmpeg (hard dep) + numpy/PIL (installed).

## Open decisions (defaults chosen, override at implementation)

- `transition_duration=0.2s`, `morph_fps=60`, `SMOOTH_RATIO=0.9` for the `smoothed` flag — tune on real
  talking-head footage.
- Auto-detecting cuts via `scene_detect` and RIFE/ML interpolation are explicitly v2.
