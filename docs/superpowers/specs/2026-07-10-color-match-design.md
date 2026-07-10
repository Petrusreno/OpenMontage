# Color Match — Design Spec

> Sub-project 2 of 4 in **Premiere automation/render features → OpenMontage**.
> Date: 2026-07-10 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-color-match`

## Context

Porting Premiere's **Color Match** (match one clip's color/tone to a reference shot) into
OpenMontage natively. Behavior reverse-engineering, not binary decompilation.

Evidence gathered 2026-07-10:
- **No color-match tool exists.** `tools/enhancement/color_grade.py` applies *subjective* looks
  (built-in profile presets + external `.cube` LUTs) — it does NOT match a target to a reference.
  Color Match is a genuine gap; this tool complements `color_grade`, it does not duplicate it.
- **Whole-clip application without per-frame Python is possible.** ffmpeg `lutrgb` accepts a
  per-channel expression; a clamped per-channel affine `clip((val-m)*g+b, 0, 255)` applies to the
  entire clip in one filter pass. Verified empirically: a source frame with mean RGB `[47,95,159]`,
  after `lutrgb` with per-channel affine, moved to `~[127,127,127]` as computed.
- **PIL 12.2.0 + numpy 2.4.4 are available; cv2 is NOT.** So statistics are computed with PIL
  (frame → array) + numpy; no OpenCV / Lab-space conversion.

Decisions locked during brainstorming:
- **Method:** per-channel **Reinhard mean+std transfer** (match both the mean and the contrast/std
  of each RGB channel). Lab-space / cross-channel transfer is deferred to v2.
- **Output:** the color-matched MP4 **plus** the transform parameters in `data` (for traceability
  and reuse). No `.cube` sidecar in v1.

## Goal

A new tool `color_match` that, given a target clip and a reference (a frame from another clip, or a
still image), extracts one frame from each, computes a per-channel affine transform that maps the
target's color statistics onto the reference's, and applies it to the whole target clip via ffmpeg
`lutrgb`. An `intensity` control (0–1) blends toward the match (like Premiere's slider). The result
is verifiable: the target frame's per-channel mean moves measurably closer to the reference's.

Non-goals (YAGNI): Lab / decorrelated-space transfer, histogram (CDF) matching, cross-channel
3×3 color correction, per-shot auto-detection of the reference, `.cube` LUT export, skin-tone
protection. All deferred to v2.

## The transform

Per channel c ∈ {R,G,B}, with `m_t,s_t` = target-frame mean/std and `m_r,s_r` = reference-frame
mean/std (in 0–255):

```
gain_c   = s_r / s_t                       (clamped: if s_t < EPS → gain 1.0; gain clamped to [0, GAIN_MAX])
match(v) = (v - m_t) * gain_c + m_r        (the full Reinhard mean+std transfer)
out(v)   = v + intensity * (match(v) - v)  (blend; intensity=1 → full match, 0 → identity)
         = clip(out(v), 0, 255)
```

`out(v)` is affine in `v` for every fixed channel, so it maps to a single ffmpeg `lutrgb`
per-channel expression — the whole clip is processed in one pass, no per-frame Python.

- `EPS` guards a flat channel (std≈0) from a div-by-zero / exploding gain.
- `GAIN_MAX` (e.g. 3.0) caps extreme gains from a near-flat target that would otherwise band.
- Both guards are reported in `data` when they fire (never a silent clamp).

## Architecture

One `BaseTool` subclass, `tools/enhancement/color_match.py`, contract-styled after
`tools/enhancement/color_grade.py`.

- **Contract:** `name="color_match"`, `version="0.1.0"`, `tier=ToolTier.CORE`,
  `capability="enhancement"`, `provider="ffmpeg+numpy"`, `stability=EXPERIMENTAL`,
  `execution_mode=SYNC`, `determinism=Determinism.DETERMINISTIC`,
  `dependencies=["cmd:ffmpeg", "python:numpy", "python:PIL"]` (the CHECKED prefixes),
  `agent_skills=["ffmpeg"]`, `capabilities=["color_match","color_transfer","reference_grade"]`.

### Data flow

```
target_clip + reference (clip or image)
  ──► extract 1 frame from each (ffmpeg -ss <t> -frames:v 1 -> PNG), default t = clip midpoint
  ──► PIL open -> numpy; per-channel mean/std for target frame and reference frame
  ──► per-channel gain/offset (Reinhard) with EPS/GAIN_MAX guards, folded with `intensity`
  ──► build lutrgb per-channel affine expression
  ──► ffmpeg apply to the WHOLE target clip -> matched MP4 (audio copied)
  ──► re-extract a frame from the OUTPUT; report before/after per-channel mean delta vs reference
```

### Helper units (each independently testable)

- `_extract_frame(path, at_seconds) -> Path | None` — one PNG frame via ffmpeg; `None` on failure.
- `_frame_stats(png_path) -> tuple[np.ndarray, np.ndarray]` — `(mean_rgb[3], std_rgb[3])` via PIL+numpy.
- `_channel_affine(m_t, s_t, m_r, s_r, intensity) -> tuple[list[float], list[float], list[dict]]` —
  returns `(gains[3], offsets[3], notes)` where `out = gain*v + offset` already folds intensity;
  `notes` lists any EPS/GAIN_MAX clamp that fired (per channel).
- `_lutrgb_expr(gains, offsets) -> str` — the ffmpeg `lutrgb=r=...:g=...:b=...` filter string with
  clamped affine expressions.
- `_midpoint(path) -> float` — clip duration/2 via ffprobe (0.0 fallback for a still image).
- `_mean_delta(a_mean, b_mean) -> float` — mean absolute per-channel difference (the metric).

Note: `out = gain*v + offset` is the folded form of `v + intensity*((v-m_t)*g + m_r - v)`:
`gain = 1 + intensity*(g-1)`, `offset = intensity*(m_r - g*m_t)`.

## Input schema

```jsonc
{
  "type": "object",
  "required": ["input_path", "reference_path"],
  "properties": {
    "input_path":      { "type": "string", "description": "Target clip to be recolored." },
    "reference_path":  { "type": "string", "description": "Reference clip or still image to match to." },
    "output_path":     { "type": "string" },
    "reference_time":  { "type": "number", "minimum": 0,
                         "description": "Seconds into the reference to sample (default: midpoint / 0 for a still)." },
    "input_time":      { "type": "number", "minimum": 0,
                         "description": "Seconds into the target to sample its stats (default: midpoint)." },
    "intensity":       { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0,
                         "description": "0 = original, 1 = full match." },
    "codec":           { "type": "string", "default": "libx264" },
    "crf":             { "type": "integer", "default": 20 }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[output_path]`
- `data`:
  - `gains`: `[float, float, float]`, `offsets`: `[float, float, float]` (the applied transform)
  - `reference_mean`: `[float×3]`, `target_mean_before`: `[float×3]`, `target_mean_after`: `[float×3]`
  - `mean_delta_before`: float, `mean_delta_after`: float (mean |target−reference| per channel)
  - `improved`: bool (`mean_delta_after < mean_delta_before`)
  - `clamp_notes`: `list[dict]` — any EPS/GAIN_MAX clamp that fired
  - `intensity`: float
- `duration_seconds`: wall time

## Error / edge-case handling

- **Missing `input_path` or `reference_path`:** `success=False`, clear error.
- **Frame extraction fails** (no video stream, unreadable, bad timestamp): `success=False` naming
  which side failed. A still image is valid input (single frame).
- **Flat channel (std < EPS):** gain forced to 1.0 for that channel (mean-only shift), recorded in
  `clamp_notes`. Never divide by zero, never fabricate.
- **Gain exceeds `GAIN_MAX`:** clamped, recorded in `clamp_notes`.
- **ffmpeg non-zero / missing or zero-byte output:** `success=False` (failure even on exit 0).
- **No audio stream in target:** apply video filter without `-c:a copy` (use `-an`).
- **`intensity = 0`:** identity transform (gains 1, offsets 0) — output is a re-encode of the input;
  `improved` may be False (delta unchanged), which is correct and honest, not a failure.
- All subprocess calls guarded (CalledProcessError/TimeoutExpired/OSError) so a missing binary /
  bad file returns `success=False`, never a traceback.
- Inputs validated at the boundary against `input_schema` before any ffmpeg work.

## Testing

`tests/tools/test_color_match.py` (pytest; import from `tools.enhancement.color_match`). Math
helpers are tested on numpy arrays directly; ffmpeg-gated tests synthesize tiny solid-color clips.

1. **Affine math (core):** given target stats `(m_t,s_t)` and reference `(m_r,s_r)` with
   `intensity=1`, `_channel_affine` returns gains/offsets such that `gain*m_t+offset == m_r`
   (the matched mean equals the reference mean) per channel.
2. **Intensity blend:** `intensity=0` → gains all 1.0, offsets all 0.0 (identity);
   `intensity=0.5` → halfway (`gain*m_t+offset == (m_t+m_r)/2` when `s_t==s_r`).
3. **Flat-channel guard:** `s_t=0` → gain 1.0 for that channel, a `clamp_notes` entry, no div-by-zero.
4. **Gain clamp:** a tiny `s_t` that would exceed `GAIN_MAX` is clamped and recorded.
5. **`_frame_stats`:** a synthesized solid-color PNG yields the expected per-channel mean (≈ the fill
   color) and near-zero std.
6. **`_lutrgb_expr`:** produces a valid `lutrgb=r=...:g=...:b=...` string with clamped affine terms.
7. **End-to-end (ffmpeg-gated):** a bluish target clip + a reddish reference clip → after match, the
   target output frame's per-channel mean is closer to the reference's than before
   (`mean_delta_after < mean_delta_before`), the MP4 exists and is non-empty, and `data.improved` is
   True. Threshold measured on the fixture, not guessed; NOT loosened to pass.
8. **No-audio target:** matches a video-only clip without error.
9. **Missing reference → failure:** `success=False`, no output written.
10. **Determinism:** same inputs → identical `gains`/`offsets` and identical output size.

## Integration touchpoints

- Discoverable via `tools/tool_registry.py` (drop-in, no wiring).
- Complements `color_grade` (subjective looks) with reference-driven matching; a downstream edit/
  compose stage can call `color_match` to keep multi-clip montages consistent.
- Reuses ffmpeg (hard dep) + numpy + PIL (already installed). No new third-party dependency, no
  schema changes.

## Open decisions (defaults chosen, override at implementation)

- `EPS = 1.0` (on the 0–255 std scale) for the flat-channel guard; `GAIN_MAX = 3.0`. Tune on real
  footage.
- Frame sampled at the clip midpoint by default to avoid black intro/outro frames; overridable via
  `input_time`/`reference_time`.
- Per-channel RGB transfer is the v1 method; Lab/decorrelated transfer and `.cube` export are v2.
