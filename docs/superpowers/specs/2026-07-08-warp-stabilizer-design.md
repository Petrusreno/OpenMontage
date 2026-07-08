# Warp Stabilizer — Design Spec

> Sub-project 1 of 4 in **Premiere automation/render features → OpenMontage**.
> Date: 2026-07-08 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-warp-stabilizer`

## Context

We are porting automation/render features from Adobe Premiere Pro 2026 into OpenMontage
natively. This is **behavior/format reverse-engineering** (replicate how the feature behaves
with free tooling), not binary decompilation.

Gap analysis against the current tool inventory (`tools/`) confirmed by evidence
(`grep` over `tools/`, 2026-07-08) found six real automation gaps. The user selected all four
of the top candidates; we decomposed them into four independent sub-projects, each its own
spec → plan → implementation cycle. Sequenced by value × effort × pattern-reuse:

1. **Warp Stabilizer** ← this spec. Establishes the "render-automation `BaseTool` + before/after
   metric test" pattern.
2. Color Match (`tools/video/`) — same ffmpeg/opencv single-clip domain.
3. Text-based editing (`tools/video/` + `edit_decisions`) — larger; ripple-cut by word timestamps.
4. Multicam auto-sync (`tools/audio/`) — audio cross-correlation.

Premiere's **Warp Stabilizer** removes camera shake from handheld footage. High value for
Petrus's use case (selfie/handheld reels; Gemini-Omni pipeline already wants stabilization).

## Goal

A single new tool, `warp_stabilizer`, that takes a shaky video clip and produces a stabilized
clip, reporting an objective before/after shakiness metric. It operates on a **clip file** in
the edit/compose stage — it does **not** mutate `edit_decisions`.

Non-goals (YAGNI): rolling-shutter correction, subframe rotation UI, per-region masks,
GPU acceleration. These are not needed for the target use case.

## Architecture

One `BaseTool` subclass, mirroring `tools/video/silence_cutter.py` (the closest existing
template: `video_post` capability, ffmpeg-backed, deterministic, contract via class attributes,
`execute()` → `ToolResult` with `artifacts`).

- **File:** `tools/video/warp_stabilizer.py`
- **Discovery:** automatic via `tools/tool_registry.py` (drop-in file, no decorator).
- **Contract:** `name="warp_stabilizer"`, `version="0.1.0"`, `tier=ToolTier.CORE`,
  `capability="video_post"`, `provider="ffmpeg"`, `stability=EXPERIMENTAL`,
  `execution_mode=SYNC`, `determinism=Determinism.DETERMINISTIC`,
  `dependencies=["cmd:ffmpeg"]`, `agent_skills=["ffmpeg"]`,
  `capabilities=["stabilization","deshake","warp_stabilizer"]`.

### Engine: ffmpeg vid.stab, 2-pass (verified available on this machine)

`ffmpeg -filters` on the target machine lists `vidstabdetect`, `vidstabtransform`, **and**
`deshake` — so the primary path and the fallback both work here.

- **Pass 1 (analyze):**
  `vidstabdetect=shakiness=S:accuracy=A:result=<tmp>/transforms.trf`
- **Pass 2 (correct):**
  `vidstabtransform=input=<tmp>/transforms.trf:smoothing=F:zoom=Z:optzoom=O` followed by
  `unsharp=5:5:0.8:3:3:0.4` when `sharpen=true` (recovers detail lost to warp, as Premiere does).
  Audio preserved with `-c:a copy`.

The two passes share one temp working dir (created with `tempfile.mkdtemp`, removed in a
`finally`).

### Dependency gate (honest failure, no fabrication)

Health-check runs `ffmpeg -filters` and checks for `vidstabdetect`/`vidstabtransform`.

- Present → primary vid.stab path.
- Absent but `deshake` present → **degraded** path using the built-in `deshake` filter
  (single pass, lower quality), and `ToolResult.data.engine="deshake"` + a warning.
- Neither present → `ToolStatus.UNAVAILABLE` with install instruction
  (`brew install ffmpeg`, which ships libvidstab on macOS). The tool never silently
  returns an unstabilized file as if it worked.

## Input schema

```jsonc
{
  "type": "object",
  "required": ["input_path"],
  "properties": {
    "input_path":  { "type": "string" },
    "output_path": { "type": "string" },                       // default: <input>_stabilized.mp4
    "smoothing":   { "type": "integer", "default": 10, "minimum": 0,
                     "description": "Frames of look-ahead/behind for camera-path smoothing." },
    "shakiness":   { "type": "integer", "default": 5,  "minimum": 1, "maximum": 10 },
    "accuracy":    { "type": "integer", "default": 15, "minimum": 1, "maximum": 15 },
    "zoom":        { "type": "number",  "default": 0,  "description": "Static extra zoom %." },
    "optzoom":     { "type": "integer", "default": 1,  "enum": [0, 1, 2],
                     "description": "0=none, 1=optimal static, 2=adaptive per-frame zoom." },
    "border":      { "type": "string",  "default": "black", "enum": ["black", "replicate"] },
    "sharpen":     { "type": "boolean", "default": true }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[output_path]`
- `data`:
  - `engine`: `"vidstab"` | `"deshake"`
  - `shakiness_before`: float
  - `shakiness_after`: float
  - `reduction_pct`: float  (`(before-after)/before*100`, clamped ≥ 0)
  - `transforms_file`: path to `.trf` (vidstab path only)
  - `params`: the resolved parameters actually used
- `duration_seconds`: wall time

### Shakiness metric (makes the test objective)

Defined as the **mean per-frame translation magnitude** the detector reports.
Compute by running `vidstabdetect` and parsing the `.trf` transforms (fields are per-frame
`x`/`y` translations); `shakiness = mean(sqrt(x² + y²))` across frames.

- `shakiness_before` = detect on `input_path`.
- `shakiness_after` = detect on `output_path` (a second detect pass on the result).

On the `deshake` fallback (no `.trf`), the metric is computed both times via a one-off
`vidstabdetect` run purely for measurement (detect works even when transform uses `deshake`);
if `vidstabdetect` is entirely absent, `data` reports `shakiness_*: null` with a note rather
than a fabricated number.

## Error / edge-case handling

- **No audio stream** (probe with `ffprobe`): omit `-c:a copy`.
- **Clip too short** (< `2 × smoothing` frames): warn, clamp `smoothing`, still attempt.
- **ffmpeg non-zero exit:** `ToolResult(success=False, error=<last stderr lines>)`; temp dir cleaned.
- **Output not created / zero bytes:** treated as failure even if ffmpeg exited 0.
- All inputs validated at the boundary against `input_schema` before spawning ffmpeg.

## Testing

Location: `tests/` following existing tool-test conventions (`tests/qa/` for output inspection,
`tests/contracts/` for contract shape).

1. **Effectiveness (core):** synthesize a shaky clip (ffmpeg applies random per-frame
   translation jitter to a still/color source), run the tool, assert
   `shakiness_after < shakiness_before` beyond a margin (e.g. ≥ 30% reduction) and that
   `output_path` exists and is non-empty.
2. **Determinism:** two runs with identical params yield identical `.trf` (byte-equal) and an
   output of identical size.
3. **Graceful degradation / gate:** monkeypatch the filter probe to report vidstab absent →
   assert `engine=="deshake"` (deshake present) or `ToolStatus.UNAVAILABLE` (neither).
4. **No-audio input:** stabilizes a video-only clip without error.
5. **Contract:** tool is discoverable via the registry with all required contract fields set.

## Integration touchpoints

- Discoverable immediately via `tools/tool_registry.py` — no orchestrator wiring needed.
- Usable by any pipeline's edit/compose stage director as a `video_post` step on a source clip.
- No `edit_decisions` schema change. No changes to other tools. No selector needed (single
  provider); a `stabilization_selector` can be added later only if a second engine appears.

## Open decisions (defaults chosen, override at implementation)

- Default `smoothing=10` and `optzoom=1` match ffmpeg/vid.stab conventional defaults and the
  Premiere "Smoothness" feel; adjustable.
- Effectiveness-test reduction threshold (30%) is a starting value; tighten once we see real
  numbers on a fixture.
