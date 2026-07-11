# Voice Isolation — Design Spec

> Automation port (Premiere/Adobe Enhance Speech · DaVinci Voice Isolation → OpenMontage).
> Sub-project 2 of the second batch (morph-cut ✓ · voice-isolation · beat-sync · react perf-refinement).
> Date: 2026-07-11 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-voice-isolation`

## Context

Porting Adobe's **Enhance Speech** / DaVinci's **Voice Isolation**: strongly clean up speech audio
(remove background noise, hum, room tone) far beyond a basic denoise. Behavior reverse-engineering.

Evidence gathered 2026-07-11 (empirical, before speccing):
- **`audio_enhance` already exists** and does spectral cleanup: presets combining `afftdn` (FFT
  denoise), `highpass`/`lowpass`, `agate`, `acompressor`, `alimiter`, `loudnorm`, EQ. Voice Isolation
  must be a MEANINGFUL step up, not a duplicate.
- **`arnndn` (RNNoise, the ML speech denoiser — the real Enhance-Speech-class engine) is available but
  needs a `.rnnn` model.** Without one it fails cleanly (`ffmpeg -af arnndn` → exit 234, "Error
  initializing filters"); with a model it runs (exit 0). No model was on the system.
- **A free, public-domain RNNoise model was fetched** (user-approved): `somnolent-hogwash` (Speech ×
  Recording noise, the best fit for talking-head) from `GregorR/rnnoise-models`, whose README states
  "none of this work is creative and thus none of it is subject to copyright." Bundled at
  `assets/rnnoise/somnolent-hogwash.rnnn` (297 KB). `arnndn=model=<that>` verified working.
- **A no-model fallback chain is genuinely stronger than `audio_enhance`'s single `afftdn`:** it adds
  `anlmdn` (Non-Local Means broadband denoise — NOT used by `audio_enhance`) + `deesser`, both
  confirmed available.

Decision locked during brainstorming: **hybrid engine + bundle the free model now.** Default runs the
RNNoise ML path; degrades to the stronger spectral chain when no model resolves; always reports which.

## Goal

A new tool `voice_isolation` that strongly isolates speech in an audio or video file. It runs the
RNNoise (`arnndn`) ML denoiser when a model is available, else a stronger-than-`audio_enhance` spectral
chain, and **honestly reports which engine ran** (`engine: "rnnoise" | "spectral"`) — never claiming ML
quality that didn't execute. A `mix` control (0–1, dry/wet) blends the isolated voice with the original,
like Enhance Speech's slider. For a video input, the cleaned audio is muxed back (video stream copied).

Non-goals (YAGNI): training/selecting among RNNoise models at runtime, de-reverb via ML, speaker
separation/diarization, auto-download of models at execution time (the model is bundled once, at build).

## Engine (both validated)

- **`rnnoise` (default when a model resolves):**
  `arnndn=model=<path>` → then `highpass=f=80, loudnorm=I=-16:LRA=11:TP=-1.5`. `mix<1` blends via
  `amix`/`sidechain`-style dry/wet (see below).
- **`spectral` (fallback / when no model / when forced):**
  `afftdn=nf=<nf>:nt=w, anlmdn, highpass=f=80, deesser, loudnorm=I=-16:LRA=11:TP=-1.5`. Adds `anlmdn`
  + `deesser` over `audio_enhance`'s `noise_reduce` preset.
- **Model resolution order:** `model_path` input → env `RNNOISE_MODEL` → bundled
  `assets/rnnoise/somnolent-hogwash.rnnn`. If none exists on disk OR `engine="spectral"` is forced OR
  `arnndn` isn't in this ffmpeg, use the spectral chain.
- **`mix` (dry/wet):** the processed signal is blended with the original at `mix` (1.0 = full
  isolation, 0.0 = original). Implemented with `amix` weights or `[dry][wet]amix=weights=...` so the
  same filtergraph works for both engines. `mix=1.0` is the default.

## Architecture

One `BaseTool` subclass, `tools/audio/voice_isolation.py`, contract-styled after
`tools/audio/audio_enhance.py`.

- **Contract:** `name="voice_isolation"`, `version="0.1.0"`, `tier=ToolTier.CORE`,
  `capability="audio_processing"`, `provider="ffmpeg"`, `stability=EXPERIMENTAL`,
  `execution_mode=SYNC`, `determinism=Determinism.DETERMINISTIC`,
  `dependencies=["cmd:ffmpeg", "python:numpy"]`, `agent_skills=["ffmpeg"]`,
  `capabilities=["voice_isolation","speech_enhance","denoise"]`.

### Helper units (each independently testable)

- `_resolve_model(model_path) -> str | None` — the resolution order above; `None` if nothing on disk.
- `_arnndn_available() -> bool` — `ffmpeg -filters` contains `arnndn`.
- `_has_video(path) -> bool` / `_has_audio(path) -> bool` — ffprobe stream probes (guarded).
- `_build_filter(engine, model, nf, mix) -> str` — the `-af` filtergraph string for the chosen engine,
  including the dry/wet `mix` blend. Pure function.
- `_noise_floor_dbfs(path, window_s) -> float` — the estimated noise floor: the minimum short-window
  RMS (dBFS) across the file (PCM via ffmpeg + numpy). The honest effectiveness metric.
- `_process(input_path, af, out_audio) -> str | None` — run the `-af` chain to a WAV; `None` on failure.
- `_mux_audio(video_in, new_audio, dest) -> str | None` — replace the video's audio track (video
  `-c copy`), `None` on failure.

## Input schema

```jsonc
{
  "type": "object",
  "required": ["input_path"],
  "properties": {
    "input_path":  { "type": "string", "description": "Audio or video file with speech to isolate." },
    "output_path": { "type": "string" },
    "engine":      { "type": "string", "enum": ["auto", "rnnoise", "spectral"], "default": "auto",
                     "description": "auto = RNNoise if a model resolves, else spectral." },
    "model_path":  { "type": "string", "description": "Override .rnnn model; else env RNNOISE_MODEL, else bundled." },
    "mix":         { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0,
                     "description": "Dry/wet: 1.0 = full isolation, 0.0 = original." },
    "noise_floor_db": { "type": "number", "default": -25,
                        "description": "Spectral-chain afftdn noise floor (nf); more negative = gentler." },
    "codec":       { "type": "string", "default": "aac" },
    "bitrate":     { "type": "string", "default": "192k" }
  }
}
```

## Output — `ToolResult`

- `success`: bool
- `artifacts`: `[output_path]`
- `data`:
  - `engine`: `"rnnoise"` | `"spectral"` (which actually ran — honest)
  - `model`: the resolved model path, or `null` for the spectral chain
  - `mix`: float
  - `noise_floor_before_db`, `noise_floor_after_db`: float (estimated noise floor, dBFS)
  - `noise_reduction_db`: float (`before - after`, clamped ≥ 0)
  - `had_video`: bool (whether the audio was muxed back into a video)
- `duration_seconds`: wall time

## Error / edge-case handling

- **Missing `input_path` / no audio stream:** `success=False`, clear error (can't isolate a silent/
  audioless file).
- **`engine="rnnoise"` forced but no model resolves / `arnndn` absent:** `success=False` telling the
  caller to supply a model or use `engine="auto"`/`"spectral"` — never silently downgrade a *forced*
  ML request (auto mode DOES fall back, and reports `engine="spectral"`).
- **Boundary-validate** `mix`∈[0,1] and numeric `noise_floor_db` before ffmpeg work; non-numeric →
  `success=False`, not a traceback.
- **ffmpeg non-zero / missing or zero-byte output:** `success=False` even on exit 0.
- **Video input:** process the audio, then mux back with video `-c copy`; audio-only input → write the
  cleaned audio directly. A video with NO audio → `success=False`.
- All subprocess calls guarded (`CalledProcessError`/`TimeoutExpired`/`OSError`) with timeouts; temp
  dir cleaned in `finally`.
- Determinism: same input + engine + model + params → identical filtergraph and identical output size.

## Testing

`tests/tools/test_voice_isolation.py` (pytest; import from `tools.audio.voice_isolation`). Filter/metric
helpers tested directly; ffmpeg-gated tests synthesize a noisy speech-proxy (tone + white noise) with a
pure-noise region.

1. **Model resolution:** `_resolve_model` returns the bundled model when it exists and no override;
   honors `model_path` and `RNNOISE_MODEL`; returns `None` when nothing is on disk.
2. **`_build_filter`:** rnnoise engine string contains `arnndn=model=`; spectral contains `afftdn` AND
   `anlmdn` AND `deesser`; `mix<1` produces a dry/wet `amix` blend; both end in `loudnorm`.
3. **Engine selection honesty:** `engine="auto"` with the bundled model present → `data.engine ==
   "rnnoise"`; `engine="spectral"` forced → `"spectral"`; `engine="rnnoise"` with a non-existent
   `model_path` → `success=False` (no silent downgrade).
4. **`_noise_floor_dbfs`:** on a signal with a loud region and a near-silent region, returns a low
   (very negative) dBFS reflecting the quiet window.
5. **e2e reduces the noise floor (honesty anchor, ffmpeg-gated):** a tone+white-noise clip → after
   `voice_isolation`, `noise_floor_after_db < noise_floor_before_db` (real reduction, measured on the
   OUTPUT), the file exists and is non-empty, and `data.noise_reduction_db > 0`. Threshold from the
   fixture, NOT loosened.
6. **Video input muxes audio back:** a video clip with a noisy audio track → output is still a video
   (`_has_video` true), `data.had_video is True`, cleaned audio present.
7. **`mix=0` ≈ identity:** the noise floor is (near) unchanged vs input; run still succeeds (honest —
   `noise_reduction_db` ≈ 0 is correct, not a failure).
8. **No-audio input → failure:** `success=False`, no output written.
9. **Determinism:** same inputs → identical `engine`/`model` and identical output size.
10. **Forced-rnnoise-without-model → clean failure:** `engine="rnnoise"`, `model_path="/no/such.rnnn"`
    → `success=False`, no traceback.

## Integration touchpoints

- Discoverable via `tools/tool_registry.py` (drop-in, no wiring).
- Complements `audio_enhance` (general presets) with speech-specific ML isolation; a talking-head /
  react / podcast pipeline calls `voice_isolation` on the raw take before mixing.
- Bundled model committed at `assets/rnnoise/somnolent-hogwash.rnnn` with a `assets/rnnoise/CREDITS.md`
  recording the public-domain source (`GregorR/rnnoise-models`). Reuses ffmpeg (hard dep) + numpy
  (installed). No new third-party dependency.

## Open decisions (defaults chosen, override at implementation)

- Bundled model = `somnolent-hogwash` (Speech × Recording), the best table fit for talking-head; a
  caller can point `model_path`/`RNNOISE_MODEL` at another `.rnnn` (e.g. `beguiling-drafter` for
  voice-with-laughter). Alternate models are NOT bundled in v1.
- Spectral `afftdn nf=-25`, `loudnorm I=-16` match `audio_enhance` conventions; tune on real footage.
- ML de-reverb, model auto-selection, and speaker separation are explicitly deferred.
