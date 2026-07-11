# RNNoise model — provenance

- **File:** `somnolent-hogwash.rnnn` (297 KB)
- **Source:** https://github.com/GregorR/rnnoise-models — `somnolent-hogwash-2018-09-01/sh.rnnn`
- **Model fit:** Speech signal × Recording noise (per the repo's model table) — best fit for
  talking-head / recorded-room speech.
- **License:** Public domain. The repository README states: *"With the exception of the tools/
  directory and this file, none of this work is creative and thus none of it is subject to
  copyright."* (Neural-network weights are not a creative work.)
- **Fetched:** 2026-07-11, for the `voice_isolation` tool (ffmpeg `arnndn=model=...`).
- Callers may point `model_path` / env `RNNOISE_MODEL` at a different `.rnnn` (e.g.
  `beguiling-drafter` for voice-with-non-speech-sounds).
