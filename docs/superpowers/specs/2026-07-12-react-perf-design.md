# React Composite Perf Refinement — Design Spec

> Automation port batch 2, sub-project 4 (morph-cut ✓ · voice-isolation ✓ · beat-sync ✓ · **react perf-refinement**).
> Date: 2026-07-12 · Status: design (pre-implementation) · Branch: `feat/premiere-automation-react-perf`

## Context

`tools/video/green_screen_composite.py` composites a keyed speaker (talking head on a solid
`#0E172A` background, e.g. from HeyGen) over a Remotion/testsrc background — the "react video"
use case (speaker in front, content rolling behind). The existing path is slow: it does **three
full disk passes** (extract every speaker frame to PNG, extract every bg frame to PNG, re-encode
composite PNGs) with **per-frame Python/PIL alpha compositing** in between. On a 4 s / 60-frame
640×360 clip that is ~762 ms; it scales as O(frames × pixels) in Python, so 1080p / minutes-long
react clips are far worse.

This sub-project adds an **ffmpeg-native single-pass fast path** — `colorkey` + `overlay` in one
filtergraph, no PNG intermediates — as a new engine on the same tool.

### Evidence gathered 2026-07-12 (empirical, before speccing)

- `colorkey` (RGB, solid-color keyer) is present in the local ffmpeg — correct for the speaker's
  solid `#0E172A` background (NOT green `chromakey`).
- A single filtergraph renders all four layouts (`full_behind`, `news_anchor`, `pip`, `split`)
  correctly in **~120 ms each** on the validation clip.
- **Keying verified by direct pixel sampling:** in `full_behind`, the top strip above the speaker
  is **0.0 % still-navy** (mean RGB distance 262 from `#0E172A`) — the navy bg is fully keyed out
  and the background shows through. A no-key overlay leaves the same strip **100 % navy** (distance
  1.0), i.e. the speaker's navy bg opaquely covers the background. So keying is real, not cosmetic.
- Speaker-track audio is preserved via `-map 0:a?` (optional map — silent speaker → no failure).
- **6.4× faster** than PIL (120 ms vs 762 ms) on the 4 s clip; the multiplier grows with resolution
  and length because PIL cost is per-pixel Python while ffmpeg is native + threaded.

### Decisions locked (user, 2026-07-12)

- **Default engine = `ffmpeg`** (the fast path). `engine="pil"` opt-in preserves the exact legacy
  per-frame behavior byte-for-byte.
- **Audio = speaker-track passthrough by default** (`-map 0:a?`); `original_audio_path`, when given,
  overrides it. This is strictly better for react (the voiceover lives in the speaker/HeyGen clip);
  the legacy PIL path drops speaker audio unless `original_audio_path` is supplied.

## Goal

Add an `engine` parameter to `green_screen_composite`. `engine="ffmpeg"` (new default) composites
the keyed speaker over the background in **one ffmpeg filtergraph pass** for all four existing
layouts, keying the solid `bg_color_hex` with `colorkey`, preserving speaker audio by default, and
honoring `original_audio_path` as an override. `engine="pil"` runs the unchanged legacy path. The
fast path never fabricates success: if ffmpeg fails or the output is missing/empty, it returns
`success=False` with the ffmpeg error — it does NOT silently fall back to PIL (the user chose an
explicit default, not `auto`).

Non-goals (YAGNI): per-frame spill suppression / edge feathering beyond `colorkey`'s `blend`;
variable/animated layouts; an `auto` engine that probes then picks (explicitly rejected — one more
code path to test for no chosen benefit); changing the PIL path's keying math or layout geometry;
GPU keying (`colorkey_opencl`).

## The filtergraphs (validated, per layout)

Output size = background size `(W, H)`; `hw = W // 2`. Keyer = `colorkey={ff_color}:{sim}:{blend}`
where `ff_color` is `bg_color_hex` as `0xRRGGBB`, `sim` = `key_similarity` (default 0.10), `blend`
= `key_blend` (default 0.08). Inputs: `[0:v]` speaker, `[1:v]` background.

- **full_behind** — speaker full-frame over full-frame bg:
  `[1:v]scale=W:H[bg];[0:v]scale=W:H,{ck}[fg];[bg][fg]overlay=0:0[v]`
- **news_anchor** — bg shifted up `bg_shift_up` px (content scrolls above the head, black fill at
  bottom), speaker scaled by `speaker_scale`, bottom-center:
  `[1:v]scale=W:H,crop=W:H-S:0:S,pad=W:H:0:0:black[bg];[0:v]scale=iw*sc:ih*sc,{ck}[fg];[bg][fg]overlay=(W-w)/2:H-h[v]`
  (crop takes the region from `y=S` down — the content that moves to the top — then `pad` restores
  full height with black at the bottom: exactly the PIL `paste(bg,(0,-S))` semantics.)
- **pip** — bg full-frame, speaker 30 % bottom-right, 20 px margin:
  `[1:v]scale=W:H[bg];[0:v]scale=W*0.30:H*0.30,{ck}[fg];[bg][fg]overlay=W-w-20:H-h-20[v]`
- **split** — keyed speaker left half over black, bg right half:
  `color=c=black:s=WxH[base];[0:v]scale=hw:H,{ck}[l];[1:v]scale=hw:H[r];[base][l]overlay=0:0[t];[t][r]overlay=hw:0[v]`

The output command maps `[v]`, maps audio (see below), caps duration with `-t {duration}`
(`duration = min(speaker, bg)`, as today), forces `-r {target_fps}` (`min` of the two source fps,
as today), and encodes `libx264 -crf 18 -preset fast -pix_fmt yuv420p`.

### Audio mapping (fast path)

- `original_audio_path` given → second input; `-map [v] -map 1:a:0 -c:a aac -b:a 192k -shortest`
  (validate the file exists first, as the tool already does).
- else → `-map [v] -map 0:a?` (speaker audio if present; the `?` makes a silent speaker safe) with
  `-c:a aac -b:a 192k`.

## Architecture

One method added to `GreenScreenComposite`, plus a thin dispatch in `execute`. The PIL code moves
verbatim into a private method; nothing about its behavior changes.

- `execute(inputs)` — validate inputs (existing checks + `engine` enum + `key_similarity`/`key_blend`
  numeric in `[0, 1]`), then dispatch on `engine`:
  - `"pil"` → `self._execute_pil(...)` — the current body, moved unchanged.
  - `"ffmpeg"` (default) → `self._execute_ffmpeg(...)`.
- `_execute_ffmpeg(speaker, background, output, *, layout, speaker_scale, bg_shift_up, bg_color_hex,
  key_similarity, key_blend, original_audio_path) -> ToolResult` — probe both (reuse `_probe_video`),
  compute `target_fps`/`out_w`/`out_h`/`duration` (reuse the existing logic), build the layout
  filtergraph via `_layout_filtergraph`, run one guarded ffmpeg call, verify output non-empty, return
  `ToolResult` with the same `data` shape as the PIL path plus `"engine": "ffmpeg"`.
- `_layout_filtergraph(layout, out_w, out_h, speaker_scale, bg_shift_up, ff_color, sim, blend) -> str`
  — pure string builder for the four graphs above; raises `ValueError` on unknown layout
  (independently unit-testable, no ffmpeg needed).
- `_ffmpeg_color(bg_color_hex) -> str` — `#0E172A` → `0x0E172A` (validate 6 hex digits).

`_probe_video`, `_parse_hex_color`, `_composite_frame`, `_encode_frames`, `_mux_audio`,
`_extract_frames`, `_cleanup_temp` stay as-is (the PIL path still uses them).

## Input schema additions

```jsonc
{
  "engine":        { "type": "string", "enum": ["ffmpeg", "pil"], "default": "ffmpeg",
                     "description": "ffmpeg = fast single-pass colorkey+overlay (default); pil = legacy per-frame path." },
  "key_similarity":{ "type": "number", "default": 0.10, "minimum": 0.0, "maximum": 1.0,
                     "description": "colorkey similarity (fast path). Higher removes more near-bg color." },
  "key_blend":     { "type": "number", "default": 0.08, "minimum": 0.0, "maximum": 1.0,
                     "description": "colorkey edge blend (fast path). Higher softens the key edge." }
}
```

`idempotency_key_fields` gains `engine`, `key_similarity`, `key_blend`.

## Output — `ToolResult`

Identical `data` shape to the PIL path (`output`, `layout`, `fps`, `frame_count`, `duration`,
`dimensions`, `speaker_scale`, `has_audio`) plus `"engine": "ffmpeg"|"pil"`. `frame_count` on the
fast path = `round(duration * target_fps)` (no PNGs to count); `has_audio` = whether an audio stream
was mapped (original_audio_path given, or the speaker had audio).

## Error / edge-case handling

- Missing speaker/background/`original_audio_path` → `success=False` (existing checks, kept).
- Invalid `engine` value → `success=False` with a clear message (enum guard) before any ffmpeg work.
- `key_similarity`/`key_blend` non-numeric or out of `[0,1]` → `success=False`, not a traceback.
- Bad `bg_color_hex` (not 6 hex digits) → `success=False` (both paths; PIL already assumes valid).
- ffmpeg failure / empty output on the fast path → `success=False` with the ffmpeg stderr; **no
  silent PIL fallback** (default is explicit, per the locked decision).
- Determinism: same inputs + params → byte-stable filtergraph and deterministic libx264 encode.
- The fast path writes no temp directory (single pass) — nothing to clean up on failure.

## Testing

`tests/tools/test_green_screen_composite.py` (new — the tool currently has no tests). Signal-level
helpers tested without ffmpeg; end-to-end tests ffmpeg-gated, synthesizing a keyed speaker (white
disc on `#0E172A`, with a tone) + a `testsrc2` background, mirroring the validation harness.

1. **`_layout_filtergraph`** returns the expected graph string for each of the 4 layouts and raises
   `ValueError` on an unknown layout. (pure, no ffmpeg)
2. **`_ffmpeg_color`** maps `#0E172A`→`0x0E172A`; rejects malformed hex.
3. **Engine dispatch:** `engine="pil"` produces a composite byte-identical to the pre-change PIL
   output (regression guard that the moved code is unchanged); default (no engine) uses ffmpeg.
4. **e2e keying (honesty anchor, ffmpeg-gated):** `full_behind` fast path → sample the top strip
   above the speaker; **< 5 % of its pixels remain within 40 RGB of `#0E172A`** (bg shows through).
   A no-key overlay of the same inputs leaves **> 90 %** — the test asserts the fast path is far
   below the no-key baseline. Thresholds from the validated measurement (0.0 % vs 100 %), not loosened.
5. **e2e all four layouts (ffmpeg-gated):** each returns `success=True`, output exists non-empty,
   dimensions = background dims, `frame_count ≈ duration*fps`.
6. **Audio passthrough (ffmpeg-gated):** speaker has audio, no `original_audio_path` → output has an
   audio stream (`has_audio=True`). `original_audio_path` given → output audio comes from that file.
   Silent speaker + no `original_audio_path` → `success=True`, no audio stream (the `?` map is safe).
7. **Speedup sanity (ffmpeg-gated, non-flaky):** fast path wall-time < PIL wall-time on the same
   clip (asserts the direction of the win, not a brittle absolute multiplier).
8. **Boundary validation:** non-numeric `key_similarity` → `success=False`, no traceback; invalid
   `engine` → `success=False`; `key_blend=1.5` → `success=False`.
9. **ffmpeg failure → clean failure:** point `background_path` at a non-video file → `success=False`
   with an error, not an exception.

## Integration touchpoints

- Same tool, same registry entry — no wiring. Existing `engine="pil"` callers are byte-unchanged;
  new callers (and the default) get the fast path.
- The react/talking-head pipeline that composites a HeyGen speaker over a Remotion background now
  runs the composite step ~6×+ faster and keeps the speaker's voiceover automatically.
- Reuses ffmpeg (hard dep) + numpy/PIL (already deps, still used by the PIL path). No new dependency.

## Open decisions (defaults chosen)

- `key_similarity=0.10`, `key_blend=0.08` — validated on the synthetic clip; tune per real footage
  (a noisier real `#0E172A` bg may want a slightly higher `similarity`). Exposed as params.
- `engine="ffmpeg"` default, `pil` opt-in; **no `auto`** (rejected as untested surface for no gain).
- Audio: speaker passthrough default, `original_audio_path` override.
