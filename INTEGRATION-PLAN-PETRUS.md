# OpenMontage ↔ Petrus Stack — Integration Plan

> Goal: (A) wire OpenMontage to our FREE cascade, (B) bridge OpenMontage→Palmier,
> swap its video/image engines to our free ones, and offload CUDA work to a GCP GPU VM
> that consumes **credits only, never the card**. Drafted 2026-06-23 from live recon.

## 0. Key discoveries (change the plan)

1. **The credit-only killswitch ALREADY EXISTS and is live.** Budget `killswitch-net-spend`
   (R$10, INCLUDE_ALL_CREDITS) → Pub/Sub `billing-killswitch` → Cloud Function gen2
   `billing-killswitch` (SA has `billing.admin`) → `updateBillingInfo` **detaches billing**
   when net spend > cap. The "no card billing" constraint is structurally satisfied. We
   reuse it; we do not rebuild it.
2. **No GPU leak (verified).** The only GCE instance is `duix-eval` (L4 spot, **TERMINATED/
   stopped**) → zero current GPU spend. (An earlier recon claim of a running `wm-l4-22082`
   was a hallucination — confirmed absent via `instances list`.) Killswitch infra verified
   live: budgets `killswitch-net-spend` R$10 + `total` R$20 + `petrus-ai-stack-monthly` R$300
   (all INCLUDE_ALL_CREDITS) + `actual-spend-only-NO-credits` R$1 (EXCLUDE_ALL_CREDITS,
   alert-only); topic `billing-killswitch`; Function `billing-killswitch` ACTIVE gen2;
   `~/gcp-killswitch/main.py` present.
3. **Swapping video/image to our free browser cascade makes OpenMontage's local CUDA
   *generation* (WAN/Hunyuan/CogVideo/LTX-local/SD-local) redundant.** So the GPU VM's real
   job shrinks to **enhancement + analysis** (Real-ESRGAN upscale, GFPGAN/CodeFormer face
   restore, bg-remove, WhisperX) + optional local-diffusion *fallback*. That means a **T4
   spot VM suffices** for most of it — cheaper, less stockout-prone than L4.
4. **Lipsync/talking-head are NOT in OpenMontage** (they live in avatar-studio's duix/wav2lip
   lane, which already has its own GPU bridge :8386). "All OpenMontage features on GPU" =
   the enhancement/analysis suite, not lipsync.
5. OpenMontage adds providers by **just dropping a tool file** (auto-discovery, no decorator).
   Remote-GPU precedent already exists: `ltx_video_modal.py` calls `MODAL_LTX2_ENDPOINT_URL`
   over HTTP → poll → download → ToolResult. That is our exact template.

## 1. Target architecture

```
                       ┌─────────────────────────── OpenMontage (local, Mac M5) ───────────────────────────┐
  brief ──> pipelines ─┤ selectors (pinned free-first, paid as fallback)                                    │
                       │   image_generation  → [gbs_image]  ───────────► GBS :8089 (Gemini browser, PNG)    │
                       │   video_generation  → [autobrowser_video] ────► auto-browser :8000 (VEO/Meta, MP4) │
                       │   tts               → [piper_local]/[el_petrus]► Piper :8004 / ElevenLabs helper    │
                       │   enhancement/analysis → [*_remote] ──HTTP/IAP─► GCP gpu-worker :8500 (T4/L4 spot)  │
                       └───────────────┬───────────────────────────────────────────────────────────────────┘
                                       │ renders/final.mp4 + assets/
                                       ▼
                       studio-to-palmier ──import_media──► Palmier (assemble/caption) ──► palmier-to-premiere
```

Principle honored: **local → free → paid** (ai_priority). Paid providers stay installed as
fallback; free is pinned default. Everything reversible (additive tools + config flags).

---

## 2. Phase 1 — Free-engine swap (image / video / TTS) + pinning
*Local, free, highest value, independent of GCP. Build first.*

### 2.1 Image adapter — `tools/graphics/gbs_image.py`  (NEW)
- `capability="image_generation"`, `provider="gbs"`, `runtime=API`.
- `execute()` calls `~/ClaudeCode/shared/gemini_browser_client.generate_to_path(prompt, out, aspect=..., postprocess=True)` → PNG. Token from `~/.config/gemini-browser-service/token` (read by the client; never hardcode).
- **Aspect mapping**: OpenMontage width/height → GBS `{16:9,9:16,1:1,4:5}`. Unsupported ratios (21:9 cinematic) → render nearest + pad/crop note in ToolResult.data.
- Dewatermark: rely on GBS `postprocess=True`; if NanoRemover :8088 is wanted, start it first (currently down).
- GBS health ✓ live (idle, concurrency 3 — note: throughput-limited for many-image pipelines).

### 2.2 Video adapter — `tools/video/autobrowser_video.py`  (NEW)
- `capability="video_generation"`, `provider="auto-browser"`, `runtime=API`.
- `execute()` = POST `http://127.0.0.1:8000/video/generate` → poll `GET /video/status/{id}` until `completed` → use returned `video_path` (cache at `~/ClaudeCode/video-pipeline-output/cache/`). Block-poll to satisfy the sync contract (same shape as `generate_ltx_modal_video`). Base URL from `AUTO_BROWSER_URL` env (default :8000).
- Cascade ElevenLabs→VEO→Meta handled server-side. Cost reported 0 (browser quotas).
- **Dependency: the auto-browser controller is DOWN right now** (:8000 refused). Phase 1 video lane needs it started (docker `auto-browser-controller-1`) — add a health-gate + clear error.

### 2.3 TTS adapters
- `tools/audio/piper_local.py` (NEW): POST `http://127.0.0.1:8004/tts` (form: text, engine=piper, voice_id=`pt_BR-faber-medium`, language=pt) → WAV 48k. Token-gated (avatar-studio token). Free default. ✓ live.
- `tools/audio/elevenlabs_petrus.py` (NEW): shell `~/Developer/video-use/helpers/elevenlabs_tts.py --voice mUH8M4GPB2nbxbxkhEvW --text … --out … --model multilingual` → WAV 48k, gets the **cache + Petrus voice + Voicebox fallback** (better than OM's stock ElevenLabs tool). For hero/narration.
- Keep OM's existing Google/OpenAI TTS as further fallback.

### 2.4 Pinning mechanism (small, reversible core touch)
- Add `tools.preferred_providers` to `config.yaml` + `lib/config_model.py`:
  ```yaml
  tools:
    preferred_providers:
      image_generation: gbs
      video_generation: auto-browser
      tts: piper          # escalate to elevenlabs_petrus for hero scenes
    fallback_enabled: true
  ```
- Patch the 3 selectors (~5 lines each): when `inputs.preferred_provider == "auto"`, default
  it from `config.tools.preferred_providers[capability]`; keep scoring for fallback when the
  pinned provider is unavailable (honors `fallback_enabled`).
- Alternative (zero core touch): pass `preferred_provider` from pipeline manifests. Chosen
  approach = config (one place, survives pipeline edits).

### 2.5 Phase 1 test
Run an `animated-explainer` short with `--dry`-style single-asset calls: one GBS image, one
auto-browser clip, one Piper line; assert artifacts land and selectors picked the free
providers. Verify fallback by stopping GBS and confirming it falls back to FLUX (if FAL_KEY set) or errors cleanly.

---

## 3. Phase 2 — OpenMontage → Palmier bridge (B)
*Reuses existing `studio-to-palmier`; minimal new code.*

- OpenMontage outputs: `projects/<name>/renders/final.mp4` + `projects/<name>/assets/{video,audio,images,music}`.
- **Mode A (review):** `import_media(path=final.mp4)` → one clip in Palmier for trim/caption.
- **Mode B (re-editable, default):** run `studio_palmier_manifest.py --dir projects/<name>/assets/video --downloads-hours 0 --fps <proj>` → import each → `add_clips` by frame. Lets you recut in Palmier.
- Deliverable: add `--om-project <name>` convenience flag to the studio-to-palmier helper (resolves the assets dir) + a short runbook section. Then → `palmier-to-premiere` for finishing.

---

## 4. Phase 3 — GCP credit-safe GPU worker (enhancement + analysis)
*Offloads the CUDA features M5 can't run. Ephemeral, spot, reuses the live killswitch.*

### 4.1 Safety (mostly already in place)
- **Keep** the live `billing-killswitch` (detach on net spend > R$10) — card protection.
- **Add** `credit-exhaustion-cap` budget ≈ 0.9× Console credit balance, INCLUDE_ALL_CREDITS,
  pointed at the SAME `billing-killswitch` topic → detach *before* credits run dry.
  (Credit balance is Console-only: billing/014FCE-93E817-95C5AB/credits — user reads it.)
- **Optional tighten:** wire the existing R$1 EXCLUDE_ALL_CREDITS budget to the topic too
  (detach at first centavo of real card spend).

### 4.2 GPU worker
- `gpu-worker` FastAPI on the VM, `:8500`, endpoints: `/health /upscale /face-restore
  /bg-remove /diffuse /whisperx`. `X-token` auth on **every** route.
- VM: **T4 spot** (n1-standard-4 + 1×T4, ~$0.11–0.18/hr) default for upscale/restore/bg/whisperx;
  **L4 spot** only for local-diffusion fallback. Reuse `duix-eval-snap` to dodge L4 stockout;
  multi-zone retry us-east1-{b,c,d}→us-central1→us-west1, then T4 fallback.
- **Ephemeral**: start on demand, stop after job; guest idle-shutdown cron (stop after 15 min
  <5% GPU util). A 4-min job ≈ R$0.10; persistent 24/7 ≈ R$1k/mo — ephemeral is mandatory.

### 4.3 Local side
- `gpu-bridge` on Mac = fork of `~/ClaudeCode/avatar-studio/tools/duix_bridge/bridge.py`:
  `_ensure_vm_running` (gcloud start + poll) → **IAP tunnel** (`gcloud compute start-iap-tunnel
  gpu-worker 8500 --local-host-port=localhost:8501`; lets the VM drop its external IP) → call
  → `/vm/stop`. Same `X-token` pattern.
- OpenMontage remote adapters (NEW), modeled on `ltx_video_modal.py`:
  `tools/enhancement/upscale_remote.py`, `face_restore_remote.py`, `bg_remove_remote.py`,
  `tools/analysis/whisperx_remote.py`, `tools/graphics/diffuse_remote.py`. Each reads
  `GPU_WORKER_URL` (=`http://127.0.0.1:8501`), POST→poll→download→ToolResult. Pin these as
  preferred for their capabilities when `GPU_WORKER_ENABLED=true`.

---

## 5. Cross-cutting
- **Reversibility:** all new providers are additive files; pinning is config flags; paid/local
  originals untouched. Set `fallback_enabled: false` to go free-only; flip flags to revert.
- **Security:** tokens read from existing locations (GBS token file, avatar token, .env,
  ELEVENLABS key) — never hardcoded, never committed (.env, tokens stay gitignored).
- **Testing:** per-phase smoke (Phase 1 §2.5; Phase 2 import + get_timeline verify; Phase 3
  `/health` + one upscale round-trip with VM auto-stop confirmed).
- **Config surface:** new env — `AUTO_BROWSER_URL`, `GPU_WORKER_URL`, `GPU_WORKER_ENABLED`,
  `GPU_WORKER_TOKEN`; new config — `tools.preferred_providers`, `tools.fallback_enabled`.

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| auto-browser controller down (:8000) | health-gate + start docker; fall back to paid video if enabled |
| GBS browser throughput (concurrency 3) | batch/serialize image calls; cache; paid fallback for bulk |
| GBS aspect set limited (no 21:9) | aspect-map + pad/crop; note in ToolResult |
| async→sync video | block-poll in adapter (proven pattern) |
| killswitch detach breaks ALL services | it's an emergency stop; rely on ephemeral + small caps routinely |
| L4 spot stockout | T4-first; reuse duix-eval-snap; multi-zone retry; on-demand last resort |
| credit balance unknown via CLI | user reads Console once to size credit-exhaustion budget |

## 7. Open decisions (need user)
1. **Free-only vs free-first+paid-fallback?** (recommend: free-first, keep paid fallback)
2. **GPU default tier** — T4 (cheap, enhancement) with L4 only for diffusion fallback? (recommend yes)
3. **Bridge B default** — re-editable assets (Mode B) vs single final.mp4 (Mode A)? (recommend Mode B)
4. **Build order** — recommend Phase 1 → 2 → 3.

## 8. Recommended sequence
Phase 0 (safety, ~mins): identify/stop `wm-l4-22082`; read credit balance; add credit-exhaustion budget.
Phase 1 (free engines): biggest value, all local. → Phase 2 (Palmier bridge): quick, reuses skill.
Phase 3 (GPU worker): last; ephemeral T4 spot + reuse killswitch & duix pattern.
