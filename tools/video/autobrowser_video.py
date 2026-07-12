"""Auto-browser controller video generation adapter.

Wraps the auto-browser controller at AUTO_BROWSER_URL (default :8000).
POST /video/generate → poll GET /video/status/{job_id} → download artifact.
Same shape as generate_ltx_modal_video in _shared.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_POLL_INTERVAL = 5.0
_DEFAULT_TIMEOUT = 600


class AutoBrowserVideo(BaseTool):
    name = "autobrowser_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "auto-browser"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    install_instructions = (
        "Start the auto-browser controller:\n"
        "  docker start auto-browser-controller-1\n"
        "  (or set AUTO_BROWSER_URL to override the base URL)\n"
        "Token (optional): set AUTO_BROWSER_TOKEN env var."
    )
    agent_skills = ["ai-video-gen", "create-video"]

    capabilities = ["text_to_video", "provider_selection"]
    supports = {
        "text_to_video": True,
        "image_to_video": False,
        "offline": False,
        "cloud_generation": True,
    }
    best_for = [
        "free video generation via browser cascade (VEO / Meta quota)",
        "zero direct API-key cost",
        "PT-BR / derma content via auto-browser controller",
    ]
    not_good_for = [
        "offline or privacy-constrained rendering",
        "image-to-video workflows",
        "when auto-browser-controller-1 is not running",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "duration_s": {
                "type": "number",
                "default": 5,
                "description": "Requested video duration in seconds.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1"],
                "default": "16:9",
            },
            "tier_hint": {
                "type": "string",
                "description": "Optional tier hint forwarded to the controller (e.g. 'veo', 'meta').",
            },
            "timeout": {
                "type": "integer",
                "default": _DEFAULT_TIMEOUT,
                "description": "Max seconds to wait for the job to complete.",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=1, backoff_seconds=10.0, retryable_errors=["timeout", "server_error"]
    )
    idempotency_key_fields = ["prompt", "aspect_ratio", "duration_s"]
    side_effects = ["writes video file to output_path", "consumes auto-browser quota"]
    user_visible_verification = ["Watch generated clip for motion quality and prompt adherence"]

    def _base_url(self) -> str:
        return os.environ.get("AUTO_BROWSER_URL", _DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        token = os.environ.get("AUTO_BROWSER_TOKEN", "")
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def get_status(self) -> ToolStatus:
        try:
            import requests
            resp = requests.get(f"{self._base_url()}/healthz", headers=self._headers(), timeout=3)
            if resp.ok:
                return ToolStatus.AVAILABLE
        except Exception:
            pass
        return ToolStatus.UNAVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import requests
        import shutil

        start = time.time()
        base = self._base_url()
        headers = self._headers()

        # Health gate — require 2xx
        try:
            health_resp = requests.get(f"{base}/healthz", headers=headers, timeout=5)
            if not health_resp.ok:
                raise RuntimeError(f"HTTP {health_resp.status_code}")
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"auto-browser controller down — start auto-browser-controller-1 ({exc})",
            )

        prompt = inputs["prompt"]
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration_s": inputs.get("duration_s", 5),
            "aspect": inputs.get("aspect_ratio", "16:9"),
            "tier_hint": inputs.get("tier_hint", ""),
        }

        # Submit job
        try:
            submit_resp = requests.post(
                f"{base}/video/generate",
                json=payload,
                headers={**headers, "Content-Type": "application/json"},
                timeout=30,
            )
            submit_resp.raise_for_status()
            job_data = submit_resp.json()
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to submit video job: {exc}")

        job_id = job_data.get("job_id") or job_data.get("id")
        if not job_id:
            return ToolResult(
                success=False,
                error=f"No job_id in response: {job_data}",
            )

        # Poll until complete — transient poll errors continue rather than abort
        timeout_s = inputs.get("timeout", _DEFAULT_TIMEOUT)
        deadline = time.time() + timeout_s
        video_path: str | None = None
        video_url: str | None = None

        while time.time() < deadline:
            time.sleep(min(_POLL_INTERVAL, max(0.0, deadline - time.time())))
            try:
                status_resp = requests.get(
                    f"{base}/video/status/{job_id}",
                    headers=headers,
                    timeout=15,
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()
            except Exception:
                # Transient network error — retry until deadline
                continue

            status = (status_data.get("status") or "").lower()

            if status in {"completed", "done"}:
                # Controller may return a URL (Docker) or a server-local path
                raw = (
                    status_data.get("video_url")
                    or status_data.get("url")
                    or status_data.get("video_path")
                    or status_data.get("output_path")
                )
                if raw and (raw.startswith("http://") or raw.startswith("https://")):
                    video_url = raw
                else:
                    video_path = raw
                break

            if status == "failed":
                err = status_data.get("error") or status_data.get("message") or "unknown"
                return ToolResult(success=False, error=f"Video job {job_id} failed: {err}")

        if video_url is None and video_path is None:
            return ToolResult(
                success=False,
                error=f"Video job {job_id} timed out after {timeout_s}s",
            )

        # Resolve final artifact path
        if inputs.get("output_path"):
            dest = Path(inputs["output_path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest = Path(f"autobrowser_video_{job_id}.mp4")

        if video_url:
            # Download from URL (controller running in Docker exposes a URL)
            try:
                dl = requests.get(video_url, timeout=120)
                dl.raise_for_status()
                dest.write_bytes(dl.content)
            except Exception as exc:
                return ToolResult(success=False, error=f"Failed to download video from {video_url}: {exc}")
        else:
            # Server-local path — verify it exists before claiming success
            src = Path(video_path)  # type: ignore[arg-type]
            if not src.exists():
                return ToolResult(
                    success=False,
                    error=f"Controller reported completion but video file not found: {src}",
                )
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)

        if not dest.exists() or dest.stat().st_size == 0:
            return ToolResult(success=False, error=f"Final artifact missing or empty: {dest}")

        return ToolResult(
            success=True,
            data={
                "provider": "auto-browser",
                "job_id": job_id,
                "prompt": prompt,
                "aspect_ratio": payload["aspect"],
                "duration_s": payload["duration_s"],
                "output": str(dest),
                "format": "mp4",
                "mode": "browser-cascade",
            },
            artifacts=[str(dest.resolve())],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="auto-browser",
        )
