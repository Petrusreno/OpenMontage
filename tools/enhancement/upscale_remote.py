"""Remote GPU upscaler — delegates to gpu-worker via gpu-bridge.

Drop-in companion to upscale.py (provider="realesrgan").
Only active when GPU_WORKER_ENABLED=true.
Base URL defaults to http://127.0.0.1:8501 (the IAP tunnel port).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_POLL_INTERVAL_S = 3.0
_POLL_TIMEOUT_S  = 600


def _worker_enabled() -> bool:
    return os.environ.get("GPU_WORKER_ENABLED", "").lower() in {"true", "1", "yes"}


def _base_url() -> str:
    return os.environ.get("GPU_WORKER_URL", "http://127.0.0.1:8501").rstrip("/")


def _worker_token() -> str:
    return os.environ.get("GPU_WORKER_TOKEN", "")


def _headers() -> dict[str, str]:
    tok = _worker_token()
    if not tok:
        raise RuntimeError("GPU_WORKER_TOKEN not set.")
    return {"X-Worker-Token": tok}


def _poll_job(job_id: str, timeout: float = _POLL_TIMEOUT_S) -> dict[str, Any]:
    """Poll /job/{id} until done or error, bounded by timeout.

    Continues on transient poll errors (connection reset, timeout) until
    the deadline is reached, then raises.
    """
    base = _base_url()
    deadline = time.time() + timeout
    interval = _POLL_INTERVAL_S
    last_exc: Exception | None = None

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{base}/job/{job_id}",
                headers=_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "done":
                return data
            if status == "error":
                raise RuntimeError(f"Worker job {job_id} failed: {data.get('error')}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        time.sleep(min(interval, max(0.0, deadline - time.time())))
        interval = min(interval * 1.3, 20.0)

    if last_exc:
        raise TimeoutError(
            f"Job {job_id} timed out after {timeout}s (last error: {last_exc})"
        )
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


def _download_result(job_id: str, output_path: Path) -> None:
    """Download /result/{id} to output_path."""
    base = _base_url()
    resp = requests.get(
        f"{base}/result/{job_id}",
        headers=_headers(),
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)


class UpscaleRemote(BaseTool):
    name = "upscale_remote"
    version = "0.1.0"
    tier = ToolTier.ENHANCE
    capability = "enhancement"
    provider = "gpu-remote"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set GPU_WORKER_ENABLED=true, GPU_WORKER_URL, and GPU_WORKER_TOKEN. "
        "Start gpu-bridge: bash ~/ClaudeCode/gpu-bridge/launcher.sh"
    )
    agent_skills = ["ffmpeg"]

    capabilities = [
        "image_upscale",
        "video_upscale",
        "face_aware_upscale",
    ]

    # Matches upscale.py input_schema exactly
    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "scale": {
                "type": "integer",
                "enum": [2, 4],
                "default": 4,
            },
            "model": {
                "type": "string",
                "enum": [
                    "RealESRGAN_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRNet_x4plus",
                ],
                "default": "RealESRGAN_x4plus",
            },
            "face_enhance": {
                "type": "boolean",
                "default": False,
            },
            "denoise_strength": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0,
        disk_mb=500, network_required=True,
    )
    idempotency_key_fields = ["input_path", "scale", "model"]
    side_effects = ["writes upscaled file to output_path", "starts GCP VM if stopped"]
    best_for = ["upscaling on CUDA GPU when local GPU unavailable"]

    def get_status(self) -> ToolStatus:
        if not _worker_enabled():
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error="GPU worker disabled. Set GPU_WORKER_ENABLED=true.",
            )

        input_path = Path(inputs["input_path"])
        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        default_output = str(input_path.with_stem(f"{input_path.stem}_upscaled"))
        output_path = Path(inputs.get("output_path", default_output))
        scale = inputs.get("scale", 4)
        model = inputs.get("model", "RealESRGAN_x4plus")

        start = time.time()
        try:
            base = _base_url()
            with input_path.open("rb") as fh:
                resp = requests.post(
                    f"{base}/upscale",
                    headers=_headers(),
                    files={"file": (input_path.name, fh)},
                    data={"scale": str(scale), "model": model},
                    timeout=30,
                )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]

            _poll_job(job_id)
            _download_result(job_id, output_path)
        except Exception as exc:
            return ToolResult(success=False, error=f"Remote upscale failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "provider": "gpu-remote",
                "input": str(input_path),
                "output": str(output_path),
                "scale": scale,
                "model": model,
            },
            artifacts=[str(output_path)],
            duration_seconds=round(time.time() - start, 2),
        )
