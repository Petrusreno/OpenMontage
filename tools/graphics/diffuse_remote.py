"""Remote GPU SDXL image generation — delegates to gpu-worker via gpu-bridge.

Drop-in companion to local_diffusion.py (provider="local_diffusion").
Only active when GPU_WORKER_ENABLED=true.
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
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S  = 600


def _worker_enabled() -> bool:
    return os.environ.get("GPU_WORKER_ENABLED", "").lower() in {"true", "1", "yes"}


def _base_url() -> str:
    return os.environ.get("GPU_WORKER_URL", "http://127.0.0.1:8501").rstrip("/")


def _headers() -> dict[str, str]:
    tok = os.environ.get("GPU_WORKER_TOKEN", "")
    if not tok:
        raise RuntimeError("GPU_WORKER_TOKEN not set.")
    return {"X-Worker-Token": tok}


def _poll_job(job_id: str, timeout: float = _POLL_TIMEOUT_S) -> dict[str, Any]:
    base = _base_url()
    deadline = time.time() + timeout
    interval = _POLL_INTERVAL_S
    last_exc: Exception | None = None

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{base}/job/{job_id}", headers=_headers(), timeout=15,
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
        interval = min(interval * 1.3, 30.0)

    if last_exc:
        raise TimeoutError(
            f"Job {job_id} timed out after {timeout}s (last error: {last_exc})"
        )
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


def _download_result(job_id: str, output_path: Path) -> None:
    base = _base_url()
    resp = requests.get(
        f"{base}/result/{job_id}", headers=_headers(), timeout=120, stream=True,
    )
    resp.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)


class DiffuseRemote(BaseTool):
    name = "diffuse_remote"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "gpu-remote"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set GPU_WORKER_ENABLED=true, GPU_WORKER_URL, and GPU_WORKER_TOKEN. "
        "Start gpu-bridge: bash ~/ClaudeCode/gpu-bridge/launcher.sh"
    )
    agent_skills = []

    capabilities = ["generate_image", "generate_illustration", "text_to_image"]
    supports = {
        "negative_prompt": True,
        "seed": True,
        "offline": False,
        "custom_size": True,
    }
    best_for = [
        "SDXL image generation on CUDA GPU (T4/L4)",
        "fallback when GBS/browser generation unavailable",
        "privacy-sensitive local-equivalent generation",
    ]
    not_good_for = [
        "highest photorealistic quality (GBS/Gemini is better)",
        "when GPU_WORKER_ENABLED is not set",
    ]

    # Matches local_diffusion.py input_schema exactly
    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "negative_prompt": {"type": "string", "default": ""},
            "width": {"type": "integer", "default": 1024},
            "height": {"type": "integer", "default": 1024},
            "model": {
                "type": "string",
                "default": "stabilityai/stable-diffusion-xl-base-1.0",
            },
            "seed": {"type": "integer"},
            "num_inference_steps": {"type": "integer", "default": 30},
            "guidance_scale": {"type": "number", "default": 7.5},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0,
        disk_mb=500, network_required=True,
    )
    retry_policy = RetryPolicy(max_retries=1)
    idempotency_key_fields = ["prompt", "width", "height", "seed", "model"]
    side_effects = [
        "writes image file to output_path",
        "starts GCP VM if stopped",
        "may download model weights on VM on first run",
    ]

    def get_status(self) -> ToolStatus:
        if not _worker_enabled():
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # T4 spot ~$0.11/hr; SDXL ~2 min/image -> ~$0.004
        return 0.005

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 120.0  # ~2 min on T4 for SDXL

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error="GPU worker disabled. Set GPU_WORKER_ENABLED=true.",
            )

        prompt = inputs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, error="prompt is required.")

        output_path = Path(inputs.get("output_path", "diffuse_remote_output.png"))
        seed = inputs.get("seed")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": inputs.get("negative_prompt", ""),
            "width": inputs.get("width", 1024),
            "height": inputs.get("height", 1024),
            "steps": inputs.get("num_inference_steps", 30),
            "model": inputs.get(
                "model", "stabilityai/stable-diffusion-xl-base-1.0"
            ),
        }
        if seed is not None:
            payload["seed"] = seed

        start = time.time()
        try:
            base = _base_url()
            resp = requests.post(
                f"{base}/diffuse",
                headers={**_headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]

            _poll_job(job_id, timeout=_POLL_TIMEOUT_S)
            _download_result(job_id, output_path)
        except Exception as exc:
            return ToolResult(success=False, error=f"Remote diffuse failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "provider": "gpu-remote",
                "model": payload["model"],
                "prompt": prompt,
                "output": str(output_path),
                "width": payload["width"],
                "height": payload["height"],
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            seed=seed,
            model=payload["model"],
        )
