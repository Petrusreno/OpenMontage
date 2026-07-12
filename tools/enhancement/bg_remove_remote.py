"""Remote GPU background removal — delegates to gpu-worker via gpu-bridge.

Drop-in companion to bg_remove.py (provider="rembg").
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
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_POLL_INTERVAL_S = 3.0
_POLL_TIMEOUT_S  = 300


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
        interval = min(interval * 1.3, 20.0)

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


class BgRemoveRemote(BaseTool):
    name = "bg_remove_remote"
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
    agent_skills = []

    capabilities = [
        "background_removal",
        "alpha_matte",
        "batch_processing",
        "custom_background",
    ]

    # Matches bg_remove.py input_schema exactly
    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Path to image or video frame",
            },
            "output_path": {
                "type": "string",
                "description": "Output path; defaults to {stem}_nobg.png",
            },
            "model": {
                "type": "string",
                "enum": ["u2net", "u2net_human_seg", "isnet-general-use"],
                "default": "u2net",
            },
            "bg_color": {
                "type": "string",
                "description": "Replacement background color hex (e.g. #00FF00). Transparent if not set.",
            },
            "alpha_matting": {
                "type": "boolean",
                "default": False,
                "description": "Use alpha matting for finer edges",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0,
        disk_mb=200, network_required=True,
    )
    idempotency_key_fields = ["input_path", "model", "bg_color", "alpha_matting"]
    side_effects = [
        "writes background-removed image to output_path",
        "starts GCP VM if stopped",
    ]
    best_for = ["background removal on CUDA GPU when local GPU unavailable"]

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

        output_path = Path(
            inputs.get(
                "output_path",
                str(input_path.with_stem(f"{input_path.stem}_nobg").with_suffix(".png")),
            )
        )
        model = inputs.get("model", "u2net")
        alpha_matting = inputs.get("alpha_matting", False)
        bg_color = inputs.get("bg_color", "")

        start = time.time()
        try:
            base = _base_url()
            with input_path.open("rb") as fh:
                resp = requests.post(
                    f"{base}/bg-remove",
                    headers=_headers(),
                    files={"file": (input_path.name, fh)},
                    data={
                        "model": model,
                        "alpha_matting": "true" if alpha_matting else "false",
                        "bg_color": bg_color or "",
                    },
                    timeout=30,
                )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]

            _poll_job(job_id)
            _download_result(job_id, output_path)
        except Exception as exc:
            return ToolResult(success=False, error=f"Remote bg-remove failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "provider": "gpu-remote",
                "input": str(input_path),
                "output": str(output_path),
                "model": model,
                "alpha_matting": alpha_matting,
                "bg_color": bg_color or None,
            },
            artifacts=[str(output_path)],
            duration_seconds=round(time.time() - start, 2),
        )
