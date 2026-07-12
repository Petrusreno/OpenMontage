"""Gemini Browser Service (GBS) image generation adapter.

Wraps ~/ClaudeCode/shared/gemini_browser_client.generate_to_path().
Token is managed by the client itself (~/.config/gemini-browser-service/token).
No API keys handled here.
"""

from __future__ import annotations

import os
import sys
import tempfile
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

# GBS supports these aspect ratios natively.
_GBS_ASPECTS = {"16:9", "9:16", "1:1", "4:5"}


def _map_aspect(inputs: dict[str, Any]) -> tuple[str, bool]:
    """Return (gbs_aspect, was_adjusted).

    Priority: explicit inputs["aspect"] → derived from width/height → default 1:1.
    """
    explicit = inputs.get("aspect") or inputs.get("aspect_ratio")
    if explicit and explicit in _GBS_ASPECTS:
        return explicit, False

    width = inputs.get("width")
    height = inputs.get("height")
    if width and height and height != 0:
        ratio = width / height
        # Map to nearest supported aspect
        candidates = {
            "16:9": 16 / 9,
            "9:16": 9 / 16,
            "1:1": 1.0,
            "4:5": 4 / 5,
        }
        nearest = min(candidates, key=lambda k: abs(candidates[k] - ratio))
        adjusted = abs(candidates[nearest] - ratio) > 0.05
        return nearest, adjusted

    if explicit:
        # Unsupported explicit aspect — pick nearest by parsing "W:H"
        try:
            w_str, h_str = explicit.split(":")
            ratio = float(w_str) / float(h_str)
            candidates = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}
            nearest = min(candidates, key=lambda k: abs(candidates[k] - ratio))
            return nearest, True
        except Exception:
            pass

    return "1:1", False


class GbsImage(BaseTool):
    name = "gbs_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "gbs"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    install_instructions = (
        "Start the Gemini Browser Service: ~/ClaudeCode/gemini-browser-service/start.sh\n"
        "Token is read automatically from ~/.config/gemini-browser-service/token."
    )
    agent_skills = ["gemini-image-stacking"]

    capabilities = ["generate_image", "text_to_image", "generate_illustration"]
    supports = {
        "negative_prompt": False,
        "seed": False,
        "custom_size": False,
        "aspect_ratio_mapping": True,
        "postprocess_watermark_removal": True,
    }
    best_for = [
        "free image generation via Gemini browser quota",
        "zero API-key cost (browser-backed)",
        "dermatology / medical illustration (Petrus stack)",
    ]
    not_good_for = [
        "high-throughput pipelines (concurrency 3)",
        "offline generation",
        "exact seed reproducibility",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "aspect": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1", "4:5"],
                "description": "Aspect ratio. If omitted, derived from width/height.",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Alias for aspect (any W:H string; mapped to nearest supported).",
            },
            "width": {"type": "integer", "description": "Used to derive aspect ratio if aspect not set."},
            "height": {"type": "integer", "description": "Used to derive aspect ratio if aspect not set."},
            "output_path": {"type": "string", "description": "Destination PNG path. Auto-generated if omitted."},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, backoff_seconds=5.0, retryable_errors=["timeout"])
    idempotency_key_fields = ["prompt", "aspect"]
    side_effects = ["writes PNG to output_path", "consumes GBS browser quota"]
    user_visible_verification = ["Inspect generated PNG for quality and prompt adherence"]

    def get_status(self) -> ToolStatus:
        """Quick health-gate against the GBS service."""
        try:
            import requests
            base = os.environ.get("GBS_URL", "http://127.0.0.1:8089")
            resp = requests.get(f"{base}/health", timeout=3)
            if resp.ok:
                return ToolStatus.AVAILABLE
        except Exception:
            pass
        return ToolStatus.UNAVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()

        # Inject shared client path
        shared_path = os.path.expanduser("~/ClaudeCode/shared")
        if shared_path not in sys.path:
            sys.path.insert(0, shared_path)

        try:
            from gemini_browser_client import generate_to_path  # type: ignore[import]
        except ImportError as exc:
            return ToolResult(
                success=False,
                error=f"gemini_browser_client not found at {shared_path}: {exc}",
            )

        aspect, was_adjusted = _map_aspect(inputs)

        # Determine output path
        if inputs.get("output_path"):
            output_path = Path(inputs["output_path"])
        else:
            fd, tmp = tempfile.mkstemp(suffix=".png", prefix="gbs_image_")
            os.close(fd)
            output_path = Path(tmp)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            generate_to_path(
                prompt=inputs["prompt"],
                output_path=str(output_path),
                aspect=aspect,
                postprocess=True,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"GBS image generation failed: {exc}")

        if not output_path.exists():
            return ToolResult(
                success=False,
                error=f"GBS returned no error but output file not found: {output_path}",
            )

        data: dict[str, Any] = {
            "provider": "gbs",
            "prompt": inputs["prompt"],
            "aspect": aspect,
            "output": str(output_path),
        }
        if was_adjusted:
            data["aspect_adjusted"] = True
            data["aspect_adjusted_note"] = (
                f"Requested aspect not supported by GBS; mapped to {aspect!r}."
            )

        return ToolResult(
            success=True,
            data=data,
            artifacts=[str(output_path.resolve())],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="gemini-browser",
        )
