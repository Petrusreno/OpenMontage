"""ElevenLabs Petrus voice TTS adapter.

Delegates to ~/Developer/video-use/helpers/elevenlabs_tts.py via subprocess.
Gets the full cache + Petrus voice + Voicebox fallback stack from that helper.
No API key handling here — the helper reads ELEVENLABS_API_KEY from its own env.
"""

from __future__ import annotations

import os
import subprocess
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

_HELPER = "~/Developer/video-use/helpers/elevenlabs_tts.py"
_DEFAULT_VOICE = "mUH8M4GPB2nbxbxkhEvW"  # Petrus professional voice
_DEFAULT_MODEL = "multilingual"


class ElevenLabsPetrus(BaseTool):
    name = "elevenlabs_petrus"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "elevenlabs-petrus"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    install_instructions = (
        "Ensure ~/Developer/video-use/helpers/elevenlabs_tts.py exists.\n"
        "Set ELEVENLABS_API_KEY in your environment or .env file.\n"
        "See: https://elevenlabs.io/app/settings/api-keys"
    )
    agent_skills = ["text-to-speech", "elevenlabs"]

    capabilities = ["text_to_speech"]
    supports = {
        "multilingual": True,
        "pt_br": True,
        "voice_cloning": True,
        "cache": True,
        "voicebox_fallback": True,
    }
    best_for = [
        "hero / presenter narration (Petrus professional voice)",
        "high-quality PT-BR TTS with caching",
        "ElevenLabs multilingual v2 model",
    ]
    not_good_for = [
        "free / zero-cost pipelines (uses ElevenLabs API credits)",
        "when ELEVENLABS_API_KEY is not configured",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice_id": {
                "type": "string",
                "default": _DEFAULT_VOICE,
                "description": "ElevenLabs voice ID. Defaults to Petrus professional voice.",
            },
            "model": {
                "type": "string",
                "default": _DEFAULT_MODEL,
                "description": "Model name passed to the helper (e.g. 'multilingual', 'turbo').",
            },
            "output_path": {"type": "string", "description": "Destination WAV path. Auto-generated if omitted."},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=2, backoff_seconds=5.0, retryable_errors=["rate_limit", "timeout"]
    )
    idempotency_key_fields = ["text", "voice_id", "model"]
    side_effects = ["writes WAV to output_path", "consumes ElevenLabs API credits"]
    user_visible_verification = ["Listen to generated audio for naturalness and Petrus voice fidelity"]

    def _helper_path(self) -> Path:
        return Path(_HELPER).expanduser()

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        """Estimate ElevenLabs spend at ~$0.00018/char (multilingual v2).

        Real billing is on the ElevenLabs account; this lets budget caps see the spend.
        """
        return len(inputs.get("text", "")) * 0.00018

    def get_status(self) -> ToolStatus:
        if not self._helper_path().exists():
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()

        helper = self._helper_path()
        if not helper.exists():
            return ToolResult(
                success=False,
                error=f"ElevenLabs helper not found at {helper}. {self.install_instructions}",
            )

        text = inputs["text"]
        voice_id = inputs.get("voice_id", _DEFAULT_VOICE)
        model = inputs.get("model", _DEFAULT_MODEL)

        if inputs.get("output_path"):
            output_path = Path(inputs["output_path"])
        else:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="el_petrus_")
            os.close(fd)
            output_path = Path(tmp)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(helper),
            "--voice", voice_id,
            "--text", text,
            "--out", str(output_path),
            "--model", model,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="ElevenLabs TTS helper timed out after 300s")
        except Exception as exc:
            return ToolResult(success=False, error=f"ElevenLabs TTS subprocess failed: {exc}")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "(no output)"
            return ToolResult(
                success=False,
                error=f"ElevenLabs TTS helper exited {proc.returncode}: {stderr}",
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            return ToolResult(
                success=False,
                error=f"ElevenLabs TTS helper produced no output at {output_path}",
            )

        return ToolResult(
            success=True,
            data={
                "provider": "elevenlabs-petrus",
                "voice_id": voice_id,
                "model": model,
                "output": str(output_path),
                "format": "wav",
            },
            artifacts=[str(output_path.resolve())],
            cost_usd=0.0,  # billed via ElevenLabs account; not tracked here
            duration_seconds=round(time.time() - start, 2),
            model=f"elevenlabs/{model}",
        )
