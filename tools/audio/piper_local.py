"""Piper TTS adapter via avatar-studio local TTS server (:8004).

POST http://127.0.0.1:8004/tts as multipart/form → WAV stream.
Token read from /tmp/avatar_token (if present) else AVATAR_TOKEN env.
"""

from __future__ import annotations

import os
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

_TTS_BASE = "http://127.0.0.1:8004"


def _read_avatar_token() -> str:
    token_file = Path("/tmp/avatar_token")
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except Exception:
            pass
    return os.environ.get("AVATAR_TOKEN", "")


class PiperLocal(BaseTool):
    name = "piper_local"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "piper-local"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    install_instructions = (
        "Start the avatar-studio TTS server (Piper engine):\n"
        "  cd ~/ClaudeCode/avatar-studio && ./start.sh\n"
        "Token: written to /tmp/avatar_token by start.sh, or set AVATAR_TOKEN env."
    )
    agent_skills = ["text-to-speech"]

    capabilities = ["text_to_speech"]
    supports = {
        "offline": True,
        "multilingual": False,
        "pt_br": True,
        "streaming": False,
    }
    best_for = [
        "free local PT-BR TTS (Piper pt_BR-faber-medium)",
        "zero cloud cost, on-device synthesis",
        "narration in Portuguese",
    ]
    not_good_for = [
        "multilingual content outside PT-BR / ES",
        "hero/presenter voice (use elevenlabs_petrus for that)",
        "when avatar-studio TTS server is not running",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice_id": {
                "type": "string",
                "default": "pt_BR-faber-medium",
                "description": "Piper voice model ID.",
            },
            "language": {
                "type": "string",
                "default": "pt",
                "description": "Language code passed to the TTS engine.",
            },
            "output_path": {"type": "string", "description": "Destination WAV path. Auto-generated if omitted."},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=50, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, backoff_seconds=2.0, retryable_errors=["timeout"])
    idempotency_key_fields = ["text", "voice_id", "language"]
    side_effects = ["writes WAV to output_path"]
    user_visible_verification = ["Listen to generated audio for naturalness and correctness"]

    def get_status(self) -> ToolStatus:
        try:
            import requests
            token = _read_avatar_token()
            headers = {"X-Avatar-Token": token} if token else {}
            resp = requests.get(f"{_TTS_BASE}/health", headers=headers, timeout=3)
            if resp.ok:
                return ToolStatus.AVAILABLE
        except Exception:
            pass
        return ToolStatus.UNAVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        start = time.time()
        token = _read_avatar_token()
        headers: dict[str, str] = {}
        if token:
            headers["X-Avatar-Token"] = token

        # Health gate
        try:
            health_resp = requests.get(f"{_TTS_BASE}/health", headers=headers, timeout=5)
            if not health_resp.ok:
                raise RuntimeError(f"HTTP {health_resp.status_code}")
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Piper TTS server unavailable — start avatar-studio TTS: {exc}",
            )

        text = inputs["text"]
        voice_id = inputs.get("voice_id", "pt_BR-faber-medium")
        language = inputs.get("language", "pt")

        if inputs.get("output_path"):
            output_path = Path(inputs["output_path"])
        else:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="piper_")
            os.close(fd)
            output_path = Path(tmp)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = requests.post(
                f"{_TTS_BASE}/tts",
                data={
                    "text": text,
                    "engine": "piper",
                    "voice_id": voice_id,
                    "language": language,
                },
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as exc:
            return ToolResult(success=False, error=f"Piper TTS request failed: {exc}")

        output_path.write_bytes(resp.content)

        if not output_path.exists() or output_path.stat().st_size == 0:
            return ToolResult(success=False, error="Piper TTS returned empty audio")

        return ToolResult(
            success=True,
            data={
                "provider": "piper-local",
                "voice_id": voice_id,
                "language": language,
                "output": str(output_path),
                "format": "wav",
            },
            artifacts=[str(output_path.resolve())],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model=f"piper/{voice_id}",
        )
