"""Voice Isolation — strongly clean up speech (Adobe Enhance Speech / DaVinci Voice Isolation).

Runs the RNNoise ML denoiser (ffmpeg `arnndn`) when a .rnnn model resolves, else a stronger
spectral chain than audio_enhance (afftdn + anlmdn + deesser). Reports which engine actually ran
and the real noise-floor reduction — never claims ML quality that did not execute. A `mix` control
blends the isolated voice with the original. For a video input, the cleaned audio is muxed back.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolResult,
    ToolStability,
    ToolTier,
)


class VoiceIsolation(BaseTool):
    name = "voice_isolation"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "audio_processing"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:numpy"]
    install_instructions = "Install FFmpeg (brew install ffmpeg) and numpy (pip install numpy)."
    agent_skills = ["ffmpeg"]

    capabilities = ["voice_isolation", "speech_enhance", "denoise"]

    # Bundled public-domain RNNoise model (see assets/rnnoise/CREDITS.md).
    BUNDLED_MODEL = str(Path(__file__).resolve().parents[2] / "assets" / "rnnoise" / "somnolent-hogwash.rnnn")

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "engine": {"type": "string", "enum": ["auto", "rnnoise", "spectral"], "default": "auto"},
            "model_path": {"type": "string"},
            "mix": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
            "noise_floor_db": {"type": "number", "default": -25},
            "codec": {"type": "string", "default": "aac"},
            "bitrate": {"type": "string", "default": "192k"},
        },
    }

    def _resolve_model(self, model_path: str | None) -> str | None:
        if model_path:
            return model_path if Path(model_path).is_file() else None
        env = os.environ.get("RNNOISE_MODEL")
        if env and Path(env).is_file():
            return env
        return self.BUNDLED_MODEL if Path(self.BUNDLED_MODEL).is_file() else None

    def _arnndn_available(self) -> bool:
        cached = getattr(self, "_arnndn_cache", None)
        if cached is None:
            try:
                proc = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                      capture_output=True, text=True, check=False, timeout=30)
                cached = "arnndn" in (proc.stdout + proc.stderr)
            except (subprocess.TimeoutExpired, OSError):
                cached = False
            self._arnndn_cache = cached
        return cached

    @staticmethod
    def _escape_filter_path(p: str) -> str:
        # Escape ffmpeg filtergraph metacharacters so a model path with ':' (Windows
        # drives), ',', '[', ']', "'", ';' or '\' can't break or inject into the graph.
        # Backslash first so the escapes we add aren't themselves re-escaped.
        for ch in ("\\", ":", "'", ",", "[", "]", ";"):
            p = p.replace(ch, "\\" + ch)
        return p

    def _base_chain(self, engine: str, model: str | None, nf: float) -> str:
        if engine == "rnnoise":
            safe = self._escape_filter_path(str(model))
            return f"arnndn=model={safe},highpass=f=80,loudnorm=I=-16:LRA=11:TP=-1.5"
        return (f"afftdn=nf={nf}:nt=w,anlmdn,highpass=f=80,deesser,"
                f"loudnorm=I=-16:LRA=11:TP=-1.5")

    def _build_filter(self, engine: str, model: str | None, nf: float, mix: float) -> str:
        chain = self._base_chain(engine, model, nf)
        if mix >= 1.0:
            return chain
        return (f"asplit=2[a][b];[a]{chain}[w];"
                f"[w]volume={mix}[wv];[b]volume={1.0 - mix}[dv];"
                f"[wv][dv]amix=inputs=2:normalize=0")

    def _ffprobe_has(self, path: str, stream: str) -> bool:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream,
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=False, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

    def _has_audio(self, path: str) -> bool:
        return self._ffprobe_has(path, "a")

    def _has_video(self, path: str) -> bool:
        return self._ffprobe_has(path, "v")

    def _process(self, input_path: str, af: str, codec: str, bitrate: str,
                 dest: str | Path) -> str | None:
        dest = Path(dest)
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(input_path),
                            "-af", af, "-c:a", codec, "-b:a", bitrate, str(dest)],
                           capture_output=True, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _mux_audio(self, video_in: str, new_audio: str, codec: str, bitrate: str,
                   dest: str | Path) -> str | None:
        dest = Path(dest)
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_in), "-i", str(new_audio),
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                            "-c:a", codec, "-b:a", bitrate, "-shortest", str(dest)],
                           capture_output=True, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _noise_floor_dbfs(self, path: str, window_s: float = 0.2) -> float:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", str(path),
                 "-f", "s16le", "-ac", "1", "-ar", "48000", "pipe:1"],
                capture_output=True, check=True, timeout=120)   # BYTES, not text
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return 0.0
        raw = proc.stdout
        raw = raw[: len(raw) - (len(raw) % 2)]
        if not raw:
            return 0.0
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        n = int(window_s * 48000)
        if n <= 0 or x.size < n:
            return 0.0
        mins = []
        for i in range(0, x.size - n, n):
            rms = float(np.sqrt(np.mean(x[i:i + n] ** 2))) + 1e-12
            mins.append(20.0 * np.log10(rms))
        return round(min(mins), 2) if mins else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
