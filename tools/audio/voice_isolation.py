"""Voice Isolation — strongly clean up speech (Adobe Enhance Speech / DaVinci Voice Isolation).

Runs the RNNoise ML denoiser (ffmpeg `arnndn`) when a .rnnn model resolves, else a stronger
spectral chain than audio_enhance (afftdn + anlmdn + deesser). Reports which engine actually ran
and the real noise-floor reduction — never claims ML quality that did not execute. A `mix` control
blends the isolated voice with the original. For a video input, the cleaned audio is muxed back.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
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

    @staticmethod
    def _audio_ext(codec: str) -> str:
        # Container extension matching the audio codec, so the output name isn't misleading.
        c = (codec or "").lower()
        if c.startswith("pcm"):
            return "wav"
        if c in ("libmp3lame", "mp3"):
            return "mp3"
        if c == "flac":
            return "flac"
        if c in ("libopus", "opus", "libvorbis", "vorbis"):
            return "ogg"
        return "m4a"          # aac and anything else -> m4a container

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
        for i in range(0, x.size - n + 1, n):   # +1 so the final full window is included
            rms = float(np.sqrt(np.mean(x[i:i + n] ** 2))) + 1e-12
            mins.append(20.0 * np.log10(rms))
        return round(min(mins), 2) if mins else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        input_path = inputs.get("input_path")
        if not input_path:
            return ToolResult(success=False, error="input_path is required.")
        try:
            mix = float(inputs.get("mix", 1.0))
            nf = float(inputs.get("noise_floor_db", -25))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="mix/noise_floor_db must be numeric.")
        if not 0.0 <= mix <= 1.0:
            return ToolResult(success=False, error="mix must be between 0.0 and 1.0.")
        engine_req = str(inputs.get("engine", "auto"))
        codec = str(inputs.get("codec", "aac"))
        bitrate = str(inputs.get("bitrate", "192k"))

        # Validate engine/model config before touching the filesystem so a forced-rnnoise
        # request with no resolvable model fails honestly ("model") rather than being masked
        # by a downstream missing-input error.
        model = self._resolve_model(inputs.get("model_path"))
        if engine_req == "rnnoise":
            if model is None or not self._arnndn_available():
                return ToolResult(success=False,
                                  error="engine='rnnoise' needs a resolvable .rnnn model and ffmpeg "
                                        "arnndn support; supply model_path or use engine='auto'.")
            engine = "rnnoise"
        elif engine_req == "spectral":
            engine, model = "spectral", None
        else:  # auto
            if model is not None and self._arnndn_available():
                engine = "rnnoise"
            else:
                engine, model = "spectral", None

        input_path = Path(input_path)
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not self._has_audio(str(input_path)):
            return ToolResult(success=False, error="Input has no audio stream to isolate.")

        had_video = self._has_video(str(input_path))
        audio_ext = self._audio_ext(codec)
        default_ext = "mp4" if had_video else audio_ext
        out_path = Path(inputs.get("output_path") or
                        input_path.with_name(f"{input_path.stem}_voice.{default_ext}"))

        af = self._build_filter(engine, model, nf, mix)
        workdir = Path(tempfile.mkdtemp(prefix="voiceiso_"))
        try:
            floor_before = self._noise_floor_dbfs(str(input_path))

            # Intermediate container extension must match the codec so ffmpeg muxes it.
            audio_out = self._process(str(input_path), af, codec, bitrate,
                                      workdir / f"clean.{'m4a' if had_video else audio_ext}")
            if audio_out is None:
                return ToolResult(success=False, error="Voice-isolation filter chain failed.")

            if had_video:
                final = self._mux_audio(str(input_path), audio_out, codec, bitrate, out_path)
                if final is None:
                    return ToolResult(success=False, error="Audio mux back into video failed.")
            else:
                try:
                    shutil.copyfile(audio_out, out_path)
                except OSError as exc:
                    return ToolResult(success=False, error=f"Failed to write output: {exc}")
            if not out_path.is_file() or out_path.stat().st_size == 0:
                return ToolResult(success=False, error="Output not created or empty.")

            floor_after = self._noise_floor_dbfs(str(out_path))
            reduction = max(0.0, floor_before - floor_after)
            return ToolResult(
                success=True,
                artifacts=[str(out_path)],
                duration_seconds=time.time() - start,
                data={
                    "engine": engine,
                    "model": model,
                    "mix": mix,
                    "noise_floor_before_db": round(floor_before, 2),
                    "noise_floor_after_db": round(floor_after, 2),
                    "noise_reduction_db": round(reduction, 2),
                    "had_video": had_video,
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
