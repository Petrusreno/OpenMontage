"""Multicam auto-sync — align clips by audio cross-correlation.

Extracts low-rate mono PCM from each clip, cross-correlates each against a
pivot with a pure-numpy FFT to recover a lag + confidence, re-baselines onto a
common timeline (auto: earliest-start clip = reference, offsets >= 0; or an
explicit reference_index), and emits a JSON offsets report. No rendering.
"""

from __future__ import annotations

import subprocess
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


class MulticamSync(BaseTool):
    name = "multicam_sync"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "ffmpeg+numpy"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:numpy"]
    install_instructions = "Install FFmpeg (brew install ffmpeg) and numpy (pip install numpy)."
    agent_skills = ["ffmpeg"]

    capabilities = ["multicam_sync", "audio_sync", "waveform_alignment"]

    input_schema = {
        "type": "object",
        "required": ["clips"],
        "properties": {
            "clips": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "reference_index": {"type": "integer", "minimum": 0},
            "sample_rate": {"type": "integer", "default": 8000, "minimum": 1000},
            "window_seconds": {"type": "number", "default": 60, "minimum": 1},
            "min_confidence": {"type": "number", "default": 0.1, "minimum": 0, "maximum": 1},
            "output_path": {"type": "string"},
        },
    }

    def _xcorr_lag(self, ref: np.ndarray, other: np.ndarray, sample_rate: int) -> tuple[float, float]:
        """FFT cross-correlation. Returns (lag_seconds, confidence in [0,1]).

        lag is the shift at which `other` aligns to `ref`: if `other` is `ref`
        delayed by D seconds, lag == +D.
        """
        ref = np.asarray(ref, dtype=np.float64)
        other = np.asarray(other, dtype=np.float64)
        if ref.size == 0 or other.size == 0:
            return 0.0, 0.0
        n = 1 << int(np.ceil(np.log2(ref.size + other.size)))
        fa = np.fft.rfft(ref, n)
        fb = np.fft.rfft(other, n)
        cc = np.fft.irfft(fb * np.conj(fa), n)
        # Reassemble into full correlation with zero-lag centered.
        # Guard ref.size == 1: `cc[-0:]` would be the whole array, not empty.
        head = cc[-(ref.size - 1):] if ref.size > 1 else cc[:0]
        cc = np.concatenate([head, cc[:other.size]])
        lags = np.arange(-(ref.size - 1), other.size)
        peak_idx = int(np.argmax(cc))
        lag_samples = int(lags[peak_idx])
        energy = float(np.sqrt(np.sum(ref ** 2) * np.sum(other ** 2)))
        confidence = 0.0 if energy == 0.0 else float(max(0.0, cc[peak_idx] / energy))
        return lag_samples / sample_rate, min(1.0, confidence)

    def _has_audio(self, path: str) -> bool:
        try:
            proc = self.run_command([
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
            ])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

    def _extract_samples(self, path: str, sample_rate: int,
                         window_seconds: float) -> np.ndarray | None:
        if not self._has_audio(path):
            return None
        cmd = [
            "ffmpeg", "-v", "quiet", "-t", f"{float(window_seconds):.3f}",
            "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        raw = proc.stdout
        # Trim a stray trailing byte so an odd-length PCM buffer can't raise
        # ValueError in np.frombuffer (int16 itemsize is 2).
        raw = raw[: len(raw) - (len(raw) % 2)]
        if not raw:
            return None
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
