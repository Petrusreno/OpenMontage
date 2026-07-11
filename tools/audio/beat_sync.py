"""Beat Sync — detect beats in a track (pure numpy) and snap cut points to them.

Builds a spectral-flux onset envelope with numpy FFT, peak-picks beat times, estimates BPM from the
median inter-beat interval (robust to the autocorrelation octave error), and snaps supplied cut
points to the nearest beat. No librosa/scipy. Output: a beat-grid report + snapped cuts.
"""

from __future__ import annotations

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


class BeatSync(BaseTool):
    name = "beat_sync"
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

    capabilities = ["beat_sync", "beat_detection", "onset_detection"]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "cut_seconds": {"type": "array", "items": {"type": "number", "minimum": 0}},
            "output_path": {"type": "string"},
            "sample_rate": {"type": "integer", "default": 22050, "minimum": 8000},
            "sensitivity": {"type": "number", "default": 1.0, "minimum": 0.0},
            "min_gap_seconds": {"type": "number", "default": 0.15, "minimum": 0.02},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")

    def _onset_envelope(self, x: "np.ndarray", sample_rate: int, win: int = 1024,
                        hop: int = 512) -> "tuple[np.ndarray, float]":
        x = np.asarray(x, dtype=np.float64)
        if x.size < win:
            return np.zeros(0), float(sample_rate) / hop
        nf = 1 + (x.size - win) // hop
        w = np.hanning(win)
        flux = np.zeros(nf)
        prev = np.zeros(win // 2 + 1)
        for i in range(nf):
            mag = np.abs(np.fft.rfft(x[i * hop:i * hop + win] * w))
            diff = mag - prev
            diff[diff < 0] = 0.0
            flux[i] = diff.sum()
            prev = mag
        std = flux.std()
        flux = (flux - flux.mean()) / (std + 1e-9)
        return flux, float(sample_rate) / hop

    def _pick_beats(self, envelope, fps: float, k: float = 1.0, min_gap_s: float = 0.15) -> list[float]:
        env = np.asarray(envelope, dtype=np.float64)
        if env.size < 3:
            return []
        thr = env.mean() + k * env.std()
        min_gap = max(1, int(min_gap_s * fps))
        peaks: list[int] = []
        i = 1
        while i < env.size - 1:
            if (env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]
                    and (not peaks or i - peaks[-1] >= min_gap)):
                peaks.append(i)
                i += min_gap
            else:
                i += 1
        return [round(p / fps, 3) for p in peaks]

    def _estimate_bpm(self, beats: list[float]) -> float:
        if len(beats) < 2:
            return 0.0
        ibi = np.diff(np.asarray(beats, dtype=np.float64))
        med = float(np.median(ibi))
        return round(60.0 / med, 1) if med > 0 else 0.0

    def _snap(self, cut_seconds: list[float], beats: list[float]) -> list[dict[str, float]]:
        if not cut_seconds or not beats:
            return []
        b = np.asarray(beats, dtype=np.float64)
        out = []
        for c in cut_seconds:
            c = float(c)
            nearest = float(b[int(np.argmin(np.abs(b - c)))])
            out.append({"original": round(c, 3), "snapped": round(nearest, 3),
                        "offset": round(nearest - c, 3)})
        return out
