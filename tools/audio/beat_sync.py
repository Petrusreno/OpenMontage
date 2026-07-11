"""Beat Sync — detect beats in a track (pure numpy) and snap cut points to them.

Builds a spectral-flux onset envelope with numpy FFT, peak-picks beat times, estimates BPM from the
median inter-beat interval (robust to the autocorrelation octave error), and snaps supplied cut
points to the nearest beat. No librosa/scipy. Output: a beat-grid report + snapped cuts.
"""

from __future__ import annotations

import json
import subprocess
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
        start = time.time()
        input_path = inputs.get("input_path")
        if not input_path:
            return ToolResult(success=False, error="input_path is required.")
        try:
            sample_rate = int(inputs.get("sample_rate", 22050))
            sensitivity = float(inputs.get("sensitivity", 1.0))
            min_gap = float(inputs.get("min_gap_seconds", 0.15))
            cuts = [float(c) for c in (inputs.get("cut_seconds") or [])]
        except (TypeError, ValueError):
            return ToolResult(success=False,
                              error="sample_rate/sensitivity/min_gap_seconds/cut_seconds must be numeric.")
        if sample_rate < 8000 or sensitivity < 0 or min_gap < 0.02:
            return ToolResult(success=False,
                              error="sample_rate >= 8000, sensitivity >= 0, min_gap_seconds >= 0.02.")
        input_path = Path(input_path)
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        samples = self._extract_samples(str(input_path), sample_rate)
        if samples is None or samples.size == 0:
            return ToolResult(success=False, error="No audio to analyze in input.")

        envelope, fps = self._onset_envelope(samples, sample_rate)
        beats = self._pick_beats(envelope, fps, k=sensitivity, min_gap_s=min_gap)
        if len(beats) < 2:
            return ToolResult(success=False,
                              error="Fewer than 2 beats detected; audio too short/silent to sync "
                                    "(try a lower sensitivity).")
        bpm = self._estimate_bpm(beats)
        snapped = self._snap(cuts, beats)

        report = {
            "version": "1.0",
            "bpm": bpm,
            "beat_count": len(beats),
            "beats": beats,
            "snapped_cuts": snapped,
            "sample_rate": sample_rate,
            "sensitivity": sensitivity,
            "min_gap_seconds": min_gap,
        }
        out_path = Path(inputs.get("output_path") or
                        input_path.with_name(f"{input_path.stem}_beats.json"))
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        return ToolResult(
            success=True,
            artifacts=[str(out_path)],
            duration_seconds=time.time() - start,
            data={
                "bpm": bpm,
                "beat_count": len(beats),
                "beats": beats,
                "snapped_cuts": snapped,
                "sample_rate": sample_rate,
                "sensitivity": sensitivity,
                "min_gap_seconds": min_gap,
            },
        )

    def _has_audio(self, path: str) -> bool:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=False, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

    def _extract_samples(self, path: str, sample_rate: int) -> "np.ndarray | None":
        if not self._has_audio(path):
            return None
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", str(path),
                 "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
                capture_output=True, check=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        raw = proc.stdout
        raw = raw[: len(raw) - (len(raw) % 2)]
        if not raw:
            return None
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

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
