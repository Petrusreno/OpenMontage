"""Multicam auto-sync — align clips by audio cross-correlation.

Extracts low-rate mono PCM from each clip, cross-correlates each against a
pivot with a pure-numpy FFT to recover a lag + confidence, re-baselines onto a
common timeline (auto: earliest-start clip = reference, offsets >= 0; or an
explicit reference_index), and emits a JSON offsets report. No rendering.
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

    def _pairwise_offsets(self, samples_list: list, sample_rate: int) -> list[tuple[float, float]]:
        pivot = samples_list[0]
        pivot_conf = 0.0 if float(np.sum(np.asarray(pivot, dtype=np.float64) ** 2)) == 0.0 else 1.0
        out = [(0.0, pivot_conf)]
        for s in samples_list[1:]:
            out.append(self._xcorr_lag(pivot, s, sample_rate))
        return out

    def _rebaseline(self, pairwise: list[tuple[float, float]],
                    reference_index: int | None = None) -> tuple[int, list[float]]:
        starts = [-lag for (lag, _conf) in pairwise]
        if reference_index is None:
            min_start = min(starts)
            ref = starts.index(min_start)
            offsets = [s - min_start for s in starts]
        else:
            ref = reference_index
            base = starts[ref]
            offsets = [s - base for s in starts]
        return ref, offsets

    def _to_report(self, clips: list[str], reference_index: int, offsets: list[float],
                   confidences: list[float], skipped: list[dict], params: dict) -> dict:
        """Assemble the offsets report. `offsets`/`confidences` are FULL-LENGTH
        arrays indexed by each clip's ORIGINAL position in `clips`, and
        `reference_index` is an ORIGINAL clip index; skipped clips carry
        placeholder slots and are emitted as null (excluded from aggregates)."""
        min_conf = float(params.get("min_confidence", 0.1))
        entries = []
        for i, src in enumerate(clips):
            if any(sk["index"] == i for sk in skipped):
                entries.append({"index": i, "source": src, "offset_seconds": None,
                                "confidence": None, "low_confidence": True})
            else:
                conf = round(float(confidences[i]), 4)
                entries.append({"index": i, "source": src,
                                "offset_seconds": round(float(offsets[i]), 4),
                                "confidence": conf, "low_confidence": conf < min_conf})
        usable = [e["confidence"] for e in entries if e["confidence"] is not None]
        return {
            "version": "1.0",
            "reference_index": reference_index,
            "reference_source": clips[reference_index],
            "sample_rate": int(params.get("sample_rate", 8000)),
            "window_seconds": float(params.get("window_seconds", 60)),
            "offsets": entries,
            "skipped": skipped,
            "max_confidence": round(max(usable), 4) if usable else 0.0,
            "min_confidence_observed": round(min(usable), 4) if usable else 0.0,
        }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        clips = inputs.get("clips") or []
        if len(clips) < 2:
            return ToolResult(success=False, error="multicam_sync needs at least 2 clips.")

        sample_rate = int(inputs.get("sample_rate", 8000))
        window_seconds = float(inputs.get("window_seconds", 60))
        params = {
            "sample_rate": sample_rate,
            "window_seconds": window_seconds,
            "min_confidence": float(inputs.get("min_confidence", 0.1)),
        }

        # Extract samples; record skips honestly.
        samples_by_index: dict[int, Any] = {}
        skipped: list[dict] = []
        for i, clip in enumerate(clips):
            s = self._extract_samples(clip, sample_rate, window_seconds)
            if s is None or s.size == 0:
                skipped.append({"index": i, "source": clip, "reason": "no audio or extraction failed"})
            else:
                samples_by_index[i] = s

        usable = sorted(samples_by_index.keys())
        if len(usable) < 2:
            return ToolResult(success=False,
                              error="Fewer than 2 clips have usable audio; cannot synchronize.")

        # Optional explicit reference must be a usable clip.
        req_ref = inputs.get("reference_index")
        if req_ref is not None and req_ref not in usable:
            return ToolResult(success=False,
                              error=f"reference_index {req_ref} is not a usable (audio-bearing) clip.")

        usable_samples = [samples_by_index[i] for i in usable]
        pairwise = self._pairwise_offsets(usable_samples, sample_rate)
        local_ref = None if req_ref is None else usable.index(req_ref)
        ref_local, offsets_local = self._rebaseline(pairwise, local_ref)

        # Map usable-local results back to original clip indices.
        offsets_full = [0.0] * len(clips)
        conf_full = [0.0] * len(clips)
        for local_i, orig_i in enumerate(usable):
            offsets_full[orig_i] = offsets_local[local_i]
            conf_full[orig_i] = pairwise[local_i][1]
        reference_index = usable[ref_local]

        report = self._to_report(clips, reference_index, offsets_full, conf_full, skipped, params)

        out_path = Path(inputs.get("output_path") or
                        Path(clips[reference_index]).with_suffix(".sync.json"))
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        usable_confs = [conf_full[i] for i in usable]
        return ToolResult(
            success=True,
            artifacts=[str(out_path)],
            duration_seconds=time.time() - start,
            data={
                "reference_index": reference_index,
                "reference_source": clips[reference_index],
                "sample_rate": sample_rate,
                "window_seconds": window_seconds,
                "offsets": report["offsets"],
                "skipped": skipped,
                "max_confidence": round(max(usable_confs), 4),
                "min_confidence_observed": round(min(usable_confs), 4),
            },
        )
