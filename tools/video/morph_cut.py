"""Morph Cut — smooth hard jump cuts with ffmpeg optical-flow interpolation.

For each specified cut, a short window around the join is re-timed with
`minterpolate` (motion-compensated interpolation) to bridge the jump, then
spliced back. Where motion interpolation cannot bridge a large content change
it degrades to the original hard cut (no warp artifact) and reports it honestly.
Completes the cut-editing cluster: silence_cutter / text_based_editor create
jump cuts; morph_cut smooths them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolResult,
    ToolStability,
    ToolTier,
)


class MorphCut(BaseTool):
    name = "morph_cut"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:numpy", "python:PIL"]
    install_instructions = "Install FFmpeg (brew install ffmpeg) and pip install numpy pillow."
    agent_skills = ["ffmpeg"]

    capabilities = ["morph_cut", "smooth_cut", "optical_flow_transition"]

    MIN_WINDOW = 0.04

    input_schema = {
        "type": "object",
        "required": ["input_path", "cut_seconds"],
        "properties": {
            "input_path": {"type": "string"},
            "cut_seconds": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 1},
            "output_path": {"type": "string"},
            "transition_duration": {"type": "number", "default": 0.2, "minimum": 0.04},
            "morph_fps": {"type": "integer", "default": 60, "minimum": 30},
            "codec": {"type": "string", "default": "libx264"},
            "crf": {"type": "integer", "default": 18},
        },
    }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")

    def _plan_windows(self, cut_seconds: list[float], duration: float,
                      transition_duration: float) -> tuple[list[dict], list[dict]]:
        # Invariant: `cursor` and `prev_morph_end` are updated together ONLY on
        # acceptance (never on a skip), so they stay equal — this is what guarantees
        # the emitted segments tile [0, duration] contiguously with no gaps/overlaps.
        half = transition_duration / 2.0
        segments: list[dict] = []
        skipped: list[dict] = []
        # Dedup while recording each dropped duplicate (never silently discarded).
        seen: set[float] = set()
        cuts: list[float] = []
        for c in cut_seconds:
            r = round(float(c), 6)
            if r in seen:
                skipped.append({"time": r, "reason": "duplicate cut time"})
                continue
            seen.add(r)
            cuts.append(r)
        cuts.sort()
        cursor = 0.0            # end of the last emitted segment
        prev_morph_end = 0.0    # end of the last morph window (for overlap detection)
        for cut in cuts:
            start = max(0.0, cut - half)
            end = min(duration, cut + half)
            if end - start < self.MIN_WINDOW:
                skipped.append({"time": cut, "reason": "window too small at clip boundary"})
                continue
            if start < prev_morph_end:
                skipped.append({"time": cut, "reason": "window overlaps a previous morph"})
                continue
            if start > cursor:
                segments.append({"kind": "pass", "start": round(cursor, 6),
                                 "end": round(start, 6), "cut": None})
            segments.append({"kind": "morph", "start": round(start, 6),
                             "end": round(end, 6), "cut": cut})
            cursor = end
            prev_morph_end = end
        if cursor < duration:
            segments.append({"kind": "pass", "start": round(cursor, 6),
                             "end": round(duration, 6), "cut": None})
        return segments, skipped

    def _run(self, cmd: list[str], timeout: int = 300):
        return subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)

    def _duration(self, path: str) -> float:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=30)
            return float(json.loads(proc.stdout)["format"]["duration"])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError, KeyError):
            return 0.0

    def _probe_fps(self, path: str) -> float:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", "-of", "default=nk=1:nw=1", str(path)],
                capture_output=True, text=True, check=True, timeout=30)
            num, den = proc.stdout.strip().split("/")
            fps = float(num) / float(den)
            return fps if fps > 0 else 30.0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError, ZeroDivisionError):
            return 30.0

    def _has_audio(self, path: str) -> bool:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=False, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return bool(proc.stdout.strip())

    def _extract_segment(self, input_path: str, start: float, end: float, out_fps: float,
                         codec: str, crf: int, dest: str | Path) -> str | None:
        dest = Path(dest)
        try:
            self._run(["ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.3f}",
                       "-to", f"{float(end):.3f}", "-i", str(input_path), "-r", str(out_fps),
                       "-c:v", codec, "-crf", str(crf), "-an", str(dest)])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _morph_window(self, input_path: str, start: float, end: float, morph_fps: int,
                      out_fps: float, codec: str, crf: int, dest: str | Path) -> str | None:
        dest = Path(dest)
        vf = (f"minterpolate=fps={morph_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
              f"fps={out_fps}")
        try:
            self._run(["ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.3f}",
                       "-to", f"{float(end):.3f}", "-i", str(input_path), "-vf", vf,
                       "-r", str(out_fps), "-c:v", codec, "-crf", str(crf), "-an", str(dest)])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _concat(self, parts: list[str], dest: str | Path) -> str | None:
        dest = Path(dest)
        list_path = dest.parent / f"{dest.stem}_concat.txt"
        # Escape single quotes for the concat-demuxer list format ('\'' inside a quoted path),
        # so a path containing an apostrophe doesn't silently break the join.
        def _q(p: str) -> str:
            return str(Path(p).resolve()).replace("'", "'\\''")
        list_path.write_text("".join(f"file '{_q(p)}'\n" for p in parts))
        try:
            self._run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                       "-i", str(list_path), "-c", "copy", str(dest)])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def _junction_max_mad(self, video_path: str, at_seconds: float, radius: float) -> float:
        workdir = Path(tempfile.mkdtemp(prefix="mad_"))
        try:
            start = max(0.0, float(at_seconds) - float(radius))
            dur = float(radius) * 2.0
            try:
                self._run(["ffmpeg", "-y", "-v", "quiet", "-ss", f"{start:.3f}",
                           "-i", str(video_path), "-t", f"{dur:.3f}",
                           str(workdir / "f_%04d.png")])
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                return 0.0
            frames = sorted(workdir.glob("f_*.png"))
            if len(frames) < 2:
                return 0.0
            arrs = [np.asarray(Image.open(f).convert("RGB")).astype(np.float64) for f in frames]
            return float(max(np.abs(arrs[i + 1] - arrs[i]).mean() for i in range(len(arrs) - 1)))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
