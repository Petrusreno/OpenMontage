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
import time
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

    SMOOTH_RATIO = 0.9

    def _mux_audio(self, video_only, original, dest) -> str | None:
        dest = Path(dest)
        try:
            self._run(["ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(original),
                       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                       "-shortest", str(dest)])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        input_path = inputs.get("input_path")
        cut_seconds = inputs.get("cut_seconds") or []
        if not input_path or not cut_seconds:
            return ToolResult(success=False, error="input_path and a non-empty cut_seconds are required.")
        try:
            transition = float(inputs.get("transition_duration", 0.2))
            morph_fps = int(inputs.get("morph_fps", 60))
            crf = int(inputs.get("crf", 18))
            cuts = [float(c) for c in cut_seconds]
        except (TypeError, ValueError):
            return ToolResult(success=False, error="cut_seconds/transition_duration/morph_fps/crf must be numeric.")
        if transition < 0.04 or morph_fps < 30:
            return ToolResult(success=False, error="transition_duration must be >= 0.04 and morph_fps >= 30.")
        codec = str(inputs.get("codec", "libx264"))
        input_path = Path(input_path)
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        duration = self._duration(str(input_path))
        if duration <= 0:
            return ToolResult(success=False, error="Could not read input duration.")
        # Uniform output fps for every segment so the concat is clean. It must be
        # >= morph_fps, otherwise the morph window's trailing `fps=out_fps` resample
        # discards the interpolated frames and the junction is never smoothed (the
        # motion-compensated frames only survive at the higher rate). Passthrough
        # segments are resampled up to match; duration is preserved throughout.
        out_fps = max(self._probe_fps(str(input_path)), float(morph_fps))
        out_path = Path(inputs.get("output_path") or
                        input_path.with_name(f"{input_path.stem}_morphed.mp4"))

        segments, skipped = self._plan_windows(cuts, duration, transition)
        morph_cuts = [s for s in segments if s["kind"] == "morph"]
        if not morph_cuts:
            return ToolResult(success=False,
                              error="No usable cuts to smooth (all skipped at boundaries/overlaps).")

        workdir = Path(tempfile.mkdtemp(prefix="morphcut_"))
        try:
            # Measure BEFORE at each morph cut (on the original).
            radius = transition
            before = {s["cut"]: self._junction_max_mad(str(input_path), s["cut"], radius)
                      for s in morph_cuts}

            parts: list[str] = []
            for i, seg in enumerate(segments):
                dest = workdir / f"seg_{i:04d}.mp4"
                if seg["kind"] == "morph":
                    out = self._morph_window(str(input_path), seg["start"], seg["end"],
                                             morph_fps, out_fps, codec, crf, dest)
                else:
                    out = self._extract_segment(str(input_path), seg["start"], seg["end"],
                                                out_fps, codec, crf, dest)
                if out is None:
                    return ToolResult(success=False, error=f"Failed to build segment {i} ({seg['kind']}).")
                parts.append(out)

            video_only = self._concat(parts, workdir / "concat.mp4")
            if video_only is None:
                return ToolResult(success=False, error="Concat of morphed/passthrough segments failed.")

            if self._has_audio(str(input_path)):
                final = self._mux_audio(video_only, str(input_path), out_path)
                if final is None:
                    return ToolResult(success=False, error="Audio mux failed.")
            else:
                shutil.copyfile(video_only, out_path)
            if not out_path.is_file() or out_path.stat().st_size == 0:
                return ToolResult(success=False, error="Output not created or empty.")

            # Measure AFTER at each cut (on the OUTPUT) — honest, re-probed, never modeled.
            per_cut = []
            for s in morph_cuts:
                mad_before = round(before[s["cut"]], 4)
                mad_after = round(self._junction_max_mad(str(out_path), s["cut"], radius), 4)
                per_cut.append({
                    "time": s["cut"],
                    "max_mad_before": mad_before,
                    "max_mad_after": mad_after,
                    "smoothed": mad_after < mad_before * self.SMOOTH_RATIO,
                })
            smoothed_count = sum(1 for c in per_cut if c["smoothed"])

            return ToolResult(
                success=True,
                artifacts=[str(out_path)],
                duration_seconds=time.time() - start,
                data={
                    "cuts_requested": len(cuts),
                    "cuts_processed": len(morph_cuts),
                    "per_cut": per_cut,
                    "smoothed_count": smoothed_count,
                    "skipped_cuts": skipped,
                    "transition_duration": transition,
                    "morph_fps": morph_fps,
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

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
        # scd=fdiff scene-change detection makes minterpolate DEGRADE HONESTLY: when the
        # content change across the junction is large enough that motion compensation cannot
        # bridge it, the frame is held (hard cut preserved) instead of cross-dissolved into a
        # fake-smooth blend — so _junction_max_mad on the output truthfully reports it as not
        # smoothed. Small, bridgeable jumps fall below the threshold and are interpolated.
        vf = (f"minterpolate=fps={morph_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:"
              f"scd=fdiff:scd_threshold=3,fps={out_fps}")
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
