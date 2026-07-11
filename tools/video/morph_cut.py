"""Morph Cut — smooth hard jump cuts with ffmpeg optical-flow interpolation.

For each specified cut, a short window around the join is re-timed with
`minterpolate` (motion-compensated interpolation) to bridge the jump, then
spliced back. Where motion interpolation cannot bridge a large content change
it degrades to the original hard cut (no warp artifact) and reports it honestly.
Completes the cut-editing cluster: silence_cutter / text_based_editor create
jump cuts; morph_cut smooths them.
"""

from __future__ import annotations

from typing import Any

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
