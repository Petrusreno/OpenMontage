"""Warp Stabilizer — remove camera shake from a clip via ffmpeg vid.stab.

Primary engine: libvidstab 2-pass (vidstabdetect -> vidstabtransform + unsharp).
Fallback engine: ffmpeg's built-in `deshake` filter (single pass, lower quality).
Reports a before/after shakiness metric measured the same way both times.
"""

from __future__ import annotations

import math
import re
import subprocess
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolResult,
    ToolStability,
    ToolTier,
)


class WarpStabilizer(BaseTool):
    name = "warp_stabilizer"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = "Install FFmpeg with libvidstab: brew install ffmpeg"
    agent_skills = ["ffmpeg"]

    capabilities = ["stabilization", "deshake", "warp_stabilizer"]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "smoothing": {"type": "integer", "default": 10, "minimum": 0},
            "shakiness": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "accuracy": {"type": "integer", "default": 15, "minimum": 1, "maximum": 15},
            "zoom": {"type": "number", "default": 0},
            "optzoom": {"type": "integer", "default": 1, "enum": [0, 1, 2]},
            "border": {"type": "string", "default": "black", "enum": ["black", "replicate"]},
            "sharpen": {"type": "boolean", "default": True},
        },
    }

    def _ffmpeg_filters(self) -> str:
        """Return the raw text of `ffmpeg -filters` (cached per instance)."""
        cached = getattr(self, "_filters_cache", None)
        if cached is None:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True, check=False,
            )
            cached = proc.stdout + proc.stderr
            self._filters_cache = cached
        return cached

    def _probe_engine(self) -> str | None:
        filters = self._ffmpeg_filters()
        if "vidstabdetect" in filters and "vidstabtransform" in filters:
            return "vidstab"
        if "deshake" in filters:
            return "deshake"
        return None

    _FRAME_RE = re.compile(r"Frame\s+\d+.*?\[\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

    def _parse_trf_shakiness(self, trf_text: str) -> float | None:
        """Mean euclidean magnitude of the first (x, y) pair on each Frame line.

        Format-version tolerant: relies only on `Frame ... [(x y ...`. Returns
        None if no frame line matches (caller reports null metric, never fake).
        """
        mags: list[float] = []
        for m in self._FRAME_RE.finditer(trf_text):
            x, y = float(m.group(1)), float(m.group(2))
            mags.append(math.hypot(x, y))
        if not mags:
            return None
        return sum(mags) / len(mags)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="not implemented")
