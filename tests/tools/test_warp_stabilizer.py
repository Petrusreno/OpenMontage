from __future__ import annotations

import math

from tools.video.warp_stabilizer import WarpStabilizer
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = WarpStabilizer()
    assert tool.name == "warp_stabilizer"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert tool.provider == "ffmpeg"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "stabilization" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path"]


def test_probe_engine_prefers_vidstab(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "vidstabdetect vidstabtransform deshake")
    assert tool._probe_engine() == "vidstab"


def test_probe_engine_falls_back_to_deshake(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "deshake scale crop")
    assert tool._probe_engine() == "deshake"


def test_probe_engine_none_when_no_stabilizer(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "scale crop overlay")
    assert tool._probe_engine() is None


def test_parse_trf_shakiness_computes_mean_magnitude():
    tool = WarpStabilizer()
    # Two frames: (3,4)->5.0 and (0,0)->0.0  => mean 2.5
    trf = "VID.STAB 1\nFrame 1 (List 1 [(3 4 0 0)])\nFrame 2 (List 1 [(0 0 0 0)])\n"
    val = tool._parse_trf_shakiness(trf)
    assert val is not None and math.isclose(val, 2.5, rel_tol=1e-6)


def test_parse_trf_shakiness_none_when_unparseable():
    tool = WarpStabilizer()
    assert tool._parse_trf_shakiness("garbage with no frames") is None
