from __future__ import annotations

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
