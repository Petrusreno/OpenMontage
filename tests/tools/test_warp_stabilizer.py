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
