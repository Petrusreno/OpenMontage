from __future__ import annotations

from tools.video.morph_cut import MorphCut
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = MorphCut()
    assert tool.name == "morph_cut"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "python:PIL" in tool.dependencies
    assert "morph_cut" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path", "cut_seconds"]
