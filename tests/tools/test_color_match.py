from __future__ import annotations

from tools.enhancement.color_match import ColorMatch
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = ColorMatch()
    assert tool.name == "color_match"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "enhancement"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "python:PIL" in tool.dependencies
    assert "color_match" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path", "reference_path"]
