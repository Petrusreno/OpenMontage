from __future__ import annotations

from tools.audio.beat_sync import BeatSync
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = BeatSync()
    assert tool.name == "beat_sync"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "analysis"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "beat_sync" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path"]
