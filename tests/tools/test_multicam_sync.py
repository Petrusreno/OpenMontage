# tests/tools/test_multicam_sync.py
from __future__ import annotations

from tools.audio.multicam_sync import MulticamSync
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = MulticamSync()
    assert tool.name == "multicam_sync"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "analysis"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "multicam_sync" in tool.capabilities
    assert tool.input_schema["required"] == ["clips"]
    assert tool.input_schema["properties"]["clips"]["minItems"] == 2
