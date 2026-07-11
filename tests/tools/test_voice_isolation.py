from __future__ import annotations

from tools.audio.voice_isolation import VoiceIsolation
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = VoiceIsolation()
    assert tool.name == "voice_isolation"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "audio_processing"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "voice_isolation" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path"]
    assert tool.input_schema["properties"]["engine"]["enum"] == ["auto", "rnnoise", "spectral"]
