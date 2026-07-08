from __future__ import annotations

from tools.video.text_based_editor import TextBasedEditor
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = TextBasedEditor()
    assert tool.name == "text_based_editor"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "text_based_editing" in tool.capabilities
    assert "input_path" in tool.input_schema["properties"]
    assert "word_timestamps" in tool.input_schema["properties"]
