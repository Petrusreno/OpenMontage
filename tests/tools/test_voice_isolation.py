from __future__ import annotations

import os

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


def test_resolve_model_prefers_explicit_then_env_then_bundled(tmp_path, monkeypatch):
    tool = VoiceIsolation()
    monkeypatch.delenv("RNNOISE_MODEL", raising=False)
    # bundled exists by default
    assert tool._resolve_model(None) == VoiceIsolation.BUNDLED_MODEL
    # explicit override wins
    m = tmp_path / "custom.rnnn"; m.write_bytes(b"x")
    assert tool._resolve_model(str(m)) == str(m)
    # env override (when no explicit)
    monkeypatch.setenv("RNNOISE_MODEL", str(m))
    assert tool._resolve_model(None) == str(m)
    # non-existent explicit -> None (not silently the bundled)
    assert tool._resolve_model("/no/such.rnnn") is None


def test_build_filter_rnnoise_and_spectral():
    tool = VoiceIsolation()
    r = tool._build_filter("rnnoise", "/m.rnnn", nf=-25, mix=1.0)
    assert "arnndn=model=/m.rnnn" in r and "loudnorm" in r and "asplit" not in r
    s = tool._build_filter("spectral", None, nf=-25, mix=1.0)
    assert "afftdn=nf=-25" in s and "anlmdn" in s and "deesser" in s and "arnndn" not in s


def test_build_filter_mix_blend():
    tool = VoiceIsolation()
    # asymmetric mix so wet (0.3) and dry (0.7) are distinct substrings
    m = tool._build_filter("rnnoise", "/m.rnnn", nf=-25, mix=0.3)
    assert "asplit=2" in m and "amix=inputs=2:normalize=0" in m
    assert "volume=0.3" in m           # wet scaled by mix
    assert "volume=0.7" in m           # dry scaled by 1-mix


def test_build_filter_escapes_model_path():
    tool = VoiceIsolation()
    # a Windows-style / metachar-laden path must not break or inject into the graph
    r = tool._build_filter("rnnoise", "C:/a,b/model.rnnn", nf=-25, mix=1.0)
    assert r.startswith("arnndn=model=")
    assert "C\\:/a\\,b/model.rnnn" in r               # ':' and ',' escaped (\: and \,)
    assert r.count("arnndn") == 1                     # no injected extra filter nodes


def test_arnndn_available_guards_subprocess_error(monkeypatch):
    tool = VoiceIsolation()

    def _boom(*a, **k):
        raise OSError("ffmpeg missing")

    monkeypatch.setattr("tools.audio.voice_isolation.subprocess.run", _boom)
    assert tool._arnndn_available() is False          # guarded -> False, not a traceback
