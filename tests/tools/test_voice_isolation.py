from __future__ import annotations

import os
import shutil
import subprocess as _sp
from pathlib import Path

import numpy as np
import pytest

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


def _make_noisy_clip(path: Path, with_video: bool = False) -> None:
    """[1s tone+noise][1s noise-only] mono audio; optionally a video stream too."""
    d = path.parent
    a = d / "na.wav"; b = d / "nb.wav"; lst = d / "nl.txt"; aud = d / "aud.wav"
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "sine=frequency=300:duration=1:sample_rate=48000", "-f", "lavfi",
             "-i", "anoisesrc=d=1:c=white:a=0.2:r=48000",
             "-filter_complex", "[0][1]amix=inputs=2:duration=shortest", "-ac", "1", str(a)],
            check=True, capture_output=True)
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "anoisesrc=d=1:c=white:a=0.2:r=48000", "-ac", "1", str(b)],
            check=True, capture_output=True)
    lst.write_text(f"file '{a.resolve()}'\nfile '{b.resolve()}'\n")
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(aud)], check=True, capture_output=True)
    if not with_video:
        aud.replace(path)
        return
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "color=c=gray:s=128x96:d=2:r=15", "-i", str(aud), "-shortest",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
            check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_video(tmp_path):
    audio = tmp_path / "a.wav"; _make_noisy_clip(audio)
    vid = tmp_path / "v.mp4"; _make_noisy_clip(vid, with_video=True)
    tool = VoiceIsolation()
    assert tool._has_audio(str(audio)) is True and tool._has_video(str(audio)) is False
    assert tool._has_audio(str(vid)) is True and tool._has_video(str(vid)) is True


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_noise_floor_dbfs_reflects_quiet_window(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    floor = VoiceIsolation()._noise_floor_dbfs(str(clip))
    assert -40 < floor < -10          # a real, finite noise floor, not 0.0 or -inf


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_runs_rnnoise_chain(tmp_path):
    clip = tmp_path / "c.wav"; _make_noisy_clip(clip)
    tool = VoiceIsolation()
    af = tool._build_filter("rnnoise", VoiceIsolation.BUNDLED_MODEL, nf=-25, mix=1.0)
    out = tool._process(str(clip), af, "pcm_s16le", "192k", tmp_path / "o.wav")
    assert out and Path(out).exists() and Path(out).stat().st_size > 0


def test_process_missing_file_returns_none(tmp_path):
    tool = VoiceIsolation()
    af = tool._build_filter("spectral", None, nf=-25, mix=1.0)
    assert tool._process("/no/such.wav", af, "aac", "192k", tmp_path / "x.aac") is None
