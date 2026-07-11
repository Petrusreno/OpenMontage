from __future__ import annotations

import shutil
import subprocess as _sp
from pathlib import Path

import numpy as np
import pytest

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


def _impulse_signal(sample_rate, dur, period, click_ms=20):
    x = np.zeros(int(dur * sample_rate), dtype=np.float64)
    cl = int(click_ms / 1000 * sample_rate)
    for k in range(int(dur / period)):
        s = int(k * period * sample_rate)
        x[s:s + cl] += np.hanning(cl) * np.sin(2 * np.pi * 1000 * np.arange(cl) / sample_rate)
    return x


def test_onset_envelope_peaks_at_impulses():
    tool = BeatSync()
    sr = 22050
    x = _impulse_signal(sr, dur=4.0, period=0.5)
    env, fps = tool._onset_envelope(x, sr)
    assert env.size > 0 and fps > 0
    # the envelope's max should be well above its median (peaks exist)
    assert float(env.max()) > float(np.median(env)) + 2.0


def test_pick_beats_recovers_spacing():
    tool = BeatSync()
    sr = 22050
    x = _impulse_signal(sr, dur=6.0, period=0.5)   # 120 BPM
    env, fps = tool._onset_envelope(x, sr)
    beats = tool._pick_beats(env, fps)
    assert len(beats) >= 10                          # ~12 beats (± boundary)
    ibi = np.diff(beats)
    assert abs(float(np.median(ibi)) - 0.5) < 0.05   # ~0.5 s spacing


def test_estimate_bpm():
    tool = BeatSync()
    assert abs(tool._estimate_bpm([0.0, 0.5, 1.0, 1.5, 2.0]) - 120.0) < 1e-6
    assert tool._estimate_bpm([1.0]) == 0.0          # < 2 beats
    # robust to one missing beat (median): 0.5,0.5,1.0,0.5 -> median 0.5 -> 120
    assert abs(tool._estimate_bpm([0.0, 0.5, 1.0, 2.0, 2.5]) - 120.0) < 1e-6


def test_snap_to_nearest_beat():
    tool = BeatSync()
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    snapped = tool._snap([0.42, 1.13, 1.9], beats)
    assert [s["snapped"] for s in snapped] == [0.5, 1.0, 2.0]
    assert abs(snapped[0]["offset"] - (0.5 - 0.42)) < 1e-9
    assert tool._snap([], beats) == [] and tool._snap([1.0], []) == []


def _make_click_wav(path: Path, bpm: int = 120, dur: float = 6.0, sr: int = 22050) -> None:
    period = 60.0 / bpm
    x = _impulse_signal(sr, dur, period)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    raw = (x * 32767).astype("<i2").tobytes()
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "s16le", "-ar", str(sr), "-ac", "1",
             "-i", "pipe:0", str(path)], input=raw, check=True, capture_output=True)


def _make_click_video(path: Path, bpm: int = 120, sr: int = 22050) -> None:
    d = path.parent
    aud = d / "click.wav"; _make_click_wav(aud, bpm=bpm, sr=sr)
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi", "-i", "color=c=gray:s=128x96:d=6:r=15",
             "-i", str(aud), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", str(path)], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_and_extract(tmp_path):
    clip = tmp_path / "c.wav"; _make_click_wav(clip)
    tool = BeatSync()
    assert tool._has_audio(str(clip)) is True
    samples = tool._extract_samples(str(clip), 22050)
    assert samples is not None and samples.dtype == np.float32
    assert samples.size > 22050 * 5                  # ~6 s at 22050
    assert float(np.max(np.abs(samples))) <= 1.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_false_on_silent_video(tmp_path):
    silent = tmp_path / "s.mp4"
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1:r=10",
             "-an", "-pix_fmt", "yuv420p", str(silent)], check=True, capture_output=True)
    assert BeatSync()._has_audio(str(silent)) is False


def test_extract_missing_file_returns_none():
    assert BeatSync()._extract_samples("/no/such.wav", 22050) is None
