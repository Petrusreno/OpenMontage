from __future__ import annotations

import numpy as np

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
