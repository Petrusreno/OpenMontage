# tests/tools/test_multicam_sync.py
from __future__ import annotations

import numpy as np

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


def _click_signal(sample_rate: int, dur: float = 1.0, at: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * 440 * t) * np.exp(-((t - at) ** 2) / 0.001)).astype(np.float32)


def test_xcorr_recovers_known_delay():
    sr = 8000
    a = _click_signal(sr)
    delay = 0.4
    b = np.concatenate([np.zeros(int(delay * sr), np.float32), a])  # b = a delayed by 0.4s
    lag, conf = MulticamSync()._xcorr_lag(a, b, sr)
    assert abs(lag - delay) < 1.5 / sr        # within ~1 sample
    assert conf > 0.5


def test_xcorr_confidence_low_for_uncorrelated():
    sr = 8000
    a = _click_signal(sr)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(a)).astype(np.float32)
    _, conf = MulticamSync()._xcorr_lag(a, noise, sr)
    assert conf < 0.3


def test_xcorr_zero_lag_for_identical():
    sr = 8000
    a = _click_signal(sr)
    lag, conf = MulticamSync()._xcorr_lag(a, a, sr)
    assert abs(lag) < 1.5 / sr
    assert conf > 0.99


def test_xcorr_single_sample_ref_does_not_crash():
    # ref.size == 1 would make cc[-0:] the whole array (IndexError) without the guard.
    sr = 8000
    lag, conf = MulticamSync()._xcorr_lag(
        np.array([-1.0], dtype=np.float32),
        np.array([1, 2, 3, 4, 5], dtype=np.float32), sr)
    assert isinstance(lag, float) and isinstance(conf, float)
