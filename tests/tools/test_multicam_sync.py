# tests/tools/test_multicam_sync.py
from __future__ import annotations

import shutil
import subprocess as _sp
from pathlib import Path

import numpy as np
import pytest

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


def _make_tone_clip(path: Path, sr: int = 8000, dur: float = 1.0) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={dur}:sample_rate={sr}",
        str(path),
    ], check=True, capture_output=True)


def _make_silent_video_no_audio(path: Path) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=160x120:rate=15:duration=1",
        "-an", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_extract_samples_returns_float_array(tmp_path):
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, sr=8000, dur=1.0)
    samples = MulticamSync()._extract_samples(str(clip), 8000, 60)
    assert samples is not None
    assert samples.dtype == np.float32
    assert 7000 < samples.size <= 8000          # ~1s at 8kHz
    assert float(np.max(np.abs(samples))) <= 1.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_extract_samples_window_bounds_length(tmp_path):
    clip = tmp_path / "tone.wav"
    _make_tone_clip(clip, sr=8000, dur=3.0)
    samples = MulticamSync()._extract_samples(str(clip), 8000, 1.0)   # 1s window of a 3s clip
    assert samples is not None and samples.size <= 8000 + 10


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_true_false(tmp_path):
    tone = tmp_path / "tone.wav"
    _make_tone_clip(tone)
    silent = tmp_path / "silent.mp4"
    _make_silent_video_no_audio(silent)
    tool = MulticamSync()
    assert tool._has_audio(str(tone)) is True
    assert tool._has_audio(str(silent)) is False


def test_extract_samples_missing_file_returns_none():
    assert MulticamSync()._extract_samples("/no/such/file.wav", 8000, 60) is None


def test_rebaseline_auto_picks_earliest_and_offsets_nonneg():
    tool = MulticamSync()
    # pivot=clip0. lag_i = _xcorr_lag(pivot, clip_i): clip1 delayed +0.4 vs pivot,
    # clip2 delayed -0.2 vs pivot (i.e. clip2 started later than pivot).
    pairwise = [(0.0, 1.0), (0.4, 0.9), (-0.2, 0.8)]
    ref, offsets = tool._rebaseline(pairwise, reference_index=None)
    # start_i = -lag_i => [0.0, -0.4, 0.2]; earliest = clip1 (start -0.4) => ref=1
    assert ref == 1
    assert offsets == pytest.approx([0.4, 0.0, 0.6])
    assert min(offsets) == pytest.approx(0.0)
    assert all(o >= -1e-9 for o in offsets)


def test_rebaseline_explicit_reference_is_zero_others_relative():
    tool = MulticamSync()
    pairwise = [(0.0, 1.0), (0.4, 0.9), (-0.2, 0.8)]   # start_i = [0.0, -0.4, 0.2]
    ref, offsets = tool._rebaseline(pairwise, reference_index=2)
    assert ref == 2
    assert offsets[2] == pytest.approx(0.0)
    # relative spacing preserved: start_i - start_2  => [-0.2, -0.6, 0.0]
    assert offsets == pytest.approx([-0.2, -0.6, 0.0])


def test_to_report_shape():
    tool = MulticamSync()
    report = tool._to_report(
        clips=["a.mp4", "b.mp4"], reference_index=0,
        offsets=[0.0, 0.4], confidences=[1.0, 0.9], skipped=[],
        params={"sample_rate": 8000, "window_seconds": 60, "min_confidence": 0.1},
    )
    assert report["reference_index"] == 0
    assert report["reference_source"] == "a.mp4"
    assert [o["index"] for o in report["offsets"]] == [0, 1]
    assert report["offsets"][1]["offset_seconds"] == pytest.approx(0.4)
    assert report["offsets"][1]["low_confidence"] is False
    assert report["sample_rate"] == 8000
