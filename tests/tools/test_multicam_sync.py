# tests/tools/test_multicam_sync.py
from __future__ import annotations

import json
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


def test_pairwise_pivot_silent_confidence_zero():
    sr = 8000
    samples_list = [np.zeros(8000, np.float32), _click_signal(sr)]
    out = MulticamSync()._pairwise_offsets(samples_list, sr)
    assert out[0] == (0.0, 0.0)


def test_pairwise_pivot_real_confidence_one():
    sr = 8000
    samples_list = [_click_signal(sr), _click_signal(sr)]
    out = MulticamSync()._pairwise_offsets(samples_list, sr)
    assert out[0] == (0.0, 1.0)


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


def test_to_report_skipped_clip_shape():
    tool = MulticamSync()
    # Full-length offsets/confidences indexed by ORIGINAL clip position; index 1
    # is a skipped (no-audio) clip whose slots carry placeholder 0.0 values.
    report = tool._to_report(
        clips=["a.mp4", "b.mp4", "c.mp4"], reference_index=0,
        offsets=[0.0, 0.0, 0.6], confidences=[1.0, 0.0, 0.8],
        skipped=[{"index": 1, "source": "b.mp4", "reason": "no audio or extraction failed"}],
        params={"sample_rate": 8000, "window_seconds": 60, "min_confidence": 0.1},
    )
    skipped_entry = next(o for o in report["offsets"] if o["index"] == 1)
    assert skipped_entry["offset_seconds"] is None
    assert skipped_entry["confidence"] is None
    assert skipped_entry["low_confidence"] is True
    # The skipped clip is excluded from the confidence aggregates.
    assert report["max_confidence"] == pytest.approx(1.0)
    assert report["min_confidence_observed"] == pytest.approx(0.8)


def test_to_report_low_confidence_clip_not_dropped():
    tool = MulticamSync()
    report = tool._to_report(
        clips=["a", "b"], reference_index=0,
        offsets=[0.0, 1.7], confidences=[1.0, 0.0], skipped=[],
        params={"sample_rate": 8000, "window_seconds": 60, "min_confidence": 0.1},
    )
    entry = next(o for o in report["offsets"] if o["index"] == 1)
    assert entry["offset_seconds"] == pytest.approx(1.7)
    assert entry["low_confidence"] is True
    assert report["min_confidence_observed"] == pytest.approx(0.0)


def _write_report_and_load(tool, inputs):
    result = tool.execute(inputs)
    return result


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_end_to_end_recovers_offset(tmp_path):
    sr = 8000
    early = tmp_path / "early.wav"
    late = tmp_path / "late.wav"
    _make_tone_clip(early, sr=sr, dur=2.0)
    # `late` = 0.5s silence + the same tone => it started 0.5s EARLIER in real time?
    # No: prepending silence means its shared content occurs 0.5s later, i.e. `late`
    # started recording 0.5s BEFORE `early`. So `late` is the earliest => reference,
    # and `early`'s offset should be +0.5.
    # Bound anullsrc via :d=0.5 — a bare anullsrc is an infinite source and a
    # misplaced `-t 0.5` binds to the NEXT input (sine), leaving anullsrc
    # unbounded so `concat` reads it forever (hang). Assertions are unchanged.
    _sp.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono:d=0.5",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration=2:sample_rate={sr}",
        "-filter_complex", "[0][1]concat=n=2:v=0:a=1", str(late),
    ], check=True, capture_output=True)
    out = tmp_path / "sync.json"
    result = MulticamSync().execute({
        "clips": [str(early), str(late)], "sample_rate": sr,
        "window_seconds": 60, "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists()
    report = json.loads(out.read_text())
    # `late` is the earliest-start clip => reference (offset 0).
    ref_src = report["reference_source"]
    assert ref_src == str(late)
    early_entry = next(o for o in report["offsets"] if o["source"] == str(early))
    assert early_entry["offset_seconds"] == pytest.approx(0.5, abs=0.05)
    assert early_entry["confidence"] > 0.3


def test_execute_requires_two_clips():
    result = MulticamSync().execute({"clips": ["only_one.wav"]})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_skips_no_audio_clip(tmp_path):
    a = tmp_path / "a.wav"; b = tmp_path / "b.wav"
    _make_tone_clip(a); _make_tone_clip(b)
    silent = tmp_path / "silent.mp4"
    _make_silent_video_no_audio(silent)
    out = tmp_path / "s.json"
    result = MulticamSync().execute({
        "clips": [str(a), str(b), str(silent)], "output_path": str(out)})
    assert result.success, result.error
    report = json.loads(out.read_text())
    assert any(sk["source"] == str(silent) for sk in report["skipped"])
    silent_entry = next(o for o in report["offsets"] if o["source"] == str(silent))
    assert silent_entry["offset_seconds"] is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_fails_when_fewer_than_two_usable(tmp_path):
    a = tmp_path / "a.wav"; _make_tone_clip(a)
    s1 = tmp_path / "s1.mp4"; s2 = tmp_path / "s2.mp4"
    _make_silent_video_no_audio(s1); _make_silent_video_no_audio(s2)
    result = MulticamSync().execute({"clips": [str(a), str(s1), str(s2)],
                                     "output_path": str(tmp_path / "x.json")})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_is_deterministic(tmp_path):
    a = tmp_path / "a.wav"; b = tmp_path / "b.wav"
    _make_tone_clip(a, dur=1.5); _make_tone_clip(b, dur=1.5)
    o1 = tmp_path / "o1.json"; o2 = tmp_path / "o2.json"
    MulticamSync().execute({"clips": [str(a), str(b)], "output_path": str(o1)})
    MulticamSync().execute({"clips": [str(a), str(b)], "output_path": str(o2)})
    assert o1.read_text() == o2.read_text()
