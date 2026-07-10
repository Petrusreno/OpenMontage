from __future__ import annotations

import math
import shutil
import subprocess as _sp
from pathlib import Path

import pytest

from tools.enhancement.color_match import ColorMatch
from tools.base_tool import ToolTier


def _make_solid_clip(path: Path, hexcolor: str, dur: float = 1.0) -> None:
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={hexcolor}:s=48x48:d={dur}:r=10", str(path),
    ], check=True, capture_output=True)


def test_contract_fields_present():
    tool = ColorMatch()
    assert tool.name == "color_match"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "enhancement"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "python:PIL" in tool.dependencies
    assert "color_match" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path", "reference_path"]


def test_channel_affine_full_intensity_matches_mean():
    tool = ColorMatch()
    m_t, s_t = [50.0, 100.0, 150.0], [20.0, 20.0, 20.0]
    m_r, s_r = [150.0, 80.0, 40.0], [40.0, 10.0, 20.0]
    gains, offsets, notes = tool._channel_affine(m_t, s_t, m_r, s_r, intensity=1.0)
    for c in range(3):
        # matched mean == reference mean
        assert math.isclose(gains[c] * m_t[c] + offsets[c], m_r[c], rel_tol=1e-6)
    assert notes == []


def test_channel_affine_zero_intensity_is_identity():
    tool = ColorMatch()
    gains, offsets, notes = tool._channel_affine(
        [50, 100, 150], [20, 20, 20], [150, 80, 40], [40, 10, 20], intensity=0.0)
    assert gains == [1.0, 1.0, 1.0]
    assert offsets == [0.0, 0.0, 0.0]


def test_channel_affine_half_intensity_when_std_equal():
    tool = ColorMatch()
    # s_t == s_r => raw gain 1.0; half intensity => matched mean = midpoint
    gains, offsets, _ = tool._channel_affine(
        [50.0], [20.0], [150.0], [20.0], intensity=0.5)
    assert math.isclose(gains[0] * 50.0 + offsets[0], 100.0, rel_tol=1e-6)  # (50+150)/2


def test_channel_affine_flat_channel_forces_gain_one():
    tool = ColorMatch()
    gains, offsets, notes = tool._channel_affine(
        [50.0], [0.0], [150.0], [30.0], intensity=1.0)   # s_t == 0 (flat)
    assert gains[0] == 1.0                    # gain not exploded
    assert math.isclose(gains[0] * 50.0 + offsets[0], 150.0, rel_tol=1e-6)  # mean still matched
    assert any(n["reason"] == "flat_channel" for n in notes)


def test_channel_affine_gain_clamped_to_max():
    tool = ColorMatch()
    # tiny s_t would give gain 100; clamp to GAIN_MAX 3.0
    gains, offsets, notes = tool._channel_affine(
        [50.0], [1.0], [150.0], [100.0], intensity=1.0)
    assert gains[0] == 3.0
    assert any(n["reason"] == "gain_clamped" for n in notes)


def test_lutrgb_expr_format():
    tool = ColorMatch()
    expr = tool._lutrgb_expr([1.0, 1.0, 1.0], [128.0, -16.0, -111.0])
    assert expr.startswith("lutrgb=")
    assert "r='clip(1.000000*val+128.000000,0,255)'" in expr
    assert "g='clip(1.000000*val-16.000000,0,255)'" in expr
    assert "b='clip(1.000000*val-111.000000,0,255)'" in expr


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_extract_frame_and_stats(tmp_path):
    clip = tmp_path / "blue.mp4"
    _make_solid_clip(clip, "0x3060A0")            # R=0x30=48, G=0x60=96, B=0xA0=160
    tool = ColorMatch()
    frame = tool._extract_frame(str(clip), 0.0, tmp_path / "f.png")
    assert frame is not None and Path(frame).exists()
    mean, std = tool._frame_stats(frame)
    assert abs(mean[0] - 48) < 3 and abs(mean[1] - 96) < 3 and abs(mean[2] - 160) < 3
    assert max(std) < 2.0                          # solid color => near-zero variance


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_midpoint_of_two_second_clip(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_solid_clip(clip, "0x808080", dur=2.0)
    assert abs(ColorMatch()._midpoint(str(clip)) - 1.0) < 0.2


def test_extract_frame_missing_file_returns_none(tmp_path):
    assert ColorMatch()._extract_frame("/no/such.mp4", 0.0, tmp_path / "x.png") is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_has_audio_true_false(tmp_path):
    silent = tmp_path / "silent.mp4"
    _make_solid_clip(silent, "0x808080")          # color source has no audio track
    tone = tmp_path / "tone.mp4"
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x808080:s=48x48:d=1:r=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", str(tone),
    ], check=True, capture_output=True)
    tool = ColorMatch()
    assert tool._has_audio(str(silent)) is False
    assert tool._has_audio(str(tone)) is True


def test_mean_delta():
    tool = ColorMatch()
    assert abs(tool._mean_delta([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])) < 1e-9
    assert abs(tool._mean_delta([0.0, 0.0, 0.0], [3.0, 6.0, 9.0]) - 6.0) < 1e-9


def test_execute_requires_both_paths(tmp_path):
    result = ColorMatch().execute({"input_path": str(tmp_path / "a.mp4")})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_matches_target_toward_reference(tmp_path):
    target = tmp_path / "target.mp4"
    reference = tmp_path / "ref.mp4"
    _make_solid_clip(target, "0x3060A0")          # bluish
    _make_solid_clip(reference, "0xB05030")       # reddish
    out = tmp_path / "matched.mp4"
    result = ColorMatch().execute({
        "input_path": str(target), "reference_path": str(reference),
        "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert result.data["improved"] is True
    assert result.data["mean_delta_after"] < result.data["mean_delta_before"]
    # after-match target mean is close to the reference mean
    for c in range(3):
        assert abs(result.data["target_mean_after"][c] - result.data["reference_mean"][c]) < 8


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_intensity_zero_is_near_identity(tmp_path):
    target = tmp_path / "t.mp4"; reference = tmp_path / "r.mp4"
    _make_solid_clip(target, "0x3060A0"); _make_solid_clip(reference, "0xB05030")
    out = tmp_path / "o.mp4"
    result = ColorMatch().execute({
        "input_path": str(target), "reference_path": str(reference),
        "output_path": str(out), "intensity": 0.0})
    assert result.success, result.error
    assert result.data["gains"] == [1.0, 1.0, 1.0]
    assert result.data["offsets"] == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_missing_reference_fails(tmp_path):
    target = tmp_path / "t.mp4"; _make_solid_clip(target, "0x3060A0")
    result = ColorMatch().execute({
        "input_path": str(target), "reference_path": str(tmp_path / "nope.mp4"),
        "output_path": str(tmp_path / "o.mp4")})
    assert not result.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_is_deterministic(tmp_path):
    target = tmp_path / "t.mp4"; reference = tmp_path / "r.mp4"
    _make_solid_clip(target, "0x3060A0"); _make_solid_clip(reference, "0xB05030")
    r1 = ColorMatch().execute({"input_path": str(target), "reference_path": str(reference),
                               "output_path": str(tmp_path / "o1.mp4")})
    r2 = ColorMatch().execute({"input_path": str(target), "reference_path": str(reference),
                               "output_path": str(tmp_path / "o2.mp4")})
    assert r1.data["gains"] == r2.data["gains"]
    assert r1.data["offsets"] == r2.data["offsets"]
