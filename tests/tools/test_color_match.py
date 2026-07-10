from __future__ import annotations

import math

from tools.enhancement.color_match import ColorMatch
from tools.base_tool import ToolTier


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
