from __future__ import annotations

from tools.video.morph_cut import MorphCut
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = MorphCut()
    assert tool.name == "morph_cut"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "python:numpy" in tool.dependencies
    assert "python:PIL" in tool.dependencies
    assert "morph_cut" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path", "cut_seconds"]


def test_plan_windows_tiles_timeline():
    tool = MorphCut()
    segs, skipped = tool._plan_windows([1.0], duration=3.0, transition_duration=0.2)
    assert skipped == []
    assert segs == [
        {"kind": "pass", "start": 0.0, "end": 0.9, "cut": None},
        {"kind": "morph", "start": 0.9, "end": 1.1, "cut": 1.0},
        {"kind": "pass", "start": 1.1, "end": 3.0, "cut": None},
    ]
    # covers [0, duration] exactly
    assert segs[0]["start"] == 0.0 and segs[-1]["end"] == 3.0
    total = sum(round(s["end"] - s["start"], 6) for s in segs)
    assert abs(total - 3.0) < 1e-6


def test_plan_windows_two_cuts():
    tool = MorphCut()
    segs, skipped = tool._plan_windows([1.0, 2.0], duration=3.0, transition_duration=0.2)
    assert skipped == []
    kinds = [s["kind"] for s in segs]
    assert kinds == ["pass", "morph", "pass", "morph", "pass"]
    assert [s["cut"] for s in segs if s["kind"] == "morph"] == [1.0, 2.0]


def test_plan_windows_clamps_at_start():
    tool = MorphCut()
    segs, skipped = tool._plan_windows([0.05], duration=2.0, transition_duration=0.2)
    # window would be [-0.05, 0.15] -> clamped to [0.0, 0.15]
    morph = [s for s in segs if s["kind"] == "morph"][0]
    assert morph["start"] == 0.0 and abs(morph["end"] - 0.15) < 1e-9
    assert segs[0]["kind"] == "morph"          # no zero-length pass before it


def test_plan_windows_skips_degenerate_window():
    tool = MorphCut()
    # a cut 0.01s from the end: clamped window < MIN_WINDOW -> skipped.
    # transition_duration=0.05 (not 0.2 like sibling tests): with half=D/2, a
    # single-side-clamped window's length is (distance_to_boundary + half), which
    # is always >= half. At half=0.1 (D=0.2) that floor (0.1) already exceeds
    # MIN_WINDOW=0.04, so no in-bounds cut could ever trigger this skip path.
    # half=0.025 (D=0.05) lets a cut 0.01s from the boundary produce a clamped
    # window of 0.035, which is genuinely < MIN_WINDOW.
    segs, skipped = tool._plan_windows([1.99], duration=2.0, transition_duration=0.05)
    assert any(abs(sk["time"] - 1.99) < 1e-9 for sk in skipped)
    assert all(s["kind"] == "pass" for s in segs)


def test_plan_windows_skips_overlapping_cut():
    tool = MorphCut()
    segs, skipped = tool._plan_windows([1.0, 1.1], duration=3.0, transition_duration=0.2)
    # windows [0.9,1.1] and [1.0,1.2] overlap -> second skipped
    assert any(abs(sk["time"] - 1.1) < 1e-9 for sk in skipped)
    assert [s["cut"] for s in segs if s["kind"] == "morph"] == [1.0]
