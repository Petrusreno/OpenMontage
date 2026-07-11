from __future__ import annotations

import shutil
import subprocess as _sp
from pathlib import Path

import numpy as np
import pytest

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


def test_plan_windows_records_duplicate_cut():
    tool = MorphCut()
    segs, skipped = tool._plan_windows([1.0, 1.0], duration=3.0, transition_duration=0.2)
    # one morph emitted; the duplicate is recorded, not silently dropped
    assert [s["cut"] for s in segs if s["kind"] == "morph"] == [1.0]
    assert any(sk["reason"] == "duplicate cut time" and abs(sk["time"] - 1.0) < 1e-9
               for sk in skipped)


def test_plan_windows_skips_overlapping_cut():
    tool = MorphCut()
    segs, skipped = tool._plan_windows([1.0, 1.1], duration=3.0, transition_duration=0.2)
    # windows [0.9,1.1] and [1.0,1.2] overlap -> second skipped
    assert any(abs(sk["time"] - 1.1) < 1e-9 for sk in skipped)
    assert [s["cut"] for s in segs if s["kind"] == "morph"] == [1.0]


def _make_jump_clip(path: Path, xa: int, xb: int, fps: int = 30, hold: float = 0.6) -> None:
    """A gray clip with a white box at xa for `hold`s, hard-cut to the box at xb for `hold`s."""
    d = path.parent
    a = d / "ja.png"; b = d / "jb.png"; sa = d / "jsa.mp4"; sb = d / "jsb.mp4"; lst = d / "jl.txt"
    for png, x in ((a, xa), (b, xb)):
        _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi", "-i", "color=c=gray:s=128x128:d=0.1",
                 "-vf", f"drawbox=x={x}:y=48:w=30:h=30:color=white:t=fill", "-frames:v", "1", str(png)],
                check=True, capture_output=True)
    for seg, png in ((sa, a), (sb, b)):
        _sp.run(["ffmpeg", "-y", "-v", "quiet", "-loop", "1", "-i", str(png), "-t", str(hold),
                 "-r", str(fps), "-pix_fmt", "yuv420p", str(seg)], check=True, capture_output=True)
    lst.write_text(f"file '{sa.resolve()}'\nfile '{sb.resolve()}'\n")
    _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(path)], check=True, capture_output=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_probe_duration_fps_audio(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 40, 52)
    tool = MorphCut()
    assert abs(tool._duration(str(clip)) - 1.2) < 0.15
    assert abs(tool._probe_fps(str(clip)) - 30.0) < 0.5
    assert tool._has_audio(str(clip)) is False        # lavfi color source has no audio


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_junction_max_mad_high_on_hard_jump(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 20, 100)                     # big jump at t=0.6
    mad = MorphCut()._junction_max_mad(str(clip), 0.6, 0.15)
    assert mad > 5.0                                   # a hard jump = a large frame-to-frame delta


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_morph_window_and_extract_and_concat(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 40, 52)
    tool = MorphCut()
    p1 = tool._extract_segment(str(clip), 0.0, 0.5, 30, "libx264", 18, tmp_path / "p1.mp4")
    mw = tool._morph_window(str(clip), 0.5, 0.7, 60, 30, "libx264", 18, tmp_path / "m.mp4")
    p3 = tool._extract_segment(str(clip), 0.7, 1.2, 30, "libx264", 18, tmp_path / "p3.mp4")
    assert p1 and mw and p3
    out = tool._concat([p1, mw, p3], tmp_path / "out.mp4")
    assert out and Path(out).exists() and Path(out).stat().st_size > 0
    assert abs(tool._duration(out) - 1.2) < 0.15       # duration preserved


def test_extract_segment_missing_file_returns_none(tmp_path):
    assert MorphCut()._extract_segment("/no/such.mp4", 0.0, 1.0, 30, "libx264", 18,
                                       tmp_path / "x.mp4") is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_concat_handles_path_with_apostrophe(tmp_path):
    # A segment path containing a single quote must not break the concat list format.
    # Build the clip in a clean dir, but write the SEGMENTS into an apostrophe dir so the
    # concat list entries contain the quote (isolates _concat's escaping).
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 40, 52)
    workdir = tmp_path / "o'brien"
    workdir.mkdir()
    tool = MorphCut()
    p1 = tool._extract_segment(str(clip), 0.0, 0.5, 30, "libx264", 18, workdir / "p1.mp4")
    p2 = tool._extract_segment(str(clip), 0.5, 1.0, 30, "libx264", 18, workdir / "p2.mp4")
    assert p1 and p2 and "o'brien" in p1
    out = tool._concat([p1, p2], workdir / "out.mp4")
    assert out and Path(out).exists() and Path(out).stat().st_size > 0


SMOOTH_ANCHOR_NOTE = "e2e honesty anchor"


def test_execute_requires_cuts(tmp_path):
    result = MorphCut().execute({"input_path": str(tmp_path / "a.mp4"), "cut_seconds": []})
    assert not result.success


def test_execute_rejects_non_numeric_transition(tmp_path):
    result = MorphCut().execute({"input_path": str(tmp_path / "a.mp4"), "cut_seconds": [1.0],
                                 "transition_duration": "x"})
    assert not result.success


def test_scd_threshold_is_exposed_and_validated(tmp_path):
    # scd_threshold is a real schema param (not a hardcoded magic number) and is
    # boundary-validated like the other numeric inputs.
    assert "scd_threshold" in MorphCut().input_schema["properties"]
    bad = MorphCut().execute({"input_path": str(tmp_path / "a.mp4"), "cut_seconds": [1.0],
                              "scd_threshold": "x"})
    assert not bad.success


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_smooths_small_jump(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 40, 52)                       # small 12px jump at t=0.6
    out = tmp_path / "smoothed.mp4"
    result = MorphCut().execute({
        "input_path": str(clip), "cut_seconds": [0.6], "output_path": str(out)})
    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert abs(result.data["per_cut"][0]["time"] - 0.6) < 1e-9
    assert result.data["per_cut"][0]["max_mad_after"] < result.data["per_cut"][0]["max_mad_before"]
    assert result.data["per_cut"][0]["smoothed"] is True
    assert result.data["smoothed_count"] == 1
    assert abs(MorphCut()._duration(str(out)) - 1.2) < 0.2      # duration preserved


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_degrades_honestly_on_large_jump(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 10, 110)                      # huge 100px jump: minterpolate can't bridge
    out = tmp_path / "o.mp4"
    result = MorphCut().execute({
        "input_path": str(clip), "cut_seconds": [0.6], "output_path": str(out)})
    assert result.success, result.error                # still succeeds (video produced)
    assert out.exists()
    assert result.data["per_cut"][0]["smoothed"] is False   # reported, not faked


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_is_deterministic(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_jump_clip(clip, 40, 52)
    r1 = MorphCut().execute({"input_path": str(clip), "cut_seconds": [0.6],
                             "output_path": str(tmp_path / "o1.mp4")})
    r2 = MorphCut().execute({"input_path": str(clip), "cut_seconds": [0.6],
                             "output_path": str(tmp_path / "o2.mp4")})
    assert [c["time"] for c in r1.data["per_cut"]] == [c["time"] for c in r2.data["per_cut"]]
    assert r1.data["cuts_processed"] == r2.data["cuts_processed"]
