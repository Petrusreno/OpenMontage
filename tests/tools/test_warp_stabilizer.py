from __future__ import annotations

import math
import shutil
import subprocess as _sp
from pathlib import Path

import pytest

from tools.video.warp_stabilizer import WarpStabilizer
from tools.base_tool import ToolTier


def _ffmpeg_has_vidstab() -> bool:
    out = _sp.run(["ffmpeg", "-hide_banner", "-filters"],
                  capture_output=True, text=True, check=False)
    return "vidstabdetect" in (out.stdout + out.stderr)


def _make_shaky_clip(path: Path) -> None:
    # Deterministic jittery crop of a moving testsrc => real inter-frame shake.
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=480x360:rate=30:duration=2",
        "-vf", ("crop=w=400:h=300:"
                "x='40+18*sin(n*1.7)+12*sin(n*3.1)':"
                "y='30+18*cos(n*2.3)+12*sin(n*4.7)'"),
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)


def test_contract_fields_present():
    tool = WarpStabilizer()
    assert tool.name == "warp_stabilizer"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert tool.provider == "ffmpeg"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "stabilization" in tool.capabilities
    assert tool.input_schema["required"] == ["input_path"]


def test_probe_engine_prefers_vidstab(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "vidstabdetect vidstabtransform deshake")
    assert tool._probe_engine() == "vidstab"


def test_probe_engine_falls_back_to_deshake(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "deshake scale crop")
    assert tool._probe_engine() == "deshake"


def test_probe_engine_none_when_no_stabilizer(monkeypatch):
    tool = WarpStabilizer()
    monkeypatch.setattr(tool, "_ffmpeg_filters", lambda: "scale crop overlay")
    assert tool._probe_engine() is None


def test_parse_trf_shakiness_computes_mean_magnitude():
    tool = WarpStabilizer()
    # Two frames: (3,4)->5.0 and (0,0)->0.0  => mean 2.5
    trf = "VID.STAB 1\nFrame 1 (List 1 [(3 4 0 0)])\nFrame 2 (List 1 [(0 0 0 0)])\n"
    val = tool._parse_trf_shakiness(trf)
    assert val is not None and math.isclose(val, 2.5, rel_tol=1e-6)


def test_parse_trf_shakiness_none_when_unparseable():
    tool = WarpStabilizer()
    assert tool._parse_trf_shakiness("garbage with no frames") is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.skipif(not _ffmpeg_has_vidstab(), reason="libvidstab required")
def test_execute_reduces_shakiness(tmp_path):
    src = tmp_path / "shaky.mp4"
    _make_shaky_clip(src)
    out = tmp_path / "stable.mp4"

    tool = WarpStabilizer()
    result = tool.execute({"input_path": str(src), "output_path": str(out)})

    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    assert result.data["engine"] == "vidstab"
    before = result.data["shakiness_before"]
    after = result.data["shakiness_after"]
    assert before and after is not None
    # Stabilization must cut shake by a clear margin.
    assert after < before
    assert result.data["reduction_pct"] >= 30.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.skipif(not _ffmpeg_has_vidstab(), reason="libvidstab required")
def test_execute_metric_is_deterministic(tmp_path):
    # The tool declares determinism=DETERMINISTIC. The reported shake metric is
    # derived from the input detection (bit-identical run to run), not from the
    # nondeterministic re-encoded output, so repeated runs must report the same
    # numbers.
    src = tmp_path / "shaky.mp4"
    _make_shaky_clip(src)
    tool = WarpStabilizer()
    r1 = tool.execute({"input_path": str(src), "output_path": str(tmp_path / "o1.mp4")})
    r2 = tool.execute({"input_path": str(src), "output_path": str(tmp_path / "o2.mp4")})
    assert r1.success and r2.success
    assert r1.data["shakiness_before"] == r2.data["shakiness_before"]
    assert r1.data["shakiness_after"] == r2.data["shakiness_after"]
    assert r1.data["reduction_pct"] == r2.data["reduction_pct"]

    # Metric is tied to the actual smoothing applied: no smoothing => no shake
    # reduction reported (honest, never a fabricated constant).
    r0 = tool.execute({"input_path": str(src), "output_path": str(tmp_path / "o0.mp4"),
                       "smoothing": 0})
    assert r0.data["reduction_pct"] == 0.0
