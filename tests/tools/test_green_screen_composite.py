import shutil
import subprocess as sp

import pytest
from tools.video.green_screen_composite import GreenScreenComposite


def test_ffmpeg_color_maps_hex():
    t = GreenScreenComposite()
    assert t._ffmpeg_color("#0E172A") == "0x0E172A"
    assert t._ffmpeg_color("0E172A") == "0x0E172A"
    for bad in ["#0E172", "#0E172AA", "#GG1234", "navy"]:
        with pytest.raises(ValueError):
            t._ffmpeg_color(bad)


def test_layout_filtergraph_full_behind():
    g = GreenScreenComposite()._layout_filtergraph("full_behind", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert g == "[1:v]scale=1920:1080[bg];[0:v]scale=1920:1080,colorkey=0x0E172A:0.1:0.08[fg];[bg][fg]overlay=0:0[v]"


def test_layout_filtergraph_news_anchor_shift_and_scale():
    g = GreenScreenComposite()._layout_filtergraph("news_anchor", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert g == (
        "[1:v]scale=1920:1080,crop=1920:780:0:300,pad=1920:1080:0:0:black[bg];"
        "[0:v]scale=iw*0.65:ih*0.65,colorkey=0x0E172A:0.1:0.08[fg];"
        "[bg][fg]overlay=(W-w)/2:H-h[v]"
    )


def test_layout_filtergraph_pip_and_split():
    t = GreenScreenComposite()
    p = t._layout_filtergraph("pip", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert p == (
        "[1:v]scale=1920:1080[bg];[0:v]scale=1920*0.30:1080*0.30,colorkey=0x0E172A:0.1:0.08[fg];"
        "[bg][fg]overlay=W-w-20:H-h-20[v]"
    )
    s = t._layout_filtergraph("split", 1920, 1080, 0.65, 300, "0x0E172A", 0.10, 0.08)
    assert s == (
        "color=c=black:s=1920x1080[base];[0:v]scale=960:1080,colorkey=0x0E172A:0.1:0.08[l];"
        "[1:v]scale=960:1080[r];[base][l]overlay=0:0[t];[t][r]overlay=960:0[v]"
    )


def test_ffmpeg_color_rejects_double_hash():
    with pytest.raises(ValueError):
        GreenScreenComposite()._ffmpeg_color("##0E172A")


def test_layout_filtergraph_unknown_raises():
    with pytest.raises(ValueError):
        GreenScreenComposite()._layout_filtergraph("bogus", 1920, 1080, 0.65, 300, "0x0E172A", 0.1, 0.08)


def _make_speaker(path, sr_audio=True, w=320, h=180, dur=2, fps=15):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
           "-i", f"color=c=0x0E172A:s={w}x{h}:d={dur}:r={fps}"]
    if sr_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    cmd += ["-vf", "drawbox=x='(w-80)/2':y=40:w=80:h=100:color=white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if sr_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    sp.run(cmd, check=True, capture_output=True)


def _make_bg(path, w=320, h=180, dur=2):
    sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", f"testsrc2=s={w}x{h}:d={dur}:r=30", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)


def _has_audio(path):
    r = sp.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def test_execute_rejects_bad_engine(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    s.write_bytes(b"x"); b.write_bytes(b"x")
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
                                        "output_path": str(tmp_path / "o.mp4"), "engine": "nope"})
    assert not r.success and "engine" in (r.error or "").lower()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_pil_engine_still_works(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
                                        "output_path": str(out), "layout": "full_behind", "engine": "pil"})
    assert r.success, r.error
    assert r.data["engine"] == "pil" and out.exists() and out.stat().st_size > 0
