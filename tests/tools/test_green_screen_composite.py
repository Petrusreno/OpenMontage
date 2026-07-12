import shutil
import subprocess as sp

import numpy as np
import pytest
from PIL import Image

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
    # Nonexistent paths: a bad engine must fail on validation BEFORE the
    # existence checks — else the error would say "not found", not "engine".
    r = GreenScreenComposite().execute({"speaker_path": "/no/such/s.mp4",
                                        "background_path": "/no/such/b.mp4",
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


def _sample_top_navy_pct(mp4, tmp_path):
    png = tmp_path / "_frame.png"
    sp.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(mp4), "-frames:v", "1", str(png)],
           check=True, capture_output=True)
    a = np.asarray(Image.open(png).convert("RGB")).astype(float)
    navy = np.array([0x0E, 0x17, 0x2A], dtype=float)
    top = a[0:30, :, :].reshape(-1, 3)
    d = np.sqrt(((top - navy) ** 2).sum(1))
    return float((d < 40).mean()) * 100


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_keying_reveals_background(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind"})
    assert r.success, r.error
    assert r.data["engine"] == "ffmpeg"
    keyed_navy = _sample_top_navy_pct(out, tmp_path)
    # No-key baseline of the SAME inputs: the speaker's navy bg opaquely covers
    # the background, so the strip stays navy. Proves the <5% is real keying,
    # not a vacuous threshold.
    nokey = tmp_path / "nokey.mp4"
    sp.run(["ffmpeg", "-y", "-v", "error", "-i", str(s), "-i", str(b),
            "-filter_complex", "[1:v]scale=320:180[bg];[0:v]scale=320:180[fg];[bg][fg]overlay=0:0[v]",
            "-map", "[v]", "-t", "2", "-r", "15", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(nokey)], check=True, capture_output=True)
    nokey_navy = _sample_top_navy_pct(nokey, tmp_path)
    assert keyed_navy < 5.0 < 90.0 < nokey_navy, (keyed_navy, nokey_navy)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.parametrize("layout", ["full_behind", "news_anchor", "pip", "split"])
def test_ffmpeg_all_layouts(tmp_path, layout):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b, w=320, h=180)
    out = tmp_path / f"o_{layout}.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": layout, "bg_shift_up": 60})  # < 180-tall bg
    assert r.success, r.error
    assert out.exists() and out.stat().st_size > 0
    assert r.data["dimensions"] == "320x180"
    assert r.data["frame_count"] >= 1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_news_anchor_rejects_oversize_shift(tmp_path):
    # bg_shift_up >= background height must fail loudly, not silently clamp.
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b, w=320, h=180)
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(tmp_path / "o.mp4"), "layout": "news_anchor", "bg_shift_up": 300})
    assert not r.success and "bg_shift_up" in (r.error or "")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_speaker_audio_passthrough(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=True); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind"})
    assert r.success, r.error
    assert r.data["has_audio"] is True and _has_audio(out)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_silent_speaker_no_audio_ok(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind"})
    assert r.success, r.error
    assert r.data["has_audio"] is False and not _has_audio(out)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_original_audio_override(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False); _make_bg(b)
    aud = tmp_path / "voice.m4a"
    sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=2",
            "-c:a", "aac", str(aud)], check=True, capture_output=True)
    out = tmp_path / "o.mp4"
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(b),
        "output_path": str(out), "layout": "full_behind", "original_audio_path": str(aud)})
    assert r.success, r.error
    assert r.data["has_audio"] is True and _has_audio(out)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_faster_than_pil(tmp_path):
    import time
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"
    _make_speaker(s, sr_audio=False, dur=3); _make_bg(b, dur=3)
    base = {"speaker_path": str(s), "background_path": str(b), "layout": "full_behind"}
    t0 = time.time(); r1 = GreenScreenComposite().execute({**base, "output_path": str(tmp_path/"f.mp4"), "engine": "ffmpeg"}); ff = time.time()-t0
    t0 = time.time(); r2 = GreenScreenComposite().execute({**base, "output_path": str(tmp_path/"p.mp4"), "engine": "pil"}); pil = time.time()-t0
    assert r1.success and r2.success
    assert ff < pil


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_ffmpeg_bad_background_fails_cleanly(tmp_path):
    s = tmp_path / "s.mp4"; _make_speaker(s, sr_audio=False)
    bad = tmp_path / "b.mp4"; bad.write_bytes(b"not a video")
    r = GreenScreenComposite().execute({"speaker_path": str(s), "background_path": str(bad),
        "output_path": str(tmp_path / "o.mp4"), "layout": "full_behind"})
    assert not r.success and r.error


def test_execute_rejects_bad_key_params(tmp_path):
    s = tmp_path / "s.mp4"; b = tmp_path / "b.mp4"; s.write_bytes(b"x"); b.write_bytes(b"x")
    base = {"speaker_path": str(s), "background_path": str(b), "output_path": str(tmp_path / "o.mp4")}
    assert not GreenScreenComposite().execute({**base, "key_similarity": "x"}).success
    assert not GreenScreenComposite().execute({**base, "key_blend": 1.5}).success
    # non-numeric geometry must fail cleanly (not a traceback) on both engines
    assert not GreenScreenComposite().execute({**base, "bg_shift_up": "abc"}).success
    assert not GreenScreenComposite().execute({**base, "speaker_scale": "big"}).success
