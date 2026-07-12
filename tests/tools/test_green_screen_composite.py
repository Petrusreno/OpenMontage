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
