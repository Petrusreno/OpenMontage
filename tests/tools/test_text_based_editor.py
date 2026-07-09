from __future__ import annotations

import json
import shutil
import subprocess as _sp
import unicodedata

import pytest

from schemas.artifacts import validate_artifact

from tools.video.text_based_editor import TextBasedEditor
from tools.base_tool import ToolTier


def test_contract_fields_present():
    tool = TextBasedEditor()
    assert tool.name == "text_based_editor"
    assert tool.tier == ToolTier.CORE
    assert tool.capability == "video_post"
    assert "cmd:ffmpeg" in tool.dependencies
    assert "text_based_editing" in tool.capabilities
    assert "input_path" in tool.input_schema["properties"]
    assert "word_timestamps" in tool.input_schema["properties"]


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_detect_fillers_pt_flags_only_disfluencies():
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.3), _w(" gato", 0.3, 0.7), _w("ãã", 0.7, 1.0)]
    lex = tool._lexicon_for("pt", None)
    spans = tool._detect_fillers(words, lex)
    got = sorted((round(s["start"], 2), s["word"]) for s in spans)
    assert got == [(0.0, "hum"), (0.7, "ãã")]
    assert all(s["reason"] == "filler" for s in spans)


def test_content_fidelity_default_keeps_ambiguous_words():
    tool = TextBasedEditor()
    words = [_w("então", 0.0, 0.4), _w("é", 0.4, 0.6), _w("né", 0.6, 0.9)]
    lex = tool._lexicon_for("pt", None)
    assert tool._detect_fillers(words, lex) == []          # default removes none
    lex2 = tool._lexicon_for("pt", ["então", "é", "né"])   # opt-in
    assert len(tool._detect_fillers(words, lex2)) == 3


def test_detect_repetitions_removes_all_but_last():
    tool = TextBasedEditor()
    words = [_w("o", 0.0, 0.2), _w("o", 0.25, 0.45), _w("gato", 0.5, 0.9)]
    spans = tool._detect_repetitions(words, 0.6)
    assert len(spans) == 1
    assert round(spans[0]["start"], 2) == 0.0 and spans[0]["reason"] == "repetition"


def test_detect_repetitions_respects_gap():
    tool = TextBasedEditor()
    words = [_w("sim", 0.0, 0.2), _w("sim", 2.0, 2.2)]   # 1.8s apart > max_gap
    assert tool._detect_repetitions(words, 0.6) == []


def test_detect_repetitions_run_of_three():
    tool = TextBasedEditor()
    words = [_w("o", 0.0, 0.2), _w("o", 0.25, 0.45), _w("o", 0.5, 0.7)]
    spans = tool._detect_repetitions(words, 0.6)
    assert len(spans) == 2
    starts = sorted(round(s["start"], 2) for s in spans)
    assert starts == [0.0, 0.25]
    assert 0.5 not in starts


def test_normalize_matches_nfd_filler():
    tool = TextBasedEditor()
    nfd_word = unicodedata.normalize("NFD", "ãã")
    words = [_w(nfd_word, 0.0, 0.3)]
    lex = tool._lexicon_for("pt", None)
    spans = tool._detect_fillers(words, lex)
    assert len(spans) == 1


def test_match_literal_words_all_occurrences():
    tool = TextBasedEditor()
    words = [_w("Tá", 0.0, 0.2), _w("bom", 0.2, 0.5), _w("tá!", 0.5, 0.7)]
    spans = tool._match_literal_words(words, ["tá"])
    assert len(spans) == 2 and all(s["reason"] == "literal" for s in spans)


def test_indices_and_ranges_to_spans():
    tool = TextBasedEditor()
    words = [_w("a", 0.0, 0.2), _w("b", 0.2, 0.4), _w("c", 0.4, 0.6)]
    spans = tool._indices_and_ranges_to_spans(
        words, indices=[1], ranges=[{"start_seconds": 1.0, "end_seconds": 1.5}])
    reasons = sorted(s["reason"] for s in spans)
    assert reasons == ["index", "range"]
    idx = next(s for s in spans if s["reason"] == "index")
    assert round(idx["start"], 2) == 0.2


def test_merge_spans_combines_overlaps():
    tool = TextBasedEditor()
    spans = [{"start": 0.0, "end": 0.3}, {"start": 0.25, "end": 0.5}, {"start": 1.0, "end": 1.2}]
    merged = tool._merge_spans(spans)
    assert merged == [{"start": 0.0, "end": 0.5}, {"start": 1.0, "end": 1.2}]


def test_keep_segments_is_complement_with_padding():
    tool = TextBasedEditor()
    merged = [{"start": 1.0, "end": 2.0}]           # remove 1..2 of a 3s clip
    keeps = tool._keep_segments(merged, duration=3.0, padding=0.0, min_gap=0.05)
    assert keeps == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]


def test_keep_segments_identity_when_no_removals():
    tool = TextBasedEditor()
    keeps = tool._keep_segments([], duration=5.0, padding=0.0, min_gap=0.05)
    assert keeps == [{"start": 0.0, "end": 5.0}]


def test_to_edit_decisions_validates_against_schema():
    tool = TextBasedEditor()
    keeps = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    ed = tool._to_edit_decisions(keeps, source="clip.mp4", removed=[], meta={"modes": ["filler"]})
    validate_artifact("edit_decisions", ed)          # raises on failure
    assert ed["version"] == "1.0"
    assert [c["in_seconds"] for c in ed["cuts"]] == [0.0, 2.0]
    assert all(c["source"] == "clip.mp4" for c in ed["cuts"])


def test_keep_segments_short_removal_still_cuts():
    tool = TextBasedEditor()
    merged = [{"start": 0.5, "end": 0.52}]            # 0.02s removal
    # A global padding of 0.08 (> half the removal) would previously cancel the
    # removal into one full-length span; the per-removal padding clamp keeps a
    # real cut → two keep segments, not one span covering the whole clip.
    keeps = tool._keep_segments(merged, duration=1.0, padding=0.08, min_gap=0.0)
    assert len(keeps) == 2
    assert keeps[0]["start"] == 0.0
    assert keeps[-1]["end"] == 1.0


def test_execute_emits_edit_decisions_from_word_timestamps(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.4), _w("olá", 0.4, 0.8), _w("mundo", 0.8, 1.2)]
    out = tmp_path / "ed.json"
    result = tool.execute({
        "word_timestamps": words, "source": "clip.mp4", "language": "pt",
        "output_path": str(out),
    })
    assert result.success, result.error
    assert out.exists()
    ed = json.loads(out.read_text())
    validate_artifact("edit_decisions", ed)
    # "hum" (0.0-0.4) removed → first kept cut starts at/after 0.4
    assert ed["cuts"][0]["in_seconds"] >= 0.4 - 1e-9
    assert result.data["removed_count"] == 1
    assert result.data["removed"][0]["reason"] == "filler"


def test_execute_identity_when_nothing_removed(tmp_path):
    tool = TextBasedEditor()
    words = [_w("olá", 0.0, 0.5), _w("mundo", 0.5, 1.0)]
    out = tmp_path / "ed.json"
    result = tool.execute({"word_timestamps": words, "source": "c.mp4",
                           "remove_fillers": False, "output_path": str(out)})
    assert result.success
    ed = json.loads(out.read_text())
    assert len(ed["cuts"]) == 1
    assert ed["cuts"][0]["in_seconds"] == 0.0
    assert result.data["removed_count"] == 0


def test_execute_fails_when_everything_removed(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 1.0)]
    out = tmp_path / "ed.json"
    result = tool.execute({"word_timestamps": words, "source": "c.mp4",
                           "output_path": str(out)})
    assert not result.success
    assert not out.exists()


def test_execute_fails_on_all_invalid_words(tmp_path):
    tool = TextBasedEditor()
    words = [{"word": "a", "start": None, "end": None}]
    out = tmp_path / "ed.json"
    result = tool.execute({"word_timestamps": words, "source": "c.mp4",
                           "output_path": str(out)})
    assert not result.success
    assert not out.exists()


def test_execute_requires_input_or_words():
    tool = TextBasedEditor()
    result = tool.execute({})
    assert not result.success and "input_path" in (result.error or "")


def test_execute_is_deterministic(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.4), _w("olá", 0.4, 0.8), _w("olá", 0.85, 1.2), _w("fim", 1.3, 1.6)]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    tool.execute({"word_timestamps": words, "source": "c.mp4", "output_path": str(a)})
    tool.execute({"word_timestamps": words, "source": "c.mp4", "output_path": str(b)})
    assert a.read_text() == b.read_text()


def test_execute_removed_seconds_uses_merged_not_raw(tmp_path):
    tool = TextBasedEditor()
    words = [_w("hum", 0.0, 0.4), _w("olá", 0.4, 0.8), _w("mundo", 0.8, 1.2)]
    out = tmp_path / "ed.json"
    # "hum" flagged by BOTH filler and literal removal → 2 raw spans, same 0.0-0.4.
    result = tool.execute({
        "word_timestamps": words, "source": "c.mp4",
        "remove_words": ["hum"], "output_path": str(out),
    })
    assert result.success, result.error
    assert result.data["removed_count"] == 2                 # raw report keeps both reasons
    assert abs(result.data["removed_seconds"] - 0.4) < 1e-6  # merged, not 0.8
    ed = json.loads(out.read_text())
    assert abs(ed["metadata"]["removed_seconds"] - 0.4) < 1e-6


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_execute_render_produces_shorter_clip(tmp_path):
    src = tmp_path / "src.mp4"
    # 3s clip with an audio track so -c:a copy path is exercised.
    _sp.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-pix_fmt", "yuv420p", "-shortest", str(src),
    ], check=True, capture_output=True)
    # Remove the middle second by explicit range.
    words = [_w("a", 0.0, 0.9), _w("b", 1.0, 2.0), _w("c", 2.1, 3.0)]
    out = tmp_path / "ed.json"
    render = tmp_path / "cut.mp4"
    result = TextBasedEditor().execute({
        "input_path": str(src), "word_timestamps": words, "source": str(src),
        "remove_fillers": False, "remove_ranges": [{"start_seconds": 1.0, "end_seconds": 2.0}],
        "output_path": str(out), "render": True, "render_path": str(render),
    })
    assert result.success, result.error
    assert render.exists() and render.stat().st_size > 0
    assert result.data["rendered"] is True
    # Rendered duration should be < source duration.
    def _dur(p):
        r = _sp.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        return float(r.stdout.strip())
    assert _dur(render) < _dur(src) - 0.3
