from __future__ import annotations

import unicodedata

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
