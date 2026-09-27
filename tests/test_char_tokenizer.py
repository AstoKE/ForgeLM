import unicodedata

import pytest

from forgelm.tokenizer import UNK, UNK_ID, CharTokenizer, analyze


def test_vocab_is_sorted_unique_chars_with_unk_first():
    tok = CharTokenizer.from_corpus("banana")

    assert tok.itos == [UNK, "a", "b", "n"]
    assert tok.vocab_size == 4


def test_ids_are_deterministic_regardless_of_char_order():
    assert CharTokenizer.from_corpus("abc").itos == CharTokenizer.from_corpus("cba").itos


def test_roundtrip_on_known_text():
    text = "hello world\n"
    tok = CharTokenizer.from_corpus(text)

    assert tok.decode(tok.encode(text)) == text


def test_encode_gives_one_id_per_char():
    tok = CharTokenizer.from_corpus("banana")

    assert tok.encode("nab") == [3, 1, 2]


def test_unknown_chars_map_to_unk_and_break_roundtrip():
    tok = CharTokenizer.from_corpus("abc")

    assert tok.encode("abz") == [1, 2, UNK_ID]
    assert tok.tokens("abz") == ["a", "b", UNK]
    assert tok.decode(tok.encode("abz")) == "ab<unk>"  # information is lost


def test_empty_text():
    tok = CharTokenizer.from_corpus("abc")

    assert tok.encode("") == []
    assert tok.decode([]) == ""


@pytest.mark.parametrize("bad_id", [-1, 4, 999])
def test_decode_rejects_ids_outside_vocab(bad_id):
    tok = CharTokenizer.from_corpus("abc")  # ids 0..3

    with pytest.raises(ValueError, match="outside vocab"):
        tok.decode([bad_id])


def test_vocab_must_start_with_unk():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "b"])


def test_tokens_are_code_points_not_visible_characters():
    # One thumbs-up with a skin tone looks like 1 character but is 2 code points.
    assert len(CharTokenizer.from_corpus("👍🏽").encode("👍🏽")) == 2

    # "é" can be stored as 1 code point (NFC) or as "e" + combining accent (NFD).
    composed = unicodedata.normalize("NFC", "é")
    decomposed = unicodedata.normalize("NFD", "é")
    assert len(CharTokenizer.from_corpus(composed).encode(composed)) == 1
    assert len(CharTokenizer.from_corpus(decomposed).encode(decomposed)) == 2


def test_analyze_reports_stats():
    tok = CharTokenizer.from_corpus("hello")
    result = analyze(tok, "hello!")

    assert result.num_chars == 6
    assert result.num_tokens == 6
    assert result.vocab_size == 5  # <unk>, e, h, l, o
    assert result.unknown_count == 1
    assert result.chars_per_token == 1.0
    assert result.roundtrip_ok is False


def test_analyze_empty_text_does_not_divide_by_zero():
    result = analyze(CharTokenizer.from_corpus("abc"), "")

    assert result.chars_per_token == 0.0
    assert result.roundtrip_ok is True
