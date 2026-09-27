import pytest

from forgelm.tokenizer import BPETokenizer, analyze
from forgelm.tokenizer.bpe import get_pair_counts, merge

A, B, C, D = 97, 98, 99, 100  # UTF-8 bytes of "a", "b", "c", "d"


def test_get_pair_counts():
    assert get_pair_counts([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}
    assert get_pair_counts([1]) == {}


def test_merge_replaces_pairs_left_to_right_without_overlap():
    assert merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]  # "aaa": first two merge, last stays


def test_one_merge_on_classic_example():
    # "aaabdaaabac" pair counts: (a,a)=4, (a,b)=2, others 1 -> merge (a,a) into 256.
    # "aaab" becomes [aa, a, b] (non-overlapping), so 11 bytes -> 9 tokens.
    tok = BPETokenizer.train("aaabdaaabac", num_merges=1)

    assert tok.merges == {(A, A): 256}
    assert tok.encode("aaabdaaabac") == [256, A, B, D, 256, A, B, A, C]


def test_three_merges_build_on_each_other():
    # 1) (a,a)->256   2) (256,a)->257 i.e. "aaa"   3) (257,b)->258 i.e. "aaab"
    tok = BPETokenizer.train("aaabdaaabac", num_merges=3)

    assert tok.vocab_size == 256 + 3
    assert tok.tokens("aaabdaaabac") == ["aaab", "d", "aaab", "a", "c"]


def test_training_stops_when_no_pair_repeats():
    tok = BPETokenizer.train("abc", num_merges=10)

    assert tok.merges == {}
    assert tok.vocab_size == 256


def test_zero_merges_is_plain_utf8_bytes():
    tok = BPETokenizer.train("anything", num_merges=0)

    assert tok.encode("İ") == [0xC4, 0xB0]  # one character, two bytes


def test_negative_merges_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", num_merges=-1)


def test_training_is_deterministic():
    corpus = "the cat sat on the mat with the hat"
    assert BPETokenizer.train(corpus, 20).merges == BPETokenizer.train(corpus, 20).merges


def test_encode_applies_learned_merges_to_new_text():
    tok = BPETokenizer.train("aaabdaaabac", num_merges=1)

    assert tok.encode("aaa") == [256, A]


@pytest.mark.parametrize("text", ["İstanbul çok güzel", "👍🏽 ok", "", "\x00\n\t"])
def test_roundtrip_on_text_never_seen_in_training(text):
    tok = BPETokenizer.train("the quick brown fox jumps over the lazy dog " * 5, 30)

    assert tok.decode(tok.encode(text)) == text


def test_nothing_is_ever_unknown():
    tok = BPETokenizer.train("english only", 5)
    result = analyze(tok, "İstanbul 👍🏽")

    assert tok.unk_id is None
    assert result.unknown_count == 0
    assert result.roundtrip_ok is True


def test_merges_shorten_sequences():
    corpus = "the cat sat on the mat. the hat is on the cat. " * 10
    text = "the cat sat on the hat"

    zero = analyze(BPETokenizer.train(corpus, 0), text)
    many = analyze(BPETokenizer.train(corpus, 50), text)

    assert zero.chars_per_token == 1.0  # ASCII: 1 byte per char
    assert many.num_tokens < zero.num_tokens


def test_decode_half_a_character_gives_replacement_char():
    tok = BPETokenizer.train("", 0)

    assert tok.decode([0xC4]) == "�"  # first byte of "İ" alone isn't valid UTF-8
    assert tok.tokens("İ") == ["\\xc4", "\\xb0"]  # display form of the raw bytes


def test_decode_rejects_ids_outside_vocab():
    tok = BPETokenizer.train("", 0)

    with pytest.raises(ValueError, match="outside vocab"):
        tok.decode([256])
