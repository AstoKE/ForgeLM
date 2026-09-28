import json
import math
import random

import pytest

from forgelm.models import BigramModel, load_checkpoint, save_checkpoint, softmax, train_on_text
from forgelm.tokenizer import CharTokenizer

# CharTokenizer.from_corpus("ab") -> ["<unk>", "a", "b"], so a=1, b=2, V=3
UNK, A, B = 0, 1, 2
V = 3


def test_train_counts_adjacent_pairs():
    model = BigramModel.train([A, B, A, B], vocab_size=V)

    assert model.counts[A] == [0, 0, 2]  # a -> b twice
    assert model.counts[B] == [0, 1, 0]  # b -> a once (last b has no successor)
    assert model.counts[UNK] == [0, 0, 0]


def test_probs_with_add_one_smoothing():
    model = BigramModel.train([A, B, A, B], vocab_size=V, smoothing=1.0)

    # row a: counts [0, 0, 2] + 1 each = [1, 1, 3], total 5
    assert model.probs(A) == pytest.approx([1 / 5, 1 / 5, 3 / 5])


@pytest.mark.parametrize("smoothing", [0.0, 0.5, 1.0])
def test_every_row_is_a_probability_distribution(smoothing):
    model = BigramModel.train([A, B, A, B, B], vocab_size=V, smoothing=smoothing)

    for prev in range(V):
        assert sum(model.probs(prev)) == pytest.approx(1.0)


def test_unseen_row_without_smoothing_is_uniform():
    model = BigramModel.train([A, B], vocab_size=V, smoothing=0.0)

    assert model.probs(UNK) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_uniform_model_loss_is_log_vocab_size():
    # No data + smoothing = every next token equally likely = the baseline.
    uniform = BigramModel([[0] * V for _ in range(V)], smoothing=1.0)

    assert uniform.loss([A, B, A, UNK]) == pytest.approx(math.log(V))


def test_trained_model_beats_baseline():
    ids = [A, B] * 50
    model = BigramModel.train(ids, vocab_size=V)

    assert model.loss(ids) < math.log(V)


def test_unseen_pair_without_smoothing_gives_infinite_loss():
    model = BigramModel.train([A, B, A, B], vocab_size=V, smoothing=0.0)

    assert model.loss([A, A]) == math.inf  # "a a" never seen -> P = 0 -> -log 0


def test_loss_needs_two_tokens():
    with pytest.raises(ValueError):
        BigramModel.train([A, B], vocab_size=V).loss([A])


def test_negative_smoothing_rejected():
    with pytest.raises(ValueError):
        BigramModel.train([A, B], vocab_size=V, smoothing=-1)


def test_softmax_sums_to_one_and_keeps_order():
    probs = softmax([1.0, 2.0, 3.0])

    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] < probs[1] < probs[2]


def test_softmax_is_stable_for_large_logits():
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])  # naive exp(1000) overflows


def test_softmax_of_logits_gives_back_probs():
    model = BigramModel.train([A, B, A, B, B], vocab_size=V)

    assert softmax(model.logits(A)) == pytest.approx(model.probs(A))


def test_temperature_sharpens_or_flattens():
    logits = [1.0, 2.0, 3.0]

    assert max(softmax(logits, 0.5)) > max(softmax(logits)) > max(softmax(logits, 5.0))


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_softmax_rejects_non_positive_temperature(temperature):
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], temperature)


def test_generate_keeps_prompt_and_adds_tokens():
    model = BigramModel.train([A, B] * 10, vocab_size=V)
    out = model.generate([A, B], max_new_tokens=5, rng=random.Random(0))

    assert out[:2] == [A, B]
    assert len(out) == 7


def test_generate_is_reproducible_with_same_seed():
    model = BigramModel.train([A, B, B, A, A, B] * 10, vocab_size=V)

    first = model.generate([A], 50, random.Random(42))
    second = model.generate([A], 50, random.Random(42))
    assert first == second


def test_greedy_generation_follows_most_likely_token():
    model = BigramModel.train([A, B] * 10, vocab_size=V)

    assert model.generate([A], 4, random.Random(0), temperature=0) == [A, B, A, B, A]


def test_generate_rejects_empty_prompt():
    with pytest.raises(ValueError):
        BigramModel.train([A, B], vocab_size=V).generate([], 5, random.Random(0))


def test_train_on_text_reports_losses():
    text = "the cat sat on the mat. " * 40
    report = train_on_text(text, val_fraction=0.1)

    assert report.train_tokens + report.val_tokens == len(text)
    assert report.baseline_loss == pytest.approx(math.log(report.tokenizer.vocab_size))
    assert report.train_loss < report.baseline_loss
    assert report.val_loss < report.baseline_loss


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.5])
def test_train_on_text_rejects_bad_val_fraction(val_fraction):
    with pytest.raises(ValueError):
        train_on_text("some text here", val_fraction=val_fraction)


def test_train_on_text_rejects_tiny_corpus():
    with pytest.raises(ValueError, match="too short"):
        train_on_text("ab")


def test_checkpoint_roundtrip(tmp_path):
    report = train_on_text("hello world, hello there. " * 10)
    path = tmp_path / "sub" / "model.json"  # parent dir is created on save

    save_checkpoint(path, report.model, report.tokenizer)
    model, tokenizer = load_checkpoint(path)

    assert tokenizer.itos == report.tokenizer.itos
    assert model.counts == report.model.counts
    assert model.probs(1) == report.model.probs(1)


def test_checkpoint_is_plain_json(tmp_path):
    report = train_on_text("abcabcabc abc")
    path = tmp_path / "model.json"
    save_checkpoint(path, report.model, report.tokenizer)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format_version"] == 1
    assert data["tokenizer"]["type"] == "char"
    assert data["model"]["type"] == "bigram"


def test_load_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"format_version": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)


def test_char_tokenizer_dict_roundtrip():
    tok = CharTokenizer.from_corpus("hello")

    assert CharTokenizer.from_dict(tok.to_dict()).itos == tok.itos


def test_char_tokenizer_from_dict_rejects_other_types():
    with pytest.raises(ValueError):
        CharTokenizer.from_dict({"type": "bpe", "merges": []})
