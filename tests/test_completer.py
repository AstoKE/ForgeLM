import pytest

torch = pytest.importorskip("torch")

from forgelm.data import build_math_dataset, parse_problems  # noqa: E402
from forgelm.eval import greedy_completer, math_accuracy  # noqa: E402
from forgelm.models.minigpt import MiniGPT  # noqa: E402
from forgelm.models.train_minigpt import train_minigpt_on_text  # noqa: E402
from forgelm.tokenizer import CharTokenizer  # noqa: E402

ALPHABET = "0123456789+=\n"


def untrained(seed=0) -> tuple[MiniGPT, CharTokenizer]:
    tokenizer = CharTokenizer.from_corpus(ALPHABET)
    model = MiniGPT(tokenizer.vocab_size, 16, 2, 1, 16, seed=seed)
    return model, tokenizer


# --- MiniGPT.generate(stop_id=...) ------------------------------------------------------


def test_generation_stops_right_after_the_stop_token():
    model, _ = untrained()
    first = model.generate([5], 1, temperature=0)[-1]  # whatever greedy writes first

    ids = model.generate([5], 10, temperature=0, stop_id=first)

    assert ids == [5, first]  # one step, not ten; the stop token itself is kept


def test_generation_without_a_stop_token_runs_to_the_limit():
    model, _ = untrained()

    assert len(model.generate([5], 10, temperature=0)) == 11


def test_a_stop_token_that_never_comes_runs_to_the_limit():
    model, tokenizer = untrained()
    unreachable = tokenizer.vocab_size + 5  # no such token, so greedy can never produce it

    assert len(model.generate([5], 10, temperature=0, stop_id=unreachable)) == 11


# --- greedy_completer -------------------------------------------------------------------


def test_the_completer_returns_text_cut_at_the_end_of_the_line():
    model, tokenizer = untrained()

    completion = greedy_completer(model, tokenizer)("3+4=")

    assert isinstance(completion, str)
    assert "\n" not in completion


def test_the_completer_is_deterministic():
    # An exam must not depend on a lucky sample, so it decodes greedily.
    model, tokenizer = untrained()
    complete = greedy_completer(model, tokenizer)

    assert complete("3+4=") == complete("3+4=")


def test_the_completer_respects_the_length_limit():
    model, tokenizer = untrained()

    assert len(greedy_completer(model, tokenizer, max_new_tokens=3)("3+4=")) <= 3


def test_the_completer_works_when_the_vocabulary_has_no_newline():
    # "\n" would encode to <unk>, which is no stop signal; the completer must not use it.
    tokenizer = CharTokenizer.from_corpus("0123456789+=")
    model = MiniGPT(tokenizer.vocab_size, 16, 2, 1, 16)

    completion = greedy_completer(model, tokenizer, max_new_tokens=4)("3+4=")

    assert len(completion) <= 4


# --- the whole chain: data -> training -> exam -----------------------------------------


def test_a_model_trained_on_addition_is_graded_correctly_by_the_exam():
    """Data, training and grading, end to end, on 1-digit sums.

    There are only 100 one-digit problems, so a small model *memorises* them. That makes
    this a test of the exam's two scores: a memoriser gets (almost) everything it was shown
    right and (almost) nothing it was not. Seen high and hidden low is exactly the signature
    the control group exists to reveal.
    """
    data = build_math_dataset(
        digits=1, holdout_fraction=0.2, lines=8000, pad=True, seed=0, seen_sample=100
    )
    model, tokenizer, _ = train_minigpt_on_text(
        data.corpus,
        steps=700,
        eval_every=700,
        batch_size=32,
        block_size=16,
        embed_dim=32,
        num_heads=2,
        num_blocks=2,
        lr=3e-3,
        device="cpu",
        seed=0,
    )
    complete = greedy_completer(model, tokenizer)
    untrained_model = MiniGPT(tokenizer.vocab_size, 32, 2, 2, 16, seed=0)

    seen = math_accuracy(parse_problems(data.seen), complete)
    hidden = math_accuracy(parse_problems(data.holdout), complete)
    before = math_accuracy(parse_problems(data.seen), greedy_completer(untrained_model, tokenizer))

    assert before.accuracy < 0.1  # it could not do it before training
    assert seen.accuracy > 0.9  # and it knows what it was shown
    assert hidden.accuracy < 0.5  # but a memoriser does not know what it was not
    assert seen.accuracy - hidden.accuracy > 0.4


# --- the prefix: asking a tagged model inside its own context ------------------------------


class Recorder:
    """A stand-in model that remembers what it was asked and then writes `reply`."""

    def __init__(self, reply_ids):
        self.reply_ids = reply_ids
        self.asked = []

    def generate(self, ids, max_new_tokens, temperature=1.0, stop_id=None, **_):
        self.asked.append(list(ids))
        return list(ids) + self.reply_ids


def test_the_prefix_is_given_to_the_model_in_front_of_the_prompt():
    tokenizer = CharTokenizer.from_corpus("<|math|>\n0123456789+=")
    model = Recorder(tokenizer.encode("7"))

    greedy_completer(model, tokenizer, prefix="<|math|>\n")("3+4=")

    assert tokenizer.decode(model.asked[0]) == "<|math|>\n3+4="


def test_the_prefix_is_not_part_of_what_comes_back():
    # The grader compares the completion to the answer; a leaked prefix would fail everything.
    tokenizer = CharTokenizer.from_corpus("<|math|>\n0123456789+=")
    model = Recorder(tokenizer.encode("7\n"))

    assert greedy_completer(model, tokenizer, prefix="<|math|>\n")("3+4=") == "7"


def test_without_a_prefix_nothing_is_added():
    tokenizer = CharTokenizer.from_corpus("0123456789+=\n")
    model = Recorder(tokenizer.encode("7"))

    greedy_completer(model, tokenizer)("3+4=")

    assert tokenizer.decode(model.asked[0]) == "3+4="
