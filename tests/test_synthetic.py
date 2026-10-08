import pytest

from forgelm.data import (
    all_problems,
    make_corpus,
    make_problem,
    needs_carry,
    parse_problem,
    parse_problems,
    render,
    split_problems,
)


def unordered(problem):
    return (min(problem.a, problem.b), max(problem.a, problem.b))


def test_every_two_digit_sum_is_generated_in_both_orders():
    problems = all_problems(digits=2)

    assert len(problems) == 100 * 100
    assert make_problem(49, 97).line == "49+97=146"
    assert {p.line for p in problems} >= {"0+0=0", "99+99=198", "5+93=98", "93+5=98"}


def test_the_answer_can_be_written_backwards():
    assert make_problem(49, 97, reverse=True).line == "49+97=641"
    assert make_problem(5, 5, reverse=True).line == "5+5=01"  # 10 written backwards is 01
    assert make_problem(1, 2, reverse=True).answer == "3"  # a single digit is its own reverse


def test_the_prompt_is_everything_up_to_and_including_the_equals_sign():
    problem = make_problem(49, 97)

    assert problem.prompt == "49+97="
    assert problem.prompt + problem.answer == problem.line


@pytest.mark.parametrize("digits", [0, 4, -1])
def test_an_unsupported_number_of_digits_is_refused(digits):
    with pytest.raises(ValueError, match="digits"):
        all_problems(digits=digits)


# --- the hidden exam ------------------------------------------------------------------


def test_the_train_and_holdout_sets_share_no_problem():
    train, holdout = split_problems(all_problems(2), holdout_fraction=0.1, seed=0)

    assert {unordered(p) for p in train}.isdisjoint({unordered(p) for p in holdout})
    assert len(train) + len(holdout) == 100 * 100


def test_a_held_out_pair_is_hidden_in_both_orders():
    # If 37+58 were held out but 58+37 trained on, "addition is commutative" would pass the
    # exam without any adding. This is the leak the unordered split exists to close.
    train, holdout = split_problems(all_problems(2), holdout_fraction=0.1, seed=0)
    trained = {(p.a, p.b) for p in train}

    for problem in holdout:
        assert (problem.a, problem.b) not in trained
        assert (problem.b, problem.a) not in trained


def test_about_the_requested_share_is_held_out():
    _, holdout = split_problems(all_problems(2), holdout_fraction=0.1, seed=0)

    assert 0.08 < len(holdout) / (100 * 100) < 0.12


def test_the_split_is_reproducible_and_the_seed_changes_it():
    problems = all_problems(1)

    first = split_problems(problems, 0.2, seed=3)
    again = split_problems(problems, 0.2, seed=3)
    other = split_problems(problems, 0.2, seed=4)

    assert first == again
    assert first[1] != other[1]


def test_both_sides_are_never_empty_even_for_an_extreme_fraction():
    train, holdout = split_problems(all_problems(1), holdout_fraction=0.001)
    assert train
    assert holdout

    train, holdout = split_problems(all_problems(1), holdout_fraction=0.999)
    assert train
    assert holdout


def test_bad_split_requests_are_refused():
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_problems(all_problems(1), holdout_fraction=0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_problems(all_problems(1), holdout_fraction=1)

    with pytest.raises(ValueError, match="at least two"):
        split_problems([make_problem(1, 1)], holdout_fraction=0.5)


# --- turning problems into text and back ----------------------------------------------


def test_render_and_parse_are_inverses():
    problems = [make_problem(49, 97), make_problem(3, 4), make_problem(0, 0)]

    assert render(problems) == "49+97=146\n3+4=7\n0+0=0\n"
    assert parse_problems(render(problems)) == problems


def test_a_reversed_answer_is_kept_exactly_as_written():
    # The parser must not "fix" the answer: the grader compares text to text.
    problem = parse_problem("49+97=641")

    assert (problem.a, problem.b, problem.answer) == (49, 97, "641")


@pytest.mark.parametrize("line", ["", "hello", "49+97", "49=146", "a+b=3", "49+=3", "+3=3"])
def test_garbage_is_not_a_problem(line):
    with pytest.raises(ValueError, match="not a problem line"):
        parse_problem(line)


def test_blank_lines_between_problems_are_ignored():
    assert len(parse_problems("1+1=2\n\n   \n2+2=4\n")) == 2


def test_the_corpus_has_the_requested_number_of_lines_and_only_training_problems():
    train, holdout = split_problems(all_problems(2), 0.1, seed=0)

    text = make_corpus(train, lines=500, seed=0)

    lines = text.splitlines()
    assert len(lines) == 500
    hidden = {p.line for p in holdout}
    assert not hidden & set(lines)  # not one hidden problem leaked into the training text


def test_the_corpus_is_reproducible():
    train, _ = split_problems(all_problems(1), 0.2)

    assert make_corpus(train, 50, seed=1) == make_corpus(train, 50, seed=1)
    assert make_corpus(train, 50, seed=1) != make_corpus(train, 50, seed=2)


def test_a_corpus_needs_something_to_draw_from():
    with pytest.raises(ValueError, match="no training problems"):
        make_corpus([], lines=5)

    with pytest.raises(ValueError, match="lines"):
        make_corpus([make_problem(1, 1)], lines=0)


# --- carries ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "carries"),
    [
        (12, 34, False),
        (49, 97, True),  # 9 + 7 = 16
        (5, 5, True),  # exactly 10 still carries
        (99, 1, True),
        (0, 0, False),
        (123, 456, False),
        (45, 55, True),
        (50, 49, False),  # 99 with no column reaching 10
        (95, 5, True),  # 5 + 5 carries; the next column then takes a 1
    ],
)
def test_needs_carry(a, b, carries):
    assert needs_carry(a, b) is carries


# --- zero-padding ---------------------------------------------------------------------


def test_padding_writes_every_number_with_the_same_width():
    assert make_problem(5, 12, pad=2).line == "05+12=017"
    assert make_problem(99, 99, pad=2).line == "99+99=198"
    assert make_problem(0, 0, pad=2).line == "00+00=000"
    assert make_problem(5, 12).line == "5+12=17"  # unpadded is unchanged


def test_padding_and_reversing_combine_in_that_order():
    # Pad first, then reverse: the answer "017" is written "710".
    assert make_problem(5, 12, reverse=True, pad=2).line == "05+12=710"


def test_every_padded_problem_has_the_same_shape():
    problems = all_problems(digits=2, pad=True)

    assert {len(p.prompt) for p in problems} == {6}  # "dd+dd="
    assert {len(p.answer) for p in problems} == {3}  # always digits + 1


def test_padding_changes_the_text_but_not_the_numbers():
    padded = make_problem(5, 12, pad=2)

    assert (padded.a, padded.b) == (5, 12)
    assert padded.prompt == "05+12="


def test_a_padded_line_survives_a_round_trip_through_text():
    problem = make_problem(5, 12, pad=2)

    parsed = parse_problem(problem.line)

    assert parsed == problem
    assert parsed.prompt == "05+12="  # the zero in front is kept, the grader needs it


def test_the_hidden_exam_is_split_the_same_way_with_padding():
    train, holdout = split_problems(all_problems(2, pad=True), 0.1, seed=0)

    assert {unordered(p) for p in train}.isdisjoint({unordered(p) for p in holdout})
    assert len(train) + len(holdout) == 100 * 100
