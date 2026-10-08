import pytest

from forgelm.data import all_problems, make_problem, needs_carry
from forgelm.eval import CARRY, FULL, NO_CARRY, SHORT, math_accuracy

PROBLEMS = all_problems(digits=2)  # 10,000 problems; 6,975 of them need a carry


def perfect(prompt: str) -> str:
    """A fake model that always knows the answer: it must score exactly 100%."""
    left = prompt.removesuffix("=")
    a, b = left.split("+")
    return str(int(a) + int(b))


def operands(prompt: str) -> tuple[int, int]:
    a, b = prompt.removesuffix("=").split("+")
    return int(a), int(b)


# --- checking the measuring instrument against known answers --------------------------


def test_a_perfect_model_scores_one_hundred_percent():
    report = math_accuracy(PROBLEMS, perfect)

    assert report.total == 10_000
    assert report.correct == 10_000
    assert report.accuracy == 1.0
    assert report.misses == []
    assert all(group.accuracy == 1.0 for group in report.groups.values())


def test_a_useless_model_scores_zero():
    report = math_accuracy(PROBLEMS, lambda prompt: "")

    assert report.correct == 0
    assert report.accuracy == 0.0
    assert len(report.misses) == 10  # capped, not 10,000 of them
    assert all(group.accuracy == 0.0 for group in report.groups.values())


def test_a_model_that_only_fails_on_carries_is_caught_by_the_carry_groups():
    # Overall accuracy would just say "about 30%". The groups say *where* it fails.
    def forgets_the_carry(prompt: str) -> str:
        return "0" if needs_carry(*operands(prompt)) else perfect(prompt)

    report = math_accuracy(PROBLEMS, forgets_the_carry)

    assert report.score(NO_CARRY) == 1.0
    assert report.score(CARRY) == 0.0
    assert report.accuracy == pytest.approx(1 - 6975 / 10_000)  # only the carry-free ones pass


def test_a_model_that_only_fails_on_short_operands_is_not_blamed_on_carries():
    """The mistake this grader once made, written as a test.

    The first model we graded scored 96% on two-digit + two-digit and 1% on anything with a
    one-digit operand. A carry-only breakdown showed "no carry: 58%, carry: 86%", which reads
    as "bad at the absence of a carry". Short operands rarely carry, so that was a confound.
    Crossing the two questions shows the truth: carries are fine, short operands are not.
    """

    def breaks_on_short_operands(prompt: str) -> str:
        a, b = operands(prompt)
        return "" if min(a, b) < 10 else perfect(prompt)

    report = math_accuracy(PROBLEMS, breaks_on_short_operands)

    assert report.score(FULL) == 1.0
    assert report.score(SHORT) == 0.0
    # Inside full-length problems the carry makes no difference, which is the real finding:
    assert report.score(f"{FULL}, {CARRY}") == 1.0
    assert report.score(f"{FULL}, {NO_CARRY}") == 1.0
    # ...while the naive carry-only view still shows a misleading gap:
    assert report.score(NO_CARRY) < report.score(CARRY)


def test_every_problem_lands_in_exactly_one_group_of_each_kind():
    report = math_accuracy(PROBLEMS, perfect)

    groups = report.groups
    assert groups[CARRY].total + groups[NO_CARRY].total == report.total
    assert groups[FULL].total + groups[SHORT].total == report.total
    assert sum(groups[f"{s}, {c}"].total for s in (FULL, SHORT) for c in (CARRY, NO_CARRY)) == (
        report.total
    )


def test_the_answer_must_match_exactly_not_merely_contain_the_right_number():
    problems = [make_problem(49, 97)]

    assert math_accuracy(problems, lambda p: "146").correct == 1
    assert math_accuracy(problems, lambda p: "1465").correct == 0  # does not stop
    assert math_accuracy(problems, lambda p: "146 ").correct == 0  # stray space
    assert math_accuracy(problems, lambda p: " 146").correct == 0
    assert math_accuracy(problems, lambda p: "14").correct == 0  # stops too early
    assert math_accuracy(problems, lambda p: "0146").correct == 0  # leading zero


def test_reversed_answers_are_graded_as_written():
    problems = [make_problem(49, 97, reverse=True)]  # the expected text is "641"

    assert math_accuracy(problems, lambda p: "641").correct == 1
    assert math_accuracy(problems, lambda p: "146").correct == 0  # the right sum, wrongly written


def test_padded_problems_are_graded_against_the_padded_answer():
    problems = [make_problem(5, 12, pad=2)]  # "05+12=017"

    assert math_accuracy(problems, lambda p: "017").correct == 1
    assert math_accuracy(problems, lambda p: "17").correct == 0  # right number, wrong format


def test_the_completer_is_given_the_prompt_as_written_including_padding():
    seen = []

    math_accuracy([make_problem(5, 12, pad=2)], lambda prompt: seen.append(prompt) or "")

    assert seen == ["05+12="]


def test_the_misses_record_what_was_expected_and_what_came_out():
    report = math_accuracy([make_problem(1, 2)], lambda p: "4")

    miss = report.misses[0]
    assert (miss.prompt, miss.expected, miss.got) == ("1+2=", "3", "4")


def test_how_many_misses_are_kept_can_be_chosen():
    report = math_accuracy(PROBLEMS, lambda p: "x", max_misses=3)

    assert len(report.misses) == 3


def test_a_group_with_no_problems_has_no_score_instead_of_a_misleading_zero():
    easy = [make_problem(10, 20), make_problem(11, 22)]  # full length, no carries

    report = math_accuracy(easy, perfect)

    assert report.score(NO_CARRY) == 1.0
    assert report.score(CARRY) is None  # "0% on carries" would claim the model failed them
    assert report.score(SHORT) is None
    assert report.score("a group nobody defined") is None


def test_grading_nothing_is_an_error_not_a_perfect_score():
    with pytest.raises(ValueError, match="no problems"):
        math_accuracy([], perfect)


def test_the_completer_sees_only_the_prompt_never_the_answer():
    seen = []

    math_accuracy([make_problem(49, 97)], lambda prompt: seen.append(prompt) or "")

    assert seen == ["49+97="]  # a grader that leaked the answer would measure nothing
