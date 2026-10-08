import pytest

from forgelm.eval import (
    code_parse_score,
    grade_code,
    parsing_prefix_fraction,
    real_snippets,
    scramble,
)

VALID = """import os


def join(a, b):
    path = os.path.join(a, b)
    return path


class Box:
    def __init__(self, items):
        self.items = list(items)
"""


# --- checking the grader against code whose answer we know ---------------------------------


def test_valid_python_scores_one():
    assert parsing_prefix_fraction(VALID) == 1.0


def test_a_sample_cut_off_mid_statement_still_gets_credit_for_what_came_before():
    cut = VALID + "    def broken(self, x"  # an unfinished last line, as generation leaves it

    score = parsing_prefix_fraction(cut)

    assert 0.8 < score < 1.0  # almost all of the lines are fine; the last one is not


def test_code_that_is_broken_from_the_start_scores_zero():
    assert parsing_prefix_fraction("def (:\n    x = = 1\n") == 0.0


def test_the_score_is_the_share_of_code_lines_before_the_first_error():
    # Four statements, the fourth is a syntax error: three of four lines are valid.
    code = "a = 1\nb = 2\nc = 3\nd = = 4\n"

    assert parsing_prefix_fraction(code) == pytest.approx(0.75)


def test_a_later_line_that_closes_an_open_bracket_is_found():
    # Prefix validity is not monotonic: the first two lines alone do not parse, all four do.
    code = "x = [\n    1,\n    2,\n]\n"

    assert parsing_prefix_fraction(code) == 1.0


def test_comments_and_blank_lines_do_not_count_as_code():
    # A model that only wrote comments would "parse" perfectly and have shown nothing.
    assert parsing_prefix_fraction("# ----\n# ----\n\n# ----\n") == 0.0
    assert parsing_prefix_fraction("") == 0.0
    assert parsing_prefix_fraction("\n\n\n") == 0.0


def test_comment_lines_do_not_dilute_the_score_either():
    code = "# header\na = 1\n# note\nb = 2\n"

    assert parsing_prefix_fraction(code) == 1.0


def test_inconsistent_indentation_is_caught():
    assert parsing_prefix_fraction("def f():\n        x = 1\n    y = 2\n") < 1.0


def test_an_unclosed_docstring_costs_the_lines_inside_it():
    code = 'a = 1\nb = 2\n"""never closed\nstill talking\n'

    assert parsing_prefix_fraction(code) < 1.0


def test_odd_input_never_crashes_the_grader():
    for text in ["\x00\x00", "(" * 5000, "a" * 100_000, "\t\t\t", "\\", "é = 1"]:
        assert 0.0 <= parsing_prefix_fraction(text) <= 1.0


def test_a_very_long_sample_is_capped_instead_of_taking_forever():
    code = "x = 1\n" * 100_000

    assert parsing_prefix_fraction(code) == 1.0  # returns promptly: only the first lines count


def test_the_score_of_several_samples_is_their_mean():
    assert code_parse_score([VALID, "def (:\n"]) == pytest.approx(0.5)


def test_scoring_nothing_is_an_error():
    with pytest.raises(ValueError, match="no samples"):
        code_parse_score([])


# --- the two reference points ---------------------------------------------------------------


def test_scrambling_keeps_the_characters_and_destroys_the_structure():
    scrambled = scramble(VALID, seed=0)

    assert sorted(scrambled) == sorted(VALID)  # the same characters
    assert scrambled != VALID
    assert parsing_prefix_fraction(scrambled) < 0.2  # no longer Python


def test_scrambling_is_reproducible_and_the_seed_changes_it():
    assert scramble(VALID, 1) == scramble(VALID, 1)
    assert scramble(VALID, 1) != scramble(VALID, 2)


def test_real_snippets_are_the_start_of_real_files_cut_to_length():
    files = [f"import os{i}\n" + "x = 1\n" * 100 for i in range(10)]

    snippets = real_snippets(files, length=40, count=4, seed=0)

    assert len(snippets) == 4
    assert all(len(s) == 40 for s in snippets)
    assert all(any(f.startswith(s) for f in files) for s in snippets)


def test_asking_for_more_snippets_than_files_gives_all_of_them():
    assert len(real_snippets(["a = 1\n" * 20] * 3 + ["b = 2\n" * 20], 10, count=50)) == 4


def test_snippets_need_files():
    with pytest.raises(ValueError, match="no files"):
        real_snippets([], 10, 3)


def test_a_grade_puts_the_model_between_a_floor_and_a_ceiling():
    real = [VALID] * 6
    perfect_model = [VALID] * 6
    useless_model = [scramble(VALID, i) for i in range(6)]

    best = grade_code(perfect_model, real)
    worst = grade_code(useless_model, real)

    assert best.ceiling == 1.0
    assert best.floor < 0.2
    assert best.position == pytest.approx(1.0)  # as good as real code
    assert worst.position == pytest.approx(0.0, abs=0.15)  # as good as shuffled characters
    assert best.position > worst.position


def test_the_position_is_zero_rather_than_a_division_by_zero_when_there_is_no_span():
    from forgelm.eval import CodeReport

    assert CodeReport(generated=0.5, ceiling=0.3, floor=0.3).position == 0.0
