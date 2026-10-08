import pytest

from forgelm.data import (
    FILE_MARKER,
    build_code_corpus,
    clean_python,
    collect_python,
    default_roots,
    split_files,
)

GOOD = "import os\n\n\ndef join(a, b):\n    return os.path.join(a, b)\n" * 6


# --- cleaning ---------------------------------------------------------------------------


def test_ordinary_python_is_kept_and_ends_with_exactly_one_newline():
    cleaned = clean_python("\n\n" + GOOD + "\n\n\n")

    assert cleaned is not None
    assert cleaned.startswith("import os")
    assert cleaned.endswith("\n")
    assert not cleaned.endswith("\n\n")


def test_windows_line_endings_become_unix_ones():
    cleaned = clean_python(GOOD.replace("\n", "\r\n"))

    assert "\r" not in cleaned
    assert cleaned == clean_python(GOOD)  # the same file, however it was saved


def test_a_few_stray_non_ascii_characters_are_removed_not_the_whole_file():
    cleaned = clean_python(GOOD + "# naïve café\n")

    assert cleaned is not None
    assert cleaned.isascii()
    assert "na" in cleaned  # the rest of the comment survives


def test_a_mostly_non_ascii_file_is_left_out():
    # A model with one embedding row per character should not spend rows on this.
    assert clean_python("中文" * 200) is None


@pytest.mark.parametrize("size", [10, 150])
def test_a_tiny_file_is_left_out(size):
    assert clean_python("a" * size, min_chars=200) is None


def test_a_huge_file_is_left_out():
    assert clean_python("x = 1\n" * 50_000, max_chars=100_000) is None


def test_a_file_that_contains_the_marker_is_left_out():
    # The marker is how files are told apart again, so it must never appear inside one.
    assert clean_python(GOOD + f"{FILE_MARKER}\n") is None


# --- collecting ---------------------------------------------------------------------------


def make_tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(GOOD, encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text(GOOD.replace("join", "link"), encoding="utf-8")
    (tmp_path / "pkg" / "copy_of_a.py").write_text(GOOD, encoding="utf-8")  # a duplicate
    (tmp_path / "pkg" / "notes.txt").write_text(GOOD, encoding="utf-8")  # not Python
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text(GOOD + "# junk\n", encoding="utf-8")
    (tmp_path / "site-packages").mkdir()
    (tmp_path / "site-packages" / "other.py").write_text(GOOD + "# other\n", encoding="utf-8")
    return tmp_path


def test_collecting_keeps_each_distinct_file_once(tmp_path):
    files = collect_python([make_tree(tmp_path)])

    # a.py and copy_of_a.py are identical, so one of them goes; b.py differs
    assert len(files) == 2


def test_collecting_skips_caches_site_packages_and_non_python_files(tmp_path):
    files = collect_python([make_tree(tmp_path)])

    assert not any("# junk" in text or "# other" in text for text in files)


def test_collecting_is_deterministic(tmp_path):
    root = make_tree(tmp_path)

    assert collect_python([root]) == collect_python([root])


def test_a_file_that_is_not_utf8_is_skipped_not_fatal(tmp_path):
    (tmp_path / "latin.py").write_bytes(b"x = '\xe9'\n" * 100)
    (tmp_path / "ok.py").write_text(GOOD, encoding="utf-8")

    assert len(collect_python([tmp_path])) == 1


def test_a_missing_root_yields_nothing(tmp_path):
    assert collect_python([tmp_path / "nowhere"]) == []


def test_the_default_roots_include_the_standard_library():
    roots = default_roots()

    assert roots
    assert any((root / "os.py").is_file() for root in roots)


# --- building the corpus ------------------------------------------------------------------


def files(n=10):
    return [f"def f{i}():\n    return {i}\n" * 10 for i in range(n)]


def test_every_file_starts_with_the_marker():
    corpus = build_code_corpus(files(5))

    assert corpus.count(FILE_MARKER) == 5
    assert corpus.startswith(FILE_MARKER)


def test_the_corpus_can_be_cut_back_into_the_same_files():
    originals = files(8)

    recovered = split_files(build_code_corpus(originals, seed=3))

    assert sorted(recovered) == sorted(originals)


def test_files_are_shuffled_so_the_validation_tail_is_not_the_last_directory():
    originals = files(30)

    corpus_files = split_files(build_code_corpus(originals, seed=0))

    assert corpus_files != originals  # not in the order they were collected


def test_the_shuffle_is_reproducible_and_the_seed_changes_it():
    originals = files(30)

    assert build_code_corpus(originals, seed=1) == build_code_corpus(originals, seed=1)
    assert build_code_corpus(originals, seed=1) != build_code_corpus(originals, seed=2)


def test_the_size_limit_stops_between_files_never_inside_one():
    originals = files(30)
    one = len(build_code_corpus(originals[:1]))

    corpus = build_code_corpus(originals, max_chars=one * 5, seed=0)

    assert len(corpus) <= one * 5 + one  # roughly five files
    assert len(split_files(corpus)) < 30
    assert all(text in originals for text in split_files(corpus))  # every kept file is whole


def test_at_least_one_file_is_kept_even_if_it_alone_exceeds_the_limit():
    assert len(split_files(build_code_corpus(files(3), max_chars=1))) == 1


def test_a_corpus_needs_files():
    with pytest.raises(ValueError, match="no files"):
        build_code_corpus([])
