import pytest

from forgelm.data import (
    FILE_MARKER,
    MATH_TAG,
    STORY_TAG,
    build_code_corpus,
    code_documents,
    math_documents,
    mix_documents,
    story_documents,
)

STORY = (
    "Once upon a time there was a little cat named Tom who loved to play with a red ball. "
    "One day the ball rolled away and Tom ran after it until the sun went down."
)


def stories(n):
    return [f"{STORY} Number {i}." for i in range(n)]


# --- turning each source into tagged documents ------------------------------------------


def test_stories_are_split_on_the_end_marker_and_tagged():
    text = f"{STORY} One.\n<|endoftext|>\n{STORY} Two.\n<|endoftext|>\n{STORY} Three."

    docs = story_documents(text)

    assert len(docs) == 3
    assert all(doc.startswith(f"{STORY_TAG}\n") for doc in docs)
    assert "<|endoftext|>" not in "".join(docs)  # the tag replaced it
    assert docs[0].endswith("One.\n")


def test_a_fragment_too_short_to_be_a_story_is_dropped():
    docs = story_documents(f"{STORY}\n<|endoftext|>\nok\n<|endoftext|>\n")

    assert len(docs) == 1


def test_code_documents_start_with_the_file_marker_and_keep_every_file():
    files = ["def f():\n    return 1\n" * 10, "class A:\n    pass\n" * 10]

    docs = code_documents(build_code_corpus(files, seed=0))

    assert len(docs) == 2
    assert all(doc.startswith(f"{FILE_MARKER}\n") for doc in docs)
    assert sorted(doc.removeprefix(f"{FILE_MARKER}\n").rstrip("\n") for doc in docs) == sorted(
        f.rstrip("\n") for f in files
    )


def test_math_lines_are_grouped_into_tagged_blocks():
    text = "".join(f"{i}+1={i + 1}\n" for i in range(45))

    docs = math_documents(text, per_document=20)

    assert len(docs) == 2  # 45 lines make two full blocks; the 5 left over are dropped
    assert all(doc.startswith(f"{MATH_TAG}\n") for doc in docs)
    assert all(len(doc.splitlines()) == 21 for doc in docs)  # the tag plus 20 sums


def test_blank_lines_are_not_counted_as_math():
    docs = math_documents("1+1=2\n\n\n2+2=4\n", per_document=2)

    assert len(docs) == 1


def test_a_math_block_must_hold_at_least_one_line():
    with pytest.raises(ValueError, match="per_document"):
        math_documents("1+1=2\n", per_document=0)


# --- the mix ----------------------------------------------------------------------------


def sources():
    return {
        "story": story_documents("\n<|endoftext|>\n".join(stories(40))),
        "code": code_documents(
            build_code_corpus([f"def f{i}():\n    return {i}\n" * 12 for i in range(40)])
        ),
        "math": math_documents("".join(f"{i}+{i}={2 * i}\n" for i in range(200)), 10),
    }


def test_a_mix_holds_documents_of_all_sources():
    mix = mix_documents(sources(), {"story": 600, "code": 600, "math": 600})

    assert STORY_TAG in mix.text
    assert FILE_MARKER in mix.text
    assert MATH_TAG in mix.text


def test_a_budget_is_a_ceiling_reached_by_whole_documents():
    mix = mix_documents(sources(), {"story": 600, "code": 600, "math": 600})

    for name in ("story", "code", "math"):
        assert mix.chars[name] <= 600
        assert mix.documents[name] >= 1


def test_the_reported_counts_match_the_text():
    mix = mix_documents(sources(), {"story": 700, "code": 900, "math": 500})

    assert mix.text.count(STORY_TAG) == mix.documents["story"]
    assert mix.text.count(FILE_MARKER) == mix.documents["code"]
    assert mix.text.count(MATH_TAG) == mix.documents["math"]
    assert len(mix.text) == sum(mix.chars.values())


def test_no_document_is_cut_in_half():
    mix = mix_documents(sources(), {"story": 700, "code": 900, "math": 500})

    # Every story that made it in is complete: it still ends with its own "Number N." line.
    for doc in mix.text.split(f"{STORY_TAG}\n")[1:]:
        first_block = doc.split(FILE_MARKER)[0].split(MATH_TAG)[0]
        assert first_block.rstrip().endswith(".")


def test_the_documents_are_interleaved_not_grouped_by_source():
    mix = mix_documents(sources(), {"story": 5000, "code": 5000, "math": 5000}, seed=0)

    tags = [line for line in mix.text.splitlines() if line in (STORY_TAG, MATH_TAG, FILE_MARKER)]

    assert tags != sorted(tags)  # not all stories, then all code, then all maths
    # The tail (the validation split of training) must hold more than one kind.
    assert len(set(tags[-10:])) > 1


def test_a_tiny_budget_still_takes_one_whole_document():
    mix = mix_documents(sources(), {"story": 1, "code": 1, "math": 1})

    assert mix.documents == {"story": 1, "code": 1, "math": 1}


def test_a_bigger_budget_takes_more_documents():
    small = mix_documents(sources(), {"story": 300, "code": 300, "math": 300})
    big = mix_documents(sources(), {"story": 3000, "code": 3000, "math": 3000})

    assert all(big.documents[name] > small.documents[name] for name in small.documents)


def test_the_mix_is_reproducible_and_the_seed_changes_it():
    budgets = {"story": 900, "code": 900, "math": 900}

    assert mix_documents(sources(), budgets, 1) == mix_documents(sources(), budgets, 1)
    assert mix_documents(sources(), budgets, 1).text != mix_documents(sources(), budgets, 2).text


def test_a_source_with_no_documents_contributes_nothing():
    sources = {"story": story_documents(STORY * 2), "code": []}

    mix = mix_documents(sources, {"story": 500, "code": 500})

    assert mix.documents["code"] == 0
    assert mix.documents["story"] == 1


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="same sources"):
        mix_documents({"a": ["x"]}, {"b": 10})

    with pytest.raises(ValueError, match="budget"):
        mix_documents({"a": ["x"]}, {"a": 0})

    with pytest.raises(ValueError, match="no documents"):
        mix_documents({"a": []}, {"a": 10})
