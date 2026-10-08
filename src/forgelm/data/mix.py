"""One training text that holds stories, Python and arithmetic, each marked by a tag line.

A model that has read three kinds of text has to know which one it is writing. We tell it
the way real base models are told: every document starts with a **tag line**,

    <|story|>      then a children's story
    # ----- file -----   then a Python file (the marker is itself a Python comment)
    <|math|>       then a block of `49+97=146` lines

At generation time the tag is the prompt that picks the skill: start with `<|math|>` and
the model continues with sums, start with the file marker and it writes Python. A small
version of what "system prompts" and chat templates do for big models.

The mix is made **per document**, never per character: cutting a story or a file in half to
hit a size target would teach the model that documents end anywhere. Documents are shuffled
together, so the validation tail (the last 10% of the text) has the same blend as the rest
and measures all three skills, not whichever source happened to come last.

Nothing here knows about models, HTTP or the terminal.
"""

import random
from dataclasses import dataclass

from forgelm.data.code import FILE_MARKER, split_files

STORY_TAG = "<|story|>"
MATH_TAG = "<|math|>"
STORY_END = "<|endoftext|>"  # how TinyStories separates its stories
MIN_STORY_CHARS = 100  # shorter than this is a fragment, not a story


def story_documents(text: str) -> list[str]:
    """Split a TinyStories file on its end-of-text marker and tag each story."""
    stories = (part.strip() for part in text.split(STORY_END))
    return [f"{STORY_TAG}\n{story}\n" for story in stories if len(story) >= MIN_STORY_CHARS]


def code_documents(corpus: str) -> list[str]:
    """The files of a `collect-code` corpus, each already starting with the file marker."""
    return [f"{FILE_MARKER}\n{text}\n" for text in split_files(corpus)]


def math_documents(text: str, per_document: int = 20) -> list[str]:
    """Group `49+97=146` lines into blocks. A block is a document: the tag, then the sums."""
    if per_document < 1:
        raise ValueError("per_document must be >= 1")
    lines = [line for line in text.splitlines() if line.strip()]
    return [
        f"{MATH_TAG}\n" + "\n".join(lines[i : i + per_document]) + "\n"
        for i in range(0, len(lines) - per_document + 1, per_document)
    ]


@dataclass(frozen=True)
class Mix:
    text: str
    documents: dict[str, int]  # how many documents of each source made it in
    chars: dict[str, int]  # and how many characters they add up to


def mix_documents(
    sources: dict[str, list[str]], budgets: dict[str, int], seed: int = 0
) -> Mix:
    """Take whole documents from each source up to its character budget, then interleave.

    A budget is a ceiling, reached by adding whole documents: a source is never cut in the
    middle of one, and always contributes at least one document if it has any.
    """
    if set(sources) != set(budgets):
        raise ValueError("sources and budgets must name the same sources")
    if any(size < 1 for size in budgets.values()):
        raise ValueError("every budget must be >= 1 character")
    if not any(sources.values()):
        raise ValueError("no documents to mix")

    rng = random.Random(seed)
    chosen: list[str] = []
    documents: dict[str, int] = {}
    chars: dict[str, int] = {}
    for name in sorted(sources):  # a fixed order, so the seed alone decides the result
        pool = list(sources[name])
        rng.shuffle(pool)
        taken, size = [], 0
        for doc in pool:
            if size + len(doc) > budgets[name] and taken:
                break
            taken.append(doc)
            size += len(doc)
        chosen.extend(taken)
        documents[name], chars[name] = len(taken), size
    rng.shuffle(chosen)
    return Mix("".join(chosen), documents, chars)
