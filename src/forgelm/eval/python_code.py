"""Grading generated Python: does it at least have the *shape* of Python?

Running what a tiny model wrote is dangerous and almost never meaningful. Parsing is the
honest and safe question: are the brackets closed, the colons where they belong, the
indentation consistent? That is exactly the structure a model has to learn before it can
write anything that works, and it is checkable with `ast.parse` and nothing else.

One number on its own means little, so every score comes with two reference points:

* the **ceiling**: real Python, cut to the same length. A sample that stops mid-statement
  does not parse even when it is perfect, so real code does not score 1.0 either.
* the **floor**: the same real code with its characters shuffled. Same characters, no
  structure. A model near the floor has learned which characters exist, not how they fit.

Nothing here knows about HTTP or the terminal. torch is only imported inside `sample_code`.
"""

import ast
import random
import warnings
from dataclasses import dataclass

from forgelm.data.code import FILE_MARKER

MAX_LINES = 200  # parsing a prefix is tried once per line, so a long sample must be capped


def _code_lines(lines: list[str]) -> int:
    """Lines that say something: not blank, not only a comment."""
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def parsing_prefix_fraction(code: str) -> float:
    """What share of the code lines belong to the longest run from the top that parses.

    A sample is cut at an arbitrary character, so the last line is usually half-written and
    the whole text rarely parses. Instead we find the longest run of complete lines, from
    the top, that does, and report the share of code lines it covers. 1.0 means everything
    parses, 0.0 means not even the first statement does.

    Blank lines and comments are not counted: a model that only writes `#####` would parse
    perfectly and has shown nothing.
    """
    lines = code.splitlines()[:MAX_LINES]
    total = _code_lines(lines)
    if total == 0:
        return 0.0
    for keep in range(len(lines), 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # e.g. invalid escape sequences: not our concern
                ast.parse("\n".join(lines[:keep]))
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        return _code_lines(lines[:keep]) / total
    return 0.0


def code_parse_score(samples: list[str]) -> float:
    """Mean `parsing_prefix_fraction` over the samples."""
    if not samples:
        raise ValueError("no samples to grade")
    return sum(parsing_prefix_fraction(sample) for sample in samples) / len(samples)


def scramble(code: str, seed: int = 0) -> str:
    """The same characters in a random order: all the characters, none of the structure."""
    chars = list(code)
    random.Random(seed).shuffle(chars)
    return "".join(chars)


def real_snippets(files: list[str], length: int, count: int, seed: int = 0) -> list[str]:
    """The first `length` characters of `count` real files, for the ceiling and the floor."""
    if not files:
        raise ValueError("no files to take snippets from")
    chosen = random.Random(seed).sample(files, min(count, len(files)))
    return [text[:length] for text in chosen]


@dataclass(frozen=True)
class CodeReport:
    generated: float  # what the model wrote
    ceiling: float  # real code, cut to the same length
    floor: float  # the same real code, shuffled

    @property
    def position(self) -> float:
        """Where the model sits between the floor (0.0) and the ceiling (1.0)."""
        span = self.ceiling - self.floor
        return (self.generated - self.floor) / span if span > 0 else 0.0


def grade_code(samples: list[str], references: list[str], seed: int = 0) -> CodeReport:
    return CodeReport(
        generated=code_parse_score(samples),
        ceiling=code_parse_score(references),
        floor=code_parse_score([scramble(text, seed + i) for i, text in enumerate(references)]),
    )


def sample_code(
    model,
    tokenizer,
    count: int,
    length: int,
    temperature: float = 0.7,
    seed: int = 0,
) -> list[str]:
    """`count` files written from scratch, each `length` tokens long at most.

    Every sample starts right after a file marker, the way every file in the corpus does.
    If the model starts a new file before `length` is up, the sample ends there.
    """
    import torch  # only needed here, so the grading above stays usable without it

    generator = torch.Generator().manual_seed(seed)
    prompt = tokenizer.encode(f"{FILE_MARKER}\n")
    samples = []
    for _ in range(count):
        ids = model.generate(prompt, length, generator, temperature)
        text = tokenizer.decode(ids[len(prompt) :])
        samples.append(text.split(FILE_MARKER)[0])
    return samples
