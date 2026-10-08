"""Training text that we generate ourselves: addition problems with a hidden exam.

A language model learns whatever it is shown. To teach it arithmetic we do not need a
download, we can write the examples:

    49+97=146
    53+5=58

One line is one problem. The model sees the prompt `49+97=` and must write `146`.

The honest question is whether it *learned to add* or only *memorised the answers*. With
two digits there are just 100 x 100 = 10,000 problems, so a few million training lines
repeat every problem hundreds of times and memorising is easy. The defence is a hidden
exam: some problems are never written into the training text, and the model is graded on
those only.

The hidden problems are chosen as **unordered pairs**. If `58+37` were trained on and
`37+58` held out, the model could pass the exam through the shortcut "addition does not
care about order" without adding anything. Holding out the pair {37, 58} hides both.

`reverse=True` writes the answer digits backwards (`49+97=641`). Addition is done from the
right, carrying leftwards, but a language model writes from the left, so it has to guess
the leading digit before it has seen any of the columns that decide it. Reversed, it
produces the digits in the order it can compute them. Whether that really helps is an
experiment, not an assumption.

`pad=True` writes every number with the same width (`05+12=017`). Without it the model must
first work out where the units column is from the length of each number, and a short
operand like `5` shifts everything. Measured on a first model: 95.8% on two-digit + two-digit
problems and 1% as soon as one operand had a single digit.

Nothing here knows about models, HTTP or the terminal.
"""

import random
from dataclasses import dataclass

MAX_DIGITS = 3  # 10**3 x 10**3 = a million problems: plenty, and still fits in memory


@dataclass(frozen=True)
class Problem:
    """One line of text, split where the model's job begins.

    `prompt` and `answer` are kept exactly as written, because the grader compares text to
    text: padding or reversing changes the written form, not the numbers a and b.
    """

    a: int
    b: int
    prompt: str  # what the model is shown, up to and including the "="
    answer: str  # exactly what it must write next (padded and/or reversed if asked)

    @property
    def line(self) -> str:
        return f"{self.prompt}{self.answer}"


def make_problem(a: int, b: int, reverse: bool = False, pad: int = 0) -> Problem:
    """`pad` > 0 writes both numbers with `pad` digits and the answer with `pad + 1`.

    Unpadded, `5+12=17` and `49+97=146` put the units digit in different places, so the model
    has to work out the column alignment from the length of each number first. Padded,
    `05+12=017` and `49+97=146` line up column for column.
    """
    prompt = f"{a:0{pad}d}+{b:0{pad}d}="
    answer = f"{a + b:0{pad + 1 if pad else 0}d}"
    return Problem(a, b, prompt, answer[::-1] if reverse else answer)


def all_problems(digits: int = 2, reverse: bool = False, pad: bool = False) -> list[Problem]:
    """Every a + b with a, b below 10**digits, in both orders."""
    if not 1 <= digits <= MAX_DIGITS:
        raise ValueError(f"digits must be between 1 and {MAX_DIGITS}, got {digits}")
    limit = 10**digits
    width = digits if pad else 0
    return [make_problem(a, b, reverse, width) for a in range(limit) for b in range(limit)]


def split_problems(
    problems: list[Problem], holdout_fraction: float = 0.1, seed: int = 0
) -> tuple[list[Problem], list[Problem]]:
    """(train, holdout). A pair {a, b} is held out in *both* orders, never just one."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    pairs = sorted({(min(p.a, p.b), max(p.a, p.b)) for p in problems})
    if len(pairs) < 2:
        raise ValueError("need at least two distinct problems to split")
    random.Random(seed).shuffle(pairs)
    # At least one pair on each side, whatever the fraction says.
    cut = min(max(1, round(len(pairs) * holdout_fraction)), len(pairs) - 1)
    hidden = set(pairs[:cut])
    train = [p for p in problems if (min(p.a, p.b), max(p.a, p.b)) not in hidden]
    holdout = [p for p in problems if (min(p.a, p.b), max(p.a, p.b)) in hidden]
    return train, holdout


def render(problems: list[Problem]) -> str:
    """One problem per line, `49+97=146\\n`."""
    return "".join(f"{p.line}\n" for p in problems)


def make_corpus(train: list[Problem], lines: int, seed: int = 0) -> str:
    """`lines` problems drawn with replacement from `train`, as training text."""
    if not train:
        raise ValueError("no training problems to draw from")
    if lines < 1:
        raise ValueError("lines must be >= 1")
    return render(random.Random(seed).choices(train, k=lines))


def parse_problem(line: str) -> Problem:
    """Read `a+b=answer` back. The answer is kept as written (possibly reversed)."""
    left, separator, answer = line.strip().partition("=")
    a, plus, b = left.partition("+")
    if not separator or not plus or not a.isdigit() or not b.isdigit():
        raise ValueError(f"not a problem line: {line!r}")
    return Problem(int(a), int(b), f"{left}=", answer)


def parse_problems(text: str) -> list[Problem]:
    return [parse_problem(line) for line in text.splitlines() if line.strip()]


def needs_carry(a: int, b: int) -> bool:
    """True if any column of the written-out sum has to carry a 1 to the next one."""
    carry = 0
    while a or b or carry:
        carry = (a % 10 + b % 10 + carry) // 10
        if carry:
            return True
        a, b = a // 10, b // 10
    return False


@dataclass(frozen=True)
class MathDataset:
    corpus: str  # the training text
    holdout: str  # problems the model never sees: the exam
    seen: str  # a sample of problems it *did* see, to tell memorising from learning
    train_problems: int  # distinct problems that can appear in the corpus
    holdout_problems: int


def build_math_dataset(
    digits: int = 2,
    holdout_fraction: float = 0.1,
    lines: int = 400_000,
    reverse: bool = False,
    seed: int = 0,
    seen_sample: int = 1_000,
    pad: bool = False,
) -> MathDataset:
    """The training text, the hidden exam, and a sample of problems it was trained on.

    The *seen* sample matters as a control. If the model scores 100% on seen problems and
    30% on hidden ones, it memorised. If both are high, it learned to add.
    """
    train, holdout = split_problems(all_problems(digits, reverse, pad), holdout_fraction, seed)
    corpus = make_corpus(train, lines, seed)
    # The control must hold problems the model *did* see. A finite corpus drawn with
    # replacement can miss some of `train`, so sample from what actually appears in it.
    written = set(corpus.splitlines())
    appeared = [p for p in train if p.line in written]
    seen = random.Random(seed + 1).sample(appeared, min(seen_sample, len(appeared)))
    return MathDataset(
        corpus=corpus,
        holdout=render(holdout),
        seen=render(seen),
        train_problems=len(train),
        holdout_problems=len(holdout),
    )
