"""Grading a model on a skill, not on loss.

Loss counts every character the same. In `49+97=146` the `+` and `=` are trivial and only
the last three digits are hard, so a model can sit at a low loss and still get half the sums
wrong. What we actually care about is *is the answer right*, and that needs its own exam.

The exam does not take a model. It takes a **completer**: a function from a prompt to the
text the model wrote after it. That keeps the grading code free of torch, and it means the
grader itself can be tested with fake models whose score we know in advance (a perfect one,
a useless one, one that only fails on carries). A measuring instrument that has not been
checked against a known answer proves nothing.

Nothing here knows about HTTP or the terminal.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from forgelm.data.synthetic import Problem, needs_carry

# prompt -> what the model wrote after it, cut at the end of the line
Completer = Callable[[str], str]

# A bigram has no reason to write more than a few characters, a real answer is short.
DEFAULT_MAX_NEW_TOKENS = 12


@dataclass(frozen=True)
class Miss:
    prompt: str
    expected: str
    got: str


@dataclass
class GroupScore:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class MathReport:
    total: int = 0
    correct: int = 0
    # Accuracy per kind of problem. A single number averages very different regimes: one
    # model scored 77% overall, which was 96% on two-digit + two-digit and 1% on the rest.
    groups: dict[str, GroupScore] = field(default_factory=dict)
    misses: list[Miss] = field(default_factory=list)  # the first few wrong answers

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def score(self, group: str) -> float | None:
        """None when no problem fell in the group: 0.0 would claim the model failed them."""
        found = self.groups.get(group)
        return found.accuracy if found is not None and found.total else None


# The groups a problem can belong to. Each problem is counted in several, so that the two
# questions "does it carry?" and "is an operand short?" can be told apart. Reading only the
# first one is a trap: short operands rarely carry, so a model that cannot handle short
# operands looks as if it cannot handle the *absence* of a carry.
CARRY, NO_CARRY = "carry", "no carry"
FULL, SHORT = "full-length operands", "a short operand"


def groups_of(problem: Problem, full_length: int) -> list[str]:
    carries = needs_carry(problem.a, problem.b)
    full = len(str(problem.a)) == len(str(problem.b)) == full_length
    shape = FULL if full else SHORT
    carry = CARRY if carries else NO_CARRY
    return [carry, shape, f"{shape}, {carry}"]


def math_accuracy(problems: list[Problem], complete: Completer, max_misses: int = 10) -> MathReport:
    """Exact match: the text after the prompt must equal the expected answer, nothing else.

    `1465` instead of `146` is wrong, and so is `146 ` with a stray space. The model has to
    know where the answer ends, because a calculator that cannot stop is not a calculator.
    """
    if not problems:
        raise ValueError("no problems to grade")
    full_length = max(max(len(str(p.a)), len(str(p.b))) for p in problems)
    report = MathReport()
    for problem in problems:
        got = complete(problem.prompt)
        right = got == problem.answer
        report.total += 1
        report.correct += right
        for name in groups_of(problem, full_length):
            group = report.groups.setdefault(name, GroupScore())
            group.total += 1
            group.correct += right
        if not right and len(report.misses) < max_misses:
            report.misses.append(Miss(problem.prompt, problem.answer, got))
    return report


def greedy_completer(
    model, tokenizer, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, prefix: str = ""
) -> Completer:
    """Wrap a MiniGPT as a completer: greedy decoding, stopping at the end of the line.

    Greedy (temperature 0) because an exam should not depend on a lucky sample.

    `prefix` is text put in front of every prompt and never shown to the grader. A model
    trained on tagged documents (`<|math|>` and then sums) is asked in the context it was
    trained in, the way a chat model is asked inside its template.
    """
    newline = tokenizer.encode("\n")
    # If "\n" is not in the vocabulary it encodes to <unk>, which is no stop signal at all.
    stop_id = newline[0] if len(newline) == 1 and newline[0] != tokenizer.unk_id else None

    def complete(prompt: str) -> str:
        ids = tokenizer.encode(prefix + prompt)
        out = model.generate(ids, max_new_tokens, temperature=0, stop_id=stop_id)
        return tokenizer.decode(out[len(ids) :]).split("\n")[0]

    return complete
