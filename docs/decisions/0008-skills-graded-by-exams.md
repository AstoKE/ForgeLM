# ADR 0008: Skills are graded by exams on hidden problems, not by loss

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

Sprint 4d tries to make the from-scratch model better at three things: writing real words,
simple arithmetic and Python. Until now the only measurement was validation loss, and loss
cannot answer the questions that matter here. It counts every character the same, so in
`49+97=146` the `+` and `=` are free and only the last three digits are hard: a model can sit
at a low loss and still get a third of the sums wrong. For code, loss says nothing about
whether the output is Python at all. And a model that memorised its training sums scores
perfectly on those sums and still cannot add.

## Decision

1. **Every skill gets an exam that asks the task's own question.** Arithmetic: is the text
   after `49+97=` exactly the answer? Python: how much of the output does `ast.parse` accept?
   Stories keep using loss, because for free text there is no better exact answer.
2. **The exam is hidden.** Some arithmetic problems are never written into the training text.
   They are chosen as *unordered pairs*, so `58+37` cannot be trained on while `37+58` is
   graded, which would let a model pass through "addition is commutative" without adding.
3. **A control group is graded next to it.** A sample of problems the model *did* train on is
   scored too. High on seen and low on hidden means it memorised; high on both means it
   learned. Without the control a single accuracy cannot tell these apart.
4. **Scores are broken down, and the breakdown crosses the questions.** One model scored 77%
   overall. Split by operand length it was 96% on two-digit + two-digit and 1% on anything
   with a one-digit operand. A carry-only breakdown showed "no carry: 58%, carry: 86%", which
   reads as a weakness at *not* carrying, but short operands rarely carry, so that was a
   confound. The groups are therefore crossed (full-length vs short, carry vs no carry), and
   a test reproduces the mistake with a fake model.
5. **Generated code is parsed, never run.** Running what a tiny model wrote is dangerous and
   rarely meaningful. The parse score is calibrated by a **ceiling** (real held-out code cut
   to the same length, which does not parse completely either) and a **floor** (the same code
   with its characters shuffled), and comment-only output counts for nothing.
6. **The grader takes a completer, not a model.** A completer is a function from a prompt to
   the text the model wrote. The grading code is then free of torch and is tested with fake
   models whose score is known in advance: a perfect one, a useless one, one that only fails
   on carries, one that only fails on short operands. A measuring instrument that has not
   been checked against a known answer proves nothing.
7. **Training data is made or collected locally.** Arithmetic is generated, Python is taken
   from the standard library and installed packages (cleaned to ASCII, deduplicated, shuffled
   by file). Stories, code and sums are blended per document under a tag line, so the tag is
   the prompt that selects the skill.

## Consequences

- Skills can be compared by the number that matters. The first use showed that a model at 84%
  overall was at 99% on full-length operands and 23% on short ones, and, digit by digit, that
  a model whose units digit was 100% right had its tens digit at 9%, which is chance. Neither
  is visible in a loss curve.
- Exact match is strict: `146 ` with a trailing space is wrong, and so is a number that does
  not stop. That is intended, but it makes early training look worse than it is.
- Parse rate measures the *shape* of Python, not whether it does anything. A model can score
  high and write useless code. Running generated code in a sandbox would be the next step and
  is deliberately not done here.
- The hidden exam only protects against memorising the training text. It says nothing about
  problems unlike the training distribution (three-digit sums for a two-digit model).
- The mix fixes the proportions (about 60% code, 30% stories, 6% sums by size). Skills share
  one set of weights and can interfere with each other; Sprint 4d measures that rather than
  assuming it away.
- The data folder is local only. Python source is read from this machine and never committed
  or redistributed, which also keeps any package licences out of the repository.
