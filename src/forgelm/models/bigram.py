"""Bigram language model, trained by counting.

The model predicts the next token from the previous token only:

    P(next | prev) = (count(prev, next) + k) / (count(prev, *) + k * V)

where k is the add-k smoothing constant and V the vocab size. No gradients,
no framework: training is one pass over the corpus that fills a V x V table.
"""

import json
import math
import random
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from forgelm.tokenizer import CharTokenizer

CHECKPOINT_FORMAT_VERSION = 1


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Turn raw scores into probabilities that sum to 1.

    Dividing by `temperature` first: T < 1 sharpens the distribution, T > 1 flattens it.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy decoding for T = 0)")
    scaled = [x / temperature for x in logits]
    # Subtracting the max doesn't change the result (it cancels in the division)
    # but keeps exp() from overflowing: math.exp(1000) raises OverflowError.
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


class BigramModel:
    def __init__(self, counts: list[list[int]], smoothing: float = 1.0) -> None:
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0")
        self.counts = counts  # counts[prev][next]: how often `next` followed `prev`
        self.smoothing = smoothing
        # Precompute every row once; loss() would otherwise redo it per token.
        self._probs = [self._row_probs(row) for row in counts]

    @classmethod
    def train(cls, ids: list[int], vocab_size: int, smoothing: float = 1.0) -> "BigramModel":
        counts = [[0] * vocab_size for _ in range(vocab_size)]
        for prev, nxt in pairwise(ids):
            counts[prev][nxt] += 1
        return cls(counts, smoothing)

    @property
    def vocab_size(self) -> int:
        return len(self.counts)

    def _row_probs(self, row: list[int]) -> list[float]:
        total = sum(row) + self.smoothing * len(row)
        if total == 0:
            # Never saw `prev` and no smoothing: we know nothing, so assume uniform.
            return [1 / len(row)] * len(row)
        return [(c + self.smoothing) / total for c in row]

    def probs(self, prev: int) -> list[float]:
        """P(next | prev) for every possible next token."""
        return self._probs[prev]

    def logits(self, prev: int) -> list[float]:
        """Log-probabilities; softmax(logits(prev)) == probs(prev)."""
        return [math.log(p) if p > 0 else -math.inf for p in self._probs[prev]]

    def loss(self, ids: list[int]) -> float:
        """Average negative log-likelihood (cross-entropy) of each next token, in nats."""
        if len(ids) < 2:
            raise ValueError("need at least 2 tokens to measure next-token loss")
        total = 0.0
        for prev, nxt in pairwise(ids):
            p = self._probs[prev][nxt]
            if p == 0:
                return math.inf  # the model called something that happened "impossible"
            total -= math.log(p)
        return total / (len(ids) - 1)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        rng: random.Random,
        temperature: float = 1.0,
    ) -> list[int]:
        """Extend `prompt_ids` one sampled token at a time. temperature=0 means greedy."""
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        ids = list(prompt_ids)
        candidates = range(self.vocab_size)
        for _ in range(max_new_tokens):
            prev = ids[-1]  # a bigram only ever looks at the last token
            if temperature == 0:
                probs = self._probs[prev]
                ids.append(max(candidates, key=probs.__getitem__))
            else:
                probs = softmax(self.logits(prev), temperature)
                ids.append(rng.choices(candidates, weights=probs)[0])
        return ids

    def to_dict(self) -> dict:
        return {"type": "bigram", "smoothing": self.smoothing, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: dict) -> "BigramModel":
        if data.get("type") != "bigram":
            raise ValueError(f"not a bigram model: type={data.get('type')!r}")
        return cls(data["counts"], data["smoothing"])


@dataclass(frozen=True)
class TrainReport:
    model: BigramModel
    tokenizer: CharTokenizer
    train_tokens: int
    val_tokens: int
    baseline_loss: float  # a uniform model: every token equally likely
    train_loss: float
    val_loss: float


def encode_and_split(
    text: str, val_fraction: float = 0.1
) -> tuple[CharTokenizer, list[int], list[int]]:
    """Char-tokenize `text` and split the ids into contiguous train/val parts."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    # The vocab comes from the whole text (it's just the character set); the
    # *statistics* only come from the train split.
    tokenizer = CharTokenizer.from_corpus(text)
    ids = tokenizer.encode(text)

    # Contiguous split, not random: a random split would put neighbouring
    # characters of the same sentence in both halves (leakage).
    split = int(len(ids) * (1 - val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError("corpus too short for a train/val split")
    return tokenizer, train_ids, val_ids


def train_on_text(text: str, smoothing: float = 1.0, val_fraction: float = 0.1) -> TrainReport:
    """Build a char tokenizer, split train/val, count bigrams, and measure loss."""
    tokenizer, train_ids, val_ids = encode_and_split(text, val_fraction)
    model = BigramModel.train(train_ids, tokenizer.vocab_size, smoothing)
    return TrainReport(
        model=model,
        tokenizer=tokenizer,
        train_tokens=len(train_ids),
        val_tokens=len(val_ids),
        baseline_loss=math.log(tokenizer.vocab_size),
        train_loss=model.loss(train_ids),
        val_loss=model.loss(val_ids),
    )


def save_checkpoint(path: str | Path, model: BigramModel, tokenizer: CharTokenizer) -> None:
    """Save model *and* tokenizer: ids are meaningless without the tokenizer that made them.

    JSON, not pickle: loading a pickle can execute arbitrary code, and JSON is
    human-readable (open the file and look at the counts).
    """
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "tokenizer": tokenizer.to_dict(),
        "model": model.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint), encoding="utf-8")


def load_checkpoint(path: str | Path) -> tuple[BigramModel, CharTokenizer]:
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version: {version!r}")
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    model = BigramModel.from_dict(checkpoint["model"])
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint model and tokenizer disagree on vocab size")
    return model, tokenizer
