"""Playground helpers shared by the CLI and API: build a tokenizer, inspect text."""

from dataclasses import dataclass
from typing import Literal

from forgelm.tokenizer.base import Tokenizer
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.tokenizer.char import CharTokenizer

TokenizerKind = Literal["char", "bpe"]


def build_tokenizer(kind: TokenizerKind, corpus: str, num_merges: int = 50) -> Tokenizer:
    if kind == "char":
        return CharTokenizer.from_corpus(corpus)
    if kind == "bpe":
        return BPETokenizer.train(corpus, num_merges)
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


@dataclass(frozen=True)
class Analysis:
    """Everything the playground shows about one piece of text."""

    tokens: list[str]
    ids: list[int]
    decoded: str
    num_chars: int
    num_tokens: int
    vocab_size: int
    unknown_count: int
    chars_per_token: float
    roundtrip_ok: bool


def analyze(tokenizer: Tokenizer, text: str) -> Analysis:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    unk_id = tokenizer.unk_id
    return Analysis(
        tokens=tokenizer.tokens(text),
        ids=ids,
        decoded=decoded,
        num_chars=len(text),
        num_tokens=len(ids),
        vocab_size=tokenizer.vocab_size,
        unknown_count=ids.count(unk_id) if unk_id is not None else 0,
        chars_per_token=len(text) / len(ids) if ids else 0.0,
        roundtrip_ok=decoded == text,
    )
