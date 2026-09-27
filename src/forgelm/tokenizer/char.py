"""Character-level tokenizer: every Unicode code point is one token.

The simplest possible tokenizer. It is useful because it makes the core ideas
visible: vocabulary, encode/decode, round-trip, and out-of-vocabulary (OOV) text.
"""

from dataclasses import dataclass

UNK = "<unk>"
UNK_ID = 0


class CharTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        # The list index *is* the token id: vocab[3] is the token with id 3.
        if not vocab or vocab[UNK_ID] != UNK:
            raise ValueError(f"vocab must start with {UNK!r} at id {UNK_ID}")
        self.itos = vocab  # id -> string
        self.stoi = {token: i for i, token in enumerate(vocab)}  # string -> id

    @classmethod
    def from_corpus(cls, corpus: str) -> "CharTokenizer":
        """Build a vocabulary from every distinct character in `corpus`.

        sorted() makes ids deterministic: the same corpus always gives the same
        ids, no matter the character order. A model trained on these ids depends
        on that.
        """
        return cls([UNK, *sorted(set(corpus))])

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokens(self, text: str) -> list[str]:
        return [ch if ch in self.stoi else UNK for ch in text]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, UNK_ID) for ch in text]

    def decode(self, ids: list[int]) -> str:
        for i in ids:
            # Explicit check: Python would silently accept -1 as "last item".
            if not 0 <= i < self.vocab_size:
                raise ValueError(f"token id {i} is outside vocab (size {self.vocab_size})")
        return "".join(self.itos[i] for i in ids)


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


def analyze(tokenizer: CharTokenizer, text: str) -> Analysis:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    return Analysis(
        tokens=tokenizer.tokens(text),
        ids=ids,
        decoded=decoded,
        num_chars=len(text),
        num_tokens=len(ids),
        vocab_size=tokenizer.vocab_size,
        unknown_count=ids.count(UNK_ID),
        chars_per_token=len(text) / len(ids) if ids else 0.0,
        roundtrip_ok=decoded == text,
    )
