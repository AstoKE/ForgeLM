"""Character-level tokenizer: every Unicode code point is one token.

The simplest possible tokenizer. It is useful because it makes the core ideas
visible: vocabulary, encode/decode, round-trip, and out-of-vocabulary (OOV) text.
"""

UNK = "<unk>"
UNK_ID = 0


class CharTokenizer:
    unk_id = UNK_ID

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
