"""Byte-level Byte Pair Encoding (BPE), trained from scratch.

Base vocabulary: the 256 possible byte values, so any UTF-8 text can be
encoded and nothing is ever unknown. Training repeatedly merges the most
frequent adjacent pair into a new token, which shortens sequences.

Known simplification: no pre-tokenization, so merges may cross word
boundaries (e.g. a token "e t"). GPT-style tokenizers split on a regex first.
"""

from itertools import pairwise

Pair = tuple[int, int]

NUM_BYTES = 256


def get_pair_counts(ids: list[int]) -> dict[Pair, int]:
    """Count adjacent pairs: [1, 2, 1, 2] -> {(1, 2): 2, (2, 1): 1}."""
    counts: dict[Pair, int] = {}
    for pair in pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right) with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    unk_id = None  # every byte is in the vocab, so nothing is ever unknown

    def __init__(self, merges: dict[Pair, int]) -> None:
        # Dicts keep insertion order, so `merges` is also the training order.
        self.merges = merges
        # id -> the bytes it stands for. Merged tokens are built from their parts.
        self.vocab = {i: bytes([i]) for i in range(NUM_BYTES)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, corpus: str, num_merges: int) -> "BPETokenizer":
        if num_merges < 0:
            raise ValueError("num_merges must be >= 0")

        ids = list(corpus.encode("utf-8"))
        merges: dict[Pair, int] = {}
        for step in range(num_merges):
            counts = get_pair_counts(ids)
            if not counts:
                break
            # Most frequent pair. On a tie, max() keeps the first pair seen in
            # the corpus, so training is deterministic.
            pair = max(counts, key=counts.__getitem__)
            if counts[pair] < 2:
                break  # no pair repeats; merging further would just memorize the corpus
            new_id = NUM_BYTES + step
            ids = merge(ids, pair, new_id)
            merges[pair] = new_id
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Apply merges in the order they were learned: among the pairs present,
            # pick the one with the lowest merge id. Frequencies in `text` don't matter.
            pairs = set(pairwise(ids))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no learned merge applies any more
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        for i in ids:
            if i not in self.vocab:
                raise ValueError(f"token id {i} is outside vocab (size {self.vocab_size})")
        data = b"".join(self.vocab[i] for i in ids)
        # Arbitrary ids can cut a multi-byte character in half; show U+FFFD for it.
        return data.decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        # A token may be part of a character (e.g. b"\xc4"); show such bytes as \xc4.
        return [self.vocab[i].decode("utf-8", errors="backslashreplace") for i in self.encode(text)]
