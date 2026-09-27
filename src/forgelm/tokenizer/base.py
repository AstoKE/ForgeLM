"""The interface every ForgeLM tokenizer provides."""

from typing import Protocol


class Tokenizer(Protocol):
    """Structural type: any class with these members *is* a Tokenizer.

    No inheritance needed. CharTokenizer and BPETokenizer never mention this
    class, yet type checkers accept both wherever a Tokenizer is expected.
    """

    @property
    def unk_id(self) -> int | None:
        """Id used for unknown input, or None if nothing can be unknown."""
        ...

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def tokens(self, text: str) -> list[str]:
        """Human-readable form of each token, for display only."""
        ...
