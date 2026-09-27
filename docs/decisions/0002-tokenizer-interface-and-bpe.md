# ADR 0002: Tokenizer protocol and byte-level BPE

- **Status:** Accepted
- **Date:** 2026-09-27

## Context

Sprint 1b added a second tokenizer (BPE) next to `CharTokenizer`. `analyze()` assumed the char
tokenizer. For example, it counted unknowns as `ids.count(0)`, but in BPE id 0 is the byte `\x00`.

## Decision

1. Define a `Tokenizer` `typing.Protocol` (`encode`, `decode`, `tokens`, `vocab_size`, `unk_id`).
   Tokenizers satisfy it structurally, without inheritance. We added the abstraction only once
   a second real implementation existed.
2. `unk_id: int | None`: `None` means the tokenizer can represent any input (BPE).
3. BPE is **byte-level** (256 base tokens) so nothing is ever out of vocabulary.
4. **No pre-tokenization** yet, so the core algorithm stays visible. Merges may cross spaces.
5. Tie-break during training: the first pair seen in the corpus wins, which keeps training
   deterministic. Training stops early when no pair occurs twice.
6. The API caps `num_merges` (≤ 1000) and text/corpus length (≤ 50k chars), because training is
   O(merges × corpus).

## Consequences

- New tokenizers only need to provide the protocol's members. The CLI and API pick one through
  `build_tokenizer(kind, ...)`.
- Tokens don't match GPT-style tokens (they end with spaces instead of starting with them).
  Revisit pre-tokenization when a model trains on these tokens.
- Merges are not saved yet. Sprint 2 needs saved merges so the model sees stable ids.
