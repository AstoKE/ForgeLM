# ADR 0003: Checkpoint format

- **Status:** Accepted
- **Date:** 2026-09-28

## Context

Sprint 2 trains the first model. A trained model has to be saved and loaded later, and its
token ids only mean something together with the exact tokenizer that produced them.

## Decision

1. A checkpoint is **one file containing both the tokenizer and the model**:
   `{"format_version": 1, "tokenizer": {...}, "model": {...}}`. Each part has a `type` field
   (`"char"`, `"bigram"`) and its own `to_dict` / `from_dict`.
2. **JSON, not pickle.** `pickle.load` can execute arbitrary code from the file, which is a known
   attack vector for shared model files (it's why the ML ecosystem moved to `safetensors`).
   JSON is also human-readable: you can open the file and see the counts.
3. `format_version` is checked on load, so an incompatible file fails clearly instead of loading
   garbage.
4. Default location is `checkpoints/`, which is gitignored.
5. Only the char tokenizer can be saved for now. BPE serialization comes when a model needs it.

## Consequences

- JSON only works for small models (a 66×66 bigram is 15 KB). Neural models in Sprint 2b/3 will
  store weight tensors in a binary format (`torch.save` with `weights_only=True`, or
  safetensors). That will need a new ADR.
