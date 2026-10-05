# ADR 0005: MiniGPT checkpoint format

- **Status:** Accepted
- **Date:** 2026-10-05

## Context

[ADR 0003](0003-checkpoint-format.md) chose JSON for the bigram and said neural models would
need a binary format and a new ADR. MiniGPT has 211,584 float weights (89 tensors). A trained
model is only usable together with its hyperparameters (to build an empty model of the right
shape) and its tokenizer (ids mean nothing without it).

## Decision

1. One file holds everything, as in ADR 0003:
   `{"format_version": 1, "tokenizer": {...}, "model": {"type": "minigpt", "config": {...},
   "weights": [tensors]}}`. `config` has `vocab_size`, `embed_dim`, `num_heads`, `num_blocks`,
   `block_size` and `hidden_dim`.
2. **`torch.save` to write, `torch.load(weights_only=True)` to read.** `weights_only=True` uses a
   restricted unpickler that only accepts tensors and basic Python values, so a shared file
   cannot run code. A test loads a file with a malicious `__reduce__` and checks it is refused.
3. Weights are a **plain list in `model.parameters()` order**. On load we rebuild an empty
   model from `config`, then check the tensor count and every shape before copying, and fail
   with a clear message on a mismatch.
4. `format_version` is checked on load, as in ADR 0003. Only the char tokenizer is supported
   for now.
5. Default location `checkpoints/minigpt.pt` (gitignored). `forgelm generate --model` picks
   the loader from the extension: `.pt` is MiniGPT, anything else is the JSON bigram.

## Consequences

- Size: 871 KB, against about 4.5 MB for the same weights as JSON text (measured, 5.2x), and no
  precision is lost in a text round trip. The file is not human-readable.
- Order-based storage is simple but fragile: if `parameters()` changes order or gains a tensor,
  old checkpoints stop loading. The shape check turns that into an error, not silent garbage.
  Bump `format_version` when the layout changes. Named parameters can replace the list if we
  ever need to load checkpoints across versions.
- **Alternative: safetensors.** The same safety guarantee (tensors only, no code) and the
  format Hugging Face uses, but a new dependency. Revisit in Sprint 5/6 when we load
  open-source models, whose weights come as safetensors.
- Optimizer state (Adam's `m` and `v`) is not saved, so training cannot be resumed exactly.
  Fine for now; revisit if a run gets long enough to need it.
