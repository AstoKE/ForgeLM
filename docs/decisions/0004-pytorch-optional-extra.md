# ADR 0004: PyTorch as an optional extra, CUDA build installed from PyTorch's index

- **Status:** Accepted
- **Date:** 2026-09-28

## Context

Sprint 2b introduces the first neural model, which needs PyTorch. Torch is large: about 2.5 GB
for the Windows CUDA build and ~200 MB for CPU-only. Most of ForgeLM (tokenizers, the counting
bigram, the API) doesn't need it. The development machine has an RTX 4060 (8 GB, driver
supports CUDA 13.0), and Sprint 3+ will need the GPU.

## Decision

1. Torch goes in an **optional extra**, `pip install -e ".[ml]"`, not in the core
   `dependencies`.
2. Modules that need torch (`forgelm.models.neural_bigram`) are **not re-exported** from
   `forgelm.models`, and the CLI imports them **lazily** inside the command. Without torch, every
   other command still works, and the torch command fails with a clear message.
3. Tests that need torch use `pytest.importorskip("torch")`, and GPU tests use
   `skipif(not torch.cuda.is_available())`. The suite passes on machines without torch or
   without a GPU. Skipped tests are reported, not hidden.
4. **CUDA build on Windows:** PyPI only ships CPU wheels for Windows, so install torch from
   PyTorch's own index *before* the extra:
   `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
   On Linux, the default PyPI wheel already includes CUDA.

## Consequences

- `pip install -e ".[dev]"` stays small and fast. CI can run all non-torch tests cheaply.
- The version pin (`torch>=2.6`) lives in `pyproject.toml`, but the CUDA variant is chosen at
  install time. This is documented in the README because pip can't express it.
- When model serving arrives (Sprint 5/11), the API process will need torch. Revisit then
  whether it stays optional.
