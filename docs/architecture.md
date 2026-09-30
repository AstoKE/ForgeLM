# ForgeLM architecture

_Last updated: Sprint 3c_

## Principle: thin adapters around core logic

```
            ┌─────────────┐      ┌──────────────┐
  terminal →│  cli.py     │      │  api.py      │← HTTP (curl, browser, later a UI)
            │  (argparse) │      │  (FastAPI)   │
            └──────┬──────┘      └──────┬───────┘
                   └────────┬───────────┘
                            ▼
                 core modules in src/forgelm/
```

- **Adapters** (`cli.py`, `api.py`) only convert input and output: they parse arguments and
  JSON, call a core function, and format the result. They don't hold any business logic.
- **Core modules** know nothing about HTTP or the terminal. This means they can be unit-tested
  directly, and every feature is available from both the CLI and the API.

## Current modules

| Module | Role |
|---|---|
| `forgelm/__init__.py` | Package marker and `__version__` (read from installed metadata) |
| `forgelm/api.py` | FastAPI `app`; `GET /health`, `POST /tokenize` |
| `forgelm/cli.py` | `forgelm` command; `--version`, `serve`, `tokenize`, `train-bigram`, `generate` |
| `forgelm/tokenizer/base.py` | `Tokenizer` protocol: `encode`, `decode`, `tokens`, `vocab_size`, `unk_id` |
| `forgelm/tokenizer/char.py` | `CharTokenizer` (vocab from corpus, `<unk>` = id 0) |
| `forgelm/tokenizer/bpe.py` | `BPETokenizer`: byte-level BPE, `train(corpus, num_merges)` ([ADR 0002](decisions/0002-tokenizer-interface-and-bpe.md)) |
| `forgelm/tokenizer/analysis.py` | `build_tokenizer(kind, corpus, merges)` and `analyze()`, shared by the CLI and API |
| `forgelm/models/bigram.py` | `BigramModel` (counting), `softmax`, `encode_and_split`, `train_on_text`, JSON checkpoints ([ADR 0003](decisions/0003-checkpoint-format.md)) |
| `forgelm/models/neural_bigram.py` | PyTorch bigram: `NeuralBigram` (V×V `W`), hand-written `cross_entropy` and `sgd_step`, `train_neural_bigram`. Needs the `ml` extra; imported explicitly, never re-exported ([ADR 0004](decisions/0004-pytorch-optional-extra.md)) |
| `forgelm/models/attention.py` | Causal mixing of past positions: `causal_mask`, causal average (loop / matmul / masked softmax), `masked_softmax_weights` (3a). `attend(q, k, v)` and `SelfAttentionHead` with learned `Wq/Wk/Wv` (3b). `MultiHeadAttention`: heads in parallel, concatenated, projected by `Wo`, returns the per-head weights (3c). Needs torch |
| `forgelm/models/minigpt.py` | `MiniGPT`: token + position embedding, a stack of `TransformerBlock`s, a final `LayerNorm` and a `(C, V)` head. `forward(ids) -> logits`, `loss` (reuses 2b's `cross_entropy`), `generate` with temperature and context cropping (3c-3). Needs torch |
| `forgelm/models/block.py` | The transformer block. `FeedForward`: per-position MLP `C -> 4C -> relu -> C`, hand-written `W1/b1/W2/b2` (3c-1). `LayerNorm`: per-position mean 0 / std 1 plus learned `gamma`/`beta`. `TransformerBlock`: pre-norm plus residuals, `x = x + attention(ln1(x))` then `x = x + feedforward(ln2(x))`, `(T, C) -> (T, C)` so blocks stack (3c-2). Needs torch |

Both `tokenize` adapters follow the same path:
`input → build_tokenizer(kind, corpus, merges) → analyze(tokenizer, text) → Analysis → print / JSON`

Training and generation (CLI only for now):
```
train-bigram:  corpus file → train_on_text() → CharTokenizer + BigramModel → checkpoints/bigram.json
generate:      checkpoint → load_checkpoint() → encode(prompt) → model.generate() → decode → print
train-neural-bigram: corpus → encode_and_split() → train_neural_bigram() (SGD on CPU/GPU) → loss history
                     (also runs train_on_text() for comparison; no checkpoint saved yet)
```
The model only sees integer ids and `vocab_size`; it never sees text.

MiniGPT's forward pass (no CLI yet; training is Sprint 3d):
```
ids (T,) -> token_embedding[ids] + position_embedding[:T] -> TransformerBlock x n
         -> ln_final -> @ head -> logits (T, V)
```
`block_size` is the height of the position table, so it is the model's context window.
`generate` crops its input to the last `block_size` tokens for that reason.

Torch boundary: nothing under `forgelm/` imports torch at module load except
`models/neural_bigram.py`, `models/attention.py` and `models/block.py`. None of them is
re-exported from `forgelm.models`, and only the CLI command that needs one imports it, so
ForgeLM still runs without the `ml` extra.

## Planned modules

```
src/forgelm/
├─ models/      Sprint 2b-3 neural bigram, MiniGPT (PyTorch)
├─ training/    Sprint 4   training loop, checkpoints, metrics
├─ inference/   Sprint 5   sampling, backends (own model, Ollama, HF)
├─ rag/         Sprint 7-8 chunking, embeddings, retrieval, reranking
├─ agents/      Sprint 9   agent loop, tools (todos, files, notes...)
└─ eval/        Sprint 10  golden datasets, metrics, tracing
```

Target end-to-end flow:
`UI → FastAPI → Orchestrator → {LLM + RAG + Tools} → Vector DB / SQL → Eval & Tracing`

## Decisions

Architecture decisions are recorded as ADRs in [decisions/](decisions/).
