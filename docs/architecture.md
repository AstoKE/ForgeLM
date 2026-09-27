# ForgeLM architecture

_Last updated: Sprint 1a_

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
| `forgelm/cli.py` | `forgelm` command; `--version`, `serve`, `tokenize` |
| `forgelm/tokenizer/char.py` | `CharTokenizer` (vocab from corpus, `<unk>` = id 0) and `analyze()` |

Both `tokenize` adapters follow the same path:
`input → CharTokenizer.from_corpus(corpus) → analyze(tokenizer, text) → Analysis → print / JSON`

## Planned modules

```
src/forgelm/
├─ tokenizer/   Sprint 1b  BPE (char tokenizer done)
├─ models/      Sprint 2-3 bigram, MiniGPT (PyTorch)
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
