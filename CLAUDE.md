# ForgeLM: instructions for AI pair programmers

ForgeLM is a long-term learning project. The owner is a software engineering student learning
LLM engineering by building one product in 12 sprints. The end goal is a personal assistant
that runs a local open-source LLM, reads the owner's documents (RAG) and does daily tasks
through tools (agents).

**Your role:** mentor and pair programmer, not a code generator. The student must be able to
understand, debug and extend everything in this repo.

## Current status

- ✅ **Sprint 0: Engineering setup.** Package, `forgelm` CLI, FastAPI `/health`, pytest,
  ruff, docs.
- ✅ **Sprint 1a: Character tokenizer.** `forgelm.tokenizer.CharTokenizer`, `analyze()`,
  `forgelm tokenize`, `POST /tokenize`.
- ✅ **Sprint 1b: Byte-level BPE.** `BPETokenizer`, `Tokenizer` protocol, `--tokenizer bpe`
  ([ADR 0002](docs/decisions/0002-tokenizer-interface-and-bpe.md)).
- ✅ **Sprint 2a: Counting bigram.** `forgelm.models.BigramModel`, `softmax`, JSON checkpoints
  ([ADR 0003](docs/decisions/0003-checkpoint-format.md)), `train-bigram` / `generate` CLI.
  Tiny Shakespeare: val loss 2.48 vs baseline 4.19.
- ✅ **Sprint 2b: Neural bigram in PyTorch.** `forgelm.models.neural_bigram` (hand-written
  cross-entropy + SGD), `train-neural-bigram` CLI, torch as `ml` extra
  ([ADR 0004](docs/decisions/0004-pytorch-optional-extra.md)). Converges to the counting model
  (train 2.456 vs 2.455) in 3000 steps / 15 s on GPU.
- ✅ **Sprint 3a: Causal averaging.** `forgelm.models.attention`: `causal_mask`, the causal
  average written 3 ways (loop, matmul, mask + softmax), `masked_softmax_weights`.
- ✅ **Sprint 3b: Single-head self-attention.** `attend(q, k, v)` and `SelfAttentionHead`
  (hand-written `Wq/Wk/Wv`, scores = Q·Kᵀ / √d, reuses `masked_softmax_weights`). Zero Q/K
  reproduces the 3a average. Quiz answered.
- ✅ **Sprint 3c: MiniGPT, the full model, untrained.** Built in three steps:
  3c-1 `MultiHeadAttention` (heads over a split of the channels, concat, `Wo`) and
  `block.FeedForward` (`C → 4C → relu → C`);
  3c-2 `block.LayerNorm` and `block.TransformerBlock` (pre-norm + residuals, `(T, C) → (T, C)`
  so blocks stack);
  3c-3 `models/minigpt.MiniGPT` (token + position embedding, block stack, `ln_final`, `(C, V)`
  head, `forward` / `loss` / `generate`).
  172 tests. Proof it works end to end: plain SGD memorises `"to be or not to be"` (loss
  4.35 → 0.0009 in 300 steps) and greedy decoding reproduces it exactly, which a bigram cannot
  do. All 3c quizzes answered.
- ✅ **Sprint 3d: Training MiniGPT.** `models/train_minigpt.py` (random windows, hand-written
  `Adam` checked against `torch.optim.Adam`, train/val loop), `save_minigpt` / `load_minigpt`
  ([ADR 0005](docs/decisions/0005-minigpt-checkpoint.md): `torch.save` + `weights_only=True`),
  `forgelm train-minigpt`, and `generate --model x.pt`. 203 tests. Tiny Shakespeare, 211,584
  params, 2000 steps on CPU (~4 min): val loss **1.85** (perplexity 6.4) vs bigram 2.48 (12).
  The 3d quiz is still unanswered (see `docs/learning-log.md`). Known limits, left on purpose:
  the batch is a Python loop over single windows (no `(B, T)` dimension), so the GPU is *slower*
  than the CPU (22 s vs 12 s per 100 steps); no best-val checkpoint; Adam state is not saved.
- 🔶 **Sprint 4a: Inference dashboard** (done, 4b pending). `forgelm/inference/local.py`
  (`ModelStore` cache, `generate_text`, `attention_maps`), endpoints `GET /models`,
  `POST /generate`, `POST /attention`, `GET /ui`, and `forgelm/ui/index.html` (vanilla JS,
  attention heatmap, no framework; [ADR 0006](docs/decisions/0006-dashboard-ui.md)). 240 tests;
  the page was also driven in headless Chrome. Models are looked up by file name only (path
  traversal is rejected). The 4a quiz is not asked yet.
- ✅ **Sprint 4b: Live training in the dashboard.** 4b-1: `on_progress` callback in
  `train_minigpt` (the CLI now prints steps live instead of 213 s of silence),
  `forgelm/training/jobs.py` (`TrainingJobStore`: background thread, one run at a time,
  in-memory progress), `GET /corpora`, `POST /train` (202 + job id), `GET /train`,
  `GET /train/{id}` ([ADR 0007](docs/decisions/0007-background-training-jobs.md)). 4b-2: a Train
  panel in `/ui` that polls once a second and draws both loss curves on a `<canvas>`, and
  refreshes the model lists when a run finishes. 284 tests; the page was driven in headless
  Chrome, running and finished. Corpus and output are file names inside one folder, never paths;
  every hyperparameter is capped (`MAX_STEPS = 5000`).
  The 4a and 4b quizzes are still unanswered (see `docs/learning-log.md`).
- ⏳ **Next: Sprint 5, an open-source LLM** (Ollama / llama.cpp / HF) behind the same
  `/generate` endpoint, so the dashboard can talk to a real assistant instead of MiniGPT. Start
  with a concept briefing, in simple language. Optional leftovers: the `(B, T)` batch dimension
  (needs approval; it is why the GPU is slower than the CPU), the 20,000-step overfitting
  experiment, a best-val checkpoint, and cancelling a running job.

Environment: torch 2.14.0+cu130 in `.venv` (CUDA 13.0 driver, RTX 4060 8 GB). Install with
`pip install torch --index-url https://download.pytorch.org/whl/cu130` then `pip install -e ".[dev,ml]"`.

Data: `data/tinyshakespeare.txt` (gitignored), downloaded from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Known environment note: on Windows, piped output uses the legacy code page (cp1254) and garbles
non-ASCII characters. Set `PYTHONUTF8=1`.

Update this section at the end of every sprint.

## Roadmap

0 Setup · 1 Tokenizer · 2 Tiny LM (bigram) · 3 MiniGPT from scratch · 4 Training dashboard ·
5 Open-source LLM (Ollama/llama.cpp/HF) · 6 LoRA/QLoRA · 7 RAG v1 · 8 RAG v2 (hybrid, rerank) ·
9 Agents (tools for daily tasks) · 10 Eval & observability · 11 Production deployment

## Language

Explain things, write summaries and ask quiz questions in **Turkish**, keeping important
technical terms in English (token, vocabulary, merge, embedding...). Code, comments, commit
messages and repo docs stay in English.

Keep explanations **simple**: start with an everyday analogy, then a tiny concrete example with
real numbers, and only then the term or the code. At most 3-5 new ideas per message, short
sentences, and split sprints into small sub-steps.

## Session loop

**Learn → Plan → Build → Test → Explain → Quiz → Document → Commit**

1. Explain the concept in 5-10 concise bullets before coding.
2. Propose the smallest implementable feature, list the files that will change and why, then
   **wait for approval**.
3. Build in small, reviewable diffs. Add tests for every meaningful behavior, including failure
   cases.
4. Afterwards, explain the architecture and the 3-5 most important code decisions.
5. Ask 3-5 checkpoint questions. If an answer is wrong, re-teach using this codebase.
6. Update `docs/learning-log.md` (concept, what we built, mistakes, lessons, next questions).
   The student writes the "Lessons" part in their own words.
7. Update `docs/architecture.md` and add an ADR in `docs/decisions/` when an architecture
   decision changes.

## Engineering rules

- Build the low-level version first; add frameworks later. PyTorch for model code; Hugging Face
  only after the from-scratch version exists.
- `cli.py` and `api.py` are thin adapters. Logic lives in core modules
  (`src/forgelm/<area>/`) that know nothing about HTTP or the terminal.
- Avoid unnecessary abstractions and dependencies. No large refactors without explaining first
  and getting approval.
- The version lives only in `pyproject.toml`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest                          # tests
ruff check . && ruff format .   # lint + format
forgelm serve --reload          # http://127.0.0.1:8000/health, /docs, /ui
```
