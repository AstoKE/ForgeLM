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
- ⏳ **Next: Sprint 2b, Neural bigram in PyTorch.** Same model as a V×V logits matrix trained
  with gradient descent. Show that its loss converges to the counting model's (~2.45). Concepts:
  tensors, one-hot/embedding lookup, autograd, backprop, optimizer, learning rate. Decide on
  CPU vs CUDA torch install (machine has an RTX 4060 8 GB). Start with a concept briefing.

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
forgelm serve --reload          # http://127.0.0.1:8000/health, /docs
```
