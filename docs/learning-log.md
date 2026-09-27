# ForgeLM learning log

This is a draft written by my AI pair programmer. **I edit it in my own words.**

---

## Sprint 0: Engineering setup (2026-09-25)

### Concepts
- `src/` layout and why tests should import the installed package
- `pyproject.toml` as the single project config (PEP 621)
- Virtual environments and editable installs (`pip install -e .`)
- Console-script entry points (`forgelm` → `forgelm.cli:main`)
- FastAPI (framework) vs Uvicorn (ASGI server); health endpoints
- Testing an API in memory with `TestClient`; faking side effects with `monkeypatch`
- Linting and formatting with ruff
- Thin adapters (CLI/API) over core logic

### What we built
- The `forgelm` package, with its version read from package metadata
- `GET /health` → `{"status": "ok", "version": "0.1.0"}`
- `forgelm --version` and `forgelm serve [--host --port --reload]`
- 6 tests covering the API and the CLI (including failure cases)
- README, architecture doc, ADR 0001

### Mistakes / surprises
- Starlette warned that plain `httpx` is deprecated for `TestClient`, so we switched the dev
  dependency to `httpx2`. Lesson: read test warnings, because they announce future breakage.
- _(fill in: anything that broke during setup or confused you)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 1a: Character tokenizer (2026-09-27)

### Concepts
- Token, vocabulary, `vocab_size`; `itos` (id → string) and `stoi` (string → id)
- Encode/decode and the **round-trip** rule: `decode(encode(text)) == text`
- Deterministic ids: `sorted(set(corpus))`, so the same corpus always gives the same ids
- Out-of-vocabulary text and `<unk>`: a lossy fix that breaks the round-trip
- Python `str` is made of Unicode **code points**, not visible characters (👍🏽 = 2, NFD é = 2)
- Pydantic request models: FastAPI returns 422 for invalid JSON before our code runs

### What we built
- `forgelm.tokenizer.CharTokenizer` (`from_corpus`, `encode`, `decode`, `tokens`)
  and `analyze()`, which returns tokens, ids and stats
- `forgelm tokenize TEXT [--corpus FILE]` and `POST /tokenize`
- 19 new tests (25 total)

### Experiment
`forgelm tokenize "İstanbul çok güzel" --corpus english.txt` gives 3 `<unk>` tokens
(`İ`, `ç`, `ü`). A vocabulary built from English text can't represent Turkish.
Sprint 1b (BPE on bytes) fixes this.

### Mistakes / surprises
- `decode([-1])` would silently return the *last* vocab entry, because Python accepts negative
  indexes. We check the range explicitly.
- _(fill in)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_
