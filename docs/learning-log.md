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

---

## Sprint 1b: Byte-level BPE from scratch (2026-09-27)

### Concepts
- UTF-8 bytes as the base vocabulary (256 ids): **no OOV and no `<unk>`**. `İ` is 2 bytes and 👍🏽 is 8
- The BPE training loop: count adjacent pairs → merge the most frequent into a new id → repeat
- A trained BPE tokenizer is an **ordered list of merges**. Later merges build on earlier ones
  (`aa` → `aaa` → `aaab`)
- Encoding replays merges in the order they were learned (lowest id first). It ignores
  frequencies in the new text
- `vocab_size = 256 + merges learned`, a hyperparameter that trades sequence length against
  embedding table size
- `typing.Protocol` (structural typing): `CharTokenizer` and `BPETokenizer` both satisfy
  `Tokenizer` without inheriting from it
- API input limits (`max_length`, `le=`) protect the server from expensive requests

### What we built
- `BPETokenizer.train/encode/decode/tokens`, `get_pair_counts`, `merge`
- `Tokenizer` protocol, plus `build_tokenizer()` and `analyze()` moved to `analysis.py`
- `forgelm tokenize --tokenizer bpe --merges N`; the API gets `tokenizer` and `num_merges`
  (0–1000) fields
- 24 new tests (49 total)

### Experiment: 300 merges trained on our own docs (~11 KB) vs GPT tokenizers

| Tokenizer | vocab | "the tokenizer learns merges" | "İstanbul çok güzel" |
|---|---|---|---|
| ForgeLM char | 109 | 1.00 chars/token | 1.00 |
| ForgeLM BPE, 0 merges | 256 | 1.00 | **0.86** (bytes > chars) |
| ForgeLM BPE, 300 merges | 556 | 3.00 (`the ` `tokeniz` `er ` `learn`…) | 1.12 |
| GPT-2 (`tiktoken`) | 50k | 4.50 | 2.00 (`İ`, `ç`, `ü` still split into bytes) |
| GPT-4o `o200k_base` | 200k | 6.75 (`the` ` tokenizer` ` learns` ` merges`) | 4.50 (` çok` ` güzel`) |

Takeaways:
- The tokenizer reflects its training corpus. Our English docs never taught Turkish merges.
  GPT-2 was English-heavy, and GPT-4o's tokenizer learned Turkish words.
- GPT tokens *start* with a space (` token`), while ours *end* with one (`the `).
  That's pre-tokenization (regex split before BPE), which we left out on purpose.
- Our first merges included `'    '` (code indentation) and `'\xe2\x94'`
  (part of the `─` box-drawing character in our diagrams): the corpus's quirks become tokens.
- Training 300 merges on 11 KB took ~0.5 s. Naive BPE is O(merges × corpus length).

### Mistakes / surprises
- My prediction for "aaabdaaabac" after 1 merge was 2 tokens. The actual result is **9**:
  one merge only replaces the single most frequent pair `(a, a)`, and merges don't overlap
  (`aaab` → `[aa, a, b]`).
- ruff B905: `zip(ids, ids[1:])` pairs lists of different lengths on purpose.
  `itertools.pairwise` states the intent.
- Windows + pipes: `forgelm tokenize "İç" | grep ...` printed `�`. With output piped, Python on
  Windows encodes stdout as the legacy code page (**cp1254** on Turkish Windows), not UTF-8.
  Fix: `$env:PYTHONUTF8 = "1"` (Linux uses UTF-8 by default).
- _(fill in)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 2a: Counting bigram language model (2026-09-28)

### Concepts
- A language model gives **P(next token | context)**. Generating text means sampling from that
  distribution, appending, and repeating
- **Bigram**: the context is only the previous token
- Training by counting: a V×V **count matrix**, and each row normalized into probabilities
- **Add-k smoothing**: without it, an unseen pair has P = 0, and −log 0 = ∞
- **Logits → softmax**: logits = log-probabilities, softmax = exp / sum. Subtract the max for
  numerical stability
- **Cross-entropy loss** = average −ln P(actual next token). **Perplexity** = exp(loss), roughly
  "how many equally likely choices"
- **Baseline**: a uniform model has loss ln V. Every model has to beat it
- **Train/val split**, kept contiguous to avoid leakage. The gap between the two losses is how
  you detect overfitting
- **Temperature**: logits / T before softmax. T=0 is greedy (argmax)
- **Checkpoints** hold the model *and* its tokenizer, stored as JSON because pickle can execute
  code on load

### What we built
- `forgelm.models.BigramModel` (`train`, `probs`, `logits`, `loss`, `generate`) and `softmax()`
- `train_on_text()` (tokenize → split → count → measure), `save_checkpoint` / `load_checkpoint`
- `CharTokenizer.to_dict/from_dict`
- `forgelm train-bigram` and `forgelm generate`; 36 new tests (85 total)

### Experiment: Tiny Shakespeare (1.1M chars, V = 66)

| | train loss | val loss | perplexity |
|---|---|---|---|
| uniform baseline | 4.190 | 4.190 | 66.0 |
| bigram, smoothing 0 | 2.452 | **∞** (val contains a pair never seen in train) | ∞ |
| bigram, smoothing 0.01 | 2.452 | 2.488 | 12.0 |
| bigram, smoothing 1 | 2.455 | 2.482 | 12.0 |
| bigram, smoothing 100 | 2.621 | 2.634 | 13.9 (too much smoothing flattens what was learned) |

- Training takes ~0.75 s, and the checkpoint is 15 KB of readable JSON
- Learned facts: after `q`, `u` has P = 0.90; after `:`, `\n` has 0.84
- Train ≈ val loss: a bigram is too simple to overfit 1M characters
- Sampling "ROMEO:" at different temperatures:
  - **T = 0 (greedy):** only newlines. After `:` the most likely token is `\n`, and after `\n` it's
    `\n` again, so greedy decoding gets stuck in a loop
  - **T = 0.5:** repetitive, `the the t the`
  - **T = 1.0:** looks like Shakespeare (names, line breaks, `'d`) but the words are nonsense
  - **T = 2.0:** close to random. `<unk>` and `$` appear because smoothing gave them a small
    probability and a high T amplifies it

### Mistakes / surprises
- My prediction: "the uniform loss is 65/66, and it can go down to 1/66". That mixes up
  **probability** and **loss**. For a uniform model, P = 1/66 and loss = −ln(1/66) = ln 66 ≈ 4.19.
  A loss of 1/66 ≈ 0.015 would mean P(correct) ≈ 0.985 on every character, which is impossible
  when many characters can follow a newline. The actual result is 2.48, perplexity 12.
- Writing Python code through a bash heredoc turned `"\n"` into a real newline, which broke
  `cli.py`. ruff and pytest caught it immediately.
- _(fill in)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_
