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

---

## Sprint 2b: Neural bigram in PyTorch (2026-09-28)

### Concepts
- **Tensor** and **device**: the same code runs on CPU or GPU (`device="cuda"`)
- **Embedding lookup**: `W[prev]` gives exactly the same result as `one_hot(prev) @ W`, and is
  much cheaper
- Hand-written **cross-entropy** via `logsumexp`, which matches `F.cross_entropy`
- **Autograd / backprop**: `loss.backward()` fills `W.grad`. For this model the gradient is
  `(softmax − one_hot) / N`, routed back to the row `prev`; a test checks it
- **Gradient descent**: `W -= lr * W.grad` inside `torch.no_grad()`, then reset the gradient.
  This is the same as `torch.optim.SGD`
- **Learning rate**: the most important hyperparameter. **Mini-batches** make each step cheap
  but noisy (SGD)
- Optional heavy dependency: torch is in the `ml` extra, imported lazily
  ([ADR 0004](decisions/0004-pytorch-optional-extra.md))

### What we built
- `forgelm.models.neural_bigram`: `NeuralBigram` (W starts at zeros), `cross_entropy`,
  `sgd_step`, `train_neural_bigram` / `train_neural_on_text`, `pick_device`
- `encode_and_split()` extracted from `train_on_text`, so both models use the same split
- `forgelm train-neural-bigram` prints the loss curve next to the counting model's numbers
- 18 new tests (103 total), including autograd vs. hand-derived gradient

### Experiment: Tiny Shakespeare, batch 32,768, RTX 4060

Learning-rate sweep (300 steps, train loss at steps 0 → 100 → 200 → 300):

| lr | loss | verdict |
|---|---|---|
| 0.1 | 4.190 → 4.138 → 4.088 → 4.040 | far too small: barely moves |
| 1 | 4.190 → 3.741 → 3.454 → 3.274 | too small |
| 10 | 4.190 → 2.843 → 2.694 → 2.630 | OK but slow |
| **50** | 4.190 → 2.571 → 2.517 → **2.497** | good |
| 500 | 4.190 → 5.581 → 6.070 → 5.481 | too large: loss goes *above* the uniform baseline and jumps around |
| 5000 | 4.190 → 107 → 116 → 122 | diverged. No `nan`, because logsumexp stays stable; the loss just explodes |

Long runs (3000 steps, ~15 s on GPU):

| | train | val |
|---|---|---|
| neural, lr 50 | 2.457 | 2.487 |
| neural, lr 100 | **2.456** | 2.485 |
| counting, smoothing 1 (2a) | 2.455 | 2.482 |
| counting, smoothing 0 = best possible train fit for *any* bigram | 2.4519 | ∞ |

- **Gradient descent found the same answer as counting.** The neural model's loss converges to the
  counting model's. No bigram can go below 2.4519 on the training data: counting with no
  smoothing gives the exact optimum (the maximum-likelihood estimate).
- **CPU vs GPU:** 300 steps took 7.0 s on CPU and 1.8 s on GPU (~3.9×), with identical losses.
  The model is tiny (4,356 params), so the GPU barely has work to do. The gap grows with model size.
- Rare characters train slowly: a row of W only gets gradient when that character appears in the
  batch. That's why plain SGD needs thousands of steps for the last 0.04 of loss.

### Mistakes / surprises
- My prediction for step 0: ln 66 ≈ 4.19. **Correct**: W = 0 means uniform probabilities.
- My prediction that the neural model could go lower than counting: only marginally. It can beat
  smoothed counting on *train* (2.455) by at most 0.003, and it can never beat 2.4519, the
  unsmoothed count solution.
- Test with lr = 1e30 expected `nan` but didn't get it. On perfectly predictable data
  (`1→2→3→4→1…`) a huge step makes the model 100% confident, which is correct, so the loss went
  to 0. Real `nan` only appears when W overflows float32 (~3.4e38); the test now uses lr = 1e39.
- A 5000-step run with lr = 500 took 54 minutes instead of ~30 s. I couldn't reproduce it; the same
  runs later took normal time. Most likely the laptop slept or throttled the GPU. (Also, lr = 500
  was a bad choice: the sweep had already shown it diverges.)
- `import torch` prints "Failed to initialize NumPy". Torch uses NumPy only for interop, and we
  don't need it. The warning is harmless.
- Quiz, checked with experiments on Tiny Shakespeare:
  - *Random init W ~ N(0, s):* step-0 loss was 4.191 (s = 0.01), 4.680 (s = 1) and 23.2 (s = 10).
    It depends on the scale, but it never beats ln V: random confidence is confidently wrong.
    That's why weights start small.
  - *Removing `W.grad = None`:* I guessed "loss wouldn't change". Wrong: gradients **accumulate**,
    so each step uses the sum of all past gradients and overshoots. After 300 steps the loss was 3.40
    instead of 2.50, and it was rising.
  - *Why lr = 500 goes above the uniform baseline:* each step overshoots the minimum and the
    model becomes confidently wrong. After a space it predicted `c` instead of `t`, and 18% of
    training pairs got P(correct) < 0.001, which costs −ln 0.001 ≈ 6.9 each. A uniform model always
    gives P = 1/66, so it can't do worse than ln 66. A confident model can.
  - *Weight decay ≈ ?* I guessed "gradient descent". Wrong: it corresponds to **smoothing**. It pulls
    W toward 0, which means toward uniform probabilities. With wd = 1, W max dropped from 10.5 to 5.0
    and train loss rose from 2.456 to 2.771, like counting with smoothing 100. It's a term added
    to the loss that gradient descent then minimizes, not gradient descent itself.
- _(fill in)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 3a: Causal averaging, the skeleton of attention (2026-09-29)

### Concepts
- The bigram's problem: it only sees the last token. "...t" could be followed by h, o or a space,
  while "I want t" is almost surely followed by "o"
- **Embedding**: each token is a list of numbers (its "profile"), like the rows of W in 2b
- Simplest way to use the past: position t takes the **average** of positions 0..t
- **Causal mask**: a lower-triangular table of who may look at whom. No looking at the future,
  because the future is exactly what we're trying to predict
- The same average three ways: a loop, one matrix multiply with a weight triangle
  (`[1,0,0] [1/2,1/2,0] [1/3,1/3,1/3]`), and **mask + softmax** (future = −inf → weight 0),
  which is the shape attention uses

### What we built
- `forgelm.models.attention`: `causal_mask`, `causal_average_loop/matmul/softmax`,
  `uniform_causal_weights`, `masked_softmax_weights`
- 16 new tests (119 total). Key test: changing a future token doesn't change earlier outputs

### Mistakes / surprises
- My prediction for `[10, 0, 5]`: "the average is 5". That's only position 2, `(10+0+5)/3 = 5`.
  Each position has its own average: **`[10, 5, 5]`**.
- My prediction: "changing 5 to 100 changes position 0". **No**: position 0 only sees itself, so
  it stays 10. Position 1 stays 5 and only position 2 changes (to 36.67). That's the causal mask:
  information flows only from the past to the future.
- Quiz:
  - `[4, 8]` → I answered "6". Again only the last position; the output has **one value per
    position**: `[4, 6]`. This is my recurring mistake: I think of *one* average for the whole
    sequence, but every position gets its own.
  - Why is the last cell of row 2 zero? "0 means don't look". Right, and the reason is that
    position 3 is in the **future** for position 2.
  - Huge score (1e9) on a future cell? I answered "the average goes up a lot". **No**: the mask
    *overwrites* that score with −inf *before* softmax, so its weight is exactly 0. Output stayed
    `[10, 5, 36.67]`. The mask doesn't compete with the score, it replaces it.
- _(fill in)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 3b: Single-head self-attention (2026-09-30)

### Concepts
- Every position makes three vectors from its `x`: **query** ("what am I looking for?"),
  **key** ("what do I contain?") and **value** ("what do I give if chosen?"), each `x @ W`
- **Score** `q_t · k_s` says how relevant position s is for position t. All pairs at once:
  `q @ k.T`, a `(T, T)` table (the learned version of 3a's triangle)
- **Scaling by √head_size**: dot products grow with the vector length, and big scores make
  softmax "winner takes all"
- Then the 3a machinery: mask + softmax, and `output = weights @ v`
- Zero Q/K means equal scores, which is exactly the 3a average, so 3b generalizes 3a
- Random init (not zeros): with equal matrices every head would learn the same thing

### What we built
- `forgelm.models.attention`: `attend(q, k, v)` and `SelfAttentionHead` (`Wq`, `Wk`, `Wv`
  written by hand, no `nn.Linear`). `forward` returns the output and the weights
- 9 new tests (128 total). Key ones: a hand-computed example, zero Q/K equals the 3a average,
  changing the future doesn't change the past, gradients reach all three matrices
- Not trained yet, so the weights are random and mean nothing. Training comes in 3d

### Experiment: hand example, 3 tokens, head_size 4
Scaled scores for the last token: `[2, 0, 1]` -> weights `[0.665, 0.090, 0.245]` ->
output `[7.876, 2.124]`. The 3a average gave `[5, 5]`. (The briefing rounded this to
`[7.875, 2.125]`; the exact result is 7.876 / 2.124.)

### Mistakes / surprises
- Quiz 1, scores `[2, 0, 1]` -> `[2, 5, 1]`: I said "the weights change proportionally to the
  0 -> 5 increase". Two things wrong. (a) It is not proportional but **exponential**:
  `exp(5) = 148`, so token 1 becomes 148x heavier before normalising. (b) The real surprise is
  **token 0**: its score never changed, yet its weight fell from 0.665 to 0.047. Every row of
  softmax is one pie that always sums to 1, so it is a **competition**, not an independent
  measurement. If one position takes more, the others must lose.
- Quiz 4, `Wq = Wk = 0`: I said "attention stays constant". Half right. The **weights** become
  constant, and exactly 3a's triangle (`[1,0,0] / [1/2,1/2,0] / [1/3,1/3,1/3]`), so attention
  collapses into the plain causal average. But the **output** is not constant, because
  `v = x @ Wv` still depends on `x`. Attention loses its "who looks at whom" decision, not the
  information it carries.

### Quiz (answered 2026-09-30)
1. Scores `[2, 0, 1]`: if token 1's score goes from 0 to 5, what happens to the weights of
   token 1 and token 0? -> **Wrong**, see above. `[0.665, 0.090, 0.245]` -> `[0.047, 0.936, 0.017]`.
2. If I change the last row of `x`, which output rows change? -> **Correct**: only the last row.
   The last row of `x` changes `q_T`, `k_T` and `v_T`, but position `t < T` only looks at
   `s <= t`, so it never sees it. That is the causal mask.
3. `head_size = 64` and no `/ sqrt`: do the scores get bigger or smaller, and what does softmax
   do? -> **Did not know.** They get **bigger**: a score is the sum of `head_size` products, so
   it grows with about `sqrt(64) = 8`. Measured on random q/k, `T=8`, `H=64`:
   raw scores std 7.43 (min -22.69, max 13.97) vs scaled std 0.93 (min -2.84, max 1.75).
   The largest weight in a row is 0.999 raw vs 0.429 scaled. So without scaling softmax becomes
   winner-takes-all at initialisation: the model commits 99.9% to a **random** token, softmax
   behaves like argmax, gradients go to ~0 and training stalls. `/ sqrt(head_size)` pulls the
   spread back to ~1 and keeps softmax soft enough to be corrected.
4. If `Wq` and `Wk` are zero, what does attention become? -> **Half right**, see above: the 3a
   causal average (`test_zero_qk_matches_3a_average` covers this).

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 3c-1: Multi-head attention and the feed-forward network (2026-09-30)

### Concepts
- One head learns **one** kind of relation. Several heads run side by side, each looking for
  something else; their outputs are concatenated and mixed once by `Wo`
- The channels are **split**, not duplicated: `head_size = embed_dim // num_heads`, so the
  concatenation is `embed_dim` wide again and the block keeps `(T, C) -> (T, C)`. That is what
  makes the residual `x = x + attention(x)` possible in 3c-2
- Every head needs its **own seed**. Identical heads would stay identical forever, which is just
  one head computed `num_heads` times
- Attention only *moves* information (`weights @ v` is a weighted sum). The **FeedForward**
  *processes* it: `relu(x @ W1 + b1) @ W2 + b2`, widening `C -> 4C` and back
- FFN is **position-wise**: each row goes through the same small network alone. Attention mixes
  positions, the FFN never does
- **Why the nonlinearity matters**: without `relu`, `(x @ W1) @ W2 = x @ (W1 @ W2)` is a single
  matrix, so widening to 4C buys nothing at all. Measured: `W1 @ W2 = [[0, 6], [3, 4]]`, and
  `x = [1, -2]` gives `[-6, -2]` instead of the network's `[1, 1]`
- Most of a transformer's parameters live in the FFN (~8C^2) rather than in attention (~4C^2)

### What we built
- `attention.MultiHeadAttention`: a list of `SelfAttentionHead`s (seed + i), `torch.cat` of their
  outputs, `@ Wo`. `forward` returns `(T, C)` and the weights stacked per head `(num_heads, T, T)`
  -- those are what an attention heatmap will draw later
- `models/block.py` (new): `FeedForward`, hand-written `W1/b1/W2/b2`, `hidden_dim = 4 * embed_dim`
  by default. Biases start at zero; the weights already break the symmetry
- 14 new tests (142 total). Key ones: `num_heads=1` with `Wo = I` reproduces a single 3b head
  exactly; different heads produce different weights; causality survives `Wo` (it mixes channels,
  never positions); changing one row of the FFN input leaves every other row untouched; equal
  rows give equal outputs

### Experiment: feed-forward by hand, C=2, hidden=4
`x = [1, -2]` -> `x @ W1 = [1, -2, 0, -3]` -> `relu` -> `[1, 0, 0, 0]` (only unit 0 fired)
-> `@ W2` -> `[1, 1]`. Without the relu the same input gives `[-6, -2]`.

### Mistakes / surprises
- `pytest.approx` does not accept nested lists: `approx([[1.0, 1.0]])` raises `TypeError`.
  Compare one row at a time, `out[0].tolist() == pytest.approx([1.0, 1.0])`
- Quiz 3: I guessed "the tests about negatives would fail". True, but I missed the point of the
  question. Deleting the `relu` really does fail exactly 2 of the 8 tests, and
  `test_every_position_is_processed_on_its_own` **keeps passing**. Position independence does not
  come from the `relu`, it comes from the shape of the matmul. The two properties are separate:
  `@ W` gives position independence, `relu` gives the capacity to compute something. Without the
  `relu` the FFN is still position-wise, only meaningless.
- Quiz 4: I did not know why `Wo` cannot leak the future. The rule is **which side the matrix
  multiplies from**: `weights @ v` has the matrix on the **left**, so it mixes rows (positions)
  and needs the mask. `concat @ Wo` has it on the **right**, so it mixes columns (channels) only.
  Every output row of `concat @ Wo` is computed from its own input row alone, so leaking is not
  possible. A `(T, T)` matrix multiplied from the left would leak -- and that is exactly what
  attention does, which is why attention is the part that needs a mask.

### Quiz (answered 2026-09-30)
1. `embed_dim = 64, num_heads = 8` -> how many channels per head, and what changes at 16?
   -> **Correct**: 8 channels, then 4. And the trade-off: more heads look for more different
   relations, but each head looks through a narrower window (`q . k` over 4 numbers is a coarse
   match). The total compute stays the same; what changes is width vs depth of matching. Real
   models keep `head_size` near 64 (GPT-2: 768 channels, 12 heads, head_size 64).
2. Same seed for every head -> **Correct**: they would stay identical forever, because equal
   weights and equal input give equal gradients. You pay for `num_heads` heads and get one.
3. Delete the `relu` -> which tests fail? -> **Half right**, see above.
4. Can `Wo` leak the future? -> **Did not know**, see above.

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 3c-2: The transformer block, residuals and LayerNorm (2026-09-30)

### Concepts
- **Residual** `x = x + block(x)`: a motorway with a service road next to it. Information flows
  along the main road untouched; the block only proposes a correction to add
- The real reason is the gradient: in the backward pass an **addition copies the gradient to
  both branches unchanged**, so there is a path from the deepest block to the input that never
  passes through a matmul. Without it the gradient is multiplied at every layer and decays
- A residual also means a block that learned nothing useful just adds ~0 and gets out of the way
- **LayerNorm**: residuals keep adding, so the numbers grow as blocks stack. Big numbers break
  softmax (winner-takes-all, see the 3b quiz) and saturate `relu`. LayerNorm pulls each position
  back to mean 0 / std 1, then applies a learned `gamma` (scale) and `beta` (shift)
- `gamma = 1`, `beta = 0` at the start, so the layer begins as the plain normalisation and the
  model can undo part of it if that helps. Position-wise, like the FFN: a row uses its own
  numbers only, never a batch's and never its neighbours'
- **Pre-norm vs post-norm**: the 2017 paper wrote `x = LN(x + block(x))`, GPT-2 and everything
  after it write `x = x + block(LN(x))`. In pre-norm the main road never passes through a
  LayerNorm, so the gradient path stays clean and deep stacks train without warmup. We use
  pre-norm

### What we built
- `block.LayerNorm`: hand-written `gamma`/`beta`, `var(unbiased=False)` (divide by C, like
  `nn.LayerNorm`), `eps = 1e-5` against a constant row
- `block.TransformerBlock`: `ln1 -> MultiHeadAttention -> residual -> ln2 -> FeedForward ->
  residual`. `forward` returns `(T, C)` plus the attention weights. The FFN gets
  `seed + num_heads + 1` so it cannot start as a copy of head 0
- 13 new tests (155 total). Key ones: the hand-written LayerNorm matches
  `F.layer_norm`; a 100x bigger input comes out the same; a constant row gives no NaN; and the
  **residual proof** -- zero `Wo` and `W2` makes the whole block the exact identity

### Experiment: 6 blocks stacked, embed_dim 32
`max|x|` layer by layer:
```
LayerNorm on : 10.2 ->  9.6 ->  9.9 ->  9.8 -> 10.4 -> 10.6 -> 10.8
LayerNorm off: 10.2 -> 15.6 -> 27.1 -> 44.9 -> 60.1 -> 108.1 -> 138.1
```
And the 20-layer gradient test: the gradient reaching the input is 0.062 without residuals
against 4.406 with them, about 70x.

Parameter split of one block (embed_dim 64, 8 heads): 49,728 total, of which attention 16,384
(= 4C^2) and feed-forward 33,088 (~8C^2). Two thirds of a transformer is the MLP.

### Quiz (answered 2026-09-30)
1. What happens if only `ln1.gamma` is zeroed? -> `ln1(x) = 0 * normalised + beta = 0`, so
   attention gets an all-zero input: `q = k = v = 0`, the scores are equal and the weights become
   the plain causal average (row 2 measured as `[1/3, 1/3, 1/3, 0, 0]`) -- but `v = 0`, so the
   branch adds exactly 0. The FFN branch keeps working on the untouched `x`, so the block is
   `x + ffn(ln2(x))`, **not** the identity. Note this lands in the same place as `Wq = Wk = 0`
   from the 3b quiz: zeroing the input and zeroing Q/K both equalise the scores.
2. Would the identity test pass with post-norm? -> **No.** `LN(x + 0) = LN(x)` is not `x`
   (verified). In post-norm a block can never step out of the way; it always rescales its input.
   That is one reason we chose pre-norm.
3. Isn't LayerNorm a loss of information? -> Two answers. (a) `gamma` and `beta` are learned, so
   the model can put the scale and the shift back; normalisation is a starting point, not a rule.
   (b) The pre-norm one: LayerNorm never touches the main road, only the **copy** handed to the
   sublayer (`x = x + attention(ln1(x))`). The residual stream carries the raw `x` through
   untouched, so nothing is lost. This would not hold in post-norm.
4. 6 blocks of 49,728 -> `298,368`, and **no**, that is not MiniGPT's total: the two embedding
   tables, `ln_final` and the head are parameters too.

### Mistakes / surprises
- _(fill in)_

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 3c-3: MiniGPT, the whole model (2026-09-30)

### Concepts
- Attention scores `q . k` are built from **content only**, so "the dog bit the man" and "the
  man bit the dog" would look identical. The causal mask carries a little order information
  ("5 tokens came before me") but not which token sat where
- **Position embedding**: a second table, one row per seat. `x = token_embedding[ids] +
  position_embedding[:T]`. One table answers "what is this token?", the other "where am I?".
  Both are learned; the model discovers by itself that seat 2 differs from seat 5
- The height of that table is the model's hard limit. No seat `block_size + 1` exists -- that
  is exactly what a **context window** is, and why `generate` crops to the last `block_size`
  tokens
- The end of the stack: a final LayerNorm, then one `(C, V)` matrix -> **logits**, the same
  output the bigram produced, except this model looked at the whole past instead of one token
- An embedding lookup is `table[ids]`, the same trick as `NeuralBigram`'s `W[prev]`: equal to
  `one_hot(ids) @ table` but much cheaper

### What we built
- `models/minigpt.py`: `MiniGPT(vocab_size, embed_dim, num_heads, num_blocks, block_size)` with
  `forward` (returns logits plus the attention weights of every block), `loss` (reuses 2b's
  hand-written `cross_entropy`), `generate` (temperature, greedy at 0, context cropping) and
  `num_parameters`
- 17 new tests (172 total). Key ones: changing the last token leaves every earlier row of logits
  untouched; the same token in two seats gives different logits, and zeroing the position table
  makes them equal again (so the difference really is the position embedding); a token the batch
  never saw gets exactly zero gradient; and the end-to-end test below

### Experiment: memorising one sentence
`"to be or not to be"`, 7-char vocab, embed_dim 32, 2 blocks, 26,688 parameters, plain SGD
(lr 0.1, no optimizer), 300 steps in 0.7 s on CPU:
```
step   0 | loss 4.3457      (ln 7 = 1.946)
step  50 | loss 0.0084
step 300 | loss 0.0009
greedy: 'to be or not to be'
```
A bigram **cannot** do this: 'o' is followed by ' ', 'r' and 't', and 't' by both 'o' and ' ',
so one token of context cannot choose. MiniGPT reads the whole past, so it reproduces the
sentence exactly. This is the first time the model is provably using context.

Sizes: `vocab 66, embed_dim 64, 4 heads, 4 blocks, block_size 128` -> 215,680 parameters in 89
tensors.

### Mistakes / surprises
- The untrained loss is 4.60 while `ln(66) = 4.19`, i.e. slightly **worse** than guessing. Random
  weights do not give uniform logits: the head is `randn * C^-0.5`, so the logits start with a
  spread of about 1 and confidently prefer the wrong tokens. Training fixes it in a few steps.
  A smaller head init would start closer to `ln(V)`

### Quiz (answered 2026-09-30)
1. `block_size = 128` and a 500-token input -> `forward` raises
   `ValueError: sequence of 500 tokens is longer than block_size 128`; `generate` keeps the last
   128 tokens. Deliberately different: in a training loop an over-long sequence is a **bug** in
   how the batch was built and should shout rather than learn something wrong, while at inference
   time cropping is the correct behaviour -- the model physically cannot see past seat 128. This
   is what a "context window" is: it cannot summarise a 500-page book because seat 129 does not
   exist.
2. How many training examples in one forward pass? -> **T**. `logits` is `(T, V)` and every row is
   its own prediction. The **causal mask** is what allows it: row `t` only saw positions `0..t`,
   so it never saw its own answer. Without the mask row 5 would already have seen token 6 -- that
   is copying, not predicting -- and you would need T separate forward passes on truncated
   inputs. This is the single biggest reason transformers train fast, and something an RNN cannot
   do.
3. Does memorising the sentence mean the model is good? -> **No.** 26,688 parameters for 18
   characters is room to store the answer, and the overfitting is deliberate. It **proves** the
   wiring: gradients reach all 89 tensors, residual / LayerNorm / attention / embeddings are
   connected correctly, and the model really uses context (a bigram cannot produce this string).
   It proves **nothing** about generalisation -- there is no validation split yet. It is a unit
   test, not a benchmark.
4. How much of the 215,680 parameters is embeddings? ->
   ```
   token_embedding         4,224    2.0%
   position_embedding      8,192    3.8%
   blocks (4x)           198,912   92.2%
   ln_final                  128    0.1%
   head                    4,224    2.0%
   both tables            12,416    5.8%
   ```
   Tiny here, because the vocabulary is 66 characters. GPT-2 small inverts it: token_embedding is
   38,597,376 of ~124M (**31.1%**) and the position table only 786,432 (0.6%). A third of the
   model sits in one table, which is exactly why GPT-2 ties the head to `token_embedding` --
   keeping them separate would add another 31%. Also worth noticing: knowing *where* you are is
   about 50x cheaper than knowing *which token* you are.

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_


---

## Sprint 3d: Training MiniGPT and saving it (2026-10-05)

### Concepts
- **Random windows**: cut `T` tokens from a random place, and `y` is the same window shifted by
  one. One window is `T` training examples (3c-3)
- **Adam** keeps two running averages per parameter, `m` (the gradient) and `v` (the gradient
  squared), and steps by `lr * m / (sqrt(v) + eps)`. Dividing by `sqrt(v)` cancels the size of the
  gradient: with a constant gradient every step is about `lr`, whether the gradient is 0.01 or
  100. SGD would move those two parameters 10,000x apart. Bias correction `1 - b^t` fixes `m`
  and `v` being too small at the start (they begin at 0)
- **Perplexity** = `e^loss`: roughly "between how many equally likely tokens the model hesitates".
  66 for guessing, 12 for the bigram, 6.4 for MiniGPT
- **Train/val gap**: train 1.68 vs val 1.85. Mild overfitting. The warning sign is val loss
  rising while train keeps falling. Fixes: stop early (keep the best-val checkpoint), more data,
  a smaller model, dropout or weight decay
- **Epochs**: 2000 steps x 16 windows x 64 tokens is about 2 passes over the 1M-character corpus
- **A checkpoint** is weights + hyperparameters + tokenizer. `torch.load(weights_only=True)`
  refuses anything but tensors and basic values, so a shared file cannot run code
  ([ADR 0005](decisions/0005-minigpt-checkpoint.md))

### What we built
- `models/train_minigpt.py`: `get_batch`, `Adam`, `estimate_loss`, `train_minigpt`,
  `train_minigpt_on_text` (3d-1, 18 tests). The Adam test follows `torch.optim.Adam` for 25 steps
- `save_minigpt` / `load_minigpt` with checks for the version, the tensor count and every shape
  (3d-2). A test shows a pickle with hidden code is refused
- `forgelm train-minigpt`, and `generate --model x.pt` (the extension picks the loader)
- 203 tests in total (75 new since 3b's 128 = 44 from 3c, 18 + 13 from 3d)

### Experiment: Tiny Shakespeare, 211,584 params, 4 blocks, embed 64, context 64, Adam lr 3e-3

| | val loss | perplexity |
|---|---|---|
| uniform | 4.190 | 66 |
| bigram | 2.482 | 12 |
| MiniGPT, 2000 steps | **1.853** | **6.4** |

- Output has Shakespeare's shape (`QUEEN ELIZABETH:` and line breaks) but is still nonsense words
- Same seed, same text before and after saving and loading, so the round trip loses nothing
- Checkpoint: 871 KB, against 4.5 MB for the same weights as JSON text (5.2x)
- **The GPU was slower than the CPU**: 22.3 s vs 11.6 s per 100 steps, with identical losses. The
  batch is a loop over 16 windows, each made of hundreds of tiny operations, so launching the
  work costs more than doing it. A `(B, T)` batch dimension would fix it

### Mistakes / surprises
- Warm-up: constant gradient 0.01, `lr = 0.1`, 1000 steps. I answered SGD 1001 and Adam 1010.
  The answers are **1.0** and **100**. I counted steps instead of distance: `1000 x step size`.
  SGD's step is `lr x gradient = 0.001`; Adam's is about `lr = 0.1`
- Prediction for 20,000 steps: "val loss goes down". Probably true for a while, but 20,000 steps
  is about 20 epochs, and a model with 211k parameters can memorise 1M characters, so I expect
  val to flatten and then rise while train keeps falling. **Not measured** (about 40 min on CPU)
- My first `get_batch` copied the whole corpus to the CPU at every step. Keeping the data on
  the device and moving only the random starts fixed it
- `torch.load` raises a different exception for each kind of garbage file (`IndexError` for
  random bytes), so `load_minigpt` catches `Exception` once and raises one clear `ValueError`
- A CLI test failed because an untrained model sampled `<unk>`, which prints as 5 characters

### Quiz (answered 2026-10-05)
1. Train loss 1.675 and val loss 1.842. Is that overfitting? -> **Right conclusion**: mild
   overfitting. My reason was only "the gap is small". The better evidence is the **trend**: val
   was still falling together with train, and the val estimate itself wobbles by about +-0.05
   (8 random windows), so one gap number says little
2. Why does `load_minigpt` build a random model first? -> **Did not know.** A checkpoint holds
   only numbers. The structure (which tensor belongs to which layer, how `forward` uses them) is
   *code*. Building the model from `config` gives the empty shelves, with the right shapes, device
   and `requires_grad=True`; `copy_` then puts the numbers in place. The random values are
   overwritten and never used
3. Why is a checkpoint without `config` useless? -> **Vague**: "it wouldn't know where it stopped".
   That is a different thing (resuming training needs the optimizer state and the step count,
   which we do not save, see ADR 0005). Without `config` we would not know the **shape** of the
   model to build. Some of it can be guessed from tensor shapes (embed_dim, vocab_size), the
   number of heads only from counting tensors (3 per head), and nothing checks the guess. The
   config makes it explicit
4. Why is the GPU slower, and what would speed it up? -> **Did not know.** A GPU is fast at one
   big operation, but every operation has a fixed launch cost. We run 16 windows one by one, each
   made of many tiny operations, so launching costs more than computing. A `(B, T)` batch would
   push all 16 windows through each operation at once

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_


---

## Sprint 4a: Inference dashboard (2026-10-07)

### Concepts
- A dashboard is a window onto the model: what it generates, and what happens inside it
- **Thin adapter again**: `api.py` only turns HTTP into calls. `forgelm/inference/` holds the
  logic, raises three error types (invalid request, not found, torch missing), and the API maps
  them to 422 / 404 / 503
- **Load once, cache**: reading and rebuilding the model on every request is wasteful. The cache
  key includes the file's modification time, so a retrained checkpoint is reloaded
- **Path traversal**: if the request carried a file path, `"../pyproject.toml"` could read files
  outside `checkpoints/`. The API only takes a *file name*, and rejects anything with a folder,
  a leading dot or an unknown extension
- **Attention heatmap**: one cell per (looking token, looked-at token), colour = weight. The
  causal mask is the empty upper triangle. 4 blocks x 4 heads = 16 different maps
- Static HTML + vanilla JS, no framework or build step ([ADR 0006](decisions/0006-dashboard-ui.md)).
  `textContent` instead of `innerHTML`, so typed text can never become markup
- FastAPI dependencies (`Depends`) let tests swap the model folder for a temporary one

### What we built
- `forgelm/inference/local.py`: `ModelStore`, `generate_text`, `attention_maps`
- `GET /models`, `POST /generate`, `POST /attention`, `GET /ui`, with input limits like `/tokenize`
- `forgelm/ui/index.html`: Tokenizer, Generate and Attention panels
- 37 new tests (240 total): store, cache and reload, path traversal, corrupt files, torch missing,
  seeds, attention rows summing to 1 with a zero future, every API status code
- Checked in a real browser by driving headless Chrome: 15 tokens gave a 15x15 grid with 105
  masked cells (15 x 14 / 2), and an empty text showed the API's error

### Experiment: what a trained MiniGPT attends to ("ROMEO:", block 3, head 0)
```
R  1.00
O  0.04 0.96
M  0.13 0.36 0.52
E  0.00 0.01 0.98 0.01          <- E looks 98% at M
O  0.00 0.00 0.05 0.01 0.94     <- O looks 94% at itself
:  0.06 0.06 0.51 0.03 0.04 0.30
```
Rows sum to 1, the future is 0 (the mask), and unlike 3b the pattern is not random: heads
specialise. In block 0, head 0 looks mostly at the token itself (the diagonal).
*Heatmap caveat:* a bright cell shows where information is read from, not why. It is a
hint about the model, not an explanation.

### Mistakes / surprises
- Ruff B008 flagged `= Depends(get_store)` in a default argument. FastAPI's current idiom is
  `Annotated[ModelStore, Depends(get_store)]`
- The Generate panel's first model was `bigram.json`, only because the list is sorted by name.
  My first reading of the screenshot was "the MiniGPT output looks bad": it was the bigram

### Quiz (not asked yet, to do next session)
1. Why can the heatmap's upper triangle never have colour, and can training change that?
2. Why does the API take a model *name* and not a path?
3. Why is the model cached, and why does the cache key include the file's modification time?
4. A heatmap shows head 0 of block 3 looking 98% at "M". Does that prove the model "uses M" to
   predict the next letter?

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 4b-1: Training in the background (2026-10-08)

### Concepts
- An HTTP request must answer in milliseconds; training takes minutes. Training inside the
  handler leaves the browser spinning until it times out, and the page can show nothing
- **The ticket pattern**: `POST /train` starts the work and returns a job id straight away
  (**202 Accepted** = started, not finished). `GET /train/{id}` answers "where are you?"
- **Polling, not WebSocket/SSE**: a plain `GET` once a second. No new protocol, trivial to test
  with `TestClient`, and a loss curve refreshed once a second does not need push
- **A progress callback is inversion of control**: `train_minigpt` must not know about terminals
  or HTTP, so the caller hands it `on_progress(step, train_loss, val_loss)` and decides what to
  do with the numbers. The hook point already existed: the `record(step)` closure
- **The GIL question**: Python threads cannot run bytecode in parallel, but torch releases the
  GIL inside its C++ kernels, so training really progresses while FastAPI answers. A
  pure-Python training loop would need a separate process
- A thread that raises **dies silently**, so a failed run must be *recorded* (`status = "failed"`
  plus the exception text), never left to vanish
- One run at a time (**409 Conflict**): two runs would compete for the same CPU/GPU and both
  would crawl ([ADR 0007](decisions/0007-background-training-jobs.md))

### What we built
- `train_minigpt(..., on_progress=...)`, called at every evaluation
- The CLI passes a callback that prints each line as it happens. Before this, `forgelm
  train-minigpt` printed the whole history *after* training: 213 seconds of silence, then 21
  lines at once
- `forgelm/training/jobs.py`: `TrainingJobStore` with `start` / `get` / `latest` / `wait`, one
  `threading.Lock` for the job dict and every job's fields
- `GET /corpora`, `POST /train` (202), `GET /train/{id}`, and a 409 handler for `JobBusy`
- 42 new tests (282 total): the 4a path-traversal rule applied to the corpus *and* the output
  name, every hyperparameter capped, a real failed run, a refused second run, an unknown job id,
  and the 4b payoff - train through the API, then generate from what was just trained

### Experiment: is the callback really live?
200 steps on 60 KB of Shakespeare, recording wall-clock time at each callback:
```
t=  0.0s  step    0  val 4.456
t=  0.8s  step   50  val 3.040
t=  1.5s  step  100  val 2.968
t=  2.2s  step  150  val 2.659
t=  3.0s  step  200  val 2.703      (total 3.0s)
```
Evenly spread across the run, so the numbers arrive *while* training happens, not after.

Also retrained the 3d model to have a checkpoint again: 211,584 params, 2000 steps, 213.8 s on
CPU, val **1.867** (perplexity 6.5) against the bigram's 2.482 (12). The logged run got 1.853;
the difference is inside the +-0.05 wobble of an 8-window val estimate.

### Mistakes / surprises
- A test used `lr = 1e30` to force a failure, but pydantic caps `lr` at 1, so the request was
  rejected with 422 and the test got a `KeyError` on `job["id"]`. Replaced with a three-character
  corpus: the *request* is valid, so the run fails only after the thread has started -- which is
  exactly the path the test is meant to cover
- In block 0 / head 0 of the retrained model, the weights sit one cell left of the diagonal
  (`O -> R`, `M -> O`, `E -> M`, `: -> O`, all 0.85-0.99): a **previous-token head**, a bigram
  learned inside the transformer. The 4a run had that head on the diagonal instead. Same seed,
  same hyperparameters, different specialisation -- which head takes which job is not in the
  architecture, it falls out of training

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 4b-2: The training panel and the loss curve (2026-10-08)

### Concepts
- The page holds the **polling loop**, not the server: `POST /train` gives a job id, then a
  `setInterval` asks `GET /train/{id}` once a second and redraws. It stops itself as soon as the
  status is no longer `running`
- A **canvas has two sizes**: the CSS box it occupies and the pixel grid it draws on. Setting
  `canvas.width = clientWidth * devicePixelRatio` and scaling the context is what keeps the
  lines sharp instead of blurry
- The x axis is `step / total_steps`, not "the points we have", so the curve **fills the box
  from left to right** as the run proceeds and the shape does not jump around
- **Reload-safe**: `GET /train` returns the most recent run, so a page opened (or refreshed)
  while a run is going attaches to it instead of showing nothing
- Colours come from the CSS variables (`--accent` for train, `--val` for val), read with
  `getComputedStyle`, so the chart follows light and dark mode like the rest of the page

### What we built
- A Train panel: corpus dropdown (from `GET /corpora`), output name, steps, learning rate and
  device, with model size and batching tucked into a `<details>`
- `drawLosses` on a `<canvas>`: two lines, min/max loss labels, a dot on the newest value, and a
  redraw on window resize
- `GET /train` (the latest run), which also stops `TrainingJobStore.latest()` being dead code
- When a run finishes, the model lists are refreshed, so the checkpoint just written appears in
  Generate and Attention without a reload. The refresh keeps the user's current selection
- 2 new tests (284 total) plus a stricter `/ui` test: the new panel, the canvas, the new routes

### Experiment: the page driven in headless Chrome
Served on port 8791, rendered with `chrome --headless --dump-dom` and `--screenshot`:
- the dropdowns are filled from the API (`tinyshakespeare.txt`, `bigram.json`, `minigpt.pt`),
  so the JavaScript really ran
- during a run: `training tinyshakespeare.txt -> ui-demo.pt | step 0/1500 | train 4.347 |
  val 4.443`, the Start button rendered `disabled`, and the curve started at the left edge
- after a run: `done in 4.4s | final val loss 2.821 (perplexity 16.8) | saved ui-demo.pt`, with
  both lines drawn across the box

### Mistakes / surprises
- The shared `run()` helper re-enables its button in a `finally`, which would have unlocked
  "Start training" the moment `POST /train` returned, while the run was still going. The Train
  button needed its own handler that keeps the button down until polling stops
- A test asserted `"innerHTML" not in page` to prove the page never builds markup from typed
  text. It failed on the **comment** that says "never innerHTML". Narrowed to `".innerHTML"`,
  which is what an actual assignment looks like
- `virtual-time-budget` in headless Chrome fast-forwards timers, so a screenshot taken during a
  run catches an early poll rather than a mid-run one. Enough to prove the live path works

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 4c: The (B, T) batch dimension and best-val weights (2026-10-08)

### Concepts
- A GPU is fast at **one big operation**, but every operation costs a fixed amount to launch.
  Looping in Python over 16 windows means 16x as many launches for the same work
- **Broadcasting is why the change is small**: matmul only ever touches the last two axes and
  carries the rest along, so `(B, T, C) @ (C, H)` already worked. Measured before the change:
  only one line was actually broken, `q @ k.T`, because on a 3-D tensor `.T` swaps the batch
  axis with T instead of the last two. Everything else was our own shape guards being too strict
- `torch.stack(weights, dim=-3)` keeps the head axis next to `(T, T)`, so unbatched input still
  gives `(num_heads, T, T)` and the attention heatmap in `/ui` needed no change at all
- Attention mixes **positions**; it must never mix **examples**. That is a separate property
  from causality and needed its own test
- **Keeping the best weights**: val loss usually turns upwards before a run ends, so the last
  model is not the one worth saving. `keep_best` snapshots the parameters at the lowest val loss
  and restores them at the end, for the price of one extra copy in memory

### What we built
- `attention.check_shape`, shared by every layer: accepts `(T, C)` and `(B, T, C)`, rejects the
  rest. `attend` now uses `transpose(-2, -1)`
- `MiniGPT.forward` takes `(T,)` or `(B, T)`; `loss` flattens `(B, T, V)` to `(B*T, V)`, because
  B windows of T positions are simply B*T predictions
- The training loop lost its `for i in range(batch_size)`: one `model.loss(x, y)` per step
- `train_minigpt(keep_best=True)` plus `TrainHistory.best_step` / `best_val_loss`; the CLI and
  the job store report the loss of the model that was actually saved
- 16 new tests (300 total): a batch of one matches the unbatched call exactly, examples in a
  batch are independent, a batched loss equals the mean of its windows, rank-4 input is refused

### Experiment: what the batch dimension bought
Big model (embed 256, 6 blocks, 8 heads, context 256, batch 32, 4.8 M params):

| | s/step | token/s |
|---|---|---|
| Python loop, GPU | 1.343 | 6,098 |
| `(B, T)` batch, GPU | **0.066** | **124,721** |

**20x**, and 5000 steps went from 112 minutes to 5.5 minutes. The GPU is now 19x faster than the
CPU on this model; before the change it was *slower* than the CPU. The underlying reason, timed
directly: 32 separate `(256, 256) @ (256, 256)` matmuls take 941 us, the one batched matmul 79 us.

Re-running the 3d config (211 k params, 2000 steps) to check for a quality regression: 213.8 s on
CPU before, **54.0 s** on GPU now, val 1.826. And best-val earned its place on the first try --
the final evaluation was 1.905 while step 1800 had reached 1.826, so the old code would have
saved the worse model.

### Mistakes / surprises
- Two existing tests had to change, and both were right to fail: one asserted that a `(B, T, C)`
  input is rejected, the other that `(2, 3)` ids are invalid. Replaced with cases that are still
  invalid (wrong channel count, rank 4, an empty sequence), so the guard is still tested
- The speed-up is not uniform: the small 211 k model on CPU went 0.100 -> 0.020 s/step (5x),
  the big model on GPU 20x. The bigger the tensors, the more a launch-bound loop was costing

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_

---

## Sprint 4d: Teaching skills, and grading them (2026-10-08)

### Concepts
- **A model learns what it is shown.** There is no "learn maths" switch: three skills are three
  kinds of text. Stories (TinyStories), Python (the standard library and installed packages on
  this machine) and arithmetic (generated by us, `49+97=146`)
- **Loss is the wrong ruler for a skill.** It counts every character the same. In `49+97=146`
  the `+` and `=` are free and only the last digits are hard. Worse, the four random operand
  digits can never be predicted, which puts a floor under the loss of about
  `4 * ln(10) / 9 = 1.02`. The padded model's val loss went from 1.063 at step 2000 to 1.049
  at step 16000 while its accuracy went from 96% to 99%: almost all of the learning happened
  where loss cannot see it
- **A hidden exam.** Some problems are never written into the training text. They are held out
  as *unordered pairs*, otherwise `58+37` could be trained on while `37+58` is graded and the
  model could pass by "addition is commutative" without adding
- **A control group.** A sample of problems the model *did* train on is graded too. High on both
  means it learned, high on seen and low on hidden means it memorised (the 1-digit test shows
  exactly that signature: seen 97-100%, hidden 0-20%)
- **Confounds.** A breakdown by "carry / no carry" blamed the missing carry for a weakness that
  was really about short operands, because short operands rarely carry. Two questions have to be
  *crossed* before either is blamed
- **Representation matters as much as size.** The same 0.8M-parameter model: 84% with plain
  `5+12=17`, 96% with zero-padded `05+12=017`. Padding gives every number the same width, so
  the units column is always in the same place
- **Why reversing the answer should help, and what actually happened.** A model writes left to
  right, so for `03+95=098` it must choose the leading digit before it has looked at any
  column, but that digit depends on a carry chain through all of them. Reversed (`890`) it
  produces digits in the order it can compute them. At 6000 steps this looked wrong (8.5%):
  digit by digit the units digit was 100% right and the tens digit 9%, which is chance. At
  20,000 steps it was **100%**. A **plateau**: loss sits flat while a circuit that needs several
  parts working together is assembled, then drops at once (1.256 at step 10,000 to 1.085 at
  12,000)
- **Code cannot be graded by running it** (a tiny model's output is rarely meaningful and
  running it is dangerous), so we ask `ast.parse` how much of it is valid Python. The score is
  placed between a **ceiling** (real code cut to the same length does not parse fully either)
  and a **floor** (the same code with its characters shuffled), and comment-only output counts
  for nothing
- **Cleaning matters.** 6,657 distinct characters across 6,000 files became 97 once files that
  are mostly non-ASCII were dropped. Each character is one embedding row: thousands of rows that
  almost never get trained would be pure noise. Identical files are kept once, and files are
  shuffled before the validation split so train and val are the same mix
- **A tag line is a prompt.** Documents start with `<|story|>`, a file marker (a Python
  comment) or `<|math|>`. At generation time the tag picks the skill: what a system prompt
  or chat template does for a big model
- **Learning-rate schedule**: a linear warm-up (Adam's first estimates rest on a few gradients)
  and a cosine decay (late steps should settle, not kick)

### What we built
- `forgelm/data/`: `synthetic.py` (problems, hidden exam, `pad`, `reverse`), `code.py`
  (collect and clean Python), `mix.py` (tagged documents, `split_documents`)
- `forgelm/eval/`: `skills.py` (`math_accuracy`, grouped scores, `greedy_completer`) and
  `python_code.py` (`parsing_prefix_fraction`, `grade_code`, `sample_code`)
- CLI: `make-math-data`, `eval-math`, `collect-code`, `eval-code`, `make-mix`,
  `train-minigpt --warmup-steps --min-lr-fraction`, `--device` on `generate` and the evals
- `MiniGPT.generate(stop_id=...)`, `train_minigpt(on_best=...)` (the best model is written to
  disk *as training runs*), `--resume` (continue a checkpoint with fresh Adam; a text with a
  different character set is refused), `ModelStore(device=...)` and `FORGELM_DEVICE`
- [ADR 0008](decisions/0008-skills-graded-by-exams.md)
- 473 tests in total (284 at the end of sprint 4b)

### Experiment 1: how to write an addition problem
0.8M parameters (embed 128, 4 blocks, context 32), batch 64, lr 1e-3, constant. Hidden-exam
accuracy on 1,001 problems the model never saw. Every row is the same model and the same data
except for the way the text is written:

| format | 6000 steps | 20,000 steps |
|---|---|---|
| `49+97=146` | 84.4% | not run |
| reversed answer `49+97=641` | 82.1% | not run |
| zero-padded `05+12=017` | 96.3% | **99.0%** |
| padded and reversed `05+12=710` | 8.5% | **100.0%** (1001/1001) |

- Plain, 6000 steps, by operand length: **99.3%** when both numbers have two digits and **23.1%**
  when one is short. Seen problems score the same as hidden ones (84.7% vs 84.4%), so it is not
  memorising
- Padded, 6000 steps: the remaining errors sit around a sum of 100 (`03+95=098` written as
  `108`): exactly the leading-digit decision described above
- Padded and reversed at 6000 steps, digit by digit: units 100%, tens 9%, hundreds 76%
- Caveat: only two-digit sums were trained and tested, and plain / reversed were not run for
  20,000 steps, so "padded and reversed is best" is a comparison with *padded* only. How the
  model behaves on three-digit sums it never saw is not measured

### Experiment 2: one 25M-parameter model, three skills
`data/mix.txt`: 64 MB, 106 distinct characters. Code 62.5% (3,124 files), stories 31.3%
(24,718), sums 6.3% (19,138 blocks of 20, padded and reversed). A 25,444,352-parameter MiniGPT
(embed 512, 8 heads, 8 blocks, context 256, batch 32 = 8,192 tokens per step), lr 8e-4 with a
500-step warm-up and a cosine decay to 10%, 24,000 steps (about 3 epochs). Measured before
launching: 55.8k token/s and 3.4 GB at this size, and `lr 1e-3` won a 500-step probe, but short
probes favour a high rate, so 8e-4 was used.

What happened to the run: the process was killed by the background time limit at step 9000. The
best model so far was already on disk (`on_best`), so the remaining 15,500 steps were run from
it as three pieces of about 5,200 steps with `--resume`, each following the original cosine
curve (6.13e-4 to 3.71e-4, to 1.61e-4, to 8e-5), fresh Adam, a 150-step warm-up and a new seed
each time. Best val per piece: 0.827 (step 8500) -> 0.772 -> 0.741 -> **0.718** (perplexity
2.05). A GPU with nothing else running did 0.10 s per step; the first part of the run had been
slowed to 0.19 by my own test runs and exams sharing it.

| exam | after step ~6000 | final model |
|---|---|---|
| sums, hidden (padded and reversed, prefix `<|math|>`) | 96.2% (400 problems) | **99.9%** (1000 / 1001) |
| sums, seen | 91.2% (400 problems) | 99.9% (999 / 1000) |
| Python, share of code lines that parse | 0.249 | **0.687** |
| ... real held-out code cut alike (ceiling) | 0.711 | 0.752 |
| ... the same code shuffled (floor) | 0.016 | 0.007 |
| ... position between floor and ceiling | 0.34 | **0.91** |

- Sums: the one error in 2001 is `16+30=` written as `540` instead of `640`. The plateau
  that took the 0.8M model 10,000 steps broke before step 6000 here, although sums are only
  6% of the text. The model sees about as many problems per step (roughly 55) as the maths-only
  run did (64)
- The mid-run seen score being *lower* than hidden (91.2 vs 96.2) is noise at n = 400, not a
  sign of anything: memorising would show the opposite
- Python: the shape is learned and the content is not. A sample at temperature 0.5 writes a
  valid `def _get_python_static_dir(path: Path, path: int) -> str:` with a docstring and
  `if / else` chains of `os.path.join(path, path)`: well-formed, and meaningless. Others loop
  (`generate_line_equations` six times) or fall into whitespace. The parse score cannot see
  that, which is the limit of `ast.parse`
- Tags work as prompts, and the model also writes them: after a story it started a
  `<|math|>` document by itself and every sum in it was right (`82+43=521` is 125 backwards)
- Greedy decoding loops: the maths prompt repeats `77+25=201` until the token limit

### Not measured
- Whether sums, code and stories interfere: the mixed model was not compared with three
  specialised models of the same size, so "no interference" is not shown, only that all three
  skills are present
- Three-digit sums, or any problem unlike the training distribution
- Stories: only loss, and there is no story exam. Loss on the mix is not comparable with loss
  on TinyStories alone (0.627 for the 4.8M model), they are different texts
- Whether code that parses runs, or does anything

### Mistakes / surprises
- My first reading of "carry 86%, no carry 58%" was "the model is bad at *not* carrying". It was
  a confound. The grader now crosses the groups, and a test reproduces the mistake with a fake
  model
- My first guess was that reversing the answer would help. At 3000 and 6000 steps it looked
  worse, and I nearly wrote it off. It was an unfinished plateau, not a bad idea
- `.gitignore` had `data/`, which matches a folder of that name **anywhere**, so
  `src/forgelm/data/` was silently left out of a commit that imports it. `git status` did not
  list the package, which is how I noticed. A clean clone is the only check that sees this
  (the tests run in my working directory, where the files exist). Anchored to `/data/`
- The "seen" control was sampled from problems that *could* appear in training, and a finite
  corpus drawn with replacement may miss some. A test caught it; the control now samples from
  what actually appears in the text
- An ADR sentence claimed "a 20x gap between two models with the same loss". I had not
  measured that, and removed it before committing
- `eval-code` took its "real code" reference by cutting the corpus at file markers, which in a
  *mixed* text appends the following story to the file. `split_documents` cuts at every tag

- I started a 60-minute job in the background and it was killed at about 29 minutes. Launching
  it again unchanged would have hit the same limit. The best-model-to-disk hook, written a
  few minutes earlier only as a precaution, is the reason nothing was lost
- `ast.parse` accepts `def f(path, path)`. Duplicate arguments are caught when the code is
  compiled, not when it is parsed. `compile()` (which does not run anything) would catch that
  and a few other things; not done, so the parse score is an upper bound on "valid Python"
- Running my own exams and tests while the job trained made it twice as slow

### Quiz (not asked yet, to do next session)
1. Why is the loss of the padded model stuck near 1.05 while its accuracy keeps rising? What is
   that number made of?
2. Why is the hidden exam split by unordered pair rather than by problem? What would a model
   learn to cheat with otherwise?
3. The seen score is 99% and the hidden score is 25%. What does that say about the model? And
   seen 99%, hidden 98%?
4. Why can writing the answer backwards help a model that writes left to right, and why can
   it look worse for thousands of steps before it looks better?
5. A tag line like `<|math|>` is "just text". What makes it work as a prompt?
6. The 25M run was killed at step 9000 and nothing was lost. Which line of code is the reason,
   and what would have been lost without it?
7. Why is the mixed model's 99.9% on sums not proof that sums, code and stories do not
   interfere? What run would show it?

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_
