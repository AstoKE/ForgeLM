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

### Lessons
- _(fill in, in your own words)_

### Open questions
- _(fill in)_
