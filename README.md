# ForgeLM

ForgeLM is an AI engineering lab that I'm building one layer at a time. It starts with a tokenizer
and a transformer written from scratch. It ends as a personal assistant that runs a local
open-source model, reads my documents (RAG), and takes actions through tools (agents).

Each sprint adds one concept to the same product. What I learned in each one is recorded in
[docs/learning-log.md](docs/learning-log.md).

## Roadmap

| Sprint | Feature | Status |
|---|---|---|
| 0 | Engineering setup: package, CLI, API, tests, lint | ✅ |
| 1 | Tokenizer playground: char tokenizer → byte-level BPE | ✅ |
| 2 | Tiny language model: counting bigram → neural bigram (PyTorch) | ✅ |
| 3 | Transformer / MiniGPT from scratch: attention → block → trained MiniGPT | ✅ |
| 4 | Training and inference dashboard: inference UI, live training, batched training | ✅ |
| 5 | Open-source LLM integration (Ollama / llama.cpp / HF) | |
| 6 | LoRA / QLoRA fine-tuning | |
| 7 | RAG v1: chunking, embeddings, vector search | |
| 8 | RAG v2: hybrid search, reranking | |
| 9 | Agents: tool calling, state, daily-task tools | |
| 10 | Evaluation and observability | |
| 11 | Production deployment | |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

### Optional: PyTorch for neural models

```powershell
# Windows + NVIDIA GPU: PyPI only has CPU wheels, so get the CUDA build from PyTorch's index.
# Pick the cuXXX that your driver supports (`nvidia-smi` shows "CUDA Version").
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,ml]"
```

On Linux, `pip install -e ".[dev,ml]"` is enough (the PyPI wheel includes CUDA). Without torch,
everything except the neural commands still works, and their tests are skipped.

## Usage

```powershell
forgelm --version
forgelm serve            # http://127.0.0.1:8000/health, docs at /docs
forgelm serve --reload   # auto-restart while developing
forgelm tokenize "hello!"                     # vocab built from the text itself
forgelm tokenize "hello!" --corpus notes.txt  # vocab from a file; unseen chars → <unk>
forgelm tokenize "İstanbul" --tokenizer bpe --merges 300 --corpus notes.txt  # never <unk>
```

### Train and sample a bigram language model

```powershell
# Tiny Shakespeare (~1 MB); data/ is gitignored
curl -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

forgelm train-bigram --corpus data/tinyshakespeare.txt          # -> checkpoints/bigram.json
forgelm generate --prompt "ROMEO:" --max-tokens 200 --seed 42 --temperature 0.8

# Same model, learned with gradient descent instead of counting (needs the ml extra)
forgelm train-neural-bigram --corpus data/tinyshakespeare.txt --steps 3000 --lr 100
```

### Train and sample MiniGPT (needs the ml extra)

```powershell
# ~55 s on a GPU for 2000 steps (val loss ~1.83, perplexity ~6.2; the bigram is 2.48 / 12)
forgelm train-minigpt --corpus data/tinyshakespeare.txt --steps 2000
forgelm generate --model checkpoints/minigpt.pt --prompt "ROMEO:" --max-tokens 250 --seed 42 --temperature 0.8
```

The weights from the lowest validation loss are the ones saved, not the ones from the last
step, so a run that starts overfitting still gives you its best model.

### A bigger model on a bigger text

Tiny Shakespeare teaches the shape of English, not its meaning. For output you can actually
read, [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is written with a
deliberately small vocabulary so that a small model can be coherent:

```powershell
# 22 MB; data/ is gitignored
curl -L -o data/tinystories.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 4.8M parameters, ~18 min on an RTX 5070
forgelm train-minigpt --corpus data/tinystories.txt --out checkpoints/tinystories.pt `
  --steps 15000 --eval-every 250 --embed-dim 256 --heads 8 --blocks 6 `
  --block-size 256 --batch-size 32 --lr 1e-3

forgelm generate --model checkpoints/tinystories.pt --prompt "Once upon a time" --max-tokens 400
```

Bigger models want a smaller learning rate: 3e-3 is right for the 211k default, 1e-3 for this
one. Train for 200 steps first and see whether the loss falls.

### Teaching it skills: arithmetic and Python

Loss cannot say whether a model can add or write Python, so each skill has its own data and an
exam that checks the answer. Nothing needs downloading beyond the stories above.

```powershell
# Arithmetic: we generate the problems. --pad writes 05+12=017, --reverse writes the answer's
# digits backwards. Some problems are never written into the training text: that is the exam.
forgelm make-math-data --out data/math-padrev.txt --pad --reverse
forgelm train-minigpt --corpus data/math-padrev.txt --out checkpoints/math.pt `
  --steps 20000 --eval-every 2000 --embed-dim 128 --blocks 4 --block-size 32 --batch-size 64 --lr 1e-3
forgelm eval-math --model checkpoints/math.pt `
  --holdout data/math-padrev-holdout.txt --seen data/math-padrev-seen.txt

# Python: read the standard library and installed packages from this machine
forgelm collect-code --out data/python.txt           # 4,059 files, 52 MB, 97 distinct characters
forgelm eval-code --model checkpoints/code.pt --corpus data/python.txt
```

`eval-math` scores the hidden problems and a sample of problems the model trained on (if it is
high on both it learned to add, if only on the second it memorised), broken down by carry and
by operand length. `eval-code` asks `ast.parse` how much of the generated code is valid
Python and shows where that sits between real code and shuffled characters. It never runs the
code.

### One model, three skills

`make-mix` blends stories, Python and sums into one text. Every document starts with a tag
line, so the tag is the prompt that selects the skill:

```powershell
forgelm make-mix --stories data/tinystories.txt --code data/python.txt --math data/math-padrev.txt `
  --out data/mix.txt --stories-mb 20 --code-mb 40 --math-mb 4

# 25M parameters, about 40 minutes on an RTX 5070 if nothing else is using the GPU. The best
# model is written to disk as it improves, and a long run can be cut into pieces with
# --resume checkpoints/forge-25m.pt (the size comes from the checkpoint; use a new --seed).
forgelm train-minigpt --corpus data/mix.txt --out checkpoints/forge-25m.pt --steps 24000 `
  --eval-every 500 --embed-dim 512 --heads 8 --blocks 8 --block-size 256 --batch-size 32 `
  --lr 8e-4 --warmup-steps 500 --min-lr-fraction 0.1

forgelm eval-math --model checkpoints/forge-25m.pt --prefix "<|math|>\n" --device cuda `
  --holdout data/math-padrev-holdout.txt --seen data/math-padrev-seen.txt
forgelm eval-code --model checkpoints/forge-25m.pt --corpus data/mix.txt --device cuda
```

On the 64 MB mix this model reaches val loss 0.718, **99.9%** on the hidden sums (1000 of 1001) and
0.91 between "shuffled characters" and "real code" on the Python parse exam. It writes
valid-looking Python that means nothing, and sometimes repeats itself: see
[docs/learning-log.md](docs/learning-log.md) for what was and was not measured.

### Dashboard

```powershell
forgelm serve            # then open http://127.0.0.1:8000/ui
```

Three panels: **Tokenizer**, **Generate** (any model in `checkpoints/`) and **Attention** (a
heatmap per block and head of a trained MiniGPT). Train a model first with the commands above.

## Development

```powershell
pytest                 # run tests
ruff check .           # lint
ruff format .          # format
```

## Project layout

```
src/forgelm/     the package (cli.py and api.py are thin adapters over core modules)
tests/           pytest tests
docs/            architecture, learning log, ADRs (docs/decisions/)
```

See [docs/architecture.md](docs/architecture.md) for details.
