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
| 2 | Tiny language model: counting bigram ✅ → neural bigram (PyTorch) | ⏳ |
| 3 | Transformer / MiniGPT from scratch | |
| 4 | Training and inference dashboard | |
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
```

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
