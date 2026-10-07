"""Running our own trained models (bigram .json, MiniGPT .pt) from a checkpoint folder.

Nothing here knows about HTTP. The API (and later other adapters) call these functions
and translate the three error types into status codes:

    InvalidRequestError  the caller asked for something impossible   (HTTP 422)
    ModelNotFoundError   no such checkpoint                          (HTTP 404)
    MLNotInstalledError  a .pt model needs torch, the `ml` extra     (HTTP 503)

torch is imported lazily, only when a MiniGPT is actually used, so ForgeLM still
runs without the `ml` extra.
"""

import random
import threading
from dataclasses import dataclass
from pathlib import Path

from forgelm.models import load_checkpoint
from forgelm.tokenizer import CharTokenizer

SUFFIXES = {".json": "bigram", ".pt": "minigpt"}


class InvalidRequestError(ValueError):
    """The request cannot be served as asked (bad name, empty prompt, wrong model kind...)."""


class ModelNotFoundError(Exception):
    """There is no checkpoint with that name."""


class MLNotInstalledError(Exception):
    """A MiniGPT checkpoint needs PyTorch, which is an optional extra."""


@dataclass(frozen=True)
class ModelInfo:
    name: str  # the file name, e.g. "minigpt.pt"
    kind: str  # "bigram" or "minigpt"
    size_bytes: int


@dataclass(frozen=True)
class LoadedModel:
    name: str
    kind: str
    model: object  # BigramModel or MiniGPT
    tokenizer: CharTokenizer


@dataclass(frozen=True)
class GenerateResult:
    model: str
    text: str  # the prompt plus the generated continuation, like the CLI prints
    num_new_tokens: int


@dataclass(frozen=True)
class AttentionResult:
    model: str
    tokens: list[str]  # one per position, the axis labels of the heatmap
    # weights[block][head][t][s]: how much position t looks at position s (0 for s > t)
    weights: list[list[list[list[float]]]]


class ModelStore:
    """A folder of checkpoints plus a cache, so a model is read from disk only once.

    A model is looked up by *file name only*. Anything with a path in it ("../secret.pt",
    "/etc/passwd") is rejected, so a request can never reach outside the folder.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._cache: dict[str, tuple[int, LoadedModel]] = {}
        self._lock = threading.Lock()  # the web server handles requests in several threads

    def list_models(self) -> list[ModelInfo]:
        if not self.directory.is_dir():
            return []
        return [
            ModelInfo(path.name, SUFFIXES[path.suffix], path.stat().st_size)
            for path in sorted(self.directory.iterdir())
            if path.is_file() and path.suffix in SUFFIXES and not path.name.startswith(".")
        ]

    def _path(self, name: str) -> Path:
        if not name or name != Path(name).name or name.startswith("."):
            raise InvalidRequestError(f"invalid model name: {name!r} (a file name, no folders)")
        if Path(name).suffix not in SUFFIXES:
            raise InvalidRequestError(f"unknown model type: {name!r} (use .json or .pt)")
        path = self.directory / name
        if not path.is_file():
            raise ModelNotFoundError(f"no model named {name!r}")
        return path

    def get(self, name: str) -> LoadedModel:
        path = self._path(name)
        stamp = path.stat().st_mtime_ns  # a retrained file has a new stamp: reload it
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None and cached[0] == stamp:
                return cached[1]
            loaded = self._load(name, path)
            self._cache[name] = (stamp, loaded)
            return loaded

    @staticmethod
    def _load(name: str, path: Path) -> LoadedModel:
        kind = SUFFIXES[path.suffix]
        try:
            if kind == "bigram":
                model, tokenizer = load_checkpoint(path)
            else:
                try:
                    from forgelm.models.minigpt import load_minigpt
                except ImportError as exc:
                    raise MLNotInstalledError(
                        'PyTorch is not installed; see README ("pip install -e .[ml]")'
                    ) from exc
                model, tokenizer = load_minigpt(path)
        except (OSError, ValueError, KeyError) as exc:
            raise InvalidRequestError(f"cannot load model {name!r}: {exc}") from exc
        return LoadedModel(name, kind, model, tokenizer)


def generate_text(
    store: ModelStore,
    name: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    seed: int | None = None,
) -> GenerateResult:
    if not prompt:
        raise InvalidRequestError("prompt must not be empty")
    if max_tokens < 0:
        raise InvalidRequestError("max_tokens must be >= 0")
    if temperature < 0:
        raise InvalidRequestError("temperature must be >= 0")
    loaded = store.get(name)
    prompt_ids = loaded.tokenizer.encode(prompt)

    if loaded.kind == "bigram":
        ids = loaded.model.generate(prompt_ids, max_tokens, random.Random(seed), temperature)
    else:
        import torch  # already imported by load_minigpt; the model exists, so torch does too

        generator = None if seed is None else torch.Generator().manual_seed(seed)
        ids = loaded.model.generate(prompt_ids, max_tokens, generator, temperature)
    return GenerateResult(name, loaded.tokenizer.decode(ids), len(ids) - len(prompt_ids))


def attention_maps(store: ModelStore, name: str, text: str) -> AttentionResult:
    """Run `text` through a MiniGPT and return every block's, every head's attention weights."""
    if not text:
        raise InvalidRequestError("text must not be empty")
    loaded = store.get(name)
    if loaded.kind != "minigpt":
        raise InvalidRequestError("attention needs a MiniGPT model; a bigram has no attention")
    model = loaded.model
    ids = loaded.tokenizer.encode(text)
    if len(ids) > model.block_size:
        raise InvalidRequestError(
            f"text has {len(ids)} tokens but this model's context window is {model.block_size}"
        )

    import torch

    with torch.no_grad():  # we only look, so no gradients are needed
        _, weights = model.forward(torch.tensor(ids, device=model.token_embedding.device))
    return AttentionResult(
        model=name,
        tokens=loaded.tokenizer.tokens(text),
        weights=[block.cpu().tolist() for block in weights],  # (heads, T, T) per block
    )
