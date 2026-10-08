"""Neural bigram language model in PyTorch.

The same model as bigram.py, but the V x V table of logits is *learned* with
gradient descent instead of counted:

    logits = W[prev]                      (row `prev` of a V x V weight matrix)
    loss   = cross_entropy(logits, next)
    W     -= lr * dloss/dW                (autograd computes the gradient)

After enough steps W ends up close to log(counts) from the counting model.
Everything is written out by hand (no nn.Module, no torch.optim) so each step
is visible; the tests show the hand-written parts match PyTorch's built-ins.

Requires the optional `ml` extra (torch). Import this module explicitly; it is
not re-exported from forgelm.models so the rest of ForgeLM works without torch.
"""

import math
import time
from dataclasses import dataclass, field

import torch

from forgelm.models.bigram import encode_and_split


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean of -log softmax(logits)[target], i.e. what F.cross_entropy computes.

    logits: (N, V) raw scores, targets: (N,) the correct next-token ids.
    """
    # log softmax = logits - log(sum(exp(logits))). logsumexp subtracts the
    # max internally, the same overflow trick as our softmax() in bigram.py.
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    # Pick, for each row i, the log-probability of its correct target.
    picked = log_probs[torch.arange(len(targets), device=logits.device), targets]
    return -picked.mean()


class NeuralBigram:
    def __init__(self, vocab_size: int, device: torch.device | str = "cpu") -> None:
        # All zeros: every row gives uniform probabilities, so the loss starts at ln V.
        # requires_grad=True tells autograd to track operations on W.
        self.W = torch.zeros((vocab_size, vocab_size), device=device, requires_grad=True)

    @property
    def vocab_size(self) -> int:
        return self.W.shape[0]

    def logits(self, prev: torch.Tensor) -> torch.Tensor:
        # Embedding lookup: picking row `prev` equals one_hot(prev) @ W, just cheaper.
        return self.W[prev]

    def loss(self, prev: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        return cross_entropy(self.logits(prev), nxt)

    def sgd_step(self, lr: float) -> None:
        """One step of gradient descent; loss.backward() must have run first."""
        with torch.no_grad():  # the update itself must not be tracked by autograd
            self.W -= lr * self.W.grad
        self.W.grad = None  # otherwise the next backward() would *add* to this gradient


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available (CPU-only torch or no GPU)")
    if requested not in ("cpu", "cuda"):
        raise ValueError(f"unknown device: {requested!r}")
    return torch.device(requested)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


@dataclass
class TrainHistory:
    device: torch.device
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0
    # Filled in by trainers that keep the best-val weights (MiniGPT); None otherwise.
    best_step: int | None = None
    best_val_loss: float | None = None


def _pairs(ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(prev, next) tensors: x[i] is followed by y[i]."""
    t = torch.tensor(ids, dtype=torch.long, device=device)
    return t[:-1], t[1:]


def train_neural_bigram(
    train_ids: list[int],
    val_ids: list[int],
    vocab_size: int,
    steps: int = 300,
    lr: float = 50.0,
    batch_size: int = 32_768,
    eval_every: int = 50,
    device: str = "auto",
    seed: int = 0,
) -> tuple[NeuralBigram, TrainHistory]:
    if steps < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError("steps, batch_size and eval_every must be >= 1")
    if not lr > 0:
        raise ValueError("lr must be > 0")

    dev = pick_device(device)
    x_train, y_train = _pairs(train_ids, dev)
    x_val, y_val = _pairs(val_ids, dev)
    model = NeuralBigram(vocab_size, dev)
    # Own generator (on CPU) so batch sampling is reproducible and independent
    # of any other random numbers the program uses.
    gen = torch.Generator().manual_seed(seed)
    history = TrainHistory(device=dev)

    def record(step: int) -> None:
        with torch.no_grad():  # evaluation only: no graph, no gradients
            history.steps.append(step)
            history.train_loss.append(model.loss(x_train, y_train).item())
            history.val_loss.append(model.loss(x_val, y_val).item())

    start = time.perf_counter()
    for step in range(steps):
        if step % eval_every == 0:
            record(step)
        # Mini-batch: a random sample of (prev, next) pairs, not the whole corpus.
        idx = torch.randint(0, len(x_train), (batch_size,), generator=gen).to(dev)
        loss = model.loss(x_train[idx], y_train[idx])  # forward pass
        loss.backward()  # backward pass: fills model.W.grad
        model.sgd_step(lr)  # update
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss became {loss.item()} at step {step}; lower --lr")
    record(steps)
    history.seconds = time.perf_counter() - start
    return model, history


def train_neural_on_text(
    text: str, val_fraction: float = 0.1, **kwargs
) -> tuple[NeuralBigram, TrainHistory]:
    """Char-tokenize and split `text` like the counting model, then train with SGD."""
    tokenizer, train_ids, val_ids = encode_and_split(text, val_fraction)
    return train_neural_bigram(train_ids, val_ids, tokenizer.vocab_size, **kwargs)
