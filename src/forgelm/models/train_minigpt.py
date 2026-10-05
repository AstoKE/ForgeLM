"""Training MiniGPT: random windows, a hand-written Adam, and a train / val loop.

    text ids -> random windows (x, y = x shifted by one) -> loss -> backward -> Adam step

One window of T tokens is T training examples (3c-3), so a batch of B windows is
B * T examples. MiniGPT still takes a single (T,) sequence, so the batch is a
Python loop over its windows; the gradients of all windows add up in `.grad`
and are averaged by dividing each loss by B.

Requires the optional `ml` extra (torch).
"""

import math
import time

import torch

from forgelm.models.bigram import encode_and_split
from forgelm.models.minigpt import MiniGPT
from forgelm.models.neural_bigram import TrainHistory, pick_device


class Adam:
    """Adam, written by hand. The tests check it against torch.optim.Adam.

    Per parameter it remembers two running averages of the gradient g:
        m = average of g        ("which way have we been pushed lately?")
        v = average of g * g    ("how big are the pushes?")
    and steps by lr * m / (sqrt(v) + eps). Dividing by sqrt(v) cancels the size of
    the gradient: a parameter with gradient 0.01 and one with gradient 100 both
    move by about lr per step. Plain SGD would move them 10,000x apart.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not lr > 0:
            raise ValueError("lr must be > 0")
        if not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must be in [0, 1)")
        self.params = list(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0  # number of steps taken so far

    def step(self) -> None:
        """One update; loss.backward() must have run first."""
        b1, b2 = self.betas
        self.t += 1
        with torch.no_grad():  # the update itself must not be tracked by autograd
            for p, m, v in zip(self.params, self.m, self.v, strict=True):
                if p.grad is None:
                    continue
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                # m and v start at 0, so early on they are too small. Dividing by
                # (1 - b^t) corrects that: at t = 1 it turns m = 0.1 * g back into g.
                m_hat = m / (1 - b1**self.t)
                v_hat = v / (1 - b2**self.t)
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None  # otherwise the next backward() would add to the old gradient


def get_batch(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows from a 1-D tensor of ids: x is (B, T), y is x shifted by one.

    ids = [r, o, m, e, o, :]  window at 1, T=3 -> x = [o, m, e], y = [m, e, o]
    Every y[i, t] is the correct next token after x[i, :t+1].
    """
    if block_size < 1 or batch_size < 1:
        raise ValueError("block_size and batch_size must be >= 1")
    if len(ids) < block_size + 1:
        raise ValueError(
            f"need at least block_size + 1 = {block_size + 1} ids to cut a window, got {len(ids)}"
        )
    # The last valid start leaves one more token after the window for y.
    # The generator lives on the CPU, so move the starts to wherever `ids` is.
    starts = torch.randint(0, len(ids) - block_size, (batch_size,), generator=generator)
    starts = starts.to(ids.device)
    offsets = torch.arange(block_size, device=ids.device)
    x = ids[starts[:, None] + offsets]  # (B, T): window i is ids[start_i : start_i + T]
    y = ids[starts[:, None] + offsets + 1]
    return x, y


@torch.no_grad()  # evaluation only: no graph, no gradients
def estimate_loss(
    model: MiniGPT,
    ids: torch.Tensor,
    batch_size: int,
    num_batches: int,
    generator: torch.Generator | None = None,
) -> float:
    """Mean loss over random windows. A full pass over the corpus would be too slow."""
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(ids, model.block_size, batch_size, generator)
        losses += [model.loss(x[i], y[i]).item() for i in range(len(x))]
    return sum(losses) / len(losses)


def train_minigpt(
    train_ids: list[int],
    val_ids: list[int],
    vocab_size: int,
    steps: int = 1000,
    lr: float = 3e-3,
    batch_size: int = 16,
    block_size: int = 64,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_blocks: int = 4,
    eval_every: int = 100,
    eval_batches: int = 4,
    device: str = "auto",
    seed: int = 0,
) -> tuple[MiniGPT, TrainHistory]:
    if steps < 1 or batch_size < 1 or eval_every < 1 or eval_batches < 1:
        raise ValueError("steps, batch_size, eval_every and eval_batches must be >= 1")

    dev = pick_device(device)
    train = torch.tensor(train_ids, dtype=torch.long, device=dev)
    val = torch.tensor(val_ids, dtype=torch.long, device=dev)
    model = MiniGPT(vocab_size, embed_dim, num_heads, num_blocks, block_size, device=dev, seed=seed)
    optimizer = Adam(model.parameters(), lr=lr)
    # Own generators (on CPU), so window sampling is reproducible and independent of
    # any other random numbers, and evaluation does not change which windows training sees.
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    history = TrainHistory(device=dev)

    def record(step: int) -> None:
        history.steps.append(step)
        history.train_loss.append(estimate_loss(model, train, batch_size, eval_batches, eval_gen))
        history.val_loss.append(estimate_loss(model, val, batch_size, eval_batches, eval_gen))

    start = time.perf_counter()
    for step in range(steps):
        if step % eval_every == 0:
            record(step)
        x, y = get_batch(train, block_size, batch_size, train_gen)
        total = 0.0
        for i in range(batch_size):
            # Dividing by B makes the summed gradients an average over the batch.
            loss = model.loss(x[i], y[i]) / batch_size
            loss.backward()  # adds this window's gradient to .grad
            total += loss.item()
        if not math.isfinite(total):
            raise FloatingPointError(f"loss became {total} at step {step}; lower --lr")
        optimizer.step()
        optimizer.zero_grad()
    record(steps)
    history.seconds = time.perf_counter() - start
    return model, history


def train_minigpt_on_text(
    text: str, val_fraction: float = 0.1, **kwargs
) -> tuple[MiniGPT, TrainHistory]:
    """Char-tokenize and split `text` like the bigram models, then train MiniGPT."""
    tokenizer, train_ids, val_ids = encode_and_split(text, val_fraction)
    return train_minigpt(train_ids, val_ids, tokenizer.vocab_size, **kwargs)
