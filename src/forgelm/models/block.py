"""The transformer block: attention, an MLP, and the wiring that makes them stackable.

    x = x + attention(layer_norm_1(x))      talk:  move information between positions
    x = x + feedforward(layer_norm_2(x))    think: process it inside each position

Three pieces live here. `FeedForward` is the thinking half, `LayerNorm` keeps the
numbers on a comparable scale, and `TransformerBlock` wires them together with
**residual** connections (`x = x + ...`), which is what lets blocks be stacked.

`FeedForward`: attention *moves* information between positions, but it never really
processes it: `weights @ v` is only a weighted sum. The processing is done here, by a
small neural network applied to each position on its own:

    hidden = relu(x @ W1 + b1)      widen   C -> 4C
    out    = hidden @ W2 + b2       narrow  4C -> C

Two things make it work:

* **position-wise**: every row of x goes through the same little network alone, so no
  information crosses between positions. That is attention's job, not this one's.
* **the nonlinearity**: without `relu`, `(x @ W1) @ W2` equals `x @ (W1 @ W2)`, a single
  matrix. Widening to 4C would buy exactly nothing. `relu` is what makes two layers two
  layers, and it is where most of a transformer's parameters end up working (~8C^2 here
  against ~4C^2 in attention).

Written by hand, like NeuralBigram and SelfAttentionHead: no nn.Linear, no nn.Module.
Requires the optional `ml` extra (torch).
"""

import torch

from forgelm.models.attention import MultiHeadAttention


class FeedForward:
    """Per-position MLP: `C -> hidden -> C`, with `hidden = 4 * embed_dim` by default.

    The output has the same width as the input, which is what lets 3c-2 write the
    residual `x = x + feedforward(x)`.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        # 4x is what the original transformer used (512 -> 2048); it is an empirical choice.
        hidden_dim = 4 * embed_dim if hidden_dim is None else hidden_dim
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        gen = torch.Generator().manual_seed(seed)

        def weights(rows: int, cols: int) -> torch.Tensor:
            # Scaled by 1/sqrt(rows) so the numbers keep a similar size layer after layer.
            scaled = torch.randn((rows, cols), generator=gen) * rows**-0.5
            return scaled.to(device).requires_grad_(True)

        def bias(size: int) -> torch.Tensor:
            # Zero: a bias has nothing to break the symmetry of, the weights already do.
            return torch.zeros(size, device=device, requires_grad=True)

        self.W1, self.b1 = weights(embed_dim, hidden_dim), bias(hidden_dim)
        self.W2, self.b2 = weights(hidden_dim, embed_dim), bias(embed_dim)

    @property
    def embed_dim(self) -> int:
        return self.W1.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.W1.shape[1]

    def parameters(self) -> list[torch.Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, C) -> (T, C). Every row is processed independently."""
        if x.ndim != 2 or x.shape[1] != self.embed_dim:
            raise ValueError(f"expected x of shape (T, {self.embed_dim}), got {tuple(x.shape)}")
        hidden = torch.relu(x @ self.W1 + self.b1)  # negatives become 0: "this one did not fire"
        return hidden @ self.W2 + self.b2


class LayerNorm:
    """Rescale each position on its own: mean 0, std 1, then a learned scale and shift.

    Residuals keep adding to `x`, so the numbers grow as blocks are stacked. Big numbers
    break the two things this model depends on: softmax turns into winner-takes-all, and
    `relu` saturates. LayerNorm pulls every position back to a comparable scale.

    `gamma` and `beta` are learned and start at 1 and 0, so the layer starts as the plain
    normalisation but the model can undo part of it if that helps. Like FeedForward this
    is **position-wise**: a row is normalised using its own numbers only, never its
    neighbours' and never a batch's.
    """

    def __init__(
        self,
        embed_dim: int,
        eps: float = 1e-5,
        device: torch.device | str = "cpu",
    ) -> None:
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        self.gamma = torch.ones(embed_dim, device=device, requires_grad=True)  # scale
        self.beta = torch.zeros(embed_dim, device=device, requires_grad=True)  # shift
        # Guards against dividing by zero when a row is constant (variance 0).
        self.eps = eps

    @property
    def embed_dim(self) -> int:
        return self.gamma.shape[0]

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, C) -> (T, C)."""
        if x.ndim != 2 or x.shape[1] != self.embed_dim:
            raise ValueError(f"expected x of shape (T, {self.embed_dim}), got {tuple(x.shape)}")
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False divides by C instead of C-1, which is what nn.LayerNorm does.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class TransformerBlock:
    """One block: attention and an MLP, each wrapped in pre-norm and a residual.

        x = x + attention(ln1(x))
        x = x + feedforward(ln2(x))

    Two details worth remembering:

    * **Residual** (`x = x + ...`): in the backward pass an addition copies the gradient
      to both branches unchanged, so there is a path from the deepest block back to the
      input that never goes through a matmul. Without it the gradient is multiplied at
      every layer and shrinks towards zero (measured on 20 layers: 0.062 against 4.406).
      It also means a block that has learned nothing useful simply adds ~0 and gets out
      of the way.
    * **Pre-norm** (`block(ln(x))`, not `ln(x + block(x))`): the main road never passes
      through a LayerNorm, so that gradient path stays perfectly clean. The 2017 paper
      used post-norm; GPT-2 and everything after it use pre-norm because it trains far
      more stably when blocks are stacked.

    Input and output are both (T, C), which is exactly what makes blocks stackable.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.ln1 = LayerNorm(embed_dim, device=device)
        self.attention = MultiHeadAttention(embed_dim, num_heads, device=device, seed=seed)
        self.ln2 = LayerNorm(embed_dim, device=device)
        # MultiHeadAttention uses seeds seed .. seed + num_heads, so start past them.
        self.feedforward = FeedForward(
            embed_dim, hidden_dim, device=device, seed=seed + num_heads + 1
        )

    @property
    def embed_dim(self) -> int:
        return self.ln1.embed_dim

    def parameters(self) -> list[torch.Tensor]:
        return [
            *self.ln1.parameters(),
            *self.attention.parameters(),
            *self.ln2.parameters(),
            *self.feedforward.parameters(),
        ]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, C) -> (output (T, C), attention weights (num_heads, T, T))."""
        attended, weights = self.attention.forward(self.ln1.forward(x))
        x = x + attended
        return x + self.feedforward.forward(self.ln2.forward(x)), weights
