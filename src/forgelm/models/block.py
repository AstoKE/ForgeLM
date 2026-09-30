"""The other half of a transformer block: thinking about what attention gathered.

Attention *moves* information between positions, but it never really processes it:
`weights @ v` is only a weighted sum. The processing is done here, by a small neural
network applied to each position on its own:

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
