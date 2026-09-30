"""Mixing information from past tokens: the skeleton of self-attention.

A bigram only sees the previous token. To see further back, every position t
needs a summary of positions 0..t. Sprint 3a uses the simplest summary, the
plain average, and computes it three equivalent ways:

    1. a Python loop          (easiest to read)
    2. one matrix multiply    (what the GPU is good at)
    3. mask + softmax         (the exact shape self-attention will use in 3b)

Shapes: x is (T, C): T positions ("time"), C numbers per position ("channels").
Position t may only look at positions 0..t, never at the future (causal).

Sprint 3b (bottom of this file) replaces the equal scores by learned ones:
single-head self-attention with query / key / value.
"""

import torch


def causal_mask(T: int, device: torch.device | str | None = None) -> torch.Tensor:
    """(T, T) bool table: mask[t, s] is True when position t may look at position s.

    T=3 ->  [[ True, False, False],
             [ True,  True, False],
             [ True,  True,  True]]
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def causal_average_loop(x: torch.Tensor) -> torch.Tensor:
    """Version 1: for each position, average itself and everything before it."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        out[t] = x[: t + 1].mean(dim=0)
    return out


def uniform_causal_weights(T: int, device: torch.device | str | None = None) -> torch.Tensor:
    """(T, T) weights: row t spreads 1.0 equally over positions 0..t.

    T=3 ->  [[1,   0,   0  ],
             [1/2, 1/2, 0  ],
             [1/3, 1/3, 1/3]]
    """
    weights = causal_mask(T, device).float()  # 1 where allowed, 0 where forbidden
    return weights / weights.sum(dim=1, keepdim=True)  # each row sums to 1 -> an average


def causal_average_matmul(x: torch.Tensor) -> torch.Tensor:
    """Version 2: all averages at once. Row t of (weights @ x) is sum_s weights[t, s] * x[s]."""
    return uniform_causal_weights(x.shape[0], x.device) @ x  # (T, T) @ (T, C) -> (T, C)


def masked_softmax_weights(scores: torch.Tensor) -> torch.Tensor:
    """Turn (T, T) scores into weights, giving the future exactly zero weight.

    Forbidden cells are set to -inf before softmax; exp(-inf) = 0, so they get
    weight 0 no matter how large their score was. Each row still sums to 1.
    In 3b the scores will be learned ("how interesting is token s for token t?").
    """
    T = scores.shape[-1]
    scores = scores.masked_fill(~causal_mask(T, scores.device), float("-inf"))
    return torch.softmax(scores, dim=-1)


def causal_average_softmax(x: torch.Tensor) -> torch.Tensor:
    """Version 3: equal scores (all zeros) + mask + softmax = the plain causal average."""
    T = x.shape[0]
    scores = torch.zeros(T, T, device=x.device)  # nobody is preferred over anybody
    return masked_softmax_weights(scores) @ x


# --- Sprint 3b: the scores are learned ------------------------------------------------
#
# Every position makes three vectors from its x (each is x @ a learned matrix):
#     query  q: "what am I looking for?"
#     key    k: "what do I contain?"          (a label other positions can match against)
#     value  v: "what do I give if chosen?"
# score[t, s] = q_t . k_s tells how relevant position s is for position t.


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores -> weights -> mix of values. q, k: (T, H), v: (T, D).

    Returns (output (T, D), weights (T, T)).
    """
    head_size = q.shape[-1]
    # (T, H) @ (H, T) -> (T, T). Dividing by sqrt(H) keeps the scores from growing
    # with H: big scores make softmax "winner takes all" and the gradients tiny.
    scores = q @ k.T / head_size**0.5
    weights = masked_softmax_weights(scores)  # the same function as in 3a
    return weights @ v, weights


class SelfAttentionHead:
    """One attention head with three learned matrices, written by hand like NeuralBigram."""

    def __init__(
        self,
        embed_dim: int,
        head_size: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if embed_dim < 1 or head_size < 1:
            raise ValueError("embed_dim and head_size must be >= 1")
        gen = torch.Generator().manual_seed(seed)
        # Small random numbers scaled by 1/sqrt(C), so q, k, v have a size similar to x.
        # (Not zeros as in NeuralBigram: with equal matrices every head would learn the same.)
        shape = (embed_dim, head_size)
        self.Wq, self.Wk, self.Wv = (
            (torch.randn(shape, generator=gen) * embed_dim**-0.5).to(device).requires_grad_(True)
            for _ in range(3)
        )

    @property
    def embed_dim(self) -> int:
        return self.Wq.shape[0]

    def parameters(self) -> list[torch.Tensor]:
        return [self.Wq, self.Wk, self.Wv]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, C) -> (output (T, head_size), weights (T, T))."""
        if x.ndim != 2 or x.shape[1] != self.embed_dim:
            raise ValueError(f"expected x of shape (T, {self.embed_dim}), got {tuple(x.shape)}")
        return attend(x @ self.Wq, x @ self.Wk, x @ self.Wv)


# --- Sprint 3c: many heads at once ----------------------------------------------------
#
# One head learns one kind of relation. Several heads, run side by side, can each look
# for something different ("which verb?", "which vowel?", "where did the quote open?").
# Their outputs are concatenated and mixed once more by Wo, so the block can decide how
# to combine what the heads found.


class MultiHeadAttention:
    """`num_heads` independent heads in parallel, concatenated and projected back.

    Each head gets `embed_dim // num_heads` channels, so the concatenation is exactly
    `embed_dim` wide again and the block keeps the shape `(T, C) -> (T, C)`. That is what
    lets 3c-2 write the residual `x = x + attention(x)`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}: "
                "the heads have to split the channels evenly"
            )
        head_size = embed_dim // num_heads
        # A different seed per head, otherwise every head would start identical and
        # learn the same thing -- num_heads copies of one head is not multi-head.
        self.heads = [
            SelfAttentionHead(embed_dim, head_size, device=device, seed=seed + i)
            for i in range(num_heads)
        ]
        gen = torch.Generator().manual_seed(seed + num_heads)
        self.Wo = (
            (torch.randn((embed_dim, embed_dim), generator=gen) * embed_dim**-0.5)
            .to(device)
            .requires_grad_(True)
        )

    @property
    def embed_dim(self) -> int:
        return self.Wo.shape[0]

    @property
    def num_heads(self) -> int:
        return len(self.heads)

    @property
    def head_size(self) -> int:
        return self.embed_dim // self.num_heads

    def parameters(self) -> list[torch.Tensor]:
        return [p for head in self.heads for p in head.parameters()] + [self.Wo]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, C) -> (output (T, C), weights (num_heads, T, T)).

        The weights of every head are returned separately: they are what an attention
        heatmap draws, and they show that different heads really do look elsewhere.
        """
        outputs, weights = zip(*(head.forward(x) for head in self.heads), strict=True)
        concatenated = torch.cat(outputs, dim=-1)  # num_heads x (T, H) -> (T, C)
        return concatenated @ self.Wo, torch.stack(weights)
