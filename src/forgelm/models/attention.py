"""Mixing information from past tokens: the skeleton of self-attention.

A bigram only sees the previous token. To see further back, every position t
needs a summary of positions 0..t. Sprint 3a uses the simplest summary, the
plain average, and computes it three equivalent ways:

    1. a Python loop          (easiest to read)
    2. one matrix multiply    (what the GPU is good at)
    3. mask + softmax         (the exact shape self-attention will use in 3b)

Shapes: x is (T, C): T positions ("time"), C numbers per position ("channels").
Position t may only look at positions 0..t, never at the future (causal).
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
