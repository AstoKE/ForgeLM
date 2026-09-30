import pytest

torch = pytest.importorskip("torch")

from forgelm.models.attention import (  # noqa: E402
    MultiHeadAttention,
    SelfAttentionHead,
    attend,
    causal_average_loop,
    causal_average_matmul,
    causal_average_softmax,
    causal_mask,
    masked_softmax_weights,
    uniform_causal_weights,
)

VERSIONS = [causal_average_loop, causal_average_matmul, causal_average_softmax]


def column(*values):
    """One number per position: shape (T, 1)."""
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1)


def test_causal_mask_is_a_lower_triangle():
    assert causal_mask(3).tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


@pytest.mark.parametrize("average", VERSIONS)
def test_average_of_2_4_6(average):
    # position 0: 2 | position 1: (2+4)/2 = 3 | position 2: (2+4+6)/3 = 4
    assert average(column(2, 4, 6)).squeeze(1).tolist() == pytest.approx([2, 3, 4])


@pytest.mark.parametrize("average", VERSIONS)
def test_average_of_10_0_5(average):
    # position 0: 10 | position 1: (10+0)/2 = 5 | position 2: (10+0+5)/3 = 5
    assert average(column(10, 0, 5)).squeeze(1).tolist() == pytest.approx([10, 5, 5])


@pytest.mark.parametrize("average", VERSIONS)
def test_changing_the_future_does_not_change_the_past(average):
    # The key property of a causal model: position 0 and 1 can't see position 2.
    before = average(column(10, 0, 5))
    after = average(column(10, 0, 100))

    assert after[0].item() == pytest.approx(before[0].item())  # still 10
    assert after[1].item() == pytest.approx(before[1].item())  # still 5
    assert after[2].item() != pytest.approx(before[2].item())  # only position 2 changes


def test_all_three_versions_agree_on_random_data():
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))

    loop = causal_average_loop(x)
    assert torch.allclose(causal_average_matmul(x), loop, atol=1e-6)
    assert torch.allclose(causal_average_softmax(x), loop, atol=1e-6)


def test_uniform_weights_rows_sum_to_one_and_ignore_the_future():
    w = uniform_causal_weights(4)

    assert torch.allclose(w.sum(dim=1), torch.ones(4))
    assert torch.equal(w.triu(diagonal=1), torch.zeros(4, 4))  # above the diagonal = future
    assert w[2].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0])


def test_softmax_of_equal_scores_gives_the_uniform_weights():
    assert torch.allclose(masked_softmax_weights(torch.zeros(5, 5)), uniform_causal_weights(5))


def test_masked_softmax_ignores_the_future_even_with_huge_scores():
    scores = torch.zeros(3, 3)
    scores[0, 2] = 1e9  # position 0 "really wants" to look at position 2

    w = masked_softmax_weights(scores)
    assert w[0].tolist() == pytest.approx([1, 0, 0])  # but it can only see itself


def test_masked_softmax_prefers_higher_scores_among_allowed_positions():
    scores = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    w = masked_softmax_weights(scores)
    assert w[2, 1] > w[2, 0]  # position 2 pays more attention to position 1
    assert w[2].sum().item() == pytest.approx(1.0)


def test_single_position_returns_itself():
    x = torch.tensor([[1.0, -2.0, 3.0]])

    for average in VERSIONS:
        assert torch.allclose(average(x), x)


# --- Sprint 3b: self-attention ---------------------------------------------------------


def test_attend_matches_the_hand_computed_example():
    # head_size 4 -> scores are divided by sqrt(4) = 2. Row 2 of q.k^T is [4, 0, 2],
    # so the scaled scores are [2, 0, 1] and the weights are softmax([2, 0, 1]).
    q = torch.zeros(3, 4)
    q[2] = 1.0
    k = torch.tensor([[1.0, 1, 1, 1], [0.0, 0, 0, 0], [1.0, 1, 0, 0]])
    v = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

    out, weights = attend(q, k, v)

    assert weights[2].tolist() == pytest.approx([0.6652, 0.0900, 0.2447], abs=1e-4)
    assert out[2].tolist() == pytest.approx([7.876, 2.124], abs=1e-3)  # 3a's average gave [5, 5]


def test_attend_with_zero_q_and_k_is_the_plain_causal_average():
    # Zero scores = nobody preferred = exactly what 3a computed.
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(1))

    out, _ = attend(torch.zeros(6, 2), torch.zeros(6, 2), x)

    assert torch.allclose(out, causal_average_matmul(x), atol=1e-6)


def test_head_output_and_weights_have_the_right_shapes():
    head = SelfAttentionHead(embed_dim=8, head_size=4)

    out, weights = head.forward(torch.randn(5, 8))

    assert out.shape == (5, 4)  # one output per position, head_size numbers each
    assert weights.shape == (5, 5)


def test_head_weights_rows_sum_to_one_and_ignore_the_future():
    head = SelfAttentionHead(embed_dim=8, head_size=4)

    _, weights = head.forward(torch.randn(5, 8))

    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.equal(weights.triu(diagonal=1), torch.zeros(5, 5))


def test_head_changing_the_future_does_not_change_the_past():
    head = SelfAttentionHead(embed_dim=8, head_size=4)
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(2))
    x_changed = x.clone()
    x_changed[4] += 100  # only the last position changes

    before, _ = head.forward(x)
    after, _ = head.forward(x_changed)

    assert torch.allclose(after[:4], before[:4])  # positions 0..3 can't see position 4
    assert not torch.allclose(after[4], before[4])


def test_head_is_reproducible_and_seed_changes_it():
    x = torch.randn(4, 8)

    a, _ = SelfAttentionHead(8, 4, seed=0).forward(x)
    b, _ = SelfAttentionHead(8, 4, seed=0).forward(x)
    c, _ = SelfAttentionHead(8, 4, seed=1).forward(x)

    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_gradients_reach_all_three_matrices():
    head = SelfAttentionHead(embed_dim=8, head_size=4)

    out, _ = head.forward(torch.randn(5, 8))
    out.sum().backward()

    for w in head.parameters():
        assert w.grad is not None and w.grad.abs().sum() > 0


def test_head_rejects_wrong_input_shape():
    head = SelfAttentionHead(embed_dim=8, head_size=4)

    with pytest.raises(ValueError, match="expected x of shape"):
        head.forward(torch.randn(5, 7))  # wrong number of channels
    with pytest.raises(ValueError, match="expected x of shape"):
        head.forward(torch.randn(2, 5, 8))  # a batch dimension we don't support (yet)


def test_head_rejects_invalid_sizes():
    with pytest.raises(ValueError, match=">= 1"):
        SelfAttentionHead(embed_dim=0, head_size=4)
    with pytest.raises(ValueError, match=">= 1"):
        SelfAttentionHead(embed_dim=8, head_size=0)


# --- Sprint 3c: multi-head -------------------------------------------------------------


def test_multi_head_keeps_the_embedding_width():
    mha = MultiHeadAttention(embed_dim=8, num_heads=4)

    out, weights = mha.forward(torch.randn(5, 8))

    assert mha.head_size == 2  # 8 channels split over 4 heads
    assert out.shape == (5, 8)  # concat(4 x 2) = 8, then Wo keeps it at 8
    assert weights.shape == (4, 5, 5)  # one (T, T) table per head


def test_one_head_with_identity_wo_is_exactly_a_single_head():
    # num_heads=1 means head_size == embed_dim, so 3c generalises 3b.
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(3))
    mha = MultiHeadAttention(embed_dim=4, num_heads=1, seed=7)
    with torch.no_grad():
        mha.Wo.copy_(torch.eye(4))  # no mixing after the concat
    head = SelfAttentionHead(embed_dim=4, head_size=4, seed=7)

    mha_out, mha_weights = mha.forward(x)
    head_out, head_weights = head.forward(x)

    assert torch.allclose(mha_out, head_out, atol=1e-6)
    assert torch.allclose(mha_weights[0], head_weights)


def test_heads_are_not_copies_of_each_other():
    mha = MultiHeadAttention(embed_dim=8, num_heads=4, seed=0)

    _, weights = mha.forward(torch.randn(5, 8, generator=torch.Generator().manual_seed(4)))

    # Every head gets its own seed, so they look at different places.
    assert not torch.allclose(weights[0], weights[1])
    assert not torch.allclose(weights[0], weights[3])


def test_multi_head_still_ignores_the_future():
    mha = MultiHeadAttention(embed_dim=8, num_heads=4)
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(5))
    x_changed = x.clone()
    x_changed[4] += 100

    before, weights = mha.forward(x)
    after, _ = mha.forward(x_changed)

    assert torch.allclose(after[:4], before[:4])  # Wo mixes channels, never positions
    assert not torch.allclose(after[4], before[4])
    assert torch.equal(weights.triu(diagonal=1), torch.zeros(4, 5, 5))


def test_multi_head_gradients_reach_every_head_and_the_projection():
    mha = MultiHeadAttention(embed_dim=8, num_heads=4)

    out, _ = mha.forward(torch.randn(5, 8))
    out.sum().backward()

    assert len(mha.parameters()) == 4 * 3 + 1  # Wq/Wk/Wv per head, plus Wo
    for parameter in mha.parameters():
        assert parameter.grad is not None
        assert not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))


def test_multi_head_rejects_an_uneven_split():
    with pytest.raises(ValueError, match="not divisible"):
        MultiHeadAttention(embed_dim=10, num_heads=4)

    with pytest.raises(ValueError, match="num_heads"):
        MultiHeadAttention(embed_dim=8, num_heads=0)
