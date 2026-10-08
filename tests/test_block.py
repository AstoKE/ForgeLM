import pytest

torch = pytest.importorskip("torch")

from forgelm.models.block import FeedForward, LayerNorm, TransformerBlock  # noqa: E402


def test_feedforward_keeps_the_shape_and_widens_by_four():
    ff = FeedForward(embed_dim=8)

    out = ff.forward(torch.randn(5, 8))

    assert ff.hidden_dim == 32  # 4 * embed_dim
    assert out.shape == (5, 8)  # same width in, same width out (needed for the residual)


def test_feedforward_matches_the_hand_computed_example():
    # x = [1, -2] -> x @ W1 = [1, -2, 0, -3] -> relu -> [1, 0, 0, 0] -> @ W2 = [1, 1]
    ff = FeedForward(embed_dim=2, hidden_dim=4)
    with torch.no_grad():
        ff.W1.copy_(torch.tensor([[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, 1.0, 1.0]]))
        ff.W2.copy_(torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))

    out = ff.forward(torch.tensor([[1.0, -2.0]]))

    assert out[0].tolist() == pytest.approx([1.0, 1.0])


def test_relu_silences_the_negative_units():
    # Without the relu the same input would give x @ (W1 @ W2) = [0 * 1 + 3 * -2, ...].
    ff = FeedForward(embed_dim=2, hidden_dim=4)
    with torch.no_grad():
        ff.W1.copy_(torch.tensor([[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, 1.0, 1.0]]))
        ff.W2.copy_(torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))
    x = torch.tensor([[1.0, -2.0]])

    linear = x @ (ff.W1 @ ff.W2)  # the two layers collapse into one matrix

    assert linear[0].tolist() == pytest.approx([-6.0, -2.0])
    assert not torch.allclose(ff.forward(x), linear)


def test_every_position_is_processed_on_its_own():
    ff = FeedForward(embed_dim=4)
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(6))
    x_changed = x.clone()
    x_changed[0] += 100  # only row 0 changes

    before = ff.forward(x)
    after = ff.forward(x_changed)

    # No information crosses positions here: that is attention's job.
    assert torch.allclose(after[1:], before[1:])
    assert not torch.allclose(after[0], before[0])


def test_the_same_row_always_gives_the_same_answer():
    # One network, reused at every position: equal rows must come out equal.
    ff = FeedForward(embed_dim=4)
    row = torch.randn(4, generator=torch.Generator().manual_seed(7))

    out = ff.forward(torch.stack([row, torch.randn(4), row]))

    assert torch.allclose(out[0], out[2])


def test_gradients_reach_both_layers_and_both_biases():
    ff = FeedForward(embed_dim=4)

    ff.forward(torch.randn(5, 4)).sum().backward()

    assert len(ff.parameters()) == 4  # W1, b1, W2, b2
    for parameter in ff.parameters():
        assert parameter.grad is not None
    assert not torch.allclose(ff.W1.grad, torch.zeros_like(ff.W1.grad))


def test_feedforward_is_reproducible_and_seed_changes_it():
    x = torch.randn(4, 8)

    a = FeedForward(8, seed=0).forward(x)
    b = FeedForward(8, seed=0).forward(x)
    c = FeedForward(8, seed=1).forward(x)

    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_feedforward_rejects_bad_sizes_and_bad_input():
    with pytest.raises(ValueError, match="embed_dim"):
        FeedForward(embed_dim=0)

    with pytest.raises(ValueError, match="hidden_dim"):
        FeedForward(embed_dim=4, hidden_dim=0)

    with pytest.raises(ValueError, match=r"\(T, 4\)"):
        FeedForward(embed_dim=4).forward(torch.randn(5, 3))


# --- LayerNorm -------------------------------------------------------------------------


def test_layer_norm_gives_every_row_mean_zero_and_std_one():
    ln = LayerNorm(embed_dim=4)

    out = ln.forward(torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 90.0]]))

    assert out.mean(dim=-1).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert out.std(dim=-1, unbiased=False).tolist() == pytest.approx([1.0, 1.0], abs=1e-4)


def test_layer_norm_starts_as_the_plain_normalisation():
    ln = LayerNorm(embed_dim=4)

    assert ln.gamma.tolist() == [1.0, 1.0, 1.0, 1.0]  # scale: no stretching yet
    assert ln.beta.tolist() == [0.0, 0.0, 0.0, 0.0]  # shift: no offset yet


def test_layer_norm_matches_torch_layer_norm():
    # The hand-written version against PyTorch's built-in, like cross_entropy in 2b.
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(8))

    ours = LayerNorm(embed_dim=8).forward(x)
    theirs = torch.nn.functional.layer_norm(x, (8,), eps=1e-5)

    assert torch.allclose(ours, theirs, atol=1e-6)


def test_layer_norm_ignores_the_scale_of_its_input():
    # This is the point of the layer: 100x bigger numbers come out the same.
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(9))
    ln = LayerNorm(embed_dim=8)

    assert torch.allclose(ln.forward(x), ln.forward(x * 100), atol=1e-4)
    assert torch.allclose(ln.forward(x), ln.forward(x + 7), atol=1e-5)  # and of its offset


def test_layer_norm_survives_a_constant_row():
    # Variance 0 would be a division by zero; eps is what keeps it finite.
    out = LayerNorm(embed_dim=4).forward(torch.tensor([[5.0, 5.0, 5.0, 5.0]]))

    assert not out.isnan().any()
    assert out[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_layer_norm_normalises_each_position_on_its_own():
    ln = LayerNorm(embed_dim=4)
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(10))
    x_changed = x.clone()
    x_changed[0] *= 50

    before = ln.forward(x)
    after = ln.forward(x_changed)

    assert torch.allclose(after[1:], before[1:])  # no batch statistics, no neighbours
    assert not torch.allclose(after[0], before[0])


def test_layer_norm_gradients_reach_gamma_and_beta():
    ln = LayerNorm(embed_dim=4)

    (ln.forward(torch.randn(5, 4)) * torch.randn(5, 4)).sum().backward()

    assert len(ln.parameters()) == 2
    for parameter in ln.parameters():
        assert parameter.grad is not None
        assert not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))


def test_layer_norm_rejects_bad_sizes_and_bad_input():
    with pytest.raises(ValueError, match="embed_dim"):
        LayerNorm(embed_dim=0)

    with pytest.raises(ValueError, match=r"\(T, 4\)"):
        LayerNorm(embed_dim=4).forward(torch.randn(5, 3))


# --- TransformerBlock ------------------------------------------------------------------


def test_block_keeps_the_shape_so_blocks_can_be_stacked():
    block = TransformerBlock(embed_dim=8, num_heads=4)

    out, weights = block.forward(torch.randn(5, 8))

    assert out.shape == (5, 8)  # same in, same out: stackable
    assert weights.shape == (4, 5, 5)  # the attention weights come back out for a heatmap


def test_block_with_silenced_sublayers_is_the_identity():
    # The residual proof: if attention and the MLP both output 0, then
    # x = x + 0 = x, and the block passes the input through untouched.
    block = TransformerBlock(embed_dim=8, num_heads=4)
    with torch.no_grad():
        block.attention.Wo.zero_()  # kills the attention branch
        block.feedforward.W2.zero_()  # kills the MLP branch
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(11))

    out, _ = block.forward(x)

    assert torch.equal(out, x)


def test_block_still_ignores_the_future():
    # LayerNorm and Wo work inside one position, so neither can leak the future.
    block = TransformerBlock(embed_dim=8, num_heads=4)
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(12))
    x_changed = x.clone()
    x_changed[4] += 100

    before, _ = block.forward(x)
    after, _ = block.forward(x_changed)

    assert torch.allclose(after[:4], before[:4])
    assert not torch.allclose(after[4], before[4])


def test_block_gradients_reach_all_twenty_one_tensors():
    block = TransformerBlock(embed_dim=8, num_heads=4)

    out, _ = block.forward(torch.randn(5, 8))
    (out * torch.randn(5, 8)).sum().backward()

    # 2 (ln1) + 4*3 + 1 (attention) + 2 (ln2) + 4 (feedforward)
    assert len(block.parameters()) == 21
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))


def test_block_sublayers_do_not_share_a_seed():
    # The MLP must not start as a copy of head 0's matrices.
    block = TransformerBlock(embed_dim=4, num_heads=4, hidden_dim=4, seed=0)

    assert not torch.allclose(block.feedforward.W1, block.attention.heads[0].Wq)


# --- the (B, T, C) batch dimension ---------------------------------------------------------


def test_layer_norm_and_feedforward_normalise_inside_a_batch():
    x = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(33)) * 10

    normed = LayerNorm(embed_dim=8).forward(x)
    through = FeedForward(embed_dim=8).forward(x)

    assert normed.shape == through.shape == (3, 5, 8)
    # Still the last axis only: every (batch, position) row separately.
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(3, 5), atol=1e-6)
    assert torch.allclose(normed.std(dim=-1, unbiased=False), torch.ones(3, 5), atol=1e-4)


def test_a_block_carries_a_batch_and_a_batch_of_one_matches():
    block = TransformerBlock(embed_dim=8, num_heads=4, seed=9)
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(34))

    flat, flat_weights = block.forward(x)
    batched, batched_weights = block.forward(x.unsqueeze(0))

    assert batched.shape == (1, 5, 8)
    assert batched_weights.shape == (1, 4, 5, 5)
    assert torch.allclose(batched[0], flat, atol=1e-6)
    assert torch.allclose(batched_weights[0], flat_weights, atol=1e-6)
