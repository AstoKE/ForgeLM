import pytest

torch = pytest.importorskip("torch")

from forgelm.models.block import FeedForward  # noqa: E402


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
