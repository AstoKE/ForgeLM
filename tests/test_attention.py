import pytest

torch = pytest.importorskip("torch")

from forgelm.models.attention import (  # noqa: E402
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
