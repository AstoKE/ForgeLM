import math

import pytest

torch = pytest.importorskip("torch")  # skip this whole file when the `ml` extra isn't installed
import torch.nn.functional as F  # noqa: E402

from forgelm.models import BigramModel  # noqa: E402
from forgelm.models.neural_bigram import (  # noqa: E402
    NeuralBigram,
    cross_entropy,
    pick_device,
    train_neural_bigram,
)

V = 5


def test_embedding_lookup_equals_one_hot_matmul():
    W = torch.randn(V, V)
    prev = torch.tensor([3, 0, 3, 1])

    one_hot = F.one_hot(prev, num_classes=V).float()  # (4, V)
    assert torch.allclose(one_hot @ W, W[prev])


def test_hand_written_cross_entropy_matches_pytorch():
    logits = torch.randn(8, V)
    targets = torch.randint(0, V, (8,))

    assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_cross_entropy_is_stable_for_large_logits():
    logits = torch.tensor([[1000.0, 0.0], [0.0, 1000.0]])

    assert cross_entropy(logits, torch.tensor([0, 1])).item() == pytest.approx(0.0)


def test_zero_weights_give_uniform_loss():
    model = NeuralBigram(V)
    prev, nxt = torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])

    assert model.loss(prev, nxt).item() == pytest.approx(math.log(V))


def test_autograd_gradient_matches_formula():
    # For logits = W[prev] and cross-entropy, dL/dlogits = (softmax(logits) - one_hot(target)) / N.
    # W[prev] routes each row's gradient back to row `prev` of W (index_add sums repeats).
    model = NeuralBigram(V)
    with torch.no_grad():
        model.W.copy_(torch.randn(V, V))
    prev = torch.tensor([0, 2, 2, 4])
    nxt = torch.tensor([1, 3, 0, 4])

    model.loss(prev, nxt).backward()

    logits = model.W.detach()[prev]
    d_logits = (logits.softmax(dim=-1) - F.one_hot(nxt, V).float()) / len(nxt)
    expected = torch.zeros(V, V).index_add_(0, prev, d_logits)
    assert torch.allclose(model.W.grad, expected, atol=1e-6)


def test_hand_written_sgd_step_matches_torch_optim():
    prev, nxt = torch.tensor([0, 1, 1]), torch.tensor([1, 2, 0])
    ours, theirs = NeuralBigram(V), NeuralBigram(V)
    optimizer = torch.optim.SGD([theirs.W], lr=0.5)

    ours.loss(prev, nxt).backward()
    ours.sgd_step(lr=0.5)

    theirs.loss(prev, nxt).backward()
    optimizer.step()

    assert torch.allclose(ours.W, theirs.W)
    assert ours.W.grad is None  # gradient reset after the step


def test_training_converges_to_counting_model():
    ids = [1, 2, 3, 1, 2, 4, 1, 3, 3, 2] * 30
    counting = BigramModel.train(ids, V, smoothing=0.0)  # smoothing 0 = best possible fit

    _, history = train_neural_bigram(
        ids, ids, V, steps=400, lr=10.0, batch_size=len(ids), eval_every=100, device="cpu"
    )

    assert history.train_loss[0] == pytest.approx(math.log(V))
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.train_loss[-1] == pytest.approx(counting.loss(ids), abs=0.02)


def test_training_is_reproducible_with_seed():
    ids = [1, 2, 3, 4, 1, 3, 2, 4] * 20
    kwargs = dict(steps=20, lr=5.0, batch_size=16, device="cpu", seed=3)

    first, _ = train_neural_bigram(ids, ids, V, **kwargs)
    second, _ = train_neural_bigram(ids, ids, V, **kwargs)
    assert torch.equal(first.W, second.W)


def test_overflowing_learning_rate_is_reported():
    # lr so large that W overflows float32 (max ~3.4e38) -> inf - inf = nan.
    # (A merely "too large" lr doesn't produce nan: logsumexp stays stable, the
    # loss just jumps around. That's why this check only catches overflow.)
    ids = [1, 2, 3, 4] * 10

    with pytest.raises(FloatingPointError, match="lower --lr"):
        train_neural_bigram(ids, ids, V, steps=50, lr=1e39, batch_size=8, device="cpu")


@pytest.mark.parametrize(
    "kwargs",
    [{"steps": 0}, {"lr": 0.0}, {"lr": -1.0}, {"batch_size": 0}, {"eval_every": 0}],
)
def test_invalid_training_arguments(kwargs):
    with pytest.raises(ValueError):
        train_neural_bigram([1, 2, 3], [1, 2], V, device="cpu", **kwargs)


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        pick_device("tpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_trains_on_cuda():
    ids = [1, 2, 3, 4] * 10
    model, history = train_neural_bigram(ids, ids, V, steps=5, batch_size=8, device="cuda")

    assert model.W.device.type == "cuda"
    assert history.device.type == "cuda"
