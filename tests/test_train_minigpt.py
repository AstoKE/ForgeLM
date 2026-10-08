import math

import pytest

torch = pytest.importorskip("torch")

from forgelm.models.minigpt import MiniGPT  # noqa: E402
from forgelm.models.train_minigpt import (  # noqa: E402
    Adam,
    estimate_loss,
    get_batch,
    lr_at,
    train_minigpt,
    train_minigpt_on_text,
)

TINY = {"embed_dim": 16, "num_heads": 2, "num_blocks": 1, "block_size": 8, "device": "cpu"}


# --- Adam ---------------------------------------------------------------------------------


def test_adam_matches_torch_adam_over_many_steps():
    # Same start, same loss, 25 steps: the hand-written Adam must follow torch's exactly.
    start = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    mine = start.clone().requires_grad_(True)
    theirs = start.clone().requires_grad_(True)
    my_opt = Adam([mine], lr=0.05)
    their_opt = torch.optim.Adam([theirs], lr=0.05)

    for _ in range(25):
        for p, opt in ((mine, my_opt), (theirs, their_opt)):
            loss = (p**3).sum() + (p * 2).pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

    assert torch.allclose(mine, theirs, atol=1e-6)


@pytest.mark.parametrize("gradient", [0.01, 100.0])
def test_adam_step_size_does_not_depend_on_the_gradient_size(gradient):
    # The warm-up question of 3d: with a constant gradient every step is about lr.
    p = torch.zeros(1, requires_grad=True)
    opt = Adam([p], lr=0.1)

    for _ in range(10):
        p.grad = torch.tensor([gradient])
        opt.step()

    assert p.item() == pytest.approx(-1.0, abs=1e-3)  # 10 steps * 0.1


def test_adam_zero_grad_clears_gradients_and_skips_unused_parameters():
    used = torch.ones(2, requires_grad=True)
    unused = torch.ones(2, requires_grad=True)  # never gets a gradient
    opt = Adam([used, unused], lr=0.1)

    used.sum().backward()
    opt.step()
    opt.zero_grad()

    assert used.grad is None
    assert torch.equal(unused, torch.ones(2))  # untouched, and no crash


def test_adam_rejects_invalid_settings():
    p = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError, match="lr"):
        Adam([p], lr=0)
    with pytest.raises(ValueError, match="betas"):
        Adam([p], betas=(1.0, 0.999))


# --- get_batch ----------------------------------------------------------------------------


def test_get_batch_y_is_x_shifted_by_one():
    ids = torch.arange(100)  # ids equal their position, so windows are easy to read

    x, y = get_batch(ids, block_size=5, batch_size=7, generator=torch.Generator().manual_seed(0))

    assert x.shape == y.shape == (7, 5)
    assert torch.equal(y, x + 1)  # on arange data "the next token" is "+1"
    assert torch.equal(x[:, 1:], y[:, :-1])  # shifted by one inside every window


def test_get_batch_stays_inside_the_data_even_at_the_edges():
    ids = torch.arange(10)  # block_size 9 leaves exactly one valid window: start 0
    x, y = get_batch(ids, block_size=9, batch_size=4)

    assert x.tolist() == [list(range(9))] * 4
    assert y.tolist() == [list(range(1, 10))] * 4


def test_get_batch_is_reproducible_with_a_seed():
    ids = torch.arange(1000)

    a, _ = get_batch(ids, 8, 4, torch.Generator().manual_seed(3))
    b, _ = get_batch(ids, 8, 4, torch.Generator().manual_seed(3))
    c, _ = get_batch(ids, 8, 4, torch.Generator().manual_seed(4))

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_get_batch_rejects_data_that_is_too_short():
    with pytest.raises(ValueError, match="need at least"):
        get_batch(torch.arange(8), block_size=8, batch_size=1)  # y needs one more token
    with pytest.raises(ValueError, match=">= 1"):
        get_batch(torch.arange(20), block_size=0, batch_size=1)
    with pytest.raises(ValueError, match=">= 1"):
        get_batch(torch.arange(20), block_size=4, batch_size=0)


# --- training loop ------------------------------------------------------------------------

REPEATED = [0, 1, 2, 3] * 40  # a perfectly predictable pattern: 0 1 2 3 0 1 2 3 ...


def train_tiny(**kwargs):
    settings = {
        "steps": 60,
        "lr": 1e-2,
        "batch_size": 4,
        "block_size": 8,
        "embed_dim": 16,
        "num_heads": 2,
        "num_blocks": 1,
        "eval_every": 20,
        "eval_batches": 2,
        "device": "cpu",
    }
    return train_minigpt(REPEATED, REPEATED, vocab_size=4, **{**settings, **kwargs})


def test_training_lowers_the_loss_well_below_the_uniform_baseline():
    _, history = train_tiny()

    assert history.train_loss[0] > 1.0  # starts near ln 4 = 1.386 (or worse)
    assert history.train_loss[-1] < 0.5  # the pattern is learned
    assert history.val_loss[-1] < 0.5


def test_history_records_every_eval_step_and_the_end():
    _, history = train_tiny(steps=60, eval_every=20)

    assert history.steps == [0, 20, 40, 60]
    assert len(history.train_loss) == len(history.val_loss) == 4
    assert history.seconds > 0


def test_training_is_reproducible_and_seed_matters():
    _, a = train_tiny(steps=10, eval_every=10, seed=0)
    _, b = train_tiny(steps=10, eval_every=10, seed=0)
    _, c = train_tiny(steps=10, eval_every=10, seed=1)

    assert a.train_loss == b.train_loss
    assert a.train_loss != c.train_loss


def test_trained_model_predicts_the_next_token_of_the_pattern():
    model, _ = train_tiny(steps=150)

    out = model.generate([0, 1, 2], max_new_tokens=5, temperature=0)

    assert out == [0, 1, 2, 3, 0, 1, 2, 3]


def test_estimate_loss_is_close_to_ln_v_for_an_untrained_model_and_does_not_train():
    model = MiniGPT(vocab_size=4, **TINY)
    before = [p.detach().clone() for p in model.parameters()]

    loss = estimate_loss(model, torch.tensor(REPEATED), batch_size=4, num_batches=2)

    assert 0.5 < loss < 4.0
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
    assert all(p.grad is None for p in model.parameters())  # no_grad: nothing was recorded


def test_train_minigpt_on_text_runs_end_to_end():
    text = "to be or not to be " * 30

    _, tokenizer, history = train_minigpt_on_text(
        text, steps=5, batch_size=2, block_size=8, embed_dim=16, num_heads=2,
        num_blocks=1, eval_every=5, eval_batches=1, device="cpu",
    )  # fmt: skip

    assert history.steps == [0, 5]
    assert tokenizer.decode(tokenizer.encode("to be")) == "to be"


def test_train_rejects_invalid_arguments():
    with pytest.raises(ValueError, match=">= 1"):
        train_tiny(steps=0)
    with pytest.raises(ValueError, match=">= 1"):
        train_tiny(batch_size=0)


def test_train_raises_when_the_loss_blows_up():
    with pytest.raises(FloatingPointError, match="lower --lr"):
        train_tiny(lr=1e30, steps=50)


def test_train_rejects_a_corpus_shorter_than_one_window():
    with pytest.raises(ValueError, match="need at least"):
        train_minigpt([0, 1, 2], [0, 1, 2], vocab_size=4, block_size=8, device="cpu")


# --- the progress callback (Sprint 4b) ----------------------------------------------------


def test_on_progress_is_called_at_every_evaluation():
    seen = []

    _, history = train_minigpt(
        list(range(60)),
        list(range(60)),
        vocab_size=60,
        steps=4,
        eval_every=2,
        batch_size=2,
        on_progress=lambda *args: seen.append(args),
        **TINY,
    )

    # Evaluations happen at 0, 2 and then once more at the end.
    assert [step for step, _, _ in seen] == [0, 2, 4]
    assert [step for step, _, _ in seen] == history.steps
    assert [train for _, train, _ in seen] == history.train_loss
    assert [val for _, _, val in seen] == history.val_loss


def test_training_without_a_callback_is_unchanged():
    # The callback is optional: the default path must stay exactly as it was.
    common = dict(steps=4, eval_every=2, batch_size=2, seed=3, **TINY)
    ids = list(range(60))

    _, quiet = train_minigpt(ids, ids, vocab_size=60, **common)
    _, loud = train_minigpt(ids, ids, vocab_size=60, on_progress=lambda *a: None, **common)

    assert quiet.steps == loud.steps
    assert quiet.train_loss == loud.train_loss
    assert quiet.val_loss == loud.val_loss


def test_a_callback_that_raises_stops_the_training():
    # Nothing is swallowed: a broken callback is the caller's bug and must surface.
    def boom(step, train_loss, val_loss):
        raise RuntimeError("callback is broken")

    with pytest.raises(RuntimeError, match="broken"):
        train_minigpt(
            list(range(60)),
            list(range(60)),
            vocab_size=60,
            steps=4,
            eval_every=2,
            batch_size=2,
            on_progress=boom,
            **TINY,
        )


# --- keeping the best-val weights (before sprint 5) ---------------------------------------


def random_ids(n=400, seed=0):
    """Ids with no structure to learn, so val loss wanders instead of falling."""
    return torch.randint(0, 40, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_the_best_val_loss_and_its_step_are_recorded():
    ids = random_ids()

    _, history = train_minigpt(
        ids, ids, vocab_size=40, steps=20, eval_every=4, batch_size=2, **TINY
    )

    assert history.best_val_loss == min(history.val_loss)
    assert history.best_step == history.steps[history.val_loss.index(min(history.val_loss))]


def test_the_model_returned_is_the_best_one_not_the_last_one():
    # On random data the last evaluation is almost never the best, so the two runs
    # must end up with different weights.
    ids = random_ids()
    common = dict(steps=20, eval_every=4, batch_size=2, seed=1, **TINY)

    best_model, history = train_minigpt(ids, ids, vocab_size=40, keep_best=True, **common)
    last_model, _ = train_minigpt(ids, ids, vocab_size=40, keep_best=False, **common)

    identical = all(
        torch.equal(a, b)
        for a, b in zip(best_model.parameters(), last_model.parameters(), strict=True)
    )
    # They may only agree if the best evaluation happened to be the final one.
    assert identical == (history.best_step == history.steps[-1])


def test_keep_best_off_records_nothing():
    ids = random_ids()

    _, history = train_minigpt(
        ids, ids, vocab_size=40, steps=8, eval_every=4, batch_size=2, keep_best=False, **TINY
    )

    assert history.best_val_loss is None
    assert history.best_step is None


# --- the (B, T) batch dimension -----------------------------------------------------------


def test_a_batch_loss_equals_the_mean_of_its_windows():
    # The batched call must compute exactly what the old per-window loop averaged.
    model = MiniGPT(
        vocab_size=40, block_size=8, **{k: v for k, v in TINY.items() if k != "block_size"}
    )
    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 40, (200,), generator=gen)
    x, y = get_batch(ids, 8, 4, gen)

    batched = model.loss(x, y)
    one_by_one = sum(model.loss(x[i], y[i]) for i in range(len(x))) / len(x)

    assert batched.item() == pytest.approx(one_by_one.item(), abs=1e-5)


# --- the learning-rate schedule (Sprint 4d) -----------------------------------------------


def test_the_defaults_give_a_constant_learning_rate():
    assert {lr_at(step, 100, 3e-3) for step in range(100)} == {3e-3}


def test_warmup_climbs_linearly_to_the_full_rate():
    rates = [lr_at(step, 100, 1e-3, warmup_steps=10) for step in range(10)]

    assert rates[0] == pytest.approx(1e-4)  # not zero: the first step must still move
    assert rates[-1] == pytest.approx(1e-3)
    assert rates == sorted(rates)
    assert rates[4] == pytest.approx(5e-4)


def test_after_warmup_the_rate_decays_to_the_floor():
    rates = [lr_at(step, 100, 1e-3, warmup_steps=10, min_fraction=0.1) for step in range(100)]

    assert rates[10] == pytest.approx(1e-3)  # the peak
    assert rates[-1] == pytest.approx(1e-4, rel=0.05)  # lr * min_fraction
    assert rates[10:] == sorted(rates[10:], reverse=True)  # never goes back up
    assert min(rates) >= 1e-4 - 1e-12  # and never below the floor


def test_the_cosine_is_at_half_height_halfway_through_the_decay():
    middle = lr_at(55, 100, 1e-3, warmup_steps=10, min_fraction=0.0)

    assert middle == pytest.approx(5e-4, rel=0.05)


def test_steps_past_the_end_stay_at_the_floor():
    assert lr_at(500, 100, 1e-3, warmup_steps=10, min_fraction=0.1) == pytest.approx(1e-4)


def test_a_warmup_longer_than_the_run_never_decays():
    rates = [lr_at(step, 10, 1e-3, warmup_steps=50, min_fraction=0.1) for step in range(10)]

    assert rates == sorted(rates)
    assert rates[-1] < 1e-3


@pytest.mark.parametrize(
    "bad", [{"min_fraction": -0.1}, {"min_fraction": 1.5}, {"warmup_steps": -1}]
)
def test_a_nonsensical_schedule_is_refused(bad):
    with pytest.raises(ValueError):
        lr_at(0, 100, 1e-3, **bad)


def test_training_with_a_schedule_runs_and_a_bad_one_is_refused_before_training():
    ids = list(range(60))

    model, history = train_minigpt(
        ids,
        ids,
        vocab_size=60,
        steps=6,
        eval_every=3,
        batch_size=2,
        warmup_steps=2,
        min_lr_fraction=0.1,
        **TINY,
    )

    assert all(math.isfinite(loss) for loss in history.val_loss)
    with pytest.raises(ValueError, match="min_fraction"):
        train_minigpt(ids, ids, vocab_size=60, steps=6, batch_size=2, min_lr_fraction=2.0, **TINY)


def test_a_schedule_changes_the_result_but_the_default_does_not():
    ids = list(range(60))
    common = dict(steps=8, eval_every=4, batch_size=2, seed=5, **TINY)

    plain, _ = train_minigpt(ids, ids, vocab_size=60, **common)
    again, _ = train_minigpt(ids, ids, vocab_size=60, **common)
    shaped = dict(warmup_steps=3, min_lr_fraction=0.1, **common)
    scheduled, _ = train_minigpt(ids, ids, vocab_size=60, **shaped)

    def same(a, b):
        return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True))

    assert same(plain, again)
    assert not same(plain, scheduled)


# --- writing the best model as training runs (before the big run) -------------------------


def test_on_best_fires_only_when_the_val_loss_reaches_a_new_low():
    calls = []
    ids = list(range(60))

    _, history = train_minigpt(
        ids,
        ids,
        vocab_size=60,
        steps=12,
        eval_every=3,
        batch_size=2,
        on_best=lambda model, step: calls.append(step),
        **TINY,
    )

    lows = [
        step
        for i, step in enumerate(history.steps)
        if history.val_loss[i] < min(history.val_loss[:i], default=float("inf"))
    ]
    assert calls == lows
    assert calls[0] == 0  # the very first evaluation is trivially a new low


def test_the_model_given_to_on_best_holds_the_weights_that_earned_it():
    snapshots = []
    ids = list(range(60))

    model, history = train_minigpt(
        ids,
        ids,
        vocab_size=60,
        steps=12,
        eval_every=3,
        batch_size=2,
        on_best=lambda m, step: snapshots.append([p.detach().clone() for p in m.parameters()]),
        **TINY,
    )

    # The last time it fired was the best step, and the returned model is the best one:
    # so the final weights must equal what the callback saw at that moment.
    assert snapshots
    assert history.best_step is not None
    assert all(
        torch.equal(final, seen)
        for final, seen in zip(model.parameters(), snapshots[-1], strict=True)
    )


def test_on_best_also_fires_when_keep_best_is_off():
    calls = []
    ids = list(range(60))

    train_minigpt(
        ids,
        ids,
        vocab_size=60,
        steps=6,
        eval_every=3,
        batch_size=2,
        keep_best=False,
        on_best=lambda model, step: calls.append(step),
        **TINY,
    )

    assert calls


def test_training_on_text_hands_the_callback_the_tokenizer_too():
    seen = []

    train_minigpt_on_text(
        "to be or not to be, that is the question.\n" * 20,
        steps=4,
        eval_every=2,
        batch_size=2,
        on_best=lambda model, tokenizer, step: seen.append((tokenizer.vocab_size, step)),
        **TINY,
    )

    assert seen
    assert all(vocab > 0 for vocab, _ in seen)


def test_a_failing_callback_stops_training_instead_of_being_swallowed():
    def disk_full(model, step):
        raise OSError("disk full")

    ids = list(range(60))

    with pytest.raises(OSError, match="disk full"):
        train_minigpt(
            ids, ids, vocab_size=60, steps=6, eval_every=3, batch_size=2, on_best=disk_full, **TINY
        )


# --- continuing a run (a long run is cut into pieces) -------------------------------------

TEXT = "to be or not to be, that is the question.\n" * 30


def trained_tiny(steps=60):
    return train_minigpt_on_text(
        TEXT, steps=steps, eval_every=steps, batch_size=4, lr=3e-3, seed=0, **TINY
    )


def test_a_resumed_run_starts_where_the_first_one_ended_not_from_random_weights():
    model, tokenizer, first = trained_tiny()

    _, _, resumed = train_minigpt_on_text(
        TEXT,
        model=model,
        resume_tokenizer=tokenizer,
        steps=2,
        eval_every=2,
        batch_size=4,
        seed=1,
        **TINY,
    )
    _, _, fresh = train_minigpt_on_text(TEXT, steps=2, eval_every=2, batch_size=4, seed=1, **TINY)

    assert resumed.val_loss[0] < fresh.val_loss[0] - 0.5  # it did not start from scratch
    assert resumed.val_loss[0] == pytest.approx(first.val_loss[-1], abs=0.5)


def test_training_goes_on_in_the_model_that_was_passed():
    model, tokenizer, _ = trained_tiny(steps=10)
    before = [p.detach().clone() for p in model.parameters()]

    returned, _, _ = train_minigpt_on_text(
        TEXT,
        model=model,
        resume_tokenizer=tokenizer,
        steps=6,
        eval_every=3,
        batch_size=4,
        keep_best=False,
        **TINY,
    )

    assert returned is model
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_the_size_of_a_resumed_model_is_its_own_not_the_flags():
    model, tokenizer, _ = trained_tiny(steps=4)
    width, context = model.embed_dim, model.block_size

    sizes = {"embed_dim": 99, "num_heads": 3, "num_blocks": 7, "block_size": 1000}
    returned, _, _ = train_minigpt_on_text(
        TEXT,
        model=model,
        resume_tokenizer=tokenizer,
        steps=2,
        eval_every=2,
        batch_size=2,
        device="cpu",
        **sizes,
    )

    assert (returned.embed_dim, returned.block_size) == (width, context)


def test_a_model_that_knows_a_different_number_of_characters_is_refused():
    model, _, _ = trained_tiny(steps=4)

    with pytest.raises(ValueError, match="characters"):
        train_minigpt(
            list(range(50)), list(range(50)), vocab_size=model.vocab_size + 5, model=model, steps=2
        )


def test_a_text_with_other_characters_than_the_checkpoint_is_refused():
    # Same vocabulary *size* would not be enough: the ids would mean different characters.
    model, tokenizer, _ = trained_tiny(steps=4)

    with pytest.raises(ValueError, match="different set of characters"):
        train_minigpt_on_text(
            "QWERTYUIOP ASDFGHJKL " * 40,
            model=model,
            resume_tokenizer=tokenizer,
            steps=2,
            **TINY,
        )
