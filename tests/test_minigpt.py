import math

import pytest

torch = pytest.importorskip("torch")

from forgelm.models.minigpt import MiniGPT  # noqa: E402

VOCAB = 66


def small_model(**overrides) -> MiniGPT:
    settings = dict(vocab_size=VOCAB, embed_dim=16, num_heads=4, num_blocks=2, block_size=8, seed=0)
    return MiniGPT(**(settings | overrides))


def test_forward_gives_one_score_per_token_per_position():
    model = small_model()

    logits, weights = model.forward(torch.arange(5))

    assert logits.shape == (5, VOCAB)  # "how likely is each token next?", per position
    assert len(weights) == 2  # one entry per block
    assert weights[0].shape == (4, 5, 5)  # per head, a (T, T) table


def test_an_untrained_model_is_about_as_good_as_guessing():
    # Random weights mean roughly uniform predictions, so the loss starts near ln(V).
    ids = torch.randint(0, VOCAB, (7,), generator=torch.Generator().manual_seed(13))

    loss = small_model().loss(ids[:-1], ids[1:])

    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.7)


def test_the_position_table_limits_the_sequence_length():
    model = small_model(block_size=8)

    model.forward(torch.arange(8))  # exactly full: fine

    with pytest.raises(ValueError, match="longer than block_size"):
        model.forward(torch.arange(9))  # there is no seat number 9


def test_earlier_positions_cannot_see_a_later_token():
    # The property that makes a GPT trainable: every position predicts using only its
    # own past, so one forward pass gives T independent training examples.
    model = small_model()
    ids = torch.tensor([3, 9, 20, 41, 7])
    changed = ids.clone()
    changed[-1] = 65

    before, _ = model.forward(ids)
    after, _ = model.forward(changed)

    assert torch.allclose(after[:-1], before[:-1])
    assert not torch.allclose(after[-1], before[-1])


def test_the_same_token_in_two_seats_is_not_the_same_input():
    # Without the position embedding these two rows would start out identical.
    model = small_model()

    logits, _ = model.forward(torch.tensor([7, 7]))

    assert not torch.allclose(logits[0], logits[1])


def test_zeroing_the_position_table_makes_the_first_two_seats_equal():
    # Proof that the difference above really comes from the position embedding: with no
    # position information, position 1 attends to two identical tokens, so its input to
    # the stack is the same as position 0's.
    model = small_model()
    with torch.no_grad():
        model.position_embedding.zero_()

    logits, _ = model.forward(torch.tensor([7, 7]))

    assert torch.allclose(logits[0], logits[1], atol=1e-5)


def test_generate_returns_the_prompt_plus_the_requested_tokens():
    model = small_model()

    ids = model.generate([5], max_new_tokens=6, generator=torch.Generator().manual_seed(0))

    assert len(ids) == 7
    assert ids[0] == 5  # the prompt is kept
    assert all(0 <= i < VOCAB for i in ids)


def test_generate_is_reproducible_and_the_seed_matters():
    model = small_model()

    a = model.generate([5], 6, torch.Generator().manual_seed(0))
    b = model.generate([5], 6, torch.Generator().manual_seed(0))
    c = model.generate([5], 6, torch.Generator().manual_seed(1))

    assert a == b
    assert a != c


def test_generate_with_temperature_zero_is_greedy_and_needs_no_generator():
    model = small_model()

    a = model.generate([5], 4, temperature=0)
    b = model.generate([5], 4, temperature=0)

    assert a == b


def test_generate_crops_a_prompt_that_is_longer_than_the_context():
    # forward() refuses a long sequence, but generate() must handle it by looking at
    # the last block_size tokens only.
    model = small_model(block_size=4)

    ids = model.generate(list(range(10)), 3, torch.Generator().manual_seed(0))

    assert len(ids) == 13
    assert ids[:10] == list(range(10))


def test_generate_rejects_an_empty_prompt_and_a_negative_temperature():
    model = small_model()

    with pytest.raises(ValueError, match="prompt"):
        model.generate([], 3)

    with pytest.raises(ValueError, match="temperature"):
        model.generate([5], 3, temperature=-1)


def test_gradients_reach_every_parameter_including_both_tables():
    model = small_model()
    ids = torch.randint(0, VOCAB, (8,), generator=torch.Generator().manual_seed(14))

    model.loss(ids[:-1], ids[1:]).backward()

    # 2 tables + 2 blocks x 21 + 2 (ln_final) + 1 (head)
    assert len(model.parameters()) == 2 + 2 * 21 + 2 + 1
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))


def test_only_the_rows_that_were_used_get_a_gradient():
    # An embedding is a row lookup, so a token the batch never saw learns nothing.
    model = small_model()

    model.loss(torch.tensor([3, 9]), torch.tensor([9, 20])).backward()

    assert not torch.allclose(model.token_embedding.grad[3], torch.zeros(model.embed_dim))
    assert torch.equal(model.token_embedding.grad[50], torch.zeros(model.embed_dim))
    # Same for seats: only positions 0 and 1 were occupied.
    assert torch.equal(model.position_embedding.grad[2], torch.zeros(model.embed_dim))


def test_blocks_do_not_share_their_seeds():
    model = small_model(num_blocks=3)

    first = model.blocks[0].attention.heads[0].Wq
    second = model.blocks[1].attention.heads[0].Wq

    assert not torch.allclose(first, second)


def test_forward_rejects_bad_input_and_loss_rejects_mismatched_targets():
    model = small_model()

    with pytest.raises(ValueError, match=r"\(T,\)"):
        model.forward(torch.zeros(2, 3, dtype=torch.long))

    with pytest.raises(ValueError, match=r"\(T,\)"):
        model.forward(torch.tensor([], dtype=torch.long))

    with pytest.raises(ValueError, match="differ"):
        model.loss(torch.tensor([1, 2]), torch.tensor([1, 2, 3]))


def test_bad_sizes_are_rejected():
    with pytest.raises(ValueError, match="vocab_size"):
        small_model(vocab_size=0)

    with pytest.raises(ValueError, match="block_size"):
        small_model(block_size=0)


def test_the_model_can_learn_one_short_sentence_end_to_end():
    """The integration test: if every part is wired correctly, plain SGD can memorise.

    A bigram cannot reproduce this sentence: 'o' is followed by ' ', 'r' and 't', and
    't' by both 'o' and ' ', so one token of context is not enough to choose. MiniGPT
    sees the whole past, so greedy decoding gets it back exactly.
    """
    text = "to be or not to be"
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text])
    model = MiniGPT(
        vocab_size=len(chars), embed_dim=32, num_heads=4, num_blocks=2, block_size=32, seed=0
    )

    first_loss = model.loss(ids[:-1], ids[1:]).item()
    for _ in range(150):
        loss = model.loss(ids[:-1], ids[1:])
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= 0.1 * parameter.grad
                parameter.grad = None

    assert first_loss > 3.0  # started off guessing
    assert loss.item() < 0.05  # and memorised the sentence
    greedy = model.generate([stoi["t"]], len(text) - 1, temperature=0)
    assert "".join(chars[i] for i in greedy) == text
