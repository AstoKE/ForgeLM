import importlib.util

import pytest

from forgelm.inference import (
    InvalidRequestError,
    MLNotInstalledError,
    ModelNotFoundError,
    ModelStore,
    attention_maps,
    generate_text,
)
from forgelm.models import save_checkpoint, train_on_text

TEXT = "to be or not to be, that is the question.\n" * 20


@pytest.fixture
def folder(tmp_path):
    """A checkpoint folder holding a tiny bigram, and a tiny MiniGPT when torch is installed."""
    report = train_on_text(TEXT)
    save_checkpoint(tmp_path / "bigram.json", report.model, report.tokenizer)
    try:
        import torch  # noqa: F401
    except ImportError:
        return tmp_path
    from forgelm.models.minigpt import MiniGPT, save_minigpt

    gpt = MiniGPT(report.tokenizer.vocab_size, 16, 2, 2, block_size=8)
    save_minigpt(tmp_path / "gpt.pt", gpt, report.tokenizer)
    return tmp_path


@pytest.fixture
def store(folder):
    return ModelStore(folder)


needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


# --- the store ------------------------------------------------------------------------------


def test_list_models_shows_kind_and_ignores_other_files(folder, store):
    (folder / "notes.txt").write_text("not a model")
    (folder / ".hidden.json").write_text("{}")

    names = {m.name: m.kind for m in store.list_models()}

    assert names.get("bigram.json") == "bigram"
    assert "notes.txt" not in names and ".hidden.json" not in names


def test_list_models_of_a_missing_folder_is_empty(tmp_path):
    assert ModelStore(tmp_path / "nope").list_models() == []


def test_a_model_is_loaded_once_and_cached(store):
    assert store.get("bigram.json") is store.get("bigram.json")


def test_a_retrained_file_is_reloaded(folder, store):
    first = store.get("bigram.json")
    report = train_on_text(TEXT + "extra text to change the counts\n" * 5)
    save_checkpoint(folder / "bigram.json", report.model, report.tokenizer)

    assert store.get("bigram.json") is not first


@pytest.mark.parametrize(
    "name", ["../secret.pt", "/etc/passwd", "sub/model.json", "..", ".hidden.json", ""]
)
def test_names_that_reach_outside_the_folder_are_rejected(store, name):
    with pytest.raises(InvalidRequestError, match="invalid model name"):
        store.get(name)


def test_unknown_extension_and_missing_file(store):
    with pytest.raises(InvalidRequestError, match="unknown model type"):
        store.get("model.txt")
    with pytest.raises(ModelNotFoundError):
        store.get("missing.json")


def test_a_corrupt_checkpoint_is_an_invalid_request_not_a_crash(folder, store):
    (folder / "broken.json").write_text("{not json")

    with pytest.raises(InvalidRequestError, match="cannot load model"):
        store.get("broken.json")


def test_a_pt_model_without_torch_says_so(folder, store, monkeypatch):
    import sys

    (folder / "any.pt").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "forgelm.models.minigpt", raising=False)

    with pytest.raises(MLNotInstalledError, match="PyTorch is not installed"):
        store.get("any.pt")


# --- generate ---------------------------------------------------------------------------------


def test_bigram_generate_is_reproducible_with_a_seed(store):
    a = generate_text(store, "bigram.json", "to", 30, 1.0, seed=7)
    b = generate_text(store, "bigram.json", "to", 30, 1.0, seed=7)

    assert a == b
    assert a.text.startswith("to")
    assert a.num_new_tokens == 30


@needs_torch
def test_minigpt_generate_is_reproducible_with_a_seed(store):
    a = generate_text(store, "gpt.pt", "to", 20, 1.0, seed=3)
    b = generate_text(store, "gpt.pt", "to", 20, 1.0, seed=3)

    assert a == b
    assert a.text.startswith("to")


@needs_torch
def test_greedy_generation_does_not_need_a_seed(store):
    a = generate_text(store, "gpt.pt", "to", 10, temperature=0)
    b = generate_text(store, "gpt.pt", "to", 10, temperature=0)

    assert a.text == b.text


def test_zero_new_tokens_returns_the_prompt(store):
    assert generate_text(store, "bigram.json", "to be", 0).text == "to be"


@pytest.mark.parametrize(
    ("prompt", "max_tokens", "temperature", "message"),
    [("", 5, 1.0, "prompt"), ("to", -1, 1.0, "max_tokens"), ("to", 5, -0.1, "temperature")],
)
def test_generate_rejects_bad_arguments(store, prompt, max_tokens, temperature, message):
    with pytest.raises(InvalidRequestError, match=message):
        generate_text(store, "bigram.json", prompt, max_tokens, temperature)


# --- attention --------------------------------------------------------------------------------


@needs_torch
def test_attention_has_one_square_table_per_block_and_head(store):
    result = attention_maps(store, "gpt.pt", "to be")

    assert result.tokens == ["t", "o", " ", "b", "e"]
    assert len(result.weights) == 2  # blocks
    assert len(result.weights[0]) == 2  # heads
    assert len(result.weights[0][0]) == 5 and len(result.weights[0][0][0]) == 5  # (T, T)


@needs_torch
def test_attention_rows_sum_to_one_and_the_future_is_zero(store):
    grid = attention_maps(store, "gpt.pt", "to be").weights[1][0]

    for t, row in enumerate(grid):
        assert sum(row) == pytest.approx(1.0, abs=1e-5)
        assert all(w == 0 for w in row[t + 1 :])  # no looking at the future


def test_attention_rejects_a_bigram_and_empty_text(store):
    with pytest.raises(InvalidRequestError, match="needs a MiniGPT"):
        attention_maps(store, "bigram.json", "to be")
    with pytest.raises(InvalidRequestError, match="empty"):
        attention_maps(store, "bigram.json", "")


@needs_torch
def test_attention_rejects_text_longer_than_the_context_window(store):
    with pytest.raises(InvalidRequestError, match="context window is 8"):
        attention_maps(store, "gpt.pt", "this text is far too long")


# --- choosing the device (before the big model) --------------------------------------------


@needs_torch
def test_a_store_loads_models_on_the_cpu_by_default(folder):
    loaded = ModelStore(folder).get("gpt.pt")

    assert loaded.model.token_embedding.device.type == "cpu"


@needs_torch
def test_a_store_can_be_told_to_use_the_gpu(folder):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU in this environment")

    loaded = ModelStore(folder, device="cuda").get("gpt.pt")

    assert loaded.model.token_embedding.device.type == "cuda"


@needs_torch
def test_generation_and_attention_work_on_the_gpu(folder):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU in this environment")
    gpu = ModelStore(folder, device="cuda")

    first = generate_text(gpu, "gpt.pt", "to", max_tokens=12, seed=3)
    again = generate_text(gpu, "gpt.pt", "to", max_tokens=12, seed=3)
    maps = attention_maps(gpu, "gpt.pt", "to be")

    assert first == again  # a seed still gives the same text
    assert first.text.startswith("to")
    assert len(maps.weights) == 2  # the weights are brought back to plain lists


@needs_torch
def test_asking_for_a_gpu_that_is_not_there_is_a_clear_error(folder, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(InvalidRequestError, match="CUDA requested but not available"):
        ModelStore(folder, device="cuda").get("gpt.pt")
