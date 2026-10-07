import pytest
from fastapi.testclient import TestClient

from forgelm import __version__
from forgelm.api import app

client = TestClient(app)


def test_health_returns_ok_and_version():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_unknown_route_returns_404():
    assert client.get("/does-not-exist").status_code == 404


def test_tokenize_builds_vocab_from_text_by_default():
    response = client.post("/tokenize", json={"text": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["tokens"] == ["h", "i"]
    assert body["ids"] == [1, 2]
    assert body["roundtrip_ok"] is True


def test_tokenize_with_corpus_reports_unknowns():
    response = client.post("/tokenize", json={"text": "hi", "corpus": "h"})

    body = response.json()
    assert body["tokens"] == ["h", "<unk>"]
    assert body["unknown_count"] == 1
    assert body["roundtrip_ok"] is False


def test_tokenize_with_bpe_has_no_unknowns():
    response = client.post(
        "/tokenize",
        json={"text": "İstanbul", "corpus": "english", "tokenizer": "bpe", "num_merges": 5},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["unknown_count"] == 0
    assert body["roundtrip_ok"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "hi", "tokenizer": "word"},  # unknown tokenizer kind
        {"text": "hi", "tokenizer": "bpe", "num_merges": -1},
        {"text": "hi", "tokenizer": "bpe", "num_merges": 1_000_000},  # too expensive
        {"text": "x" * 50_001},  # text too long
    ],
)
def test_tokenize_rejects_invalid_or_too_expensive_requests(payload):
    assert client.post("/tokenize", json=payload).status_code == 422


def test_tokenize_without_text_is_rejected():
    response = client.post("/tokenize", json={"corpus": "abc"})

    assert response.status_code == 422  # Pydantic validation error


# --- models, generate, attention, ui (Sprint 4a) ---------------------------------------------


@pytest.fixture
def models_client(tmp_path):
    """A client whose model folder is a temporary one holding a tiny bigram (+ a MiniGPT)."""
    from forgelm.api import get_store
    from forgelm.inference import ModelStore
    from forgelm.models import save_checkpoint, train_on_text

    report = train_on_text("to be or not to be, that is the question.\n" * 20)
    save_checkpoint(tmp_path / "bigram.json", report.model, report.tokenizer)
    try:
        import torch  # noqa: F401

        from forgelm.models.minigpt import MiniGPT, save_minigpt

        gpt = MiniGPT(report.tokenizer.vocab_size, 16, 2, 1, block_size=8)
        save_minigpt(tmp_path / "gpt.pt", gpt, report.tokenizer)
    except ImportError:
        pass
    app.dependency_overrides[get_store] = lambda: ModelStore(tmp_path)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_models_lists_the_checkpoints(models_client):
    response = models_client.get("/models")

    assert response.status_code == 200
    assert {"name": "bigram.json", "kind": "bigram"}.items() <= response.json()[0].items()


def test_generate_returns_text_that_starts_with_the_prompt(models_client):
    response = models_client.post(
        "/generate", json={"model": "bigram.json", "prompt": "to", "max_tokens": 10, "seed": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"].startswith("to")
    assert body["num_new_tokens"] == 10


def test_generate_unknown_model_is_404(models_client):
    response = models_client.post("/generate", json={"model": "nope.json", "prompt": "to"})

    assert response.status_code == 404


@pytest.mark.parametrize("model", ["../x.pt", "/etc/passwd", "sub/m.json"])
def test_generate_refuses_paths_instead_of_names(models_client, model):
    response = models_client.post("/generate", json={"model": model, "prompt": "to"})

    assert response.status_code == 422
    assert "invalid model name" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"model": "bigram.json", "prompt": ""},  # empty prompt
        {"model": "bigram.json", "prompt": "to", "max_tokens": 5000},  # over the limit
        {"model": "bigram.json", "prompt": "to", "temperature": -1},
        {"prompt": "to"},  # model missing
    ],
)
def test_generate_validates_the_request(models_client, body):
    assert models_client.post("/generate", json=body).status_code == 422


def test_attention_on_a_bigram_is_422(models_client):
    response = models_client.post("/attention", json={"model": "bigram.json", "text": "to be"})

    assert response.status_code == 422
    assert "needs a MiniGPT" in response.json()["detail"]


def test_attention_returns_tokens_and_weights(models_client):
    pytest.importorskip("torch")

    response = models_client.post("/attention", json={"model": "gpt.pt", "text": "to be"})

    assert response.status_code == 200
    body = response.json()
    assert body["tokens"] == ["t", "o", " ", "b", "e"]
    assert len(body["weights"][0][0]) == 5  # block 0, head 0: five rows


def test_ui_page_is_served_and_wired_to_the_api():
    response = client.get("/ui")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    for element in ("tokenizer-panel", "generate-panel", "attention-panel", "att-table"):
        assert f'id="{element}"' in page
    for route in ("/tokenize", "/generate", "/attention", "/models"):
        assert route in page
