import sys

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
    for element in (
        "tokenizer-panel",
        "train-panel",
        "generate-panel",
        "attention-panel",
        "att-table",
        "tr-canvas",
        "tr-corpus",
    ):
        assert f'id="{element}"' in page
    for route in ("/tokenize", "/generate", "/attention", "/models", "/corpora", "/train"):
        assert route in page
    # Typed text must never be able to become markup, so the page never assigns innerHTML.
    assert ".innerHTML" not in page


# --- training in the background (Sprint 4b) --------------------------------------------------

TRAIN_BODY = {
    "corpus": "tiny.txt",
    "out": "gpt.pt",
    "steps": 4,
    "eval_every": 2,
    "batch_size": 2,
    "block_size": 8,
    "embed_dim": 16,
    "num_heads": 2,
    "num_blocks": 1,
    "device": "cpu",
}


@pytest.fixture
def train_client(tmp_path):
    """A client whose corpus and checkpoint folders are temporary ones."""
    from forgelm.api import get_jobs
    from forgelm.training import TrainingJobStore

    corpora = tmp_path / "data"
    corpora.mkdir()
    (corpora / "tiny.txt").write_text("to be or not to be.\n" * 40, encoding="utf-8")
    (corpora / "short.txt").write_text("ab\n", encoding="utf-8")  # too small to train on
    store = TrainingJobStore(corpora, tmp_path / "checkpoints")
    app.dependency_overrides[get_jobs] = lambda: store
    yield TestClient(app), store
    app.dependency_overrides.clear()


def test_corpora_lists_the_texts_available_for_training(train_client):
    client, _ = train_client

    response = client.get("/corpora")

    assert response.status_code == 200
    assert response.json() == ["short.txt", "tiny.txt"]


def test_train_returns_202_with_a_ticket_and_then_finishes(train_client):
    pytest.importorskip("torch")
    client, store = train_client

    started = client.post("/train", json=TRAIN_BODY)

    # 202 Accepted: the run has started, not finished.
    assert started.status_code == 202
    job = started.json()
    assert job["status"] == "running"
    assert job["steps"] == 4

    store.wait(job["id"], timeout=120)
    finished = client.get(f"/train/{job['id']}").json()

    assert finished["status"] == "done", finished["error"]
    assert [p["step"] for p in finished["progress"]] == [0, 2, 4]
    assert finished["val_loss"] == pytest.approx(finished["progress"][-1]["val_loss"])
    assert (store.checkpoint_dir / "gpt.pt").is_file()


def test_the_new_checkpoint_is_then_servable(train_client, tmp_path):
    """The point of 4b: train from the page, then generate from what you trained."""
    pytest.importorskip("torch")
    from forgelm.api import get_store
    from forgelm.inference import ModelStore

    client, store = train_client
    store.wait(client.post("/train", json=TRAIN_BODY).json()["id"], timeout=120)
    app.dependency_overrides[get_store] = lambda: ModelStore(store.checkpoint_dir)

    response = client.post("/generate", json={"model": "gpt.pt", "prompt": "to", "seed": 1})

    assert response.status_code == 200
    assert response.json()["text"].startswith("to")


def test_a_second_run_gets_409_while_one_is_going(train_client):
    pytest.importorskip("torch")
    client, store = train_client

    first = client.post("/train", json={**TRAIN_BODY, "steps": 200}).json()
    try:
        second = client.post("/train", json=TRAIN_BODY)
        assert second.status_code == 409
        assert "already in progress" in second.json()["detail"]
    finally:
        store.wait(first["id"], timeout=120)


def test_a_failed_run_is_reported_through_the_status_endpoint(train_client):
    pytest.importorskip("torch")
    client, store = train_client

    # A corpus of three characters cannot be split into train and val. The *request* is
    # valid, so this can only fail once the thread is already running.
    started = client.post("/train", json={**TRAIN_BODY, "corpus": "short.txt"})
    assert started.status_code == 202

    job_id = started.json()["id"]
    store.wait(job_id, timeout=120)
    status = client.get(f"/train/{job_id}").json()

    # The thread died, but the caller still gets a clean answer instead of a hang.
    assert status["status"] == "failed"
    assert "corpus too short" in status["error"]
    assert not (store.checkpoint_dir / "gpt.pt").exists()


def test_an_unknown_job_id_is_404(train_client):
    client, _ = train_client

    assert client.get("/train/nosuchjob").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"corpus": "../pyproject.toml"},  # path traversal on the corpus
        {"corpus": "sub/tiny.txt"},
        {"corpus": "missing.txt"},
        {"out": "../evil.pt"},  # ...and on the output
        {"out": "gpt.json"},
        {"steps": 999_999},  # above MAX_STEPS
        {"steps": 0},
        {"embed_dim": 10, "num_heads": 4},  # not divisible
        {"lr": 0},
        {"device": "tpu"},
    ],
)
def test_a_bad_training_request_is_422_and_starts_nothing(train_client, body):
    pytest.importorskip("torch")
    client, store = train_client

    response = client.post("/train", json={**TRAIN_BODY, **body})

    assert response.status_code == 422
    assert store.latest() is None


def test_train_without_torch_returns_503(train_client, monkeypatch):
    client, _ = train_client
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "forgelm.models.train_minigpt", raising=False)

    response = client.post("/train", json=TRAIN_BODY)

    assert response.status_code == 503
    assert "PyTorch is not installed" in response.json()["detail"]


def test_latest_training_is_404_before_anything_has_run(train_client):
    client, _ = train_client

    assert client.get("/train").status_code == 404


def test_latest_training_lets_a_reloaded_page_find_the_run(train_client):
    pytest.importorskip("torch")
    client, store = train_client

    started = client.post("/train", json=TRAIN_BODY).json()
    store.wait(started["id"], timeout=120)

    latest = client.get("/train")

    assert latest.status_code == 200
    assert latest.json()["id"] == started["id"]
    assert latest.json()["status"] == "done"
