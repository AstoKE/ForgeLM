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


def test_tokenize_without_text_is_rejected():
    response = client.post("/tokenize", json={"corpus": "abc"})

    assert response.status_code == 422  # Pydantic validation error
