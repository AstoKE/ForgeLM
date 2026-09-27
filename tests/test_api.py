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
