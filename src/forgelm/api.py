"""HTTP adapter: exposes ForgeLM over a FastAPI app.

Keep this file thin. Routes translate HTTP <-> Python calls; real logic lives
in other modules so it can be tested without a web server.
"""

from fastapi import FastAPI

from forgelm import __version__

app = FastAPI(title="ForgeLM", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by humans, Docker and load balancers."""
    return {"status": "ok", "version": __version__}
