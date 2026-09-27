"""HTTP adapter: exposes ForgeLM over a FastAPI app.

Keep this file thin. Routes translate HTTP <-> Python calls; real logic lives
in other modules so it can be tested without a web server.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from forgelm import __version__
from forgelm.tokenizer import Analysis, CharTokenizer, analyze

app = FastAPI(title="ForgeLM", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by humans, Docker and load balancers."""
    return {"status": "ok", "version": __version__}


class TokenizeRequest(BaseModel):
    # Pydantic validates the JSON body: a missing or non-string `text`
    # is rejected with 422 before our function ever runs.
    text: str
    corpus: str | None = None  # text to build the vocab from; defaults to `text`


@app.post("/tokenize")
def tokenize(request: TokenizeRequest) -> Analysis:
    corpus = request.corpus if request.corpus is not None else request.text
    return analyze(CharTokenizer.from_corpus(corpus), request.text)
