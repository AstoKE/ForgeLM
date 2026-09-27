"""HTTP adapter: exposes ForgeLM over a FastAPI app.

Keep this file thin. Routes translate HTTP <-> Python calls; real logic lives
in other modules so it can be tested without a web server.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from forgelm import __version__
from forgelm.tokenizer import Analysis, TokenizerKind, analyze, build_tokenizer

# BPE training is O(num_merges x corpus length). Without limits, one request
# could keep the server busy for minutes.
MAX_TEXT_CHARS = 50_000
MAX_MERGES = 1_000

app = FastAPI(title="ForgeLM", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by humans, Docker and load balancers."""
    return {"status": "ok", "version": __version__}


class TokenizeRequest(BaseModel):
    # Pydantic validates the JSON body: a missing or non-string `text`
    # is rejected with 422 before our function ever runs.
    text: str = Field(max_length=MAX_TEXT_CHARS)
    # Text to build the vocab / train merges on; defaults to `text`.
    corpus: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    tokenizer: TokenizerKind = "char"
    num_merges: int = Field(default=50, ge=0, le=MAX_MERGES)  # BPE only


@app.post("/tokenize")
def tokenize(request: TokenizeRequest) -> Analysis:
    corpus = request.corpus if request.corpus is not None else request.text
    tokenizer = build_tokenizer(request.tokenizer, corpus, request.num_merges)
    return analyze(tokenizer, request.text)
