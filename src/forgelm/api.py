"""HTTP adapter: exposes ForgeLM over a FastAPI app.

Keep this file thin. Routes translate HTTP <-> Python calls; real logic lives
in other modules so it can be tested without a web server.
"""

import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from forgelm import __version__
from forgelm.inference import (
    AttentionResult,
    GenerateResult,
    InvalidRequestError,
    MLNotInstalledError,
    ModelInfo,
    ModelNotFoundError,
    ModelStore,
    attention_maps,
    generate_text,
)
from forgelm.tokenizer import Analysis, TokenizerKind, analyze, build_tokenizer
from forgelm.training import MAX_STEPS, JobBusy, JobNotFoundError, TrainingJob, TrainingJobStore

# BPE training is O(num_merges x corpus length). Without limits, one request
# could keep the server busy for minutes.
MAX_TEXT_CHARS = 50_000
MAX_MERGES = 1_000
MAX_NEW_TOKENS = 1_000
UI_FILE = Path(__file__).parent / "ui" / "index.html"

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


# --- Models: list, generate, look inside (Sprint 4a) ---------------------------------------

_store = ModelStore(os.environ.get("FORGELM_CHECKPOINT_DIR", "checkpoints"))


def get_store() -> ModelStore:
    """A dependency, so tests can swap in a store that points at a temporary folder."""
    return _store


Store = Annotated[ModelStore, Depends(get_store)]


# The core raises its own error types; here they become status codes.
@app.exception_handler(InvalidRequestError)
def invalid_request(_: Request, exc: InvalidRequestError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(ModelNotFoundError)
def model_not_found(_: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(MLNotInstalledError)
def ml_not_installed(_: Request, exc: MLNotInstalledError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=503)


class GenerateRequest(BaseModel):
    model: str = Field(max_length=200)
    prompt: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    max_tokens: int = Field(default=200, ge=0, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=1.0, ge=0, le=5)  # 0 = greedy
    seed: int | None = None


class AttentionRequest(BaseModel):
    model: str = Field(max_length=200)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.get("/models")
def list_models(store: Store) -> list[ModelInfo]:
    return store.list_models()


@app.post("/generate")
def generate(request: GenerateRequest, store: Store) -> GenerateResult:
    return generate_text(
        store, request.model, request.prompt, request.max_tokens, request.temperature, request.seed
    )


@app.post("/attention")
def attention(request: AttentionRequest, store: Store) -> AttentionResult:
    return attention_maps(store, request.model, request.text)


# --- Training in the background (Sprint 4b) -----------------------------------------------

_jobs = TrainingJobStore(
    os.environ.get("FORGELM_CORPUS_DIR", "data"),
    os.environ.get("FORGELM_CHECKPOINT_DIR", "checkpoints"),
)


def get_jobs() -> TrainingJobStore:
    """A dependency, so tests can swap in a store over temporary folders."""
    return _jobs


Jobs = Annotated[TrainingJobStore, Depends(get_jobs)]


@app.exception_handler(JobNotFoundError)
def job_not_found(_: Request, exc: JobNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(JobBusy)
def job_busy(_: Request, exc: JobBusy) -> JSONResponse:
    # 409 Conflict: the request is fine, the server is just not free to do it now.
    return JSONResponse({"detail": str(exc)}, status_code=409)


class TrainRequest(BaseModel):
    """Hyperparameters, all capped: this endpoint is reachable from a browser."""

    corpus: str = Field(max_length=200)  # a file name in the corpus folder, never a path
    out: str = Field(default="minigpt.pt", max_length=200)
    steps: int = Field(default=500, ge=1, le=MAX_STEPS)
    lr: float = Field(default=3e-3, gt=0, le=1)
    batch_size: int = Field(default=16, ge=1, le=64)
    block_size: int = Field(default=64, ge=1, le=256)
    embed_dim: int = Field(default=64, ge=1, le=256)
    num_heads: int = Field(default=4, ge=1, le=16)
    num_blocks: int = Field(default=4, ge=1, le=8)
    eval_every: int = Field(default=50, ge=1, le=MAX_STEPS)
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"


@app.get("/corpora")
def list_corpora(jobs: Jobs) -> list[str]:
    """The texts available to train on, so the page can offer a list."""
    return jobs.list_corpora()


@app.post("/train", status_code=202)
def train(request: TrainRequest, jobs: Jobs) -> TrainingJob:
    """Start a run and return its ticket. 202 Accepted: started, not finished."""
    return jobs.start(
        request.corpus,
        out=request.out,
        steps=request.steps,
        lr=request.lr,
        batch_size=request.batch_size,
        block_size=request.block_size,
        embed_dim=request.embed_dim,
        num_heads=request.num_heads,
        num_blocks=request.num_blocks,
        eval_every=request.eval_every,
        device=request.device,
        seed=request.seed,
    )


@app.get("/train")
def latest_training(jobs: Jobs) -> TrainingJob:
    """The most recent run, so a reloaded page can pick up one that is still going."""
    job = jobs.latest()
    if job is None:
        raise JobNotFoundError("no training run has been started yet")
    return job


@app.get("/train/{job_id}")
def train_status(job_id: str, jobs: Jobs) -> TrainingJob:
    """Where is the run? The page polls this and redraws the loss curve."""
    return jobs.get(job_id)


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    """The dashboard: one static HTML page that calls the endpoints above."""
    return UI_FILE.read_text(encoding="utf-8")
