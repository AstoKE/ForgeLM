"""Running a training run in the background, so an HTTP request can start it and leave.

A request has to answer in milliseconds; training takes minutes. So `start()` kicks off
a thread and immediately returns a job id, like a ticket at a counter, and `get()`
answers "where are you?". The caller (the dashboard) polls that until the job is done.

Why a plain thread is enough: Python threads cannot run bytecode in parallel because of
the GIL, but torch releases the GIL inside its C++ kernels, so training really does make
progress while FastAPI serves other requests. A pure-Python training loop would not.

Nothing here knows about HTTP. The error types map to status codes in `api.py`:

    InvalidRequestError  bad name, bad hyperparameters      (HTTP 422)
    JobNotFoundError     no job with that id                (HTTP 404)
    JobBusy              a run is already going             (HTTP 409)
    MLNotInstalledError  training needs torch               (HTTP 503)

Only one run at a time, on purpose: two runs would compete for the same CPU or GPU and
both would crawl. A queue would be the next step, and is not needed yet.
"""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# These two already mean "422" and "503" to the API. A shared `forgelm/errors.py` would be
# the tidier home for them once a third area needs them; importing is cheaper than moving.
from forgelm.inference import InvalidRequestError, MLNotInstalledError

CORPUS_SUFFIXES = {".txt"}
# A browser must not be able to start a ten-hour run.
MAX_STEPS = 5_000


class JobNotFoundError(Exception):
    """There is no training job with that id."""


class JobBusy(Exception):
    """A training run is already in progress."""


@dataclass(frozen=True)
class Progress:
    step: int
    train_loss: float
    val_loss: float


@dataclass
class TrainingJob:
    """One training run. `progress` grows while the thread works."""

    id: str
    corpus: str  # file name inside the corpus folder
    out: str  # checkpoint file name to write
    steps: int  # total requested, so a caller can show step / steps
    status: str = "running"  # "running" | "done" | "failed"
    progress: list[Progress] = field(default_factory=list)
    error: str | None = None  # set when status is "failed"
    seconds: float | None = None  # set when status is "done"
    val_loss: float | None = None  # final val loss, when done


class TrainingJobStore:
    """Starts training runs and keeps their progress in memory.

    In-memory on purpose: a restart loses the progress history but not the model, which
    the job itself writes to disk.
    """

    def __init__(self, corpus_dir: str | Path, checkpoint_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self._jobs: dict[str, TrainingJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        # One lock for the dict *and* for every job's fields: the training thread writes
        # them while request threads read them.
        self._lock = threading.Lock()

    def list_corpora(self) -> list[str]:
        if not self.corpus_dir.is_dir():
            return []
        return sorted(
            path.name
            for path in self.corpus_dir.iterdir()
            if path.is_file() and path.suffix in CORPUS_SUFFIXES and not path.name.startswith(".")
        )

    def corpus_path(self, name: str) -> Path:
        """Resolve a corpus *file name*. A path like "../pyproject.toml" is refused.

        The same rule as `ModelStore`: a request names a file, never a location.
        """
        if not name or name != Path(name).name or name.startswith("."):
            raise InvalidRequestError(f"invalid corpus name: {name!r} (a file name, no folders)")
        if Path(name).suffix not in CORPUS_SUFFIXES:
            raise InvalidRequestError(f"unknown corpus type: {name!r} (use .txt)")
        path = self.corpus_dir / name
        if not path.is_file():
            raise InvalidRequestError(f"no corpus named {name!r}")
        return path

    def checkpoint_path(self, name: str) -> Path:
        """Same rule for where the checkpoint is written: a file name, not a path."""
        if not name or name != Path(name).name or name.startswith("."):
            raise InvalidRequestError(f"invalid output name: {name!r} (a file name, no folders)")
        if Path(name).suffix != ".pt":
            raise InvalidRequestError(f"output must be a .pt file, got {name!r}")
        return self.checkpoint_dir / name

    def get(self, job_id: str) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"no training job with id {job_id!r}")
        return job

    def latest(self) -> TrainingJob | None:
        """The most recently started job, so a reloaded page can find a running job."""
        with self._lock:
            return next(reversed(list(self._jobs.values())), None)

    def start(self, corpus: str, out: str = "minigpt.pt", **hyper) -> TrainingJob:
        """Validate everything, then run `train_minigpt_on_text` on a background thread."""
        try:
            from forgelm.models.train_minigpt import train_minigpt_on_text
        except ImportError as exc:
            raise MLNotInstalledError(
                'PyTorch is not installed; see README ("pip install -e .[ml]")'
            ) from exc

        steps = int(hyper.get("steps", 1000))
        if not 1 <= steps <= MAX_STEPS:
            raise InvalidRequestError(f"steps must be between 1 and {MAX_STEPS}, got {steps}")
        if hyper.get("embed_dim", 64) % hyper.get("num_heads", 4) != 0:
            raise InvalidRequestError("embed_dim must be divisible by num_heads")
        # Validate the names before starting the thread, so a bad request fails loudly
        # at the request instead of inside a job nobody is watching yet.
        corpus_file = self.corpus_path(corpus)
        out_file = self.checkpoint_path(out)

        with self._lock:
            if any(job.status == "running" for job in self._jobs.values()):
                raise JobBusy("a training run is already in progress")
            job = TrainingJob(id=uuid.uuid4().hex[:12], corpus=corpus, out=out, steps=steps)
            self._jobs[job.id] = job

        def record(step: int, train_loss: float, val_loss: float) -> None:
            with self._lock:
                job.progress.append(Progress(step, train_loss, val_loss))

        def run() -> None:
            try:
                from forgelm.models.minigpt import save_minigpt

                text = corpus_file.read_text(encoding="utf-8")
                model, tokenizer, history = train_minigpt_on_text(text, on_progress=record, **hyper)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                save_minigpt(out_file, model, tokenizer)
            except Exception as exc:
                # A thread that raises dies silently, so the failure is recorded instead.
                with self._lock:
                    job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                return
            with self._lock:
                job.status = "done"
                job.seconds = history.seconds
                # The checkpoint on disk holds the best-val weights, so report that loss.
                job.val_loss = (
                    history.best_val_loss
                    if history.best_val_loss is not None
                    else history.val_loss[-1]
                )

        thread = threading.Thread(target=run, name=f"train-{job.id}", daemon=True)
        with self._lock:
            self._threads[job.id] = thread
        thread.start()
        return job

    def wait(self, job_id: str, timeout: float = 60.0) -> TrainingJob:
        """Block until the job finishes. For tests and for any synchronous caller."""
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is None:
            raise JobNotFoundError(f"no training job with id {job_id!r}")
        thread.join(timeout)
        return self.get(job_id)
