# ADR 0007: Training runs in a background thread, and the page polls for progress

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

Sprint 4b lets the dashboard start a training run and watch the loss curve. Training Tiny
Shakespeare for 2000 steps takes about 3.5 minutes, while an HTTP request has to answer in
milliseconds. Doing the work inside the request handler would leave the browser spinning
until it timed out, and the page could show nothing while it waited.

Three questions had to be answered: where the work runs (in the request, on a thread, in a
separate process, or in a task queue like Celery/RQ), how progress reaches the page (polling,
Server-Sent Events, or a WebSocket), and how `train_minigpt` reports progress at all, given
that it must not know about HTTP.

## Decision

1. **A plain `threading.Thread` inside the API process**, managed by
   `forgelm/training/jobs.py` (`TrainingJobStore`). `start()` validates, spawns the thread and
   returns immediately with a job id; `get(job_id)` answers where the run is.
   Python threads cannot run bytecode in parallel, but torch releases the GIL inside its C++
   kernels, so training really progresses while FastAPI serves other requests. A pure-Python
   training loop would not work this way, and would need a process.
2. **`POST /train` returns 202 Accepted** with the job as its body, and the page polls
   `GET /train/{id}` about once a second. Polling needs no new protocol, is trivial to test
   with `TestClient`, and a loss curve that updates once a second does not need push. SSE or a
   WebSocket would add machinery for no gain here.
3. **Progress is reported through an optional callback**, not by printing or by writing to a
   file: `train_minigpt(..., on_progress=lambda step, train_loss, val_loss: ...)`, called at
   every evaluation. The training module stays unaware of terminals and HTTP; the caller
   decides what to do with the numbers. The CLI passes a callback that prints each line as it
   happens, which also fixed 3.5 minutes of silence in `forgelm train-minigpt`.
4. **One run at a time.** A second `POST /train` while one is going gets 409 Conflict. Two runs
   would compete for the same CPU or GPU and both would crawl. A queue is the obvious next step
   and is not needed yet.
5. **Job state lives in memory**, not on disk. A restart loses the loss history but not the
   model, which the job writes to `checkpoints/` itself.
6. **The corpus and the output are file names, not paths**, resolved inside one folder
   (`data/`, or `FORGELM_CORPUS_DIR`) exactly like models in ADR 0006. `"../pyproject.toml"` is
   rejected. Every hyperparameter is capped, including `steps` (`MAX_STEPS = 5000`), so a
   browser cannot start a ten-hour run.
7. **A failing run is recorded, not raised.** A thread that raises dies silently, so the job
   gets `status = "failed"` and the exception text in `error`, and the page can show it.

## Consequences

- No new dependency, no broker, no worker process to run alongside the server.
- Progress is lost on restart, and a job cannot be cancelled. Both are acceptable for a
  single-user learning tool; cancellation would need a flag the training loop checks.
- The run shares the process with the API, so a training run makes requests a little slower,
  and an out-of-memory error in training takes the server down with it.
- One run at a time means a second user would be refused rather than queued. The API is meant
  for `127.0.0.1` (ADR 0006), so there is no second user yet.
- `jobs.py` imports `InvalidRequestError` and `MLNotInstalledError` from `forgelm.inference`,
  because `api.py` already maps them to 422 and 503. A shared `forgelm/errors.py` is the tidier
  home once a third area needs them.
- Polling costs one request per second per open page. Negligible locally; a public deployment
  (Sprint 11) would want SSE or a longer interval.
