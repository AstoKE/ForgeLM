# ADR 0006: The dashboard is one static page served by FastAPI

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

Sprint 4 needs a UI to try the tokenizer, generate text and look at attention. The roadmap
ends with a product UI later, but now the goal is learning, with as few moving parts as
possible. The choices were a JavaScript framework (React/Vue with npm and a build step), a
server-rendered template engine, or plain HTML and JavaScript.

## Decision

1. **One HTML file with vanilla JavaScript** at `src/forgelm/ui/index.html`, served by
   `GET /ui`. No npm, no build step, no CDN, so it works offline and ships inside the package
   (`package-data` in `pyproject.toml`).
2. **The page only calls the JSON API** (`/tokenize`, `/models`, `/generate`, `/attention`).
   Every capability exists as an endpoint first, so any other client can use it too.
3. **Core logic lives in `forgelm/inference/`**, which knows nothing about HTTP. It raises three
   error types, and `api.py` maps them to status codes: `InvalidRequestError` 422,
   `ModelNotFoundError` 404, `MLNotInstalledError` 503.
4. **Models are chosen by file name only**, inside one folder (`checkpoints/`, or
   `FORGELM_CHECKPOINT_DIR`). Names with a path, a leading dot or an unknown extension are
   rejected, so a request cannot read files outside the folder.
5. **A model is loaded once and cached** per file, keyed by its modification time, so
   retraining a checkpoint is picked up without restarting the server.
6. **The page never uses `innerHTML`.** Text typed by a user (and token strings) is inserted
   with `textContent`, so it cannot become markup.
7. The attention heatmap is an HTML table with one cell per pair (a `title` tooltip gives the
   exact weight). The loss chart in Sprint 4b will use a `<canvas>`.

## Consequences

- No frontend tooling to learn or maintain, and the whole UI can be read in one sitting.
- Without a framework, state and DOM updates are written by hand. This is fine for one page
  with three panels; if the UI grows a lot, revisit it (Sprint 11 deployment is the natural point).
- The HTML itself cannot be unit-tested in Python. Tests check that the page is served and wired
  to the right routes; the behaviour was checked by driving headless Chrome by hand.
- The model cache is per process and holds models in memory. Fine for one user; a multi-user
  deployment would need a size limit.
- The page and `/generate` have no authentication. They are meant for `127.0.0.1`. Before any
  public deployment (Sprint 11) this must change.
