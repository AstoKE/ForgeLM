# ADR 0001: Project structure and tooling

- **Status:** Accepted
- **Date:** 2026-09-25

## Context

ForgeLM is a long-lived learning project that will grow into a real service. We need a setup
that is standard and production-friendly, while keeping the tooling simple enough that it's
clear what each piece does.

## Decision

1. **`src/` layout** (`src/forgelm/`). Tests have to import the *installed* package, so
   packaging mistakes, like a missing file or a wrong package path, show up immediately instead
   of in production.
2. **One `pyproject.toml`** for metadata, dependencies, the CLI entry point, and pytest and
   ruff config. The build backend is setuptools.
3. **pip + venv** for environments. Both are built into Python, so every step is visible. We can
   move to uv later if speed or lockfiles start to matter.
4. **argparse** for the CLI. It's in the standard library, adds no dependency, and shows how
   argument parsing actually works. We can revisit Typer later.
5. **FastAPI + Uvicorn** for HTTP. It's the standard for Python ML/LLM services, and it gives us
   typed request models and auto-generated OpenAPI docs.
6. **ruff** for both linting and formatting (one tool instead of flake8 + isort + black).
7. **Version lives only in `pyproject.toml`**. Code reads it through `importlib.metadata`.

## Consequences

- You need `pip install -e ".[dev]"` before running tests or the CLI.
- Docker, CI and pre-commit are left out on purpose. They'll be added in small follow-ups.
