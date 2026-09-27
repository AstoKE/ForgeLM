"""Terminal adapter: the `forgelm` command.

Like api.py, this only parses input and calls into the rest of the package.
"""

import argparse

from forgelm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgelm", description="ForgeLM command line.")
    parser.add_argument("--version", action="version", version=f"forgelm {__version__}")

    subcommands = parser.add_subparsers(dest="command")

    serve = subcommands.add_parser("serve", help="Run the HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Restart on code changes (dev).")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. `argv` is injectable so tests can call main() directly."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        # Imported lazily so `forgelm --version` stays fast.
        import uvicorn

        uvicorn.run("forgelm.api:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
