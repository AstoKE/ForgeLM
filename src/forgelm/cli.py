"""Terminal adapter: the `forgelm` command.

Like api.py, this only parses input and calls into the rest of the package.
"""

import argparse

from forgelm import __version__
from forgelm.tokenizer import Analysis, CharTokenizer, analyze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgelm", description="ForgeLM command line.")
    parser.add_argument("--version", action="version", version=f"forgelm {__version__}")

    subcommands = parser.add_subparsers(dest="command")

    serve = subcommands.add_parser("serve", help="Run the HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Restart on code changes (dev).")

    tokenize = subcommands.add_parser("tokenize", help="Show how text is split into tokens.")
    tokenize.add_argument("text")
    tokenize.add_argument(
        "--corpus",
        help="UTF-8 file to build the vocabulary from (default: the text itself).",
    )

    return parser


def format_analysis(result: Analysis) -> str:
    roundtrip = "OK" if result.roundtrip_ok else f"FAILED, {result.unknown_count} unknown"
    return "\n".join(
        [
            f"tokens:  {result.tokens}",
            f"ids:     {result.ids}",
            f"stats:   {result.num_chars} chars -> {result.num_tokens} tokens"
            f" | vocab {result.vocab_size} | unknown {result.unknown_count}"
            f" | {result.chars_per_token:.2f} chars/token",
            f"decoded: {result.decoded!r}   (round-trip: {roundtrip})",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. `argv` is injectable so tests can call main() directly."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        # Imported lazily so `forgelm --version` stays fast.
        import uvicorn

        uvicorn.run("forgelm.api:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "tokenize":
        if args.corpus is None:
            corpus = args.text
        else:
            try:
                with open(args.corpus, encoding="utf-8") as f:
                    corpus = f.read()
            except OSError as exc:
                parser.error(f"cannot read corpus file: {exc}")  # exits with code 2
        print(format_analysis(analyze(CharTokenizer.from_corpus(corpus), args.text)))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
