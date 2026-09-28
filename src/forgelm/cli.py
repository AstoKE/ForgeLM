"""Terminal adapter: the `forgelm` command.

Like api.py, this only parses input and calls into the rest of the package.
"""

import argparse
import math
import random

from forgelm import __version__
from forgelm.models import TrainReport, load_checkpoint, save_checkpoint, train_on_text
from forgelm.tokenizer import Analysis, analyze, build_tokenizer


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
    tokenize.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    tokenize.add_argument(
        "--merges", type=int, default=50, help="Number of BPE merges to learn (bpe only)."
    )

    train = subcommands.add_parser("train-bigram", help="Train a bigram model by counting.")
    train.add_argument("--corpus", required=True, help="UTF-8 training text.")
    train.add_argument("--out", default="checkpoints/bigram.json")
    train.add_argument("--smoothing", type=float, default=1.0, help="Add-k smoothing constant.")
    train.add_argument("--val-fraction", type=float, default=0.1)

    generate = subcommands.add_parser("generate", help="Generate text from a trained model.")
    generate.add_argument("--model", default="checkpoints/bigram.json")
    generate.add_argument("--prompt", default="\n", help="Text to continue (default: newline).")
    generate.add_argument("--max-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=1.0, help="0 = greedy.")
    generate.add_argument("--seed", type=int, help="Fix for reproducible output.")

    return parser


def read_text_file(parser: argparse.ArgumentParser, path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        parser.error(f"cannot read file: {exc}")  # exits with code 2


def format_train_report(r: TrainReport) -> str:
    return "\n".join(
        [
            f"vocab {r.tokenizer.vocab_size} | train {r.train_tokens:,} tokens"
            f" | val {r.val_tokens:,} tokens",
            f"uniform baseline loss {r.baseline_loss:.3f}"
            f" (perplexity {math.exp(r.baseline_loss):.1f})",
            f"train loss {r.train_loss:.3f} | val loss {r.val_loss:.3f}"
            f" (perplexity {math.exp(r.val_loss):.1f})",
        ]
    )


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
        if args.merges < 0:
            parser.error("--merges must be >= 0")
        corpus = args.text if args.corpus is None else read_text_file(parser, args.corpus)
        tokenizer = build_tokenizer(args.tokenizer, corpus, args.merges)
        print(format_analysis(analyze(tokenizer, args.text)))
        return 0

    if args.command == "train-bigram":
        text = read_text_file(parser, args.corpus)
        try:
            report = train_on_text(text, args.smoothing, args.val_fraction)
        except ValueError as exc:
            parser.error(str(exc))
        save_checkpoint(args.out, report.model, report.tokenizer)
        print(format_train_report(report))
        print(f"saved {args.out}")
        return 0

    if args.command == "generate":
        if args.temperature < 0:
            parser.error("--temperature must be >= 0")
        if args.max_tokens < 0:
            parser.error("--max-tokens must be >= 0")
        if not args.prompt:
            parser.error("--prompt must not be empty (a bigram needs a previous token)")
        try:
            model, tokenizer = load_checkpoint(args.model)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"cannot load model: {exc}")
        ids = model.generate(
            tokenizer.encode(args.prompt),
            args.max_tokens,
            random.Random(args.seed),
            args.temperature,
        )
        print(tokenizer.decode(ids))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
