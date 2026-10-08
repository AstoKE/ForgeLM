"""Terminal adapter: the `forgelm` command.

Like api.py, this only parses input and calls into the rest of the package.
"""

import argparse
import math
import os
import random
from pathlib import Path

from forgelm import __version__
from forgelm.data import (
    FILE_MARKER,
    build_code_corpus,
    build_math_dataset,
    code_documents,
    collect_python,
    default_roots,
    math_documents,
    mix_documents,
    parse_problems,
    split_documents,
    story_documents,
)
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

    neural = subcommands.add_parser(
        "train-neural-bigram", help="Train a bigram with gradient descent (needs torch)."
    )
    neural.add_argument("--corpus", required=True, help="UTF-8 training text.")
    neural.add_argument("--steps", type=int, default=300)
    neural.add_argument("--lr", type=float, default=50.0, help="Learning rate.")
    neural.add_argument("--batch-size", type=int, default=32_768)
    neural.add_argument("--eval-every", type=int, default=50)
    neural.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    neural.add_argument("--seed", type=int, default=0)
    neural.add_argument("--val-fraction", type=float, default=0.1)

    gpt = subcommands.add_parser("train-minigpt", help="Train MiniGPT (needs torch).")
    gpt.add_argument("--corpus", required=True, help="UTF-8 training text.")
    gpt.add_argument("--out", default="checkpoints/minigpt.pt")
    gpt.add_argument(
        "--resume",
        help="Continue training this checkpoint instead of starting from random weights. "
        "Its size is used and --embed-dim/--heads/--blocks/--block-size are ignored. Give a "
        "different --seed than the first run, or it sees the same windows again; Adam starts "
        "afresh, so add a short --warmup-steps.",
    )
    gpt.add_argument("--steps", type=int, default=1000)
    gpt.add_argument("--lr", type=float, default=3e-3, help="Learning rate (Adam).")
    gpt.add_argument(
        "--warmup-steps", type=int, default=0, help="Ramp the learning rate up over N steps."
    )
    gpt.add_argument(
        "--min-lr-fraction",
        type=float,
        default=1.0,
        help="Cosine-decay the learning rate down to this share of --lr (1.0 = constant).",
    )
    gpt.add_argument("--batch-size", type=int, default=16, help="Windows per step.")
    gpt.add_argument("--block-size", type=int, default=64, help="Context window in tokens.")
    gpt.add_argument("--embed-dim", type=int, default=64)
    gpt.add_argument("--heads", type=int, default=4)
    gpt.add_argument("--blocks", type=int, default=4)
    gpt.add_argument("--eval-every", type=int, default=100)
    gpt.add_argument("--eval-batches", type=int, default=4)
    gpt.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    gpt.add_argument("--seed", type=int, default=0)
    gpt.add_argument("--val-fraction", type=float, default=0.1)

    generate = subcommands.add_parser("generate", help="Generate text from a trained model.")
    generate.add_argument(
        "--model",
        default="checkpoints/bigram.json",
        help="A bigram checkpoint (.json) or a MiniGPT checkpoint (.pt).",
    )
    generate.add_argument("--prompt", default="\n", help="Text to continue (default: newline).")
    generate.add_argument("--max-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=1.0, help="0 = greedy.")
    generate.add_argument("--seed", type=int, help="Fix for reproducible output.")
    generate.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Where to run a MiniGPT. A big model is far faster on a GPU.",
    )

    make_math = subcommands.add_parser(
        "make-math-data", help="Write addition problems to train on, plus a hidden exam."
    )
    make_math.add_argument("--out", default="data/math.txt", help="The training text.")
    make_math.add_argument("--digits", type=int, default=2, help="Digits per number (1-3).")
    make_math.add_argument(
        "--holdout", type=float, default=0.1, help="Share of problems kept out of training."
    )
    make_math.add_argument("--lines", type=int, default=400_000, help="Training lines to write.")
    make_math.add_argument(
        "--reverse", action="store_true", help="Write answer digits backwards (49+97=641)."
    )
    make_math.add_argument(
        "--pad", action="store_true", help="Zero-pad every number to the same width (05+12=017)."
    )
    make_math.add_argument("--seed", type=int, default=0)

    eval_math = subcommands.add_parser(
        "eval-math", help="Grade a MiniGPT on addition: exact-match accuracy (needs torch)."
    )
    eval_math.add_argument("--model", required=True, help="A MiniGPT checkpoint (.pt).")
    eval_math.add_argument(
        "--holdout", default="data/math-holdout.txt", help="Problems the model never saw."
    )
    eval_math.add_argument(
        "--seen", default="data/math-seen.txt", help="Problems it trained on (the control)."
    )
    eval_math.add_argument("--limit", type=int, help="Grade only the first N problems of each.")
    eval_math.add_argument(
        "--prefix",
        default="",
        help=r"Text placed before every problem, e.g. '<|math|>\n' for a model trained on "
        "tagged documents. Backslash escapes such as \\n are understood.",
    )
    eval_math.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Where to run a MiniGPT. A big model is far faster on a GPU.",
    )

    collect = subcommands.add_parser(
        "collect-code", help="Gather Python source from this machine into one training text."
    )
    collect.add_argument("--out", default="data/python.txt")
    collect.add_argument(
        "--root",
        action="append",
        help="A folder to read .py files from (repeatable). Default: the standard library "
        "plus the installed packages in forgelm.data.DEFAULT_PACKAGES.",
    )
    collect.add_argument("--max-mb", type=float, default=60.0, help="Stop at about this size.")
    collect.add_argument("--seed", type=int, default=0, help="Seeds the file shuffle.")

    mix = subcommands.add_parser(
        "make-mix",
        help="Blend stories, Python and arithmetic into one tagged training text.",
    )
    mix.add_argument("--stories", help="A TinyStories text file.")
    mix.add_argument("--code", help="The text written by collect-code.")
    mix.add_argument("--math", help="The training text written by make-math-data.")
    mix.add_argument("--stories-mb", type=float, default=20.0)
    mix.add_argument("--code-mb", type=float, default=40.0)
    mix.add_argument("--math-mb", type=float, default=4.0)
    mix.add_argument("--out", default="data/mix.txt")
    mix.add_argument("--seed", type=int, default=0)

    eval_code = subcommands.add_parser(
        "eval-code",
        help="Grade a MiniGPT on writing Python: does it parse? (needs torch)",
    )
    eval_code.add_argument("--model", required=True, help="A MiniGPT checkpoint (.pt).")
    eval_code.add_argument(
        "--corpus", default="data/python.txt", help="The text from collect-code."
    )
    eval_code.add_argument("--samples", type=int, default=40, help="Files to generate.")
    eval_code.add_argument("--length", type=int, default=400, help="Characters per sample.")
    eval_code.add_argument("--temperature", type=float, default=0.7)
    eval_code.add_argument("--seed", type=int, default=0)
    eval_code.add_argument(
        "--val-fraction", type=float, default=0.1, help="Must match the training run."
    )
    eval_code.add_argument(
        "--show", type=int, default=1, help="How many generated samples to print."
    )
    eval_code.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Where to run a MiniGPT. A big model is far faster on a GPU.",
    )

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


def format_math_report(label: str, report) -> str:
    """The overall score, then the same score per kind of problem, then a few wrong answers."""
    lines = [f"{label}: {report.correct}/{report.total} = {100 * report.accuracy:.1f}%"]
    for name, group in sorted(report.groups.items()):
        lines.append(
            f"    {name:<36} {group.correct:>5}/{group.total:<5} = {100 * group.accuracy:5.1f}%"
        )
    lines += [
        f"    wrong: {miss.prompt}  expected {miss.expected!r}  got {miss.got!r}"
        for miss in report.misses[:5]
    ]
    return "\n".join(lines)


def format_code_report(report, length: int, samples: int, references: int) -> str:
    return "\n".join(
        [
            f"generated {report.generated:.3f}   ({samples} samples of up to {length} characters)",
            f"ceiling   {report.ceiling:.3f}   (real held-out code, cut alike: {references})",
            f"floor     {report.floor:.3f}   (the same real code with its characters shuffled)",
            f"position  {report.position:.2f}    (0 = no better than shuffled characters, "
            "1 = as parseable as real code)",
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

    if args.command == "train-neural-bigram":
        try:
            # Lazy import: torch is an optional extra and takes seconds to load.
            from forgelm.models.neural_bigram import device_name, train_neural_on_text
        except ImportError:
            parser.error('PyTorch is not installed; see README ("pip install -e .[ml]")')
        text = read_text_file(parser, args.corpus)
        try:
            model, history = train_neural_on_text(
                text,
                val_fraction=args.val_fraction,
                steps=args.steps,
                lr=args.lr,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                device=args.device,
                seed=args.seed,
            )
            counting = train_on_text(text, smoothing=1.0, val_fraction=args.val_fraction)
        except (ValueError, FloatingPointError) as exc:
            parser.error(str(exc))
        print(
            f"device {device_name(history.device)} | vocab {model.vocab_size}"
            f" | params {model.W.numel():,}"
        )
        for step, train_loss, val_loss in zip(
            history.steps, history.train_loss, history.val_loss, strict=True
        ):
            print(f"step {step:>5} | train {train_loss:.3f} | val {val_loss:.3f}")
        print(f"trained in {history.seconds:.1f}s")
        print(
            f"counting bigram (2a): train {counting.train_loss:.3f}"
            f" | val {counting.val_loss:.3f} | baseline {counting.baseline_loss:.3f}"
        )
        return 0

    if args.command == "train-minigpt":
        try:
            from forgelm.models.minigpt import save_minigpt
            from forgelm.models.neural_bigram import device_name
            from forgelm.models.train_minigpt import train_minigpt_on_text
        except ImportError:
            parser.error('PyTorch is not installed; see README ("pip install -e .[ml]")')
        if args.embed_dim % args.heads != 0:
            parser.error("--embed-dim must be divisible by --heads")
        text = read_text_file(parser, args.corpus)

        out_path = Path(args.out)

        def keep(model, tokenizer, step: int) -> None:
            # The best model so far goes to disk as training runs. Written beside the real
            # file and swapped in, so a crash mid-write cannot leave a half-written checkpoint.
            partial = out_path.with_name(out_path.name + ".partial")
            save_minigpt(partial, model, tokenizer)
            os.replace(partial, out_path)

        def show(step: int, train_loss: float, val_loss: float) -> None:
            # Printed while training runs; 2000 steps are minutes of silence otherwise.
            print(f"step {step:>5} | train {train_loss:.3f} | val {val_loss:.3f}", flush=True)

        resume = {}
        if args.resume:
            try:
                from forgelm.models.minigpt import load_minigpt
                from forgelm.models.neural_bigram import pick_device

                start, start_tokenizer = load_minigpt(args.resume, device=pick_device(args.device))
            except (OSError, ValueError, KeyError) as exc:
                parser.error(f"cannot resume from {args.resume}: {exc}")
            resume = {"model": start, "resume_tokenizer": start_tokenizer}
            print(
                f"resuming from {args.resume}: {start.num_parameters():,} parameters, "
                f"context {start.block_size} (the size flags are ignored)",
                flush=True,
            )

        try:
            model, tokenizer, history = train_minigpt_on_text(
                text,
                **resume,
                on_progress=show,
                on_best=keep,
                val_fraction=args.val_fraction,
                steps=args.steps,
                lr=args.lr,
                batch_size=args.batch_size,
                block_size=args.block_size,
                embed_dim=args.embed_dim,
                num_heads=args.heads,
                num_blocks=args.blocks,
                eval_every=args.eval_every,
                eval_batches=args.eval_batches,
                device=args.device,
                seed=args.seed,
                warmup_steps=args.warmup_steps,
                min_lr_fraction=args.min_lr_fraction,
            )
        except (ValueError, FloatingPointError) as exc:
            parser.error(str(exc))
        # The step lines were printed live by `show`; what is left is the summary.
        print(
            f"device {device_name(history.device)} | vocab {model.vocab_size}"
            f" | params {model.num_parameters():,} | context {model.block_size} tokens"
        )
        # The saved model is the best-val one, so that is the number to report first.
        kept = history.best_val_loss if history.best_val_loss is not None else history.val_loss[-1]
        print(
            f"trained in {history.seconds:.1f}s | kept val loss {kept:.3f}"
            f" (perplexity {math.exp(kept):.1f}) from step {history.best_step}"
            f" | last {history.val_loss[-1]:.3f}"
            f" | uniform baseline loss {math.log(model.vocab_size):.3f}"
        )
        save_minigpt(args.out, model, tokenizer)
        print(f"saved {args.out}")
        return 0

    if args.command == "generate":
        if args.temperature < 0:
            parser.error("--temperature must be >= 0")
        if args.max_tokens < 0:
            parser.error("--max-tokens must be >= 0")
        if not args.prompt:
            parser.error("--prompt must not be empty (a bigram needs a previous token)")
        if args.model.endswith(".pt"):
            try:
                from forgelm.models.minigpt import load_minigpt
                from forgelm.models.neural_bigram import pick_device

                model, tokenizer = load_minigpt(args.model, device=pick_device(args.device))
            except ImportError:
                parser.error('PyTorch is not installed; see README ("pip install -e .[ml]")')
            except (OSError, ValueError, KeyError) as exc:
                parser.error(f"cannot load model: {exc}")
            import torch

            generator = None if args.seed is None else torch.Generator().manual_seed(args.seed)
            ids = model.generate(
                tokenizer.encode(args.prompt), args.max_tokens, generator, args.temperature
            )
        else:
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

    if args.command == "make-math-data":
        try:
            data = build_math_dataset(
                args.digits, args.holdout, args.lines, args.reverse, args.seed, pad=args.pad
            )
        except ValueError as exc:
            parser.error(str(exc))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        files = {
            out: data.corpus,
            out.with_name(f"{out.stem}-holdout{out.suffix}"): data.holdout,
            out.with_name(f"{out.stem}-seen{out.suffix}"): data.seen,
        }
        for path, text in files.items():
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}  ({len(text.splitlines()):,} lines)")
        print(
            f"{data.train_problems:,} problems can appear in training; "
            f"{data.holdout_problems:,} are hidden and only used for the exam"
        )
        return 0

    if args.command == "eval-math":
        try:
            from forgelm.eval import greedy_completer, math_accuracy
            from forgelm.models.minigpt import load_minigpt
            from forgelm.models.neural_bigram import pick_device

            model, tokenizer = load_minigpt(args.model, device=pick_device(args.device))
        except ImportError:
            parser.error('PyTorch is not installed; see README ("pip install -e .[ml]")')
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"cannot load model: {exc}")
        prefix = args.prefix.encode("utf-8").decode("unicode_escape")
        complete = greedy_completer(model, tokenizer, prefix=prefix)
        for label, path in (("hidden", args.holdout), ("seen", args.seen)):
            try:
                problems = parse_problems(read_text_file(parser, path))
            except ValueError as exc:
                parser.error(f"{path}: {exc}")
            if args.limit is not None:
                problems = problems[: args.limit]
            if not problems:
                parser.error(f"{path} contains no problems")
            print(format_math_report(label, math_accuracy(problems, complete)))
        return 0

    if args.command == "collect-code":
        roots = [Path(root) for root in args.root] if args.root else default_roots()
        files = collect_python(roots)
        if not files:
            parser.error("no usable .py files found under: " + ", ".join(map(str, roots)))
        corpus = build_code_corpus(files, int(args.max_mb * 1_000_000), args.seed)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(corpus, encoding="utf-8")
        print(
            f"{len(files):,} distinct files from {len(roots)} folders; wrote "
            f"{corpus.count(FILE_MARKER):,} of them: {len(corpus) / 1e6:.1f} MB, "
            f"{len(set(corpus))} distinct characters -> {out}"
        )
        return 0

    if args.command == "make-mix":
        makers = {
            "story": (args.stories, args.stories_mb, story_documents),
            "code": (args.code, args.code_mb, code_documents),
            "math": (args.math, args.math_mb, math_documents),
        }
        sources, budgets = {}, {}
        for name, (path, megabytes, to_documents) in makers.items():
            if path is None:
                continue
            if megabytes <= 0:
                parser.error(f"--{name if name != 'story' else 'stories'}-mb must be > 0")
            sources[name] = to_documents(read_text_file(parser, path))
            budgets[name] = int(megabytes * 1_000_000)
        if not sources:
            parser.error("give at least one of --stories, --code, --math")
        try:
            mix = mix_documents(sources, budgets, args.seed)
        except ValueError as exc:
            parser.error(str(exc))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(mix.text, encoding="utf-8")
        for name in sorted(mix.documents):
            share = 100 * mix.chars[name] / max(len(mix.text), 1)
            print(
                f"{name:<6} {mix.documents[name]:>8,} documents "
                f"{mix.chars[name] / 1e6:>7.1f} MB  {share:4.1f}%"
            )
        print(
            f"wrote {out}: {len(mix.text) / 1e6:.1f} MB, {len(set(mix.text))} distinct characters"
        )
        return 0

    if args.command == "eval-code":
        if args.samples < 1 or args.length < 1:
            parser.error("--samples and --length must be >= 1")
        if not 0 < args.val_fraction < 1:
            parser.error("--val-fraction must be between 0 and 1")
        try:
            from forgelm.eval import grade_code, real_snippets, sample_code
            from forgelm.models.minigpt import load_minigpt
            from forgelm.models.neural_bigram import pick_device

            model, tokenizer = load_minigpt(args.model, device=pick_device(args.device))
        except ImportError:
            parser.error('PyTorch is not installed; see README ("pip install -e .[ml]")')
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"cannot load model: {exc}")
        corpus = read_text_file(parser, args.corpus)
        # Real code from the part of the corpus the model was *not* trained on: the tail,
        # which is the validation split of train-minigpt. The document cut by the boundary is
        # half seen and is dropped by split_documents. In a mixed text the code files are
        # picked out by their tag, so a story that follows a file never leaks into it.
        tail = corpus[int(len(corpus) * (1 - args.val_fraction)) :]
        held_out = [body for tag, body in split_documents(tail) if tag == FILE_MARKER]
        if not held_out:
            parser.error("the held-out part of the corpus holds no complete code file")
        references = real_snippets(held_out, args.length, args.samples, args.seed)
        samples = sample_code(
            model, tokenizer, args.samples, args.length, args.temperature, args.seed
        )
        report = grade_code(samples, references, args.seed)
        print(format_code_report(report, args.length, len(samples), len(references)))
        for number, sample in enumerate(samples[: max(args.show, 0)], 1):
            print(f"--- sample {number} ---")
            print(sample)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
