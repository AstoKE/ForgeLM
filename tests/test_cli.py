import re
import sys

import pytest

from forgelm import __version__
from forgelm.cli import main


def test_version_flag_prints_version(capsys):
    # argparse's "version" action prints and then exits with code 0.
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: forgelm" in capsys.readouterr().out


def test_unknown_command_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["fly-to-the-moon"])

    assert exc.value.code == 2  # argparse's code for usage errors
    assert "invalid choice" in capsys.readouterr().err


def test_serve_starts_uvicorn_with_our_app(monkeypatch):
    # We don't want a real server in tests, so replace uvicorn.run with a fake.
    calls = {}

    def fake_run(app, **kwargs):
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    assert main(["serve", "--port", "9000"]) == 0
    assert calls == {"app": "forgelm.api:app", "host": "127.0.0.1", "port": 9000, "reload": False}


def test_tokenize_prints_tokens_and_roundtrip(capsys):
    assert main(["tokenize", "hi"]) == 0

    out = capsys.readouterr().out
    assert "tokens:  ['h', 'i']" in out
    assert "round-trip: OK" in out


def test_tokenize_with_corpus_file_shows_unknowns(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello", encoding="utf-8")

    assert main(["tokenize", "hello!", "--corpus", str(corpus)]) == 0
    assert "round-trip: FAILED, 1 unknown" in capsys.readouterr().out


def test_tokenize_with_bpe_roundtrips_unseen_characters(tmp_path, capsys):
    corpus = tmp_path / "english.txt"
    corpus.write_text("the quick brown fox", encoding="utf-8")

    args = ["tokenize", "İstanbul", "--tokenizer", "bpe", "--corpus", str(corpus)]
    assert main(args) == 0

    out = capsys.readouterr().out
    assert "unknown 0" in out
    assert "round-trip: OK" in out


def test_tokenize_negative_merges_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["tokenize", "hi", "--tokenizer", "bpe", "--merges", "-1"])

    assert exc.value.code == 2
    assert "--merges must be >= 0" in capsys.readouterr().err


@pytest.fixture
def trained_model(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be, that is the question.\n" * 20, encoding="utf-8")
    model = tmp_path / "ckpt" / "bigram.json"

    assert main(["train-bigram", "--corpus", str(corpus), "--out", str(model)]) == 0
    out = capsys.readouterr().out
    assert "uniform baseline loss" in out
    assert f"saved {model}" in out
    return model


def test_generate_is_reproducible_with_seed(trained_model, capsys):
    args = ["generate", "--model", str(trained_model), "--prompt", "to", "--seed", "7"]

    main(args)
    first = capsys.readouterr().out
    main(args)
    second = capsys.readouterr().out

    assert first == second
    assert first.startswith("to")


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--temperature", "-1"], "--temperature must be >= 0"),
        (["--prompt", ""], "--prompt must not be empty"),
    ],
)
def test_generate_rejects_bad_arguments(trained_model, capsys, extra_args, message):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--model", str(trained_model), *extra_args])

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_generate_missing_model_fails(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--model", str(tmp_path / "nope.json")])

    assert exc.value.code == 2
    assert "cannot load model" in capsys.readouterr().err


def test_train_bigram_tiny_corpus_fails(tmp_path, capsys):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("ab", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["train-bigram", "--corpus", str(corpus), "--out", str(tmp_path / "m.json")])

    assert exc.value.code == 2
    assert "too short" in capsys.readouterr().err


def test_train_neural_bigram_prints_progress(tmp_path, capsys):
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be, that is the question.\n" * 20, encoding="utf-8")

    args = ["train-neural-bigram", "--corpus", str(corpus), "--steps", "10", "--device", "cpu"]
    assert main([*args, "--batch-size", "64", "--eval-every", "5"]) == 0

    out = capsys.readouterr().out
    assert "device cpu" in out
    assert "step     0" in out
    assert "step    10" in out
    assert "counting bigram (2a)" in out


def test_train_neural_bigram_without_torch_fails_clearly(tmp_path, capsys, monkeypatch):
    # Pretend torch isn't installed: a None entry in sys.modules makes `import` fail.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "forgelm.models.neural_bigram", raising=False)

    with pytest.raises(SystemExit) as exc:
        main(["train-neural-bigram", "--corpus", str(tmp_path / "x.txt")])

    assert exc.value.code == 2
    assert "PyTorch is not installed" in capsys.readouterr().err


def test_tokenize_missing_corpus_file_fails(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["tokenize", "hi", "--corpus", str(tmp_path / "nope.txt")])

    assert exc.value.code == 2
    assert "cannot read file" in capsys.readouterr().err


@pytest.fixture
def trained_minigpt(tmp_path, capsys):
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be, that is the question.\n" * 20, encoding="utf-8")
    model = tmp_path / "ckpt" / "minigpt.pt"
    args = ["train-minigpt", "--corpus", str(corpus), "--out", str(model), "--steps", "10"]
    tiny = ["--embed-dim", "16", "--heads", "2", "--blocks", "1", "--block-size", "8"]

    assert main([*args, *tiny, "--batch-size", "2", "--eval-every", "5", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "device cpu" in out
    assert "uniform baseline loss" in out
    assert "step     0" in out
    assert "step    10" in out
    assert f"saved {model}" in out
    return model


def test_generate_from_a_minigpt_checkpoint_is_reproducible(trained_minigpt, capsys):
    args = ["generate", "--model", str(trained_minigpt), "--prompt", "to", "--max-tokens", "20"]

    main([*args, "--seed", "7"])
    first = capsys.readouterr().out
    main([*args, "--seed", "7"])
    second = capsys.readouterr().out

    assert first == second
    assert first.startswith("to")
    assert len(first.rstrip("\n")) > len("to")  # something was generated after the prompt


def test_generate_rejects_a_broken_pt_file(tmp_path, capsys):
    pytest.importorskip("torch")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")

    with pytest.raises(SystemExit) as exc:
        main(["generate", "--model", str(bad)])

    assert exc.value.code == 2
    assert "cannot load model" in capsys.readouterr().err


def test_train_minigpt_rejects_embed_dim_not_divisible_by_heads(tmp_path, capsys):
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world " * 50, encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["train-minigpt", "--corpus", str(corpus), "--embed-dim", "10", "--heads", "4"])

    assert exc.value.code == 2
    assert "divisible" in capsys.readouterr().err


def test_train_minigpt_without_torch_fails_clearly(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    for name in ("minigpt", "neural_bigram", "train_minigpt", "block", "attention"):
        monkeypatch.delitem(sys.modules, f"forgelm.models.{name}", raising=False)

    with pytest.raises(SystemExit) as exc:
        main(["train-minigpt", "--corpus", str(tmp_path / "x.txt")])

    assert exc.value.code == 2
    assert "PyTorch is not installed" in capsys.readouterr().err


# --- make-math-data and eval-math (Sprint 4d) ---------------------------------------------


@pytest.fixture
def math_files(tmp_path, capsys):
    out = tmp_path / "data" / "m.txt"
    assert main(["make-math-data", "--out", str(out), "--digits", "1", "--lines", "200"]) == 0
    capsys.readouterr()
    return out


def test_make_math_data_writes_the_corpus_the_exam_and_the_control(math_files, capsys):
    folder = math_files.parent

    corpus = math_files.read_text(encoding="utf-8").splitlines()
    hidden = (folder / "m-holdout.txt").read_text(encoding="utf-8").splitlines()
    seen = (folder / "m-seen.txt").read_text(encoding="utf-8").splitlines()

    assert len(corpus) == 200
    assert hidden
    assert not set(hidden) & set(corpus)  # the exam never appears in the training text
    assert set(seen) <= set(corpus)  # the control only holds problems it trained on


def test_make_math_data_reports_what_it_wrote(tmp_path, capsys):
    out = tmp_path / "m.txt"

    main(["make-math-data", "--out", str(out), "--digits", "1", "--lines", "30"])

    printed = capsys.readouterr().out
    assert f"wrote {out}" in printed
    assert "m-holdout.txt" in printed
    assert "hidden" in printed


def test_make_math_data_can_pad_and_reverse(tmp_path):
    out = tmp_path / "m.txt"

    main(["make-math-data", "--out", str(out), "--digits", "2", "--lines", "20", "--pad"])
    padded = out.read_text(encoding="utf-8").splitlines()
    main(["make-math-data", "--out", str(out), "--digits", "2", "--lines", "20", "--reverse"])
    reversed_ = out.read_text(encoding="utf-8").splitlines()

    assert {len(line) for line in padded} == {9}  # "dd+dd=ddd"
    for line in reversed_:
        prompt, _, answer = line.partition("=")
        a, b = prompt.split("+")
        assert answer == str(int(a) + int(b))[::-1]


def test_make_math_data_refuses_bad_settings(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["make-math-data", "--out", str(tmp_path / "m.txt"), "--digits", "9"])

    assert exc.value.code == 2
    assert "digits" in capsys.readouterr().err


def test_eval_math_prints_a_score_for_the_exam_and_for_the_control(math_files, tmp_path, capsys):
    pytest.importorskip("torch")
    model = tmp_path / "ckpt" / "m.pt"
    main(
        ["train-minigpt", "--corpus", str(math_files), "--out", str(model), "--steps", "5"]
        + ["--embed-dim", "16", "--heads", "2", "--blocks", "1", "--block-size", "16"]
        + ["--batch-size", "2", "--eval-every", "5", "--device", "cpu"]
    )
    capsys.readouterr()
    folder = math_files.parent

    code = main(
        ["eval-math", "--model", str(model)]
        + ["--holdout", str(folder / "m-holdout.txt"), "--seen", str(folder / "m-seen.txt")]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "hidden:" in out
    assert "seen:" in out
    assert "%" in out


def test_eval_math_limit_caps_the_number_of_problems(math_files, tmp_path, capsys):
    pytest.importorskip("torch")
    model = tmp_path / "m.pt"
    main(
        ["train-minigpt", "--corpus", str(math_files), "--out", str(model), "--steps", "2"]
        + ["--embed-dim", "16", "--heads", "2", "--blocks", "1", "--block-size", "16"]
        + ["--batch-size", "2", "--eval-every", "2", "--device", "cpu"]
    )
    capsys.readouterr()
    folder = math_files.parent

    main(
        ["eval-math", "--model", str(model), "--limit", "3"]
        + ["--holdout", str(folder / "m-holdout.txt"), "--seen", str(folder / "m-seen.txt")]
    )

    out = capsys.readouterr().out
    assert re.search(r"hidden: \d+/3 = ", out)  # three problems graded, not the whole file
    assert re.search(r"seen: \d+/3 = ", out)


def test_eval_math_rejects_a_file_that_is_not_made_of_problems(math_files, tmp_path, capsys):
    pytest.importorskip("torch")
    model = tmp_path / "m.pt"
    main(
        ["train-minigpt", "--corpus", str(math_files), "--out", str(model), "--steps", "2"]
        + ["--embed-dim", "16", "--heads", "2", "--blocks", "1", "--block-size", "16"]
        + ["--batch-size", "2", "--eval-every", "2", "--device", "cpu"]
    )
    capsys.readouterr()
    garbage = tmp_path / "garbage.txt"
    garbage.write_text("this is not arithmetic\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["eval-math", "--model", str(model), "--holdout", str(garbage)])

    assert exc.value.code == 2
    assert "not a problem line" in capsys.readouterr().err


def test_eval_math_fails_clearly_for_a_missing_model(tmp_path, capsys):
    pytest.importorskip("torch")

    with pytest.raises(SystemExit) as exc:
        main(["eval-math", "--model", str(tmp_path / "nope.pt")])

    assert exc.value.code == 2
    assert "cannot load model" in capsys.readouterr().err


# --- collect-code and eval-code (Sprint 4d) -----------------------------------------------

SNIPPET = "import os\n\n\ndef join(a, b):\n    return os.path.join(a, b)\n" * 8


@pytest.fixture
def code_corpus(tmp_path, capsys):
    tree = tmp_path / "src"
    tree.mkdir()
    for i in range(30):
        (tree / f"mod{i}.py").write_text(SNIPPET.replace("join", f"join{i}"), encoding="utf-8")
    out = tmp_path / "data" / "python.txt"
    assert main(["collect-code", "--out", str(out), "--root", str(tree)]) == 0
    capsys.readouterr()
    return out


def test_collect_code_writes_one_corpus_and_reports_what_it_found(tmp_path, capsys):
    tree = tmp_path / "src"
    tree.mkdir()
    (tree / "a.py").write_text(SNIPPET, encoding="utf-8")
    (tree / "b.py").write_text(SNIPPET.replace("join", "link"), encoding="utf-8")
    out = tmp_path / "data" / "python.txt"

    assert main(["collect-code", "--out", str(out), "--root", str(tree)]) == 0

    printed = capsys.readouterr().out
    assert "2 distinct files" in printed
    assert "distinct characters" in printed
    assert out.read_text(encoding="utf-8").count("# ----- file -----") == 2


def test_collect_code_respects_the_size_limit(tmp_path, capsys):
    tree = tmp_path / "src"
    tree.mkdir()
    for i in range(20):
        (tree / f"m{i}.py").write_text(SNIPPET.replace("join", f"j{i}"), encoding="utf-8")
    out = tmp_path / "python.txt"

    main(["collect-code", "--out", str(out), "--root", str(tree), "--max-mb", "0.001"])

    assert out.stat().st_size < 3_000  # about one file's worth, not twenty


def test_collect_code_with_nothing_to_collect_fails_clearly(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["collect-code", "--out", str(tmp_path / "x.txt"), "--root", str(tmp_path)])

    assert exc.value.code == 2
    assert "no usable .py files" in capsys.readouterr().err


def test_eval_code_reports_the_model_the_ceiling_and_the_floor(code_corpus, tmp_path, capsys):
    pytest.importorskip("torch")
    model = tmp_path / "ckpt" / "code.pt"
    main(
        ["train-minigpt", "--corpus", str(code_corpus), "--out", str(model), "--steps", "3"]
        + ["--embed-dim", "16", "--heads", "2", "--blocks", "1", "--block-size", "16"]
        + ["--batch-size", "2", "--eval-every", "3", "--device", "cpu"]
    )
    capsys.readouterr()

    code = main(
        ["eval-code", "--model", str(model), "--corpus", str(code_corpus)]
        + ["--samples", "3", "--length", "60", "--show", "1"]
    )

    assert code == 0
    out = capsys.readouterr().out
    for word in ("generated", "ceiling", "floor", "position", "--- sample 1 ---"):
        assert word in out


def test_eval_code_rejects_bad_settings(tmp_path, capsys):
    for bad in (["--samples", "0"], ["--length", "0"], ["--val-fraction", "1"]):
        with pytest.raises(SystemExit) as exc:
            main(["eval-code", "--model", str(tmp_path / "m.pt"), *bad])
        assert exc.value.code == 2


def test_eval_code_needs_a_corpus_with_a_held_out_file(code_corpus, tmp_path, capsys):
    pytest.importorskip("torch")
    model = tmp_path / "code.pt"
    main(
        ["train-minigpt", "--corpus", str(code_corpus), "--out", str(model), "--steps", "2"]
        + ["--embed-dim", "16", "--heads", "2", "--blocks", "1", "--block-size", "16"]
        + ["--batch-size", "2", "--eval-every", "2", "--device", "cpu"]
    )
    capsys.readouterr()
    no_markers = tmp_path / "plain.txt"
    no_markers.write_text("just some text without any file marker\n" * 50, encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["eval-code", "--model", str(model), "--corpus", str(no_markers)])

    assert exc.value.code == 2
    assert "no complete file" in capsys.readouterr().err
