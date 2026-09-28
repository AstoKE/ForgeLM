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


def test_tokenize_missing_corpus_file_fails(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["tokenize", "hi", "--corpus", str(tmp_path / "nope.txt")])

    assert exc.value.code == 2
    assert "cannot read file" in capsys.readouterr().err
