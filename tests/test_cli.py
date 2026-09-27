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
