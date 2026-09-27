"""ForgeLM: an AI engineering lab built step by step."""

from importlib.metadata import PackageNotFoundError, version

try:
    # pyproject.toml is the single source of truth for the version;
    # we read it from the installed package metadata.
    __version__ = version("forgelm")
except PackageNotFoundError:  # running from source without `pip install -e .`
    __version__ = "0.0.0"
