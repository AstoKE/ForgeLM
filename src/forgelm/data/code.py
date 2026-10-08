"""Python source code as training text, collected from what is already on this machine.

The standard library and the packages in `.venv` are a hundred megabytes of Python written
by people who knew what they were doing, and nothing has to be downloaded.

Raw files are not usable as they are, for three reasons the cleaning below deals with:

* **Characters.** A character-level model gives every distinct character its own embedding
  row. Across 6,000 files there are 6,657 distinct characters and 6,475 of them are rare
  (names in comments, box-drawing, emoji): thousands of rows that are almost never trained.
  Files that are mostly non-ASCII are dropped and the odd stray character is removed.
* **Repeats.** Packages vendor each other and copy files. A model that sees the same file
  forty times memorises it, and the validation set then flatters it. Identical files are
  kept once.
* **Order.** If files were concatenated in directory order, the validation split (the tail)
  would be whatever sorts last, say `xml/` and `zipfile.py`, a different kind of code from
  the training part. Files are shuffled first, so train and validation are the same mix and
  the split falls *between* files.

Files are joined with a marker line, which is itself a valid Python comment. That gives the
model a clear "a new file starts here" signal, and gives the grader a way to cut the corpus
back into files.

Nothing here knows about models, HTTP or the terminal.
"""

import hashlib
import importlib.util
import random
import sysconfig
from pathlib import Path

FILE_MARKER = "# ----- file -----"

# Installed packages worth learning from: readable, mature, and varied (sympy is maths code).
# A name that is not installed is simply skipped.
DEFAULT_PACKAGES = (
    "sympy",
    "networkx",
    "pygments",
    "pydantic",
    "fastapi",
    "starlette",
    "_pytest",
    "setuptools",
    "pip",
)
SKIP_DIRS = {"__pycache__", "site-packages", ".git", "node_modules"}
MAX_NON_ASCII = 0.01  # a file with more than 1% non-ASCII characters is not worth keeping


def clean_python(text: str, min_chars: int = 200, max_chars: int = 100_000) -> str | None:
    """The cleaned file, or None if it should be left out."""
    if not min_chars <= len(text) <= max_chars:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if non_ascii / len(text) > MAX_NON_ASCII:
        return None
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # The marker is how files are told apart again later, so it must not occur inside one.
    if FILE_MARKER in text:
        return None
    return text.strip("\n") + "\n"


def default_roots() -> list[Path]:
    """The standard library plus the installed packages in DEFAULT_PACKAGES."""
    roots = [Path(sysconfig.get_paths()["stdlib"])]
    for name in DEFAULT_PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            roots.extend(Path(location) for location in spec.submodule_search_locations)
    return [root for root in roots if root.is_dir()]


def collect_python(roots: list[Path], min_chars: int = 200, max_chars: int = 100_000) -> list[str]:
    """Every cleaned, distinct .py file under `roots`, in a deterministic order."""
    files, seen = [], set()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if SKIP_DIRS & set(path.relative_to(root).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            cleaned = clean_python(text, min_chars, max_chars)
            if cleaned is None:
                continue
            digest = hashlib.sha1(cleaned.encode("ascii")).digest()
            if digest in seen:
                continue
            seen.add(digest)
            files.append(cleaned)
    return files


def build_code_corpus(files: list[str], max_chars: int | None = None, seed: int = 0) -> str:
    """Shuffle the files, then join them with the marker, stopping before `max_chars`."""
    if not files:
        raise ValueError("no files to build a corpus from")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    parts, size = [], 0
    for text in shuffled:
        piece = f"{FILE_MARKER}\n{text}\n"
        if max_chars is not None and size + len(piece) > max_chars and parts:
            break
        parts.append(piece)
        size += len(piece)
    return "".join(parts)


def split_files(corpus: str) -> list[str]:
    """Cut a corpus built by `build_code_corpus` back into its files."""
    return [piece.strip("\n") + "\n" for piece in corpus.split(FILE_MARKER + "\n") if piece.strip()]
