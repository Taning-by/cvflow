"""Solution-relative path resolution.

File and directory parameters (images, models, templates, capture folders,
plugin dirs) may be stored relative to the solution file, so a solution folder
can be moved, zipped or cloned from git and still run. ``Solution.load`` /
``Solution.save`` set the base directory; nodes call ``resolve`` when they open
something.
"""
from __future__ import annotations

import os
from pathlib import Path

_base: Path | None = None


def set_base_dir(directory: str | os.PathLike | None) -> None:
    global _base
    _base = Path(directory).resolve() if directory else None


def base_dir() -> Path:
    return _base if _base is not None else Path.cwd()


def resolve(path: str | os.PathLike | None) -> str:
    """Absolute path for ``path``; relative paths are taken from the solution directory."""
    if not path:
        return ""
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute():
        return str(p)
    return str((base_dir() / p).resolve())


def make_relative(path: str | os.PathLike | None, base: Path | None = None) -> str:
    """Relative form of ``path`` when it is near the base directory, else the path unchanged."""
    if not path:
        return "" if path is None else str(path)
    try:
        rel = os.path.relpath(Path(path).resolve(), base or base_dir())
    except (ValueError, OSError):  # different drive on Windows, broken path ...
        return str(path)
    if rel.startswith("..") and rel.count("..") > 2:
        return str(path)
    return rel.replace(os.sep, "/")
