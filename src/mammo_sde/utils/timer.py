"""Simple timing context manager."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timed(name: str = "block") -> Iterator[dict]:
    """Context manager that yields a dict whose ``elapsed`` key is set on exit."""
    holder: dict = {"name": name, "elapsed": None}
    t0 = time.time()
    try:
        yield holder
    finally:
        holder["elapsed"] = time.time() - t0
