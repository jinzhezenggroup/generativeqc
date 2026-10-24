"""Atomic publication of immutable, content-addressed generated sources."""

import os
import tempfile
import typing
from pathlib import Path


def cache_source(path: typing.Any, source: typing.Any) -> None:
    """Publish exact UTF-8 bytes without newline or locale conversion."""
    data = source.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("native source identity mismatch")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".source-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
