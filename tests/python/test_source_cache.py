"""Generated source identity is the caller's exact UTF-8 bytes."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc_compiler.common import source_cache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "source",
    (
        "// first\n// second\n",
        "// crlf\r\n",
        "// mixed\r\n// lf\n// cr\r",
        "// no final newline",
        "// caf\u00e9 \u03bb \u4e2d\n",
    ),
)
@pytest.mark.parametrize("text_defaults", ("native", "windows-legacy"))
def test_exact_source_bytes_and_existing_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, text_defaults: str
) -> None:
    if text_defaults == "windows-legacy":
        original_fdopen = os.fdopen

        def legacy_fdopen(fd: int, mode: str = "r", **kwargs: Any) -> Any:
            # Exercise real I/O under a non-UTF-8 locale and Windows newlines,
            # even on Linux CI. Explicit binary/text contracts remain effective.
            if "b" not in mode:
                kwargs.setdefault("encoding", "cp1252")
                kwargs.setdefault("newline", "\r\n")
            return original_fdopen(fd, mode, **kwargs)

        monkeypatch.setattr(os, "fdopen", legacy_fdopen)
    path = tmp_path / "source.cu"
    expected = source.encode("utf-8")
    source_cache.cache_source(path, source)
    assert path.read_bytes() == expected
    source_cache.cache_source(path, source)
    assert path.read_bytes() == expected
    assert not list(tmp_path.glob(".source-*"))


def test_existing_newline_corruption_is_not_normalized(tmp_path: Path) -> None:
    path = tmp_path / "source.cu"
    damaged = b"// first\r\n// second\r\n"
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="identity mismatch"):
        source_cache.cache_source(path, "// first\n// second\n")
    assert path.read_bytes() == damaged


def test_failed_publication_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(*args: Any) -> None:
        raise OSError("publication unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)
    path = tmp_path / "source.cu"
    with pytest.raises(OSError, match="publication unavailable"):
        source_cache.cache_source(path, "// intact\n")
    assert not path.exists()
    assert not list(tmp_path.glob(".source-*"))
