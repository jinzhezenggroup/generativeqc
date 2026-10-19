"""Linux opened-object and fixed-width ABI gates, never CUDA/science evidence."""

from __future__ import annotations

import os
import struct
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from benchmarks.cc_trace_reader import InvalidTrace, open_trace

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux nofollow/FIFO trace admission"
)


def empty_activity(path: Path) -> None:
    metadata = [0] * 14
    metadata[1] = 1
    metadata[7] = 28
    metadata[8] = 1
    path.write_bytes(struct.pack("<19Q", 0x434354494D453031, 1, 20, 14, 0, *metadata))


def test_opened_regular_object_and_exact_size(tmp_path: Path) -> None:
    path = tmp_path / "activity.bin"
    empty_activity(path)
    trace = open_trace(path)
    assert list(trace.records()) == []
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(InvalidTrace, match="size mismatch"):
        open_trace(path)
    with pytest.raises(InvalidTrace, match="changed"):
        list(trace.records())


def test_symlink_and_fifo_are_refused_without_follow_or_block(tmp_path: Path) -> None:
    path = tmp_path / "activity.bin"
    empty_activity(path)
    alias = tmp_path / "alias.bin"
    alias.symlink_to(path)
    with pytest.raises(InvalidTrace, match="open refused"):
        open_trace(alias)
    fifo = tmp_path / "pipe.bin"
    os.mkfifo(fifo)
    with pytest.raises(InvalidTrace, match="regular"):
        open_trace(fifo)


def test_replacement_after_hash_is_not_the_same_opened_object(tmp_path: Path) -> None:
    path = tmp_path / "activity.bin"
    empty_activity(path)
    trace = open_trace(path)
    replacement = tmp_path / "replacement.bin"
    empty_activity(replacement)
    replacement.replace(path)
    with pytest.raises(InvalidTrace, match="changed"):
        list(trace.records())


def test_unsupported_abi_and_short_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid.bin"
    path.write_bytes(struct.pack("<5Q", 0x434354494D453031, 2, 20, 14, 0))
    with pytest.raises(InvalidTrace, match="unsupported"):
        open_trace(path)
    path.write_bytes(struct.pack("<5Q", 0x434354494D453031, 1, 20, 14, 0))
    with pytest.raises(InvalidTrace, match="short metadata"):
        open_trace(path)
