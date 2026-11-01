"""Decode exact FP64 benchmark evidence while authenticating original bytes.

This storage reader makes no numerical acceptance decisions. Callers must still
replay their independent scientific gates and retain all failed populations.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def require(condition: bool, detail: str) -> None:
    """Enforce integrity even when Python assertions are disabled."""
    if not condition:
        raise ValueError(detail)


def load(folder: Path) -> tuple[dict, dict[str, str]]:
    """Reconstruct exact receipts, including the sequential-marker v2 format.

    V2 omits redundant cursor integers from the JSON stream. Traversal order,
    the final float count and each original byte checksum still authenticate
    every FP64 word; neither representation rounds numerical evidence.
    """
    for line in (folder / "sha256.txt").read_text().splitlines():
        digest, name = line.split()
        require(
            hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
        )
    bundle = json.loads(lzma.decompress((folder / "receipts.json.xz").read_bytes()))
    require(
        bundle["schema"]
        in (
            "generativeqc.bounded-projection-evidence.v1",
            "generativeqc.exact-fp64-evidence.v2",
        ),
        "schema",
    )
    sequential = bundle["schema"] == "generativeqc.exact-fp64-evidence.v2"
    planes = lzma.decompress((folder / "fp64.xz").read_bytes())
    count = bundle["float_count"]
    require(len(planes) == 8 * count, "FP64 storage size")
    last: dict[tuple, int] = {}
    cursor = 0

    def decode(value: object, path: tuple) -> object:
        nonlocal cursor
        if isinstance(value, dict) and "$fp64" in value:
            require(
                value == {"$fp64": None if sequential else cursor} and cursor < count,
                "FP64 ordering",
            )
            delta = int.from_bytes(
                bytes(planes[lane * count + cursor] for lane in range(8)), "little"
            )
            word = delta ^ last.get(path, 0)
            last[path] = word
            cursor += 1
            return struct.unpack("<d", word.to_bytes(8, "little"))[0]
        if isinstance(value, list):
            return [decode(v, path + (i,)) for i, v in enumerate(value)]
        if isinstance(value, dict):
            return {k: decode(v, path + (k,)) for k, v in value.items()}
        return value

    files = {}
    for row in bundle["files"]:
        text = (
            (json.dumps(decode(row["json"], (row["anchor"],)), indent=2) + "\n")
            if "json" in row
            else row["text"]
        )
        require(hashlib.sha256(text.encode()).hexdigest() == row["sha256"], row["path"])
        require(row["path"] not in files, "duplicate receipt")
        files[row["path"]] = text
    require(cursor == count, "unconsumed FP64 values")
    return bundle, files
