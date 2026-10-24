"""Restore the actual failed-but-partially-returned v2 task, never execute it."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import os
import re
from pathlib import Path, PurePosixPath

if not __debug__:
    raise RuntimeError("actual evidence recovery requires non-optimized Python")

HERE = Path(__file__).resolve().parent
CAPSULE = HERE / "actual-terminal-originals-v2.json.xz"
CAPSULE_SHA = "2e9a5aa83bd998bcf2b63703f590f97c93a67b87fc0e573c034fbb438510816f"
MAX_STORED = 1 << 20
MAX_EXPANDED = 8 << 20


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(packed: bytes) -> dict:
    if len(packed) > MAX_STORED:
        raise ValueError("stored capsule exceeds bounded reader limit")
    decoder = lzma.LZMADecompressor(memlimit=96 << 20)
    raw = decoder.decompress(packed, max_length=MAX_EXPANDED + 1)
    if len(raw) > MAX_EXPANDED or not decoder.eof or decoder.unused_data:
        raise ValueError("oversized, truncated or trailing compressed capsule")
    return json.loads(raw)


def validate_members(value: dict) -> dict[str, tuple[bytes, int]]:
    rows = value.get("members")
    if not isinstance(rows, list) or not 0 < len(rows) <= 4096:
        raise ValueError("invalid original member count")
    result = {}
    total = 0
    for row in rows:
        name = row["path"]
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not relative.parts
            or any(":" in part for part in relative.parts)
            or name in result
            or relative.name == ".env"
        ):
            raise ValueError("unsafe or repeated original path")
        size = row["bytes"]
        if type(size) is not int or not 0 <= size <= MAX_STORED:
            raise ValueError("invalid original byte count")
        encoded = row["base64"]
        if len(encoded) > ((MAX_STORED + 2) // 3) * 4:
            raise ValueError("oversized encoded original")
        data = base64.b64decode(encoded, validate=True)
        if len(data) != size or digest(data) != row["sha256"]:
            raise ValueError("original member size/hash mismatch")
        if not isinstance(row["mode"], str) or not re.fullmatch(
            r"0o[0-7]{1,3}", row["mode"]
        ):
            raise ValueError("invalid original mode witness")
        total += size
        if total > 4 << 20:
            raise ValueError("original members exceed bounded aggregate")
        result[name] = (data, int(row["mode"], 8))
    if len(rows) != value.get("raw_members") or total != value.get("raw_bytes"):
        raise ValueError("original closure counters differ")
    return result


def members(path: Path = CAPSULE) -> dict[str, tuple[bytes, int]]:
    packed = path.read_bytes()
    if digest(packed) != CAPSULE_SHA:
        raise ValueError("actual terminal capsule identity differs")
    return validate_members(decode(packed))


def restore(output: Path, path: Path = CAPSULE) -> dict:
    originals = members(path)
    output.mkdir(exist_ok=False)
    for name, (data, mode) in originals.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            target.chmod(mode)
        if target.read_bytes() != data:
            raise ValueError("restored readback differs")
        if os.name == "posix" and target.stat().st_mode & 0o777 != mode:
            raise ValueError("restored Linux mode differs")
    return {
        "scope": "storage recovery only; no scientific execution",
        "raw_members": len(originals),
        "raw_bytes": sum(len(data) for data, _ in originals.values()),
        "byte_hash_readback": "PASS",
        "Linux_modes": "APPLIED"
        if os.name == "posix"
        else "retained witnesses; not applied on Windows",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory; existing paths are refused",
    )
    args = parser.parse_args()
    print(json.dumps(restore(args.output), indent=2))
