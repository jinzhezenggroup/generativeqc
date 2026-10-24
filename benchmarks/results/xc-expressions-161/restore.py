"""Verify and optionally restore every original XC #161 evidence byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath

DIRECTORY = Path(__file__).resolve().parent


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate evidence member")
        result[key] = value
    return result


def read_records(directory: Path = DIRECTORY) -> dict[str, bytes]:
    """Authenticate the exact historical manifest, including unchanged summary."""
    manifest = json.loads(
        (directory / "manifest.json").read_text(), object_pairs_hook=_unique
    )
    packed = (directory / "records.json.xz").read_bytes()
    decoder = lzma.LZMADecompressor(memlimit=64 << 20)
    raw = decoder.decompress(packed, max_length=4 << 20)
    if not decoder.eof or decoder.unused_data:
        raise ValueError("evidence exceeds bounds or has trailing data")
    texts = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(texts, dict) or set(texts) != set(manifest) - {"summary.json"}:
        raise ValueError("evidence members differ from original manifest")
    records = {"summary.json": (directory / "summary.json").read_bytes()}
    for name, text in texts.items():
        path = PurePosixPath(name)
        if (
            not isinstance(text, str)
            or path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or len(path.parts) != 2
            or path.name not in {"report.json", "contract.json"}
        ):
            raise ValueError("unsafe evidence member")
        records[name] = text.encode("utf-8")
    for name, data in records.items():
        if hashlib.sha256(data).hexdigest() != manifest[name]:
            raise ValueError(f"original evidence checksum mismatch: {name}")
    return records


def restore(output: Path, directory: Path = DIRECTORY) -> int:
    """Validate all members first; never overwrite an existing destination."""
    records = read_records(directory)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in records.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for exact originals")
    args = parser.parse_args()
    count = restore(args.output) if args.output else len(read_records())
    print(
        f"Verified {count} original files"
        + (f"; restored to {args.output}" if args.output else "")
    )
