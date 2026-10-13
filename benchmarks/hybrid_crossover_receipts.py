"""Read losslessly packed crossover receipts without extracting filesystem paths.

Member names are the historical receipt names, including former ``.gz`` suffixes.
Returned bytes are always the exact original decoded receipt, never reserialized
JSON. Storage authentication is separate from scientific acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
from pathlib import Path, PurePosixPath
from typing import Any

FORMAT = "generativeqc.concatenated-receipts-xz.v1"
MAX_PAYLOAD_BYTES = 64 << 20
MAX_CONTAINER_BYTES = 1 << 20


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\0" in name
        or ":" in name
        or PurePosixPath(name).is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"unsafe receipt name: {name!r}")


class ReceiptBundle:
    """Authenticate one complete packed inventory and return named raw bytes."""

    def __init__(self, root: Path, expected_members: set[str]) -> None:
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("receipt manifest cannot be a link")
        self.manifest = json.loads(
            manifest_path.read_bytes(), object_pairs_hook=_unique_object
        )
        storage = self.manifest["storage"]
        if storage.get("format") != FORMAT:
            raise ValueError("unknown receipt storage format")
        if (
            type(storage["bytes"]) is not int
            or not 0 <= storage["bytes"] <= MAX_CONTAINER_BYTES
            or type(storage["content_bytes"]) is not int
            or not 0 <= storage["content_bytes"] <= MAX_PAYLOAD_BYTES
        ):
            raise ValueError("invalid or excessive receipt container size")
        _safe_name(storage["path"])
        archive_path = root / storage["path"]
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if (
            archive_path.is_symlink()
            or not archive_path.resolve().is_relative_to(root.resolve())
            or actual != {"manifest.json", storage["path"]}
        ):
            raise ValueError("missing, linked or unexpected stored receipt file")
        if archive_path.stat().st_size != storage["bytes"]:
            raise ValueError("compressed receipt container size mismatch")
        encoded = archive_path.read_bytes()
        if (len(encoded), _digest(encoded)) != (
            storage["bytes"],
            storage["sha256"],
        ):
            raise ValueError("compressed receipt container integrity failed")
        length = storage["content_bytes"]
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 << 20)
        payload = decoder.decompress(encoded, max_length=length + 1)
        if (
            not decoder.eof
            or decoder.unused_data
            or (len(payload), _digest(payload)) != (length, storage["content_sha256"])
        ):
            raise ValueError("decoded receipt container integrity failed")
        self.members: dict[str, bytes] = {}
        offset = 0
        for entry in self.manifest["files"]:
            if set(entry) != {"path", "offset", "bytes", "sha256"}:
                raise ValueError("unknown receipt member fields; links are unsupported")
            name = entry["path"]
            _safe_name(name)
            count = entry["bytes"]
            if (
                name in self.members
                or type(count) is not int
                or count < 0
                or type(entry["offset"]) is not int
                or entry["offset"] != offset
            ):
                raise ValueError("duplicate, overlapping or invalid receipt member")
            data = payload[offset : offset + count]
            if (len(data), _digest(data)) != (count, entry["sha256"]):
                raise ValueError(f"receipt member integrity failed: {name}")
            self.members[name] = data
            offset += count
        if offset != len(payload) or set(self.members) != expected_members:
            raise ValueError("missing or unexpected receipt member or payload bytes")

    def raw(self, name: str) -> bytes:
        """Return exact original bytes; historical compression suffixes are names."""
        return self.members[name]

    def read(self, name: str) -> dict[str, Any]:
        """Decode a JSON receipt after storage authentication."""
        return json.loads(self.raw(name), object_pairs_hook=_unique_object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("member", help="historical member name from manifest.json")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_bytes())
    bundle = ReceiptBundle(args.directory, {row["path"] for row in manifest["files"]})
    sys.stdout.buffer.write(bundle.raw(args.member))


if __name__ == "__main__":
    main()
