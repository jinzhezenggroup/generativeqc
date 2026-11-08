"""Recover exact offline evidence while sharing repeated source-file censuses.

Only source path/hash maps use base-plus-delta storage. Every other JSON token
and embedded string stays in original UTF-8 fragments. Ordinary evidence files
retain their existing representation; small authenticated member descriptors
select the shared storage explicitly. No Git, network, native code or scientific
acceptance decision is involved in recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath

from benchmarks._retention import raw_output_path
from tools.generativeqc_validation.storage_json import decode_storage_json

ROOT = Path(__file__).resolve().parents[1]
MEMBER_SCHEMA = "generativeqc.shared-source-evidence.member.v1"
MANIFEST_SCHEMA = "generativeqc.shared-source-evidence.v1"
MAP_SCHEMA = "generativeqc.source-census-maps.v1"
FRAGMENT_SCHEMA = "generativeqc.source-census-fragments.v1"
PREFIX = ('{"schema":"' + MEMBER_SCHEMA + '"').encode()
MAX_STORED = 1 << 20
MAX_EXPANDED = 64 << 20
MAX_METADATA = 128 << 10
MAX_ENTRIES = 32768
MAX_MAPS = 128
HASH = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _integer(value: object, maximum: int, *, positive: bool = False) -> bool:
    return type(value) is int and (1 if positive else 0) <= value <= maximum


def _local(directory: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "invalid storage path")
    name = PurePosixPath(relative)
    _require(
        not name.is_absolute()
        and name.as_posix() == relative
        and ".." not in name.parts
        and "\\" not in relative
        and ":" not in relative
        and not any(ord(char) < 32 for char in relative),
        "unsafe storage path",
    )
    path = directory.joinpath(*name.parts)
    _require(
        not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()),
        "linked or escaping storage path",
    )
    return path


def _expand(stored: bytes, limit: int) -> bytes:
    decoder = lzma.LZMADecompressor(memlimit=96 << 20)
    raw = decoder.decompress(stored, max_length=limit + 1)
    _require(
        decoder.eof and not decoder.unused_data and len(raw) <= limit,
        "oversized, truncated or trailing XZ evidence",
    )
    return raw


@lru_cache(maxsize=4)
def _metadata(raw: bytes) -> dict:
    value = decode_storage_json(raw)
    _require(isinstance(value, dict), "storage metadata is not an object")
    return value


def _checked(directory: Path, binding: dict) -> bytes:
    _require(isinstance(binding, dict), "invalid shared-storage binding")
    size, expanded = binding.get("bytes"), binding.get("decoded_bytes")
    _require(
        _integer(size, MAX_STORED, positive=True)
        and _integer(expanded, MAX_EXPANDED, positive=True)
        and _hash(binding.get("sha256"))
        and _hash(binding.get("decoded_sha256")),
        "invalid shared-storage size or digest",
    )
    path = _local(directory, binding.get("path"))
    _require(
        path.is_file() and path.stat().st_size == size, "shared-storage size differs"
    )
    stored = path.read_bytes()
    _require(_sha(stored) == binding["sha256"], "shared-storage checksum differs")
    raw = _expand(stored, expanded)
    _require(
        len(raw) == expanded and _sha(raw) == binding["decoded_sha256"],
        "decoded shared-storage checksum differs",
    )
    return raw


@lru_cache(maxsize=2)
def _maps(raw: bytes) -> dict[str, dict[str, str]]:
    packet = _metadata(raw)
    _require(packet.get("schema") == MAP_SCHEMA, "unsupported source-map schema")
    paths, values, records = (
        packet.get("paths"),
        packet.get("base_values"),
        packet.get("maps"),
    )
    _require(
        isinstance(paths, list)
        and 0 < len(paths) <= MAX_ENTRIES
        and all(isinstance(p, str) and 0 < len(p) <= 4096 for p in paths)
        and sum(len(p) for p in paths) <= 8 << 20
        and len(set(paths)) == len(paths)
        and isinstance(values, list)
        and len(values) == len(paths)
        and all(v is None or _hash(v) for v in values)
        and isinstance(records, dict)
        and 0 < len(records) <= MAX_MAPS,
        "invalid source-map inventory",
    )
    base = {index: value for index, value in enumerate(values) if value is not None}
    result = {}
    for name, record in records.items():
        _require(
            isinstance(name, str) and isinstance(record, dict),
            "invalid source-map record",
        )
        removed, changed = record.get("removed"), record.get("changed")
        _require(
            isinstance(removed, list)
            and len(removed) <= len(paths)
            and all(_integer(i, len(paths) - 1) and i in base for i in removed)
            and len(set(removed)) == len(removed)
            and isinstance(changed, list)
            and len(changed) <= len(paths),
            "invalid source-map removals or changes",
        )
        updates = {}
        for row in changed:
            _require(
                isinstance(row, list)
                and len(row) == 2
                and _integer(row[0], len(paths) - 1)
                and row[0] not in updates
                and _hash(row[1]),
                "invalid or duplicate source-map change",
            )
            updates[row[0]] = row[1]
        excluded = set(removed)
        restored = {
            index: value for index, value in base.items() if index not in excluded
        }
        restored.update(updates)
        count = record.get("count")
        _require(
            _integer(count, MAX_ENTRIES) and len(restored) == count,
            "source-map count differs",
        )
        order = record.get("order", sorted(restored, key=lambda i: paths[i]))
        _require(
            isinstance(order, list)
            and len(order) == len(restored)
            and all(_integer(i, len(paths) - 1) for i in order)
            and len(set(order)) == len(order)
            and set(order) == set(restored),
            "source-map ordering differs",
        )
        value = {paths[index]: restored[index] for index in order}
        _require(
            _hash(record.get("sha256"))
            and _sha(json.dumps(value, separators=(",", ":")).encode())
            == record["sha256"],
            "source-map checksum differs",
        )
        result[name] = value
    return result


def _render(value: dict[str, str], part: dict) -> str:
    fmt, depth = part.get("format"), part.get("escape_depth")
    _require(
        isinstance(fmt, dict) and _integer(depth, 2), "invalid source-map formatting"
    )
    _require(
        set(fmt) <= {"indent", "ensure_ascii", "line_prefix", "separators"}
        and (
            fmt.get("indent") is None
            or (type(fmt.get("indent")) is int and fmt["indent"] in (0, 2, 4))
        )
        and type(fmt.get("ensure_ascii")) is bool
        and isinstance(fmt.get("line_prefix"), str)
        and len(fmt["line_prefix"]) <= 64
        and not fmt["line_prefix"].strip(" ")
        and ("separators" not in fmt or fmt["separators"] == [",", ":"]),
        "unsupported source-map formatting",
    )
    options = {key: item for key, item in fmt.items() if key != "line_prefix"}
    text = json.dumps(value, **options).replace("\n", "\n" + fmt["line_prefix"])
    for _ in range(depth):
        text = json.dumps(text, ensure_ascii=True)[1:-1]
    return text


def _original(path: Path, repository: Path) -> tuple[bytes, dict | None]:
    path = Path(os.path.abspath(path))
    repository = Path(os.path.abspath(repository))
    _require(
        not path.is_symlink() and path.is_file(), "evidence must be a regular file"
    )
    stored = path.read_bytes()
    if not path.name.endswith(".xz"):
        return stored, None
    decoder = lzma.LZMADecompressor(memlimit=96 << 20)
    prefix = decoder.decompress(stored, max_length=len(PREFIX))
    if prefix != PREFIX:
        return _expand(stored, MAX_EXPANDED), None
    _require(len(stored) <= 4096, "oversized shared-evidence descriptor")
    descriptor = _metadata(_expand(stored, 4096))
    _require(descriptor.get("schema") == MEMBER_SCHEMA, "unsupported member descriptor")
    manifest_path = _local(repository, descriptor.get("manifest"))
    _require(
        manifest_path.stat().st_size <= MAX_METADATA,
        "oversized shared-evidence manifest",
    )
    manifest_bytes = manifest_path.read_bytes()
    _require(
        _hash(descriptor.get("manifest_sha256"))
        and _sha(manifest_bytes) == descriptor["manifest_sha256"],
        "shared-evidence manifest checksum differs",
    )
    manifest = _metadata(manifest_bytes)
    _require(
        manifest.get("schema") == MANIFEST_SCHEMA,
        "unsupported shared-evidence manifest",
    )
    _require(
        path.resolve().is_relative_to(repository.resolve()), "escaping evidence member"
    )
    name = path.relative_to(repository).as_posix()
    entries = manifest.get("files")
    _require(
        descriptor.get("member") == name
        and isinstance(entries, dict)
        and 0 < len(entries) <= 128
        and all(isinstance(value, dict) for value in entries.values())
        and name in entries,
        "shared-evidence member differs",
    )
    entry = entries[name]
    _require(isinstance(entry, dict), "invalid shared-evidence member")
    directory = manifest_path.parent
    maps = _maps(_checked(directory, manifest.get("source_maps")))
    groups = manifest.get("groups")
    _require(
        isinstance(groups, dict) and entry.get("group") in groups,
        "unknown evidence group",
    )
    group = _metadata(_checked(directory, groups[entry["group"]]))
    _require(
        group.get("schema") == FRAGMENT_SCHEMA, "unsupported evidence-fragment schema"
    )
    members = group.get("files")
    expected = {
        key for key, value in entries.items() if value.get("group") == entry["group"]
    }
    _require(
        isinstance(members, dict) and set(members) == expected,
        "evidence group inventory differs",
    )
    record = members[name]
    fields = ("json_bytes", "json_sha256", "xz_bytes", "xz_sha256", "xz_recipe")
    _require(
        isinstance(record, dict)
        and all(record.get(key) == entry.get(key) for key in fields)
        and _integer(record.get("json_bytes"), MAX_EXPANDED, positive=True)
        and _hash(record.get("json_sha256"))
        and _integer(record.get("xz_bytes"), MAX_STORED, positive=True)
        and _hash(record.get("xz_sha256")),
        "original evidence identity differs",
    )
    parts = record.get("parts")
    _require(
        isinstance(parts, list) and 0 < len(parts) <= 1024, "invalid evidence fragments"
    )
    chunks, size = [], 0
    for part in parts:
        if isinstance(part, str):
            text = part
        else:
            _require(
                isinstance(part, dict) and part.get("map") in maps,
                "unknown source-map fragment",
            )
            text = _render(maps[part["map"]], part)
        data = text.encode("utf-8")
        size += len(data)
        _require(size <= record["json_bytes"], "oversized reconstructed evidence")
        chunks.append(data)
    raw = b"".join(chunks)
    _require(
        size == record["json_bytes"] and _sha(raw) == record["json_sha256"],
        "original JSON checksum differs",
    )
    return raw, record


def read_json_bytes(path: Path, *, repository: Path = ROOT) -> bytes:
    """Read exact original JSON, including untouched numeric and string tokens."""
    return _original(path, repository)[0]


def read_xz_bytes(path: Path, *, repository: Path = ROOT) -> bytes:
    """Recover and hash-check original XZ bytes for compressed-identity consumers."""
    raw, record = _original(path, repository)
    if record is None:
        return Path(path).read_bytes()
    recipe = record["xz_recipe"]
    _require(
        isinstance(recipe, dict)
        and set(recipe) == {"preset"}
        and type(recipe["preset"]) is int
        and recipe["preset"] in (6, 9, 9 | lzma.PRESET_EXTREME),
        "unsupported original XZ recipe",
    )
    stored = lzma.compress(raw, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, **recipe)
    _require(
        len(stored) == record["xz_bytes"] and _sha(stored) == record["xz_sha256"],
        "original XZ checksum differs; the linked liblzma encoder may be incompatible",
    )
    return stored


def main() -> None:
    """Restore one requested original without overwriting an existing file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="existing evidence member path")
    parser.add_argument(
        "--xz", action="store_true", help="restore its original XZ transport"
    )
    parser.add_argument(
        "--output", type=raw_output_path, required=True, help="new output file"
    )
    args = parser.parse_args()
    data = (read_xz_bytes if args.xz else read_json_bytes)(args.path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(data)
    print(f"Restored {len(data)} bytes; SHA-256 {_sha(data)}")


if __name__ == "__main__":
    main()
