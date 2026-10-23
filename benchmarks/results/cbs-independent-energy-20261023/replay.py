"""Verify storage, restore exact canonical files, or explicitly replay energies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

LIMIT = 1 << 20


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pairs(items: list) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def loads(raw: bytes) -> dict:
    return json.loads(raw, object_pairs_hook=pairs)


def inflate(raw: bytes) -> bytes:
    decoder = lzma.LZMADecompressor(memlimit=128 << 20)
    data = decoder.decompress(raw, max_length=LIMIT + 1)
    if len(data) > LIMIT or not decoder.eof or decoder.unused_data:
        raise ValueError("oversize, unfinished or concatenated XZ stream")
    return data


def safe(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name == "."
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 for c in name)
    ):
        raise ValueError("unsafe member path")
    p = PurePosixPath(name)
    if p.is_absolute() or p.as_posix() != name:
        raise ValueError("noncanonical member path")
    for part in p.parts:
        if part in {".", ".."} or part.endswith((".", " ")):
            raise ValueError("unsafe member component")
        if re.fullmatch(
            r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", part.split(".")[0], re.IGNORECASE
        ):
            raise ValueError("reserved member component")
    return name


def snapshot(root: Path) -> tuple[dict, dict, dict[str, bytes]]:
    if (root / "publication.json").is_symlink() or (
        root / "publication.json"
    ).stat().st_size > LIMIT:
        raise ValueError("linked or oversized storage manifest")
    manifest = loads((root / "publication.json").read_bytes())
    required = {
        "README.md",
        "replay.py",
        "canonical.json",
        "records.json.xz",
        "evidence.json.xz",
        "oracle-source.patch.xz",
    }
    if {e["path"] for e in manifest["files"]} != required:
        raise ValueError("unexpected storage inventory")
    stored = {}
    for e in manifest["files"]:
        name = safe(e["path"])
        path = root / name
        if name in stored or path.is_symlink() or not path.is_file():
            raise ValueError("duplicate, linked or missing stored member")
        if path.stat().st_size > LIMIT:
            raise ValueError("oversize stored member")
        raw = path.read_bytes()
        if len(raw) != e["bytes"] or digest(raw) != e["sha256"]:
            raise ValueError("stored checksum/size mismatch")
        stored[name] = raw
    index = loads(stored["canonical.json"])
    bindings = {
        "records": {"path": "records.json.xz", "encoding": "base64"},
        "evidence": {"path": "evidence.json.xz", "encoding": "xz"},
        "patch": {"path": "oracle-source.patch.xz", "encoding": "raw"},
    }
    if index["bindings"] != bindings or index["default_storage"] != "records":
        raise ValueError("invalid storage bindings")

    def binding(entry: dict) -> dict:
        tag = entry.get("storage", index["default_storage"])
        if tag not in bindings:
            raise ValueError("unknown storage binding")
        return bindings[tag]

    entries = index["files"] + index.get("checks", [])
    if (
        index["schema"] != "generativeqc.cbs-canonical-snapshot.v1"
        or not 0 < len(entries) <= 64
    ):
        raise ValueError("invalid canonical inventory")
    names = [safe(e["path"]) for e in entries]
    folded = {n.casefold() for n in names}
    if len(folded) != len(names) or any(
        str(p) in folded
        for n in folded
        for p in PurePosixPath(n).parents
        if str(p) != "."
    ):
        raise ValueError("duplicate or file/directory collision")
    bundle = loads(inflate(stored["records.json.xz"]))
    if bundle["schema"] != "generativeqc.cbs-canonical-files.v1":
        raise ValueError("invalid file container schema")
    encoded = bundle["files"]
    expected = {e["path"] for e in entries if binding(e)["encoding"] == "base64"}
    if set(encoded) != expected:
        raise ValueError("encoded inventory mismatch")
    result = {}
    for e in entries:
        name = e["path"]
        if (
            e["mode"] != "100644"
            or type(e["bytes"]) is not int
            or not 0 <= e["bytes"] <= LIMIT
        ):
            raise ValueError("nonregular mode or invalid member size")
        spec = binding(e)
        if name in expected:
            raw = base64.b64decode(encoded[name], validate=True)
        elif name == "evidence.json" and spec == {
            "path": "evidence.json.xz",
            "encoding": "xz",
        }:
            raw = inflate(stored["evidence.json.xz"])
        elif name == "oracle-source.patch.xz" and spec == {
            "path": name,
            "encoding": "raw",
        }:
            raw = stored[name]
        else:
            raise ValueError("unsupported canonical storage binding")
        if len(raw) != e["bytes"] or digest(raw) != e["sha256"]:
            raise ValueError("canonical checksum/size mismatch")
        result[name] = raw
    if (
        index["file_count"] != len(index["files"])
        or index["total_bytes"] != sum(len(result[e["path"]]) for e in index["files"])
        or sum(map(len, result.values())) > LIMIT
    ):
        raise ValueError("canonical totals mismatch")
    return manifest, stored, result


def restore(files: dict[str, bytes], destination: Path) -> None:
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing existing destination")
    for parent in destination.parents:
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("linked destination parent")
    destination.mkdir(mode=0o700, exist_ok=False)
    for name, raw in files.items():
        target = destination / safe(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        target.chmod(0o644)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--unpack", type=Path)
    modes.add_argument("--require-pass", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    manifest, stored, files = snapshot(root)
    if args.unpack:
        restore(files, args.unpack)
    elif args.require_pass:
        if args.output is None or args.output.exists() or args.output.is_symlink():
            raise ValueError("provide a fresh --output file")
        repo = next(parent for parent in root.parents if (parent / ".git").exists())
        sys.path[:0] = [str(repo), str(repo / "python")]
        from tools.generativeqc_validation.publication import validate_publication

        validate_publication(manifest, stored)
        with tempfile.TemporaryDirectory(prefix="cbs-canonical-") as folder:
            temp = Path(folder)
            if not temp.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
                raise ValueError("temporary root escapes its parent")
            target = temp / "canonical"
            restore(files, target)
            original = loads(files["publication.json"])
            validate_publication(
                original, {e["path"]: files[e["path"]] for e in original["files"]}
            )
            subprocess.run(
                [
                    sys.executable,
                    "-E",
                    str(target / "replay.py"),
                    "--receipts",
                    str(target / "receipts"),
                    "--output",
                    str(temp / "result.json"),
                    "--require-pass",
                ],
                cwd=repo,
                check=True,
                timeout=120,
            )
            raw = (temp / "result.json").read_bytes()
            if loads(raw) != loads(files["comparison.json"]):
                raise ValueError("canonical replay differs from retained comparison")
            with args.output.open("xb") as stream:
                stream.write(raw)
    print(
        json.dumps(
            {
                "restored_files": len(files),
                "canonical_bytes_verified": True,
                "numerical_replay_executed": bool(args.require_pass),
            }
        )
    )


if __name__ == "__main__":
    main()
