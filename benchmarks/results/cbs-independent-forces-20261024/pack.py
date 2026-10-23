"""Losslessly retain scientific text outputs in per-task JSON/XZ records."""

import argparse
import hashlib
import json
import lzma
import re
from pathlib import Path

HERE = Path(__file__).parent
TASK_FILES = (
    "stdout.log",
    "outcome.json",
    "science/receipt.json",
    "science/basis-2.log",
    "science/basis-3.log",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inventory(root: Path, names: tuple, modes: dict) -> list:
    records = []
    for name in names:
        path = root / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Only regular files: {path}")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        assert text.encode("utf-8") == raw
        records.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": sha(raw),
                "original_mode": modes.get(name, "UNAVAILABLE"),
                "utf8": text,
            }
        )
    return records


def restore(bundle: dict, target: Path, allowed: tuple) -> None:
    """Validate the complete record before creating a fresh restore directory."""
    names = set()
    for record in bundle["files"]:
        name = record["path"]
        if name not in allowed or name in names:
            raise ValueError("Unexpected or duplicate relative path")
        names.add(name)
        raw = record["utf8"].encode("utf-8")
        if len(raw) != record["bytes"] or sha(raw) != record["sha256"]:
            raise ValueError("Raw byte identity mismatch")
        if (
            record["original_mode"] != "UNAVAILABLE"
            and re.fullmatch(r"100[0-7]{3}", record["original_mode"]) is None
        ):
            raise ValueError("Unsupported original mode")
    target.mkdir(parents=True, exist_ok=False)
    for record in bundle["files"]:
        path = target / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(record["utf8"].encode("utf-8"))
        if record["original_mode"] != "UNAVAILABLE":
            path.chmod(int(record["original_mode"][-3:], 8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    tasks = json.loads(lzma.decompress((HERE / "inputs.json.xz").read_bytes()))["tasks"]
    if args.restore is not None:
        if args.restore.exists():
            raise FileExistsError(args.restore)
        for key in tasks:
            path = HERE / "receipts" / f"{key}.json.xz"
            if path.is_file():
                restore(
                    json.loads(lzma.decompress(path.read_bytes())),
                    args.restore / key,
                    TASK_FILES,
                )
        return
    if args.raw is None:
        parser.error("Specify --raw or --restore")
    target = HERE / "receipts"
    target.mkdir(exist_ok=True)
    for key in tasks:
        root = args.raw / key
        mode_path = root / "modes.json"
        modes = json.loads(mode_path.read_bytes()) if mode_path.is_file() else {}
        records = inventory(root, TASK_FILES, modes)
        if not records:
            continue
        bundle = {
            "schema": "generativeqc.independent-cbs-force-raw.v1",
            "task": key,
            "files": records,
        }
        raw = (
            json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        with (target / f"{key}.json.xz").open("xb") as stream:
            stream.write(lzma.compress(raw, preset=9))


if __name__ == "__main__":
    main()
