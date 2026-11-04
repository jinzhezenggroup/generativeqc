"""Replay retained CPU receipts without executing or timing molecular work."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_publication(root: Path) -> None:
    """Authenticate scripts and receipts before replaying the retained packet."""
    manifest = json.loads((root / "publication.json").read_text())
    for member in manifest["files"]:
        path = root / member["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError("publication member escapes its directory")
        content = path.read_bytes()
        if len(content) != member["bytes"] or (
            hashlib.sha256(content).hexdigest() != member["sha256"]
        ):
            raise ValueError(f"publication member changed: {member['path']}")


def frozen_sources(root: Path) -> dict[str, bytes]:
    """Restore lossless source text without making frozen bytes formatter inputs."""
    with lzma.open(root / "frozen-sources.json.xz", "rt") as stream:
        archive = json.load(stream)
    if archive["schema"] != "generativeqc.frozen-source-text.v1":
        raise ValueError("unsupported frozen source container")
    sources = {}
    for entry in archive["files"]:
        name = entry["path"]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in sources:
            raise ValueError("unsafe or duplicate frozen source path")
        content = entry["content"].encode()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"frozen source bytes changed: {name}")
        sources[name] = content
    return sources


def restore_records(destination: Path, records: list[dict]) -> None:
    """Restore exact original JSON bytes and reject unsafe or altered members."""
    names = set()
    for record in records:
        name = record["name"]
        if Path(name).name != name or name in names:
            raise ValueError("unsafe or duplicate receipt name")
        names.add(name)
        if record["format"] == "jsonl":
            encoded = "".join(
                json.dumps(value, allow_nan=False) + "\n" for value in record["value"]
            )
        elif record["format"] == "json":
            encoded = (
                json.dumps(record["value"], indent=record["indent"], allow_nan=False)
                + "\n"
            )
        else:
            raise ValueError("unknown receipt format")
        content = encoded.encode()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"receipt bytes changed: {name}")
        (destination / name).write_bytes(content)


def main() -> int:
    """Keep an incomplete campaign incomplete, including its original failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-sources", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify_publication(root)
    sources = frozen_sources(root)
    if arguments.extract_sources is not None:
        arguments.extract_sources.mkdir(parents=True, exist_ok=False)
        for name, content in sources.items():
            destination = arguments.extract_sources / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return 0
    with lzma.open(root / "samples.json.xz", "rt") as stream:
        samples = json.load(stream)
    if samples["schema"] != "generativeqc.required-native-policy-receipts.v1":
        raise ValueError("unsupported receipt container")
    with tempfile.TemporaryDirectory(prefix="native-policy-replay-") as temporary:
        restored = Path(temporary)
        previous = restored / "failed-protocol-v1"
        previous.mkdir()
        auditor = restored / "audit_receipts.py"
        auditor.write_bytes(sources["audit_receipts.py.txt"])
        restore_records(restored, samples["protocol_v2"])
        restore_records(previous, samples["protocol_v1"])
        replay = subprocess.run(
            [sys.executable, str(auditor), str(restored)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if replay.returncode not in (0, 2):
            sys.stderr.write(replay.stderr)
            sys.stdout.write(replay.stdout)
            return replay.returncode
        report = json.loads(replay.stdout)
        with lzma.open(root / "summary.json.xz", "rt") as stream:
            expected = json.load(stream)["receipt_replay"]
        if report != expected or replay.returncode != 2:
            raise ValueError("retained incomplete qualification result changed")
        print(json.dumps(report, indent=2, allow_nan=False))
        return replay.returncode


if __name__ == "__main__":
    raise SystemExit(main())
