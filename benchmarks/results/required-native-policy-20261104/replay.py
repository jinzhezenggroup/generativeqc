"""Replay retained CPU receipts without executing or timing molecular work."""

from __future__ import annotations

import gzip
import hashlib
import json
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
    root = Path(__file__).resolve().parent
    verify_publication(root)
    with gzip.open(root / "samples.json.gz", "rt") as stream:
        samples = json.load(stream)
    if samples["schema"] != "generativeqc.required-native-policy-receipts.v1":
        raise ValueError("unsupported receipt container")
    with tempfile.TemporaryDirectory(prefix="native-policy-replay-") as temporary:
        restored = Path(temporary)
        previous = restored / "failed-protocol-v1"
        previous.mkdir()
        restore_records(restored, samples["protocol_v2"])
        restore_records(previous, samples["protocol_v1"])
        replay = subprocess.run(
            [sys.executable, str(root / "audit_receipts.py.txt"), str(restored)],
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
        expected = json.loads((root / "summary.json").read_text())["receipt_replay"]
        if report != expected or replay.returncode != 2:
            raise ValueError("retained incomplete qualification result changed")
        print(json.dumps(report, indent=2, allow_nan=False))
        return replay.returncode


if __name__ == "__main__":
    raise SystemExit(main())
