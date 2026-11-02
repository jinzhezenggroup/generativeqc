"""Authenticate rejected pilot evidence and replay its independent gates."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]


def checked(raw: bytes, digest: str, name: str) -> bytes:
    """Reject altered bytes even when Python assertions are disabled."""
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"evidence checksum mismatch: {name}")
    return raw


def main() -> None:
    """Recompute every saved gate without rewriting original receipts."""
    folder = Path(__file__).parent
    archive = checked(
        (folder / "receipts.json.xz").read_bytes(),
        (folder / "sha256.txt").read_text().split()[0],
        "receipts.json.xz",
    )
    bundle = json.loads(lzma.decompress(archive))
    if bundle["schema"] != "generativeqc.rejected-xc-map-groups.v1":
        raise ValueError("unsupported evidence schema")
    members = {
        name: checked(entry["text"].encode(), entry["sha256"], name).decode()
        for name, entry in bundle["files"].items()
    }
    references = {
        name: json.loads(
            lzma.decompress(checked((ROOT / name).read_bytes(), digest, name))
        )
        for name, digest in bundle["references"].items()
    }
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]
    saved = json.loads(members["pilot-qualification.json"])
    for row in saved["checks"]:
        name = row["file"].split("xc-cooperative-map-groups-20261101/", 1)[1]
        device = name.split("-", 1)[0]
        record = json.loads(members[name])
        expected_identity = bundle[
            "candidate_identity" if "/candidate-" in name else "parent_identity"
        ]
        if record["native_build"]["source_identity"] != expected_identity:
            raise ValueError(f"source identity mismatch: {name}")
        actual = qualify(
            [record],
            oracles,
            reference_reuse=reuse[device]["analysis"]["reference_reuse"],
        )[0]
        if actual != row["qualification"] or actual["status"] != "qualified":
            raise ValueError(f"independent gate mismatch: {name}")
    if len(saved["checks"]) != 24:
        raise ValueError("incomplete pilot population census")
    for device, job in (("5090", "1395"), ("pro6000", "3823")):
        prefix = f"{device}-pilot-{job}/"
        for checker in ("synccheck", "memcheck"):
            if "ERROR SUMMARY: 0 errors" not in members[prefix + checker + ".log"]:
                raise ValueError(f"missing successful {device} {checker}")
        if members[prefix + "exit-code.txt"].strip() != "0":
            raise ValueError(f"failed {device} scheduler population")
    print(members["qualification-summary.txt"], end="")
    print("All original member hashes and 24 independent qualifications verified.")


if __name__ == "__main__":
    main()
