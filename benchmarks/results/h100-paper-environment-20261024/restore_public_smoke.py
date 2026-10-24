"""Restore exact H100 setup returns and optionally replay them on a CPU.

This opt-in artifact utility never starts a scientific or reference calculation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
CAPSULE = HERE / "public-device-originals-v1.json.xz"
CAPSULE_SHA256 = "a42b1ad8e5f5752a7b446dd81ff65921a549b0f2e7f5b7459211a5a7a7746db5"
MAX_BYTES = 96 << 20
MAX_MEMBER_BYTES = 16 << 20


def validated_members(payload: dict) -> dict:
    if payload.get("schema") != "generativeqc.h100-actual-output-export.v1":
        raise ValueError("wrong actual-output schema")
    source = PurePosixPath(payload["output_root"]).parents[1]
    rows = payload["files"]
    if not source.is_absolute() or not 0 < len(rows) <= 128:
        raise ValueError("invalid source root or member count")
    result = {}
    total = 0
    for row in rows:
        name = row["path"]
        path = PurePosixPath(name)
        if ".." in path.parts or "\\" in name:
            raise ValueError("unsafe original path")
        relative = path.relative_to(source)
        if not relative.parts or any(":" in x for x in relative.parts):
            raise ValueError("unsafe restored path")
        if relative.name == ".env" or relative.as_posix() in result:
            raise ValueError("forbidden or repeated retained path")
        size = row["bytes"]
        if type(size) is not int or not 0 <= size <= MAX_MEMBER_BYTES:
            raise ValueError("retained member exceeds the size bound")
        encoded = row["bytes_b64"]
        if len(encoded) > ((MAX_MEMBER_BYTES + 2) // 3) * 4:
            raise ValueError("oversized encoded member")
        data = base64.b64decode(encoded, validate=True)
        if len(data) != size or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("retained original byte/hash mismatch")
        mode = int(row["mode"], 8)
        if not 0 <= mode <= 0o777:
            raise ValueError("invalid retained file mode")
        total += size
        if total > MAX_BYTES:
            raise ValueError("retained originals exceed the aggregate bound")
        result[relative.as_posix()] = (data, mode)
    return result


def members(capsule: Path = CAPSULE) -> dict:
    if capsule.stat().st_size > MAX_MEMBER_BYTES:
        raise ValueError("oversized compressed capsule")
    packed = capsule.read_bytes()
    if hashlib.sha256(packed).hexdigest() != CAPSULE_SHA256:
        raise ValueError("frozen actual-output capsule hash mismatch")
    decoder = lzma.LZMADecompressor()
    raw = decoder.decompress(packed, max_length=MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("oversized, truncated or concatenated capsule")
    return validated_members(json.loads(raw))


def restore(destination: Path | None = None, capsule: Path = CAPSULE) -> dict:
    files = members(capsule)
    if destination is not None:
        destination.mkdir(exist_ok=False)
        for name, (data, mode) in files.items():
            path = destination.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name == "posix":
                path.chmod(mode)
            if path.read_bytes() != data:
                raise ValueError("restored original readback mismatch")
    return {
        "raw_files": len(files),
        "raw_bytes": sum(len(data) for data, _ in files.values()),
        "original_byte_integrity": "PASS",
        "scope": "offline recovery only; no scientific execution",
    }


def replay_all(destination: Path) -> dict:
    import oracle_replay
    import replay

    hf = oracle_replay.reference(
        HERE.parent / "readme-repaired-20261021/oracles.json.xz",
        oracle_replay.HF_REFERENCE_SHA,
    )
    cbs = oracle_replay.reference(
        HERE.parent / "cbs-fast-memory-20261021/summary.json.xz",
        oracle_replay.CBS_REFERENCE_SHA,
    )
    result = {}
    for endpoint in sorted(destination.glob("output/*/*-process/endpoint")):
        raw = replay.load_json(endpoint / "raw.json")
        if len(raw["samples"]) != 1 or len(raw["numeric_originals"]) != 1:
            raise ValueError("expected one returned sample per setup arm")
        sample = replay.recover_sample(
            endpoint, raw["samples"][0], raw["numeric_originals"][0]
        )
        comparison = (
            oracle_replay.compare_cbs(raw, sample, cbs)
            if raw["arm"] == "cbs-c2"
            else oracle_replay.compare_hf(raw, sample, hf)
        )
        result[raw["arm"]] = {
            "numeric_replay": replay.replay(endpoint),
            "reference_comparison": comparison,
        }
    if set(result) != {"native-hf", "gpu4pyscf-hf", "cbs-c2"}:
        raise ValueError("missing or extra actual setup arm")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay and args.destination is None:
        parser.error("--replay requires a new --destination")
    report = restore(args.destination)
    if args.replay:
        report["replay"] = replay_all(args.destination)
    print(json.dumps(report, indent=2, allow_nan=False))
