"""CPU-only integrity/finite-value replay of the setup adapter's returned data.

This reader verifies original numeric bytes and reports finite setup outcomes.
It does not promote an endpoint to scientific or performance acceptance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
from pathlib import Path

from qz_smoke import checked_file, digest, finite_endpoint, load_json


def artifact(root: Path, name: str, checksum: str) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("numeric artifact escapes its attempt directory")
    return checked_file(path, checksum)


def same_value(display: object, original: object) -> bool:
    if isinstance(original, list):
        return (
            isinstance(display, list)
            and len(display) == len(original)
            and all(same_value(x, y) for x, y in zip(display, original, strict=True))
        )
    if isinstance(original, float):
        if not math.isfinite(original):
            return display == {"raw_nonfinite": str(original)}
        return (
            not isinstance(display, bool)
            and isinstance(display, (int, float))
            and struct.pack("<d", display) == struct.pack("<d", original)
        )
    return display == original


def reshape(flat: list, dimensions: list[int], offset: int = 0) -> object:
    if not dimensions:
        return flat[offset]
    stride = math.prod(dimensions[1:])
    return [
        reshape(flat, dimensions[1:], offset + n * stride) for n in range(dimensions[0])
    ]


def recover_sample(root: Path, sample: dict, manifest: dict) -> dict:
    values = artifact(
        root, manifest["values_file"], manifest["values_sha256"]
    ).read_bytes()
    index = load_json(artifact(root, manifest["index_file"], manifest["index_sha256"]))
    recovered = copy.deepcopy(sample)
    expected_offset = 0
    for entry in index:
        if entry["offset"] != expected_offset:
            raise ValueError("overlapping or omitted numeric bytes")
        dtype = entry["dtype"]
        code = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "u8": "Q", "u4": "I"}.get(
            dtype[1:]
        )
        if code is None or dtype[0] not in "<>=":
            raise ValueError(f"unsupported original numeric dtype: {dtype}")
        shape = entry["shape"]
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape):
            raise ValueError("invalid original array shape")
        count = math.prod(shape)
        if entry["bytes"] != count * struct.calcsize(dtype[0] + code):
            raise ValueError("original shape/byte count mismatch")
        end = expected_offset + entry["bytes"]
        if end > len(values):
            raise ValueError("truncated numeric original")
        flat = list(
            struct.unpack(dtype[0] + str(count) + code, values[expected_offset:end])
        )

        original = reshape(flat, shape)
        target = recovered
        for part in entry["field"][:-1]:
            target = target[part]
        key = entry["field"][-1]
        if not same_value(target[key], original):
            raise ValueError(f"JSON differs from numeric original at {entry['field']}")
        target[key] = original
        expected_offset = end
    if expected_offset != len(values):
        raise ValueError("unindexed numeric bytes")
    return recovered


def replay(root: Path) -> dict:
    raw = load_json(root / "raw.json")
    if raw.get("schema") != "generativeqc.h100-environment-smoke.v1":
        raise ValueError("wrong setup receipt schema")
    samples, originals = raw["samples"], raw["numeric_originals"]
    if len(samples) != len(originals):
        raise ValueError("missing original numeric artifact")
    results = []
    for sample, original in zip(samples, originals, strict=True):
        recovered = recover_sample(root, sample, original)
        atoms = len(
            raw["protocol"]["atoms_bohr"]
            if raw["arm"] == "cbs-c2"
            else raw["geometry_bohr"]
        )
        try:
            finite_endpoint(recovered["energy"], recovered["forces"], atoms)
            finite = True
        except (ValueError, TypeError):
            finite = False
        results.append(
            {"finite_complete_energy_forces": finite, "retained_original_bytes": True}
        )
    return {
        "raw_sha256": digest(root / "raw.json"),
        "arm": raw["arm"],
        "recorded_outcome": raw["status"],
        "sample_count": len(samples),
        "samples": results,
        "scientific_acceptance": "INCOMPLETE",
        "performance_result": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.attempt_directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
