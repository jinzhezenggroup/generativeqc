"""Authenticate the retained diagnostic and recompute its work-count summaries.

This offline check neither loads CUDA nor establishes molecular E/F accuracy or
performance. Embedded probe/script text is evidence and is never executed here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FIELDS = (
    "candidates",
    "admitted_ao_quartets",
    "primitive_product_extent",
    "unique_shell_quartets",
    "shell_primitive_product_extent",
    "warp_shell_groups",
    "warp_primitive_product_extent",
)


def require(condition: bool, message: str) -> None:
    """Keep all integrity gates active under Python optimization too."""
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(directory: Path) -> dict:
    """Check retained receipts and aggregate formulas without inferring timings."""
    summary = json.loads((directory / "summary.json").read_bytes())
    raw = (directory / "evidence.json").read_bytes()
    require(digest(raw) == summary["evidence_sha256"], "evidence digest differs")
    evidence = json.loads(raw)
    files = evidence["files"]
    for name, member in files.items():
        data = member["text"].encode()
        require(len(data) == member["bytes"], f"member length differs: {name}")
        require(digest(data) == member["sha256"], f"member digest differs: {name}")

    def text(name: str) -> str:
        return files[name]["text"]

    def load(name: str) -> dict:
        return json.loads(text(name))

    require(load("run/outcome.json") == {"exit_code": 0, "job": "5748"}, "run failed")
    require(load("build/build-outcome.json")["exit_code"] == 0, "build failed")
    require(summary["job"] == "5748" and summary["node"] == "n1", "run differs")
    identity = load("run/source-identity.json")
    require(identity["source_identity"] == summary["source_identity"], "source differs")
    require(identity["commit"] == summary["frozen_commit"], "commit differs")
    native_path = "build/cuda-release-sm120/libgenerativeqc.so"
    native_manifest = dict(
        line.split(maxsplit=1)[::-1]
        for line in text("native/binaries.sha256").splitlines()
    )
    require(native_manifest[native_path] == summary["library_sha256"], "library differs")
    require(
        f"{native_path}: OK" in text("run/native-binaries.txt"),
        "native library check did not pass",
    )
    manifest = text("run/files.sha256").splitlines()
    expected_files = dict(line.split(maxsplit=1)[::-1] for line in manifest)
    require(expected_files["census.cpp"] == summary["probe_sha256"], "probe differs")
    require(
        expected_files["census"] == summary["probe_binary_sha256"],
        "probe executable differs",
    )
    require(
        files["probe/census.cpp"]["sha256"] == summary["probe_sha256"],
        "archived probe source differs",
    )
    provenance = evidence["input_provenance"]
    require(
        expected_files["input.txt"]
        == files["probe/input.txt"]["sha256"]
        == provenance["input_sha256"],
        "input differs",
    )
    require(
        provenance == load("run/input-provenance.json"), "input provenance differs"
    )
    census = load("run/census.json")
    require(census == summary["census"], "raw census differs")
    protocol = provenance["protocol"]
    require(
        (census["atoms"], census["public_aos"], summary["basis"])
        == (protocol["atoms"], protocol["aos"], protocol["basis"]),
        "scientific shape differs",
    )
    rows = census["orders"]
    require([row["order"] for row in rows] == list(range(13)), "orders differ")
    for row in rows:
        require(
            all(type(row[key]) is int and row[key] >= 0 for key in FIELDS),
            "missing or invalid work count",
        )
        require(
            row["unique_shell_quartets"]
            <= row["warp_shell_groups"]
            <= row["admitted_ao_quartets"]
            <= row["candidates"],
            "invalid quartet grouping",
        )
        require(
            row["shell_primitive_product_extent"]
            <= row["warp_primitive_product_extent"]
            <= row["primitive_product_extent"],
            "invalid primitive grouping",
        )
        require(
            sum(row["ao_quartets_by_unique_atoms"]) == row["admitted_ao_quartets"],
            "unique-atom distribution differs",
        )
    total = sum(row["candidates"] for row in rows)
    admitted = sum(row["admitted_ao_quartets"] for row in rows)
    for prefix, label in (("", "full"), ("lr_", "LR")):
        require(census[prefix + "actual_candidates"] == total, "candidate gate differs")
        require(
            census[prefix + "actual_radial_evaluations"] == admitted,
            "actual radial gate differs",
        )
        require(
            f"{label} resident counters: {total}/{admitted}; host candidate/admitted enumeration: {total}/{admitted}"
            in text("run/census.log"),
            "native count log differs",
        )
    for label, subset in (("all", rows), ("orders5-8", rows[5:9])):
        expected = {key: sum(row[key] for row in subset) for key in FIELDS}
        for key, divisor in (
            ("shell_preparation_reuse_bound", "shell_primitive_product_extent"),
            ("current_warp_preparation_reuse_bound", "warp_primitive_product_extent"),
        ):
            expected[key] = expected["primitive_product_extent"] / expected[divisor]
        require(expected == summary["aggregates"][label], "derived work differs")
    require(census["flops"] is None, "census does not measure FLOPs")
    for key in (
        "flops",
        "force_work",
        "dynamic_traffic",
        "complete_endpoint_seconds",
        "speedup",
    ):
        require(summary[key] is None, f"unmeasured quantity must remain null: {key}")
    return {"job": summary["job"], "admitted_per_operator": admitted, "scope": summary["scope"]}


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    print(json.dumps(verify(directory), indent=2))
