"""Offline integrity and scientific-gate recomputation for #2054 pilot evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.benchmark_hybrid_provider_crossover import summarize

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
MANIFEST = EVIDENCE / "manifest.json"
SCHEMA = "generativeqc.hybrid-provider-evidence-manifest.v1"
CASES = ("water-48", "formaldehyde")
SUPPORTED = ("direct", "df-jk-occupied")


def read(path: Path) -> dict:
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    return json.loads(raw)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def required_paths() -> set[str]:
    paths = {"platform/job-metrics.json"}
    for case in CASES:
        paths.add(f"native/{case}-df-j-exact-k.json.gz")
        for arm in SUPPORTED:
            paths.add(f"profiles/{case}-{arm}-profile.json.gz")
            for repeat in (0, 1):
                paths.add(f"native/{case}-{arm}-{repeat}.json.gz")
                if case != "water-48" or arm != "direct":
                    paths.add(f"oracles/{case}-{arm}-{repeat}.json.gz")
        for repeat in (0, 1):
            paths.add(f"summaries/{case}-pair{repeat}.json")
    return paths


def verify() -> dict:
    manifest = read(MANIFEST)
    if manifest.get("schema") != SCHEMA or manifest.get("source_dirty") is not False:
        raise ValueError("wrong #2054 evidence manifest schema")
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or {row.get("path") for row in entries} != required_paths()
    ):
        raise ValueError("#2054 retained evidence inventory is incomplete")
    if len(entries) != len(required_paths()):
        raise ValueError("duplicate #2054 evidence member")
    for row in entries:
        relative = row["path"]
        path = (EVIDENCE / relative).resolve()
        if not path.is_relative_to(EVIDENCE.resolve()) or not path.is_file():
            raise ValueError(f"unsafe or missing #2054 evidence path: {relative}")
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"#2054 evidence checksum mismatch: {relative}")
        if path.suffix == ".gz":
            content = gzip.decompress(path.read_bytes())
            if len(content) != row.get("content_bytes") or hashlib.sha256(
                content
            ).hexdigest() != row.get("content_sha256"):
                raise ValueError(f"#2054 decompressed evidence mismatch: {relative}")

    metrics = read(EVIDENCE / "platform/job-metrics.json")
    if (
        metrics.get("success") is not True
        or metrics.get("data", {}).get("name") != "i2054-h200-48-holdout"
        or not any(
            row.get("metric") == "gpu_memory_usage_rate"
            and isinstance(row.get("max"), (int, float))
            and 0 <= row["max"] <= 1
            for row in metrics.get("data", {}).get("series", ())
        )
    ):
        raise ValueError("missing bounded sampled H200 memory receipt")

    outcomes = {}
    for case in CASES:
        for repeat in (0, 1):
            native = [
                read(EVIDENCE / f"native/{case}-{arm}-{repeat}.json.gz")
                for arm in SUPPORTED
            ]
            for record in native:
                if (
                    record.get("source", {}).get("revision")
                    != manifest["source_revision"]
                    or record.get("source", {}).get("dirty")
                    or record.get("library", {}).get("sha256")
                    != manifest["installed_library_sha256"]
                ):
                    raise ValueError(
                        f"#2054 {case} pair{repeat} source/binary mismatch"
                    )
            native.append(read(EVIDENCE / f"native/{case}-df-j-exact-k.json.gz"))
            oracles = [
                read(EVIDENCE / f"oracles/{case}-{arm}-{repeat}.json.gz")
                for arm in SUPPORTED
                if (EVIDENCE / f"oracles/{case}-{arm}-{repeat}.json.gz").is_file()
            ]
            for oracle in oracles:
                native_path = (
                    EVIDENCE / f"native/{case}-{oracle['arm']}-{repeat}.json.gz"
                )
                native_raw = gzip.decompress(native_path.read_bytes())
                if (
                    oracle.get("native_record_sha256")
                    != hashlib.sha256(native_raw).hexdigest()
                ):
                    raise ValueError(f"#2054 {case} pair{repeat} oracle/raw mismatch")
            profiles = [
                read(EVIDENCE / f"profiles/{case}-{arm}-profile.json.gz")
                for arm in SUPPORTED
            ]
            recomputed = summarize(native, oracles, profiles)
            stored = read(EVIDENCE / f"summaries/{case}-pair{repeat}.json")
            if recomputed != stored:
                raise ValueError(
                    f"#2054 {case} pair{repeat} summary differs from raw evidence"
                )
            expected = "INCOMPLETE" if case == "water-48" else "PILOT_ACCEPTED"
            if stored["status"] != expected or stored["crossover_claim_eligible"]:
                raise ValueError(
                    f"#2054 {case} pair{repeat} has an invalid acceptance claim"
                )
            if case == "water-48":
                failed = native[0]
                if failed.get("status") != "FAILED" or any(
                    item.get("forces_hartree_per_bohr") is not None
                    for item in failed.get("attempts", ())
                ):
                    raise ValueError("Direct 48 force failure was not retained")
                if not any(
                    "prepared native integral derivatives are unavailable"
                    in item.get("status_message", "")
                    for item in failed.get("attempts", ())
                ):
                    raise ValueError("Direct 48 capacity blocker was not retained")
                if native[1].get("status") != "MEASURED":
                    raise ValueError("DF-JK 48 source did not complete")
            outcomes[f"{case}/pair{repeat}"] = {
                "status": stored["status"],
                "failures": stored["failures"],
                "independent_oracle_errors": stored["independent_oracle_errors"],
            }
    return {
        "source_revision": manifest["source_revision"],
        "installed_library_sha256": manifest["installed_library_sha256"],
        "members": len(entries),
        "outcomes": outcomes,
        "crossover_claim_eligible": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
