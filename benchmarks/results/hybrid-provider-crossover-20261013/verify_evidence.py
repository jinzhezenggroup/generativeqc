"""Authenticate the installed RTX 5090 receipts and recompute unchanged gates.

Every oracle is bound to its original native receipt. Failed references and missing
endpoints remain incomplete; repeated timings cannot replace scientific gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.hybrid_crossover_receipts import ReceiptBundle
from tools.benchmark_hybrid_provider_crossover import summarize

HERE = Path(__file__).resolve().parent / "evidence"
CASES = ("water-96", "water-48", "formaldehyde")
ARMS = ("direct", "df-jk-occupied")


def required_paths() -> set[str]:
    required = {
        "native/campaign.json.gz",
        "build/binaries.sha256.gz",
        "build/build-identity.json.gz",
        "build/cache-after.txt.gz",
        "build/cache-before.txt.gz",
        "build/cache-version.txt.gz",
        "platform/allocation.txt",
        "platform/binaries.sha256",
        "platform/device.json",
        "platform/node2-compatibility.json",
        "platform/slurm.txt",
        "platform/source.txt",
    }
    for case in CASES:
        required.add(f"native/{case}-df-j-exact-k.json.gz")
        if case == "water-48":
            for repeat in (0, 1):
                required.add(f"oracles/{case}-direct-{repeat}.termination.json")
                required.add(f"oracles/{case}-direct-{repeat}.exit")
        for repeat in (0, 1):
            required.add(f"summaries/{case}-pair{repeat}.json.gz")
            for arm in ARMS:
                required.add(f"native/{case}-{arm}-{repeat}.json.gz")
                if case != "water-96":
                    required.add(f"oracles/{case}-{arm}-{repeat}.json.gz")
        for arm in ARMS:
            required.add(f"profiles/{case}-{arm}-profile.json.gz")
            if case != "water-96" or arm == "direct":
                required.add(f"traces/{case}-{arm}-trace.jsonl.gz")
    return required


def verify() -> dict:
    bundle = ReceiptBundle(HERE, required_paths())
    manifest = bundle.manifest
    read, raw = bundle.read, bundle.raw
    if manifest.get("schema") != "generativeqc.hybrid-provider-rtx5090-evidence.v1":
        raise ValueError("unexpected evidence schema")
    rows = manifest["files"]
    outcomes = {}
    for case in CASES:
        profiles = [read(f"profiles/{case}-{arm}-profile.json.gz") for arm in ARMS]
        for profile in profiles:
            # An authentic failed endpoint must still belong to this campaign;
            # an INCOMPLETE pair cannot hide a foreign diagnostic receipt.
            if (
                profile.get("source")
                != {
                    "revision": manifest["source_revision"],
                    "dirty": False,
                }
                or profile.get("library", {}).get("sha256")
                != manifest["library_sha256"]
            ):
                raise ValueError("diagnostic profile source/library identity mismatch")
        for repeat in (0, 1):
            native = [read(f"native/{case}-{arm}-{repeat}.json.gz") for arm in ARMS]
            native.append(read(f"native/{case}-df-j-exact-k.json.gz"))
            for row in native:
                if row["source"] != {
                    "revision": manifest["source_revision"],
                    "dirty": False,
                }:
                    raise ValueError("scientific source identity mismatch")
                if (
                    row.get("library")
                    and row["library"]["sha256"] != manifest["library_sha256"]
                ):
                    raise ValueError("installed library identity mismatch")
            oracles = []
            if case != "water-96":
                for arm in ARMS:
                    oracle = read(f"oracles/{case}-{arm}-{repeat}.json.gz")
                    native_bytes = raw(f"native/{case}-{arm}-{repeat}.json.gz")
                    if (
                        oracle["native_record_sha256"]
                        != hashlib.sha256(native_bytes).hexdigest()
                    ):
                        raise ValueError(
                            "oracle is bound to a different native receipt"
                        )
                    if case == "water-48" and arm == "direct":
                        failure = read(
                            f"oracles/{case}-{arm}-{repeat}.termination.json"
                        )
                        oracle_bytes = raw(f"oracles/{case}-{arm}-{repeat}.json.gz")
                        if (
                            failure.get("status")
                            != "INTERRUPTED_AFTER_REQUIRED_REFERENCE_FAILURE"
                            or failure.get("oracle_record_sha256")
                            != hashlib.sha256(oracle_bytes).hexdigest()
                            or not any(
                                s.get("geometry") == "original"
                                and s.get("converged") is False
                                and s.get("status") == "FAIL"
                                for s in oracle.get("samples", [])
                            )
                        ):
                            raise ValueError(
                                "Direct 48 reference failure disposition is missing"
                            )
                    oracles.append(oracle)
            result = summarize(native, oracles, profiles)
            if result != read(f"summaries/{case}-pair{repeat}.json.gz"):
                raise ValueError("stored summary disagrees with raw scientific gates")
            expected = "PILOT_ACCEPTED" if case == "formaldehyde" else "INCOMPLETE"
            if result["status"] != expected or result["crossover_claim_eligible"]:
                raise ValueError("invalid crossover or acceptance claim")
            if case == "water-48":
                # One rejected arm must not hide regression of the independently
                # qualified DF arm in an overall INCOMPLETE pair summary.
                errors = result["independent_oracle_errors"].get("df-jk-occupied", [])
                if (
                    native[0].get("status") != "MEASURED"
                    or native[1].get("status") != "MEASURED"
                    or profiles[1].get("occupied_reuse_verified") is not True
                    or {e["phase"] for e in errors}
                    != {"cold", "warm-0", "changed-geometry", "moved-warm"}
                    or any(
                        not math.isfinite(e["energy_error"])
                        or not math.isfinite(e["force_max_abs_error"])
                        or e["energy_error"] > result["gates"]["energy_hartree"]
                        or e["force_max_abs_error"]
                        > result["gates"]["force_max_abs_hartree_per_bohr"]
                        for e in errors
                    )
                ):
                    raise ValueError(
                        "the completed 48-atom DF qualification was not retained"
                    )
            if case == "water-96":
                # These failures belong to this frozen capacity experiment;
                # later resource fixes must retain it and add a new population.
                if not any(
                    "out of memory" in a["status_message"]
                    for a in native[0]["attempts"]
                ):
                    raise ValueError("Direct 96 force OOM was not retained")
                if "DF memory budget cannot hold" not in native[1].get("error", ""):
                    raise ValueError("DF 96 preparation rejection was not retained")
            outcomes[f"{case}/pair{repeat}"] = result["status"]
    return {
        "members": len(rows),
        "outcomes": outcomes,
        "crossover_claim_eligible": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
