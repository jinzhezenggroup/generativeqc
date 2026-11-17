"""Recompute the bounded stock/tight GPU4PySCF accuracy comparison.

This historical experiment does not promote a default or benchmark an optimized
GPU4PySCF installation. Timings retain the installed CuPy contraction fallback.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
from collections import defaultdict
from pathlib import Path


def replay(path: Path) -> dict:
    """Check identities and every force component, retaining failed samples."""
    bundle = json.loads(lzma.decompress(path.read_bytes()))
    assert bundle["schema"] == "generativeqc.gpu4pyscf-force-comparator/v1"
    files = bundle["files"]
    for entry in files.values():
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
    recipe_sha = files["endpoint.py"]["sha256"]
    groups = defaultdict(list)
    endpoint_count = 0
    for name in sorted(files):
        if not (name.startswith(("pilot/", "qualified/")) and name.endswith(".json")):
            continue
        record = json.loads(files[name]["text"])
        if name.startswith("pilot/"):
            basis, arm = Path(name).stem, "gpu4pyscf-default"
        else:
            basis, _, label = Path(name).stem.rsplit("-", 2)
            arm = (
                f"gpu4pyscf-{label}"
                if label in ("default", "strict")
                else f"repair-{label}"
            )
        fixture_entry = files[f"fixtures/rhf-direct-{basis}.json"]
        fixture = json.loads(fixture_entry["text"])
        assert record["fixture_sha256"] == fixture_entry["sha256"]
        assert record["slurm_job_id"] in ("17690", "17691")
        assert record["device_uuid"] == bundle["device_uuid"]
        assert record["cuda_visible_devices"]
        if arm.startswith("gpu4pyscf-"):
            assert record["recipe_sha256"] == recipe_sha
            assert record["versions"]["gpu4pyscf-cuda12x"] == "1.8.1"
            assert record["controls"]["conv_tol"] == 1e-9
            assert record["controls"]["conv_tol_grad"] == (
                1e-9 if arm.endswith("strict") else None
            )
        else:
            assert (
                record["native_probe"]["source_identity"]
                == bundle["repair_source_identity"]
            )
            assert (
                record["library_sha256"]
                == bundle["repair_library_sha256"][record["profile"]]
            )
        assert [sample["phase"] for sample in record["samples"]] == [
            "cold",
            "warm",
            "moved",
            "moved_warm",
            "energy_only",
            "energy_to_forces",
        ]
        for sample in record["samples"]:
            expected = fixture["oracle"]["records"][sample["geometry"]]
            errors = {"energy": abs(sample["energy"] - expected["energy"])}
            forces = sample["forces"]
            if forces is not None:
                assert len(forces) == len(expected["forces"])
                assert all(len(row) == 3 for row in forces)
                errors["forces"] = max(
                    abs(value - reference)
                    for row, reference_row in zip(
                        forces, expected["forces"], strict=True
                    )
                    for value, reference in zip(row, reference_row, strict=True)
                )
                errors["translation"] = max(
                    abs(sum(row[axis] for row in forces)) for axis in range(3)
                )
            valid = bool(
                sample["converged"]
                and all(
                    error <= (1e-9 if key == "energy" else 1e-7)
                    for key, error in errors.items()
                )
            )
            assert errors == sample["errors"]
            assert valid == sample["scientifically_valid"]
            assert sample["endpoint_ns"] == sample["execute_ns"] + (
                record["prepare_ns"] if sample["phase"] == "cold" else 0
            )
            endpoint_count += 1
            groups[(arm, basis, sample["phase"])].append({**sample, "errors": errors})
    summaries = []
    for (arm, basis, phase), samples in sorted(groups.items()):
        assert len(samples) == (3 if arm == "gpu4pyscf-default" else 2)
        summaries.append(
            {
                "arm": arm,
                "basis": basis,
                "phase": phase,
                "samples": len(samples),
                "valid_samples": sum(
                    sample["scientifically_valid"] for sample in samples
                ),
                "max_errors": {
                    key: max(sample["errors"][key] for sample in samples)
                    for key in samples[0]["errors"]
                },
                "median_endpoint_ms": statistics.median(
                    sample["endpoint_ns"] for sample in samples
                )
                / 1e6,
                "iterations": sorted({sample["iterations"] for sample in samples}),
                "work": [
                    sample["scf_call_counts"]
                    if arm.startswith("gpu4pyscf-")
                    else {"fock_builds": sample["fock_builds"]}
                    for sample in samples
                ],
            }
        )
    assert endpoint_count == 108
    return {
        "endpoints": endpoint_count,
        "groups": summaries,
        "scope": "accuracy of two direct-RHF water cases on one RTX5090; no all-system accuracy guarantee or optimized-reference speed ranking",
    }


if __name__ == "__main__":
    print(json.dumps(replay(Path(__file__).with_name("gpu4pyscf.json.xz")), indent=2))
