"""Verify every retained E/F pairing and the complete, unnormalized timing rows.

Run from the repository root with PYTHONPATH=python:.; no GPU is needed.
This does not qualify an automatic provider/profile or establish direct parity.
"""

from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path

from benchmarks.readme_omol25 import check_record
from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parent


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def read(name: str) -> dict:
    data = (ROOT / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def verify() -> dict:
    """Recompute numerical maxima and validate work, histories and provenance."""
    manifest = read("publication.json")
    validate_publication(
        manifest,
        {row["path"]: (ROOT / row["path"]).read_bytes() for row in manifest["files"]},
    )
    summary = read("summary.json")
    receipts = read("receipts.json")
    require(receipts["endpoint/job.exit"].strip() == "0", "incomplete Slurm campaign")
    maxima = {"energy": 0.0, "force": 0.0, "pairings": 0}
    for atoms in (48, 96):
        reference = read(f"reference-{atoms}.json.gz")
        require(
            reference["status"] == "measured" and len(reference["records"]) == 12,
            "incomplete independent reference",
        )
        for arm in ("control", "candidate"):
            run = read(f"{arm}-{atoms}.json.gz")
            require(
                run["status"] == "measured" and len(run["records"]) == 12,
                "incomplete native endpoint",
            )
            require(
                run["protocol"] == reference["protocol"], "scientific protocol differs"
            )
            require(
                run["environment"]["git"]["commit"] == summary[f"{arm}_head"]
                and run["environment"]["git"]["dirty"] is False,
                "source provenance differs",
            )
            receipt = receipts[f"endpoint/source-{arm}-{atoms}.txt"].split()
            require(
                receipt
                == [
                    "SOURCE_MATCH",
                    summary[f"{arm}_head"],
                    run["native_build"]["probe"]["source_identity"],
                ],
                "source/library receipt differs",
            )
            for row in run["records"]:
                diagnostic = row["native_ks_diagnostic"]
                require(
                    diagnostic["fock_builds"]
                    == row["fock_builds"]
                    == len(diagnostic["history"]),
                    "actual SCF history is incomplete",
                )
                require(
                    0 <= row["physical_residual_rms"] <= 1e-10,
                    "physical residual failed",
                )
                force = row["native_force_components"]
                require(
                    force["stationary_integral_derivative_route"]
                    == "prepared-native-complete",
                    "native derivative route missing",
                )
                work = force["work_counts"]["observed"]
                grid = force["grid_work_plan"]
                require(
                    work["geometry_batches"]
                    == work["phased_becke_batches"]
                    == grid["tile_count"],
                    "phased work changed",
                )
                require(
                    work["becke_pair_state_evaluations"]
                    == grid["grid_points"] * atoms * (atoms - 1) // 2,
                    "pair production changed",
                )
                require(
                    force["force_active_ao_policy"]["profile_id"]
                    == "sm120-ordinary-rks-second-jet-v1",
                    "composed qualification profile missing",
                )
                for oracle in reference["records"]:
                    if oracle["geometry"] != row["geometry"]:
                        continue
                    result = check_record(row, oracle)
                    require(result["gate"], "independent all-sample E/F gate failed")
                    maxima["energy"] = max(maxima["energy"], result["energy_error"])
                    maxima["force"] = max(maxima["force"], result["force_error"])
                    maxima["pairings"] += 1
            for phase, count in (
                ("cold", 1),
                ("warm", 5),
                ("moved", 1),
                ("moved-warm", 5),
            ):
                rows = [r for r in run["records"] if r["phase"] == phase]
                require(len(rows) == count, "phase inventory differs")
                retained = summary["cases"][str(atoms)][phase][arm]
                seconds = [r["complete_seconds"] for r in rows]
                require(
                    retained["seconds"] == seconds
                    and retained["median_seconds"] == statistics.median(seconds),
                    "timing summary differs",
                )
                require(
                    retained["iterations"] == [r["iterations"] for r in rows],
                    "SCF count changed",
                )
    require(maxima["pairings"] == 288, "not every independent pairing was checked")
    transplant = read("shipping-transplant.json")
    retained = transplant["receipts"]
    require(transplant["endpoint_measured"] is False, "transplant scope changed")
    require(retained["gpu/job.exit"].strip() == "0", "transplant GPU gate incomplete")
    require(
        retained["gpu/source-check.txt"].split()
        == ["SOURCE_MATCH", transplant["revision"], transplant["source_identity"]],
        "transplant source receipt differs",
    )
    require(
        retained["build/source-head.txt"].strip() == transplant["revision"]
        and retained["build/source-identity.txt"].strip()
        == transplant["source_identity"],
        "transplant build identity differs",
    )
    for sanitizer in ("memcheck", "initcheck"):
        require(
            "ERROR SUMMARY: 0 errors" in retained[f"gpu/{sanitizer}.log"],
            "transplant sanitizer gate failed",
        )
    require(
        "through-f derivatives PASS" in retained["gpu/native-through-f.log"],
        "transplant native gate missing",
    )
    require("47 passed" in retained["host-focused.log"], "transplant host gate missing")
    require(
        len(retained["build/aot-binaries.sha256"].splitlines()) == 12,
        "six-profile AOT build receipt missing",
    )
    return maxima


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
