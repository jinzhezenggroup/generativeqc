"""Recompute all retained endpoint gates separately from the intrusive census."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import audit_census
from generativeqc_compiler.common.evidence import block_error, canonical_hash

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_record

ROOT = Path(__file__).resolve().parent
IDENTITIES = {
    "control": (
        "7e5343ff567deae29e4feb6232cdb8620e35ec4f",
        "cb81f4c481e5cadd85999a7ff791f93a95a61f9fcf7bb5917ba3c3bfffa849b8",
        "3a119b7400171754d50fdf1a8bc3f27ce20c8d3b6a3780fb8101aaca138ff69c",
    ),
    "candidate": (
        "5f957d696a8786d5fb0eb319145289bea9225c34",
        "35a26a50a7e794b2b21ac9b5210fdc6af4d7b93bbf5dc37d28831b68951bf316",
        "5cf7ed0ffe1c871930f95c02bb3704b005cdd996b1d917c1a454d311e94843aa",
    ),
}
PHASES = ["cold", *(["warm"] * 5), "moved", *(["moved-warm"] * 5)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(name: str) -> dict:
    """Read lossless stored observations without rewriting their scientific hashes."""
    return load_record(ROOT / name)


def compare(actual: dict, reference: dict) -> dict:
    """Check every same-geometry pairing, including the five individual replays."""
    pairs = []
    for row in actual["records"]:
        for oracle in reference["records"]:
            if row["geometry"] != oracle["geometry"]:
                continue
            errors = {
                key: block_error([row[key]], [oracle[key]], atol=atol, rtol=0)
                for key, atol in [("energy", 1e-8), ("forces", 1e-7)]
            }
            require(all(e["passed"] for e in errors.values()), "E/F gate failed")
            pairs.append(errors)
    require(len(pairs) == 72, "pairing inventory changed")
    return {
        "pairings": len(pairs),
        **{
            f"max_{key}_error": max(e[key]["max_absolute_error"] for e in pairs)
            for key in ("energy", "forces")
        },
    }


def analyze() -> dict:
    """Do not infer missing clean-endpoint work counts from the separate probe."""
    receipts = read("provenance.json")
    require(receipts["endpoint"]["job.exit"].strip() == "0", "endpoint job failed")
    summary = {
        "decision": "rejected-no-demonstrated-endpoint-benefit",
        "ordered_processes_not_causal_estimates": True,
        "cases": {},
    }
    for atoms in (48, 96):
        runs = {
            arm: read(f"{arm}-{atoms}.json.gz")
            for arm in ("control", "candidate", "reference")
        }
        protocol = runs["reference"]["protocol"]
        require(protocol["density_fitting"] is False, "not exact direct")
        require(
            protocol["reference_fock_policy"] == "full-density-rebuild",
            "reference policy changed",
        )
        require(
            protocol["energy_gate"] == 1e-8
            and protocol["force_gate"] == 1e-7
            and protocol["repeats"] == 5,
            "gates changed",
        )
        for arm, run in runs.items():
            require(
                run["status"] == "measured" and run["protocol"] == protocol,
                "protocol mismatch",
            )
            require(
                [r["phase"] for r in run["records"]] == PHASES,
                "phase inventory mismatch",
            )
            for row in run["records"]:
                require(
                    row["status"] == 0 and row["converged"] and row["gate"],
                    "sample failed",
                )
                require(
                    math.isfinite(row["complete_seconds"])
                    and row["complete_seconds"] > 0,
                    "invalid duration",
                )
                require(
                    row["geometry"] == (0 if row["phase"] in ("cold", "warm") else 1),
                    "geometry order changed",
                )
                if arm == "reference":
                    backend = row["reference_xc_backend"]
                    require(
                        backend["backend"] == "cuda-libxc"
                        and all(c["on_gpu"] is True for c in backend["components"]),
                        "reference fallback",
                    )
                    continue
                require(row["fock_builds"] is None, "historical unknown count invented")
                scf = row["native_scf_ao_work"]
                require(
                    scf["requested"]
                    and scf["selected"]
                    and scf["xc_evaluations"] > 0
                    and scf["cutoff"] == 1e-16,
                    "SCF map missed",
                )
                require(
                    0 < scf["point_ao_square_sum"] < scf["dense_point_ao_square_sum"],
                    "sparse SCF work absent",
                )
                force = row["native_force_components"]
                require(
                    force["stationary_integral_derivative_route"]
                    == "prepared-native-complete"
                    and force["wall_seconds"]["stationary_integral_derivatives"] > 0,
                    "force route missed",
                )
                grid = force["grid_work_plan"]
                batches = (grid["grid_points"] + grid["tile_points"] - 1) // grid[
                    "tile_points"
                ]
                counts = force["work_counts"]["observed"]
                require(
                    counts["geometry_batches"]
                    == counts["phased_becke_batches"]
                    == batches,
                    "phased coverage missed",
                )
                ao = force["resident_ao_selection"]
                require(
                    ao["mode"] == "explicit-sampled-jet-cutoff"
                    and ao["cutoff"] == 1e-16,
                    "force map missed",
                )
                require(
                    ao["work"]["tile_count"] == batches
                    and 0
                    < ao["work"]["point_ao_square_sum"]
                    < ao["work"]["dense_point_ao_square_sum"],
                    "sparse force work absent",
                )
            if arm == "reference":
                continue
            revision, source, library = IDENTITIES[arm]
            require(
                run["environment"]["git"]
                == {
                    "commit": revision,
                    "dirty": False,
                    "pending_generated_benchmark_artifacts": 0,
                },
                "source checkout changed",
            )
            require(
                run["native_build"]["probe"]["source_identity"] == source
                and run["native_build"]["library_sha256"] == library,
                "native identity changed",
            )
            require(
                receipts[arm]["source-head.txt"].strip() == revision
                and receipts[arm]["source-identity.txt"].strip() == source,
                "receipt identity changed",
            )
            require(
                receipts[arm]["binaries.sha256"].split()[0] == library,
                "library receipt changed",
            )
            require(
                run["native_schedule_settings"]["GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE"]
                == "indexed",
                "schedule changed",
            )
            require(
                run["qualification_policy"]
                == {
                    "name": "local-indexed-phased",
                    "indexed_force_requested": True,
                    "resident_ao_cutoff": 1e-16,
                    "resident_ao_cache_bytes": 16 << 20,
                    "phased_becke_requested": True,
                    "scf_active_ao_requested": True,
                },
                "composition changed",
            )
        phases = {}
        for phase in dict.fromkeys(PHASES):
            selected = {
                a: [r for r in run["records"] if r["phase"] == phase]
                for a, run in runs.items()
            }
            phases[phase] = {
                a: {
                    "seconds": [r["complete_seconds"] for r in rows],
                    "median_seconds": statistics.median(
                        r["complete_seconds"] for r in rows
                    ),
                    "iterations": [r["iterations"] for r in rows],
                    "fock_builds": [r.get("fock_builds") for r in rows],
                }
                for a, rows in selected.items()
            }
            phases[phase]["candidate_over_control"] = (
                phases[phase]["candidate"]["median_seconds"]
                / phases[phase]["control"]["median_seconds"]
            )
            phases[phase]["candidate_over_reference"] = (
                phases[phase]["candidate"]["median_seconds"]
                / phases[phase]["reference"]["median_seconds"]
            )
        summary["cases"][str(atoms)] = {
            "inputs_hash": canonical_hash(protocol),
            "phases": phases,
            "comparisons": {
                f"{a}-{b}": compare(runs[a], runs[b])
                for a, b in [
                    ("control", "reference"),
                    ("candidate", "reference"),
                    ("candidate", "control"),
                ]
            },
        }
    summary["census"] = audit_census.audit()
    return summary


def verify() -> dict:
    """Authenticate retained files and then independently recompute the decision data."""
    manifest = read("publication.json")
    validate_publication(
        manifest,
        {e["path"]: (ROOT / e["path"]).read_bytes() for e in manifest["files"]},
    )
    result = analyze()
    require(result == read("summary.json"), "retained summary differs from samples")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
