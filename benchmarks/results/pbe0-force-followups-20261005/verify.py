"""Authenticate follow-up force observations; no default or parity promotion.

All numerical pairings and phase summaries are recomputed from retained samples.
Run with PYTHONPATH=python:. from the repository root; no GPU is needed.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from generativeqc_compiler.common.evidence import block_error
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_record
ROOT=Path(__file__).resolve().parent

def require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)

def read(name: str) -> dict:
    return load_record(ROOT/name)

def validate_run(run: dict, protocol: dict, identity: dict, arm: str) -> None:
    """Require actual selected work, complete final forces and real Fock counts."""
    require(run["status"] == "measured", "incomplete endpoint")
    require(run["protocol"] == protocol, "scientific protocol differs")
    require([r["phase"] for r in run["records"]] == (["cold"] + ["warm"] * protocol["repeats"] + ["moved"] + ["moved-warm"] * protocol["repeats"]), "sample inventory")
    for row in run["records"]:
        require(row["status"] == 0 and row["converged"] and row["gate"], "failed sample")
        require(math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0,
                "invalid duration")
        require(row["geometry"] == (0 if row["phase"] in ("cold", "warm") else 1),
                "geometry ordering changed")
        if arm == "reference":
            backend = row["reference_xc_backend"]
            require(backend["backend"] == "cuda-libxc", "reference XC fallback")
            require(all(c["on_gpu"] is True for c in backend["components"]),
                    "reference component fallback")
            continue
        require(type(row["fock_builds"]) is int and row["fock_builds"] > 0,
                "actual Fock count missing")
        require(not row["warm_start_fallback"], "unexpected warm retry")
        scf = row["native_scf_ao_work"]
        require(scf["requested"] and scf["selected"] and scf["xc_evaluations"] > 0,
                "SCF local AO route missing")
        require(scf["cutoff"] == 1e-16, "SCF cutoff changed")
        require(0 < scf["point_ao_square_sum"] < scf["dense_point_ao_square_sum"],
                "SCF sparse work missing")
        force = row["native_force_components"]
        require(force["stationary_integral_derivative_route"] == "prepared-native-complete",
                "incomplete native force route")
        grid = force["grid_work_plan"]
        batches = (grid["grid_points"] + grid["tile_points"] - 1) // grid["tile_points"]
        counts = force["work_counts"]["observed"]
        require(counts["geometry_batches"] == counts["phased_becke_batches"] == batches,
                "phased Becke coverage missing")
        policy = force["force_active_ao_policy"]
        require(policy is not None and policy["requested_mode"] == "auto",
                "public policy receipt missing")
        ao = force["resident_ao_selection"]
        if arm == "default":
            require(policy["profile_id"] == "sm120-ordinary-rks-second-jet-v1"
                    and policy["actual_mode"] == "selected", "candidate profile missing")
            require(ao["mode"] == "sampled-jet-cutoff" and ao["cutoff"] == 1e-16,
                    "force cutoff changed")
            work = ao["work"]
            require(work["tile_count"] == batches, "force map coverage missing")
            require(0 < work["point_ao_square_sum"] < work["dense_point_ao_square_sum"],
                    "force sparse work missing")
        else:
            require(policy["profile_id"] is None and policy["actual_mode"] == "dense",
                    "control selected maps")
            require(ao["mode"] == "disabled" and ao["cutoff"] is None and ao["work"] is None,
                    "control observer changed")
    if arm == "reference":
        return
    build = run["native_build"]
    require(build["probe"]["source_identity"] == identity["source"], "source mismatch")
    require(build["library_sha256"] == identity["library"], "library mismatch")
    require(run["environment"]["git"] == {
        "commit": identity["revision"], "dirty": False,
        "pending_generated_benchmark_artifacts": 0,
    }, "measured checkout differs")
    require(run["qualification_policy"]["force_policy"] == arm, "arm mislabelled")
    for selector in ("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE"):
        require(run["native_schedule_settings"].get(selector) is None,
                "diagnostic schedule was enabled")


def compare(actual: dict, reference: dict) -> dict:
    """Check every same-geometry pairing, retaining all solver trajectories."""
    pairs = [(a, b) for a in actual["records"] for b in reference["records"]
             if a["geometry"] == b["geometry"]]
    require(len(pairs) == 2 * (actual["protocol"]["repeats"] + 1) ** 2,
            "pairing inventory changed")
    errors = {}
    for key, atol in (("energy", 1e-8), ("forces", 1e-7)):
        error = block_error([a[key] for a, _ in pairs], [b[key] for _, b in pairs],
                            atol=atol, rtol=0)
        require(error["passed"], f"{key} gate failed")
        errors[key] = error["max_absolute_error"]
    return {"pairings": len(pairs), "max_errors": errors}


def phase_summary(run: dict) -> dict:
    """Preserve individual times and counts; never normalize by iterations."""
    result = {}
    for phase in ("cold", "warm", "moved", "moved-warm"):
        rows = [r for r in run["records"] if r["phase"] == phase]
        result[phase] = {
            "seconds": [r["complete_seconds"] for r in rows],
            "median_seconds": statistics.median(r["complete_seconds"] for r in rows),
            "iterations": [r["iterations"] for r in rows],
            "fock_builds": [r.get("fock_builds") for r in rows],
        }
    return result


def check_protocol(protocol: dict, repeats: int) -> None:
    require(protocol["repeats"] == repeats and protocol["density_fitting"] is False
            and protocol["reference_fock_policy"] == "full-density-rebuild"
            and protocol["energy_gate"] == 1e-8 and protocol["force_gate"] == 1e-7,
            "scientific protocol or gates changed")


def analyze() -> dict:
    identities = read("identities.json")
    for name, digest in identities["patches"].items():
        require(hashlib.sha256(gzip.decompress((ROOT / name).read_bytes())).hexdigest()
                == digest, "source patch receipt differs")
    result = {"default_promoted": False, "root_reuse": {}, "cold_diagnostics": {}}
    for atoms in (48, 96):
        runs = {arm: read(f"root-reuse/{arm}-{atoms}.json.gz")
                for arm in ("control", "candidate", "reference")}
        protocol = runs["reference"]["protocol"]
        check_protocol(protocol, 5)
        validate_run(runs["reference"], protocol, {}, "reference")
        # Both arms enable the same public force profile. Only the generated
        # primitive-root implementation differs between their native sources.
        for arm, identity in (("control", "control"), ("candidate", "root")):
            validate_run(runs[arm], protocol, identities[identity], "default")
            require(all(r["fock_builds"] == 1 for r in runs[arm]["records"]
                        if r["phase"] in ("warm", "moved-warm")), "warm work changed")
        phases = {arm: phase_summary(run) for arm, run in runs.items()}
        result["root_reuse"][str(atoms)] = {
            "phases": phases,
            "candidate_over_control": {
                phase: phases["candidate"][phase]["median_seconds"]
                / phases["control"][phase]["median_seconds"] for phase in phases["control"]},
            "comparisons": {f"{a}-{b}": compare(runs[a], runs[b]) for a, b in
                            (("control", "reference"), ("candidate", "reference"),
                             ("candidate", "control"))},
        }
    for cohort, order in (
        ("shared-cache", ("default", "dense-control", "dense-control", "default")),
        ("fresh-cache", ("dense-control", "default", "default", "dense-control")),
    ):
        reference = read(f"{cohort}/reference-96.json.gz")
        protocol = reference["protocol"]
        check_protocol(protocol, 1)
        validate_run(reference, protocol, {}, "reference")
        rows = []
        runs = []
        for index, arm in enumerate(order, 1):
            run = read(f"{cohort}/{index}-{arm}.json.gz")
            validate_run(run, protocol, identities["control"], arm)
            runs.append(run)
            rows.append({"index": index, "arm": arm, "phases": phase_summary(run),
                         "comparison": compare(run, reference)})
        result["cold_diagnostics"][cohort] = {
            "initially_empty_cache_per_arm": cohort == "fresh-cache",
            "repeats_per_phase": 1, "ordered_observations": rows,
            "replaces_five_warm_qualification": False,
            "native_comparisons": {f"{a+1}-{b+1}": compare(runs[a], runs[b])
                                   for a in range(4) for b in range(a+1, 4)},
        }
    receipts = read("receipts.json")
    for campaign in ("root-reuse", "shared-cache", "fresh-cache", "native"):
        require(receipts[campaign]["job.exit"].strip() == "0", f"{campaign} job failed")
    for sanitizer in ("memcheck", "initcheck"):
        require("ERROR SUMMARY: 0 errors" in receipts["native"][f"{sanitizer}.log"],
                "native sanitizer failed")
    reconstruction = read("reconstruction.json")
    for name in ("control", "root"):
        require(reconstruction[name]["matched"] is True
                and reconstruction[name]["source"] == identities[name]["source"],
                "source reconstruction mismatch")
    return result


def verify() -> dict:
    """Authenticate all retained bytes before deriving the phase summaries."""
    manifest = read("publication.json")
    validate_publication(manifest, {e["path"]: (ROOT / e["path"]).read_bytes()
                                    for e in manifest["files"]})
    result = analyze()
    require(result == read("summary.json.gz"), "summary differs from raw records")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
