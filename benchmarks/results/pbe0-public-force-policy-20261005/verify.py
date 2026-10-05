"""Recheck complete policy endpoints and separately aggregate intrusive profiling.

Run with PYTHONPATH=python:. from the repository root. No GPU is needed. Raw
records are losslessly compressed; historical iteration changes stay visible.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from generativeqc_compiler.common.evidence import block_error, canonical_hash

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_record

ROOT = Path(__file__).resolve().parent
PHASES = ["cold", *(["warm"] * 5), "moved", *(["moved-warm"] * 5)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(name: str) -> dict:
    path = ROOT / name
    return load_record(path if path.exists() else ROOT / (name + ".gz"))


def compare(actual: dict, reference: dict) -> dict:
    """Include every same-geometry pair, regardless of SCF iteration count."""
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


def validate_run(run: dict, protocol: dict, identity: dict, arm: str) -> None:
    """Require actual selected work, complete final forces and real Fock counts."""
    require(run["status"] == "measured", "incomplete endpoint")
    require(run["protocol"] == protocol, "scientific protocol differs")
    require([r["phase"] for r in run["records"]] == PHASES, "sample inventory")
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


def profile_summary(identity: dict, atoms: int) -> dict:
    """Group observed kernel names; keep unknown owners and overlap explicit."""
    folder = "profile48" if atoms == 48 else "profile"
    require((ROOT / f"{folder}/job.exit").read_text().strip() == "0", "profile job failed")
    run = read(f"{folder}/profiled-native-{atoms}.json.gz")
    reference = read(f"{folder}/reference-{atoms}.json.gz")
    validate_run(run, reference["protocol"], identity, "default")
    groups = defaultdict(lambda: {"nanoseconds": 0, "launches": 0})
    rules = (
        ("bounded_direct_shell_quartet_kernel", "direct_derivative"),
        ("phased_becke_", "becke_partition_derivative"),
        ("geometry_cooperative_kernel", "ao_grid_geometry_response"),
        ("shell_warp_gradient", "one_electron_derivative"),
        ("_shell_class_fock_", "generated_direct_fock"),
        ("ao_kernel(", "ao_values_and_jets"),
        ("diagnostic_kernel", "ks_physical_diagnostics"),
        ("evaluate_points<", "xc_density_points_potential"),
        ("tiled_density_product<", "xc_density_points_potential"),
        ("tiled_potential(", "xc_density_points_potential"),
        ("cooperative_feature_kernel", "xc_density_points_potential"),
        ("cutlass::Kernel2", "gemm_caller_unclassified"),
    )
    with (ROOT / f"{folder}/statistics_cuda_gpu_kern_sum.csv").open() as stream:
        for row in csv.DictReader(stream):
            group = next((group for token, group in rules if token in row["Name"]), "other")
            groups[group]["nanoseconds"] += int(row["Total Time (ns)"])
            groups[group]["launches"] += int(row["Instances"])
    total = sum(g["nanoseconds"] for g in groups.values())
    with (ROOT / f"{folder}/statistics_nvtx_sum.csv").open() as stream:
        ranges = list(csv.DictReader(stream))
    require(len(ranges) == 1 and ranges[0]["Range"] == ":public.warm.energy_force",
            "capture scope changed")
    # The pinned strict-Direct owner submits generated J, then generated K, on
    # the same stream (direct_jk.cpp::enqueue_cuda_direct_jk_device_impl).
    # These are source-order attributions of value kernels only; no NVTX J/K
    # ranges were captured and setup/scatter/synchronization remain outside.
    with (ROOT / f"{folder}/ordered-fock-kernels.csv").open() as stream:
        values = list(csv.DictReader(stream))
    require(run["records"][1]["fock_builds"] == 1 and len(values) == 40,
            "single-build J/K launch inventory changed")
    require(len({r["stream"] for r in values}) == 1, "J/K stream changed")
    require([r["name"] for r in values[:20]] == [r["name"] for r in values[20:]],
            "J/K class submission order changed")
    split = {label: sum(int(r["end_ns"]) - int(r["start_ns"]) for r in rows)
             for label, rows in (("J", values[:20]), ("K", values[20:]))}
    require(sum(split.values()) == groups["generated_direct_fock"]["nanoseconds"],
            "ordered trace and kernel summary disagree")
    return {
        "clean_endpoint_timing": False,
        "captured_calls": 1,
        "nvtx_nanoseconds": int(ranges[0]["Total Time (ns)"]),
        "summed_gpu_nanoseconds": total,
        "summed_gpu_time_is_not_wall_time": True,
        "groups": {key: {**value, "fraction_of_summed_gpu_time": value["nanoseconds"] / total}
                   for key, value in sorted(groups.items())},
        "generated_value_kernel_nanoseconds": split,
        "jk_attribution": "pinned-source-submission-order; excludes setup and other kernels",
        "numerical_comparisons": compare(run, reference),
    }


def analyze() -> dict:
    """Retain ordered observations without claiming causal or normalized timings."""
    identities = read("identities.json")
    result = {
        "default_promoted": False,
        "ordered_processes_not_causal_estimates": True,
        "cohorts": {},
    }
    for cohort, identity in identities["cohorts"].items():
        reconstruction = read("reconstruction.json")[cohort]
        patch = gzip.decompress((ROOT / cohort / "source.patch.gz").read_bytes())
        require(reconstruction["source"] == identity["source"]
                and reconstruction["matched"]
                and reconstruction["reconstruction_base"] == identities["base"]
                and hashlib.sha256(patch).hexdigest() == reconstruction["patch_sha256"],
                "source reconstruction receipt changed")
        receipts = read(f"{cohort}/receipts.json")
        require(receipts["endpoint"]["job.exit"].strip() == "0", "endpoint job failed")
        cases = {}
        for atoms in (48, 96):
            runs = {arm: read(f"{cohort}/{arm}-{atoms}.json.gz")
                    for arm in ("dense-control", "default", "reference")}
            protocol = runs["reference"]["protocol"]
            require(protocol["density_fitting"] is False
                    and protocol["reference_fock_policy"] == "full-density-rebuild",
                    "not exact-direct/full-reference evidence")
            require(protocol["repeats"] == 5 and protocol["energy_gate"] == 1e-8
                    and protocol["force_gate"] == 1e-7, "acceptance gates changed")
            for arm, run in runs.items():
                validate_run(run, protocol, identity, arm)
            phases = {}
            for phase in dict.fromkeys(PHASES):
                phases[phase] = {}
                for arm, run in runs.items():
                    rows = [r for r in run["records"] if r["phase"] == phase]
                    phases[phase][arm] = {
                        "seconds": [r["complete_seconds"] for r in rows],
                        "median_seconds": statistics.median(r["complete_seconds"] for r in rows),
                        "iterations": [r["iterations"] for r in rows],
                        "fock_builds": [r.get("fock_builds") for r in rows],
                    }
                phases[phase]["default_over_control"] = (
                    phases[phase]["default"]["median_seconds"]
                    / phases[phase]["dense-control"]["median_seconds"])
                phases[phase]["default_over_reference"] = (
                    phases[phase]["default"]["median_seconds"]
                    / phases[phase]["reference"]["median_seconds"])
            cases[str(atoms)] = {
                "inputs_hash": canonical_hash(protocol), "phases": phases,
                "comparisons": {f"{a}-{b}": compare(runs[a], runs[b]) for a, b in (
                    ("dense-control", "reference"), ("default", "reference"),
                    ("default", "dense-control"))},
                "force_work": {arm: [r["native_force_components"]["resident_ao_selection"]
                                      for r in runs[arm]["records"]]
                               for arm in ("dense-control", "default")},
            }
        result["cohorts"][cohort] = cases
    result["intrusive_profiles"] = {
        str(atoms): profile_summary(identities["cohorts"]["current"], atoms)
        for atoms in (48, 96)
    }
    return result


def verify() -> dict:
    """Authenticate every stored byte and recompute the summary from raw samples."""
    manifest = read("publication.json")
    validate_publication(manifest, {e["path"]: (ROOT / e["path"]).read_bytes()
                                    for e in manifest["files"]})
    result = analyze()
    require(result == read("summary.json"), "summary differs from raw observations")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
