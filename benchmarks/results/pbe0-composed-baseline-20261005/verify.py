"""Verify every composed-baseline sample, without a CUDA device.

Run from a checkout with PYTHONPATH=python:. . Publication checks authenticate
the stored bytes; the independent comparisons below recompute the scientific
gates, actual route observations and phase medians from all retained samples.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from generativeqc_compiler.common.evidence import block_error, canonical_hash

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_record

ROOT = Path(__file__).resolve().parent
SOURCE = "e858a3537dd389dafbf952f9c70f5f803de37fed8b719776511c8e6b5876b6d4"
REVISION = "cd121e3af089b0d59f9cffaeff381b1ac01ffb31"
LIBRARY = "b9c449f51744ca1f77ba35124fc07be67ba1b72b1321b7245b540bcf9d0f94f3"
PHASES = ["cold", *(["warm"] * 5), "moved", *(["moved-warm"] * 5)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        path = ROOT / (name + ".gz")
    return load_record(path)


def analyze() -> dict:
    """Keep diagnostic event timings separate from complete E/F wall time."""
    summary = {"endpoint_comparisons": 0, "component_samples": 0, "cases": {}}
    for atoms in (48, 96):
        native, reference, profile = (
            read(f"{prefix}-{atoms}.json")
            for prefix in ("composed", "reference", "components")
        )
        protocol = native["protocol"]
        require(protocol == reference["protocol"], "different scientific protocol")
        require(protocol == profile["scientific_protocol"], "profile identity drift")
        require(protocol["density_fitting"] is False, "not exact-direct evidence")
        require(
            protocol["reference_fock_policy"] == "full-density-rebuild",
            "incremental reference",
        )
        require(protocol["repeats"] == 5, "missing five replay protocol")
        require(
            protocol["energy_gate"] == 1e-8 and protocol["force_gate"] == 1e-7,
            "changed acceptance gates",
        )
        for run in (native, reference):
            require(run["status"] == "measured", "incomplete endpoint")
            require(
                [r["phase"] for r in run["records"]] == PHASES,
                "sample inventory changed",
            )
            for row in run["records"]:
                require(
                    row["status"] == 0 and row["converged"] and row["gate"],
                    "failed sample",
                )
                require(
                    math.isfinite(row["complete_seconds"])
                    and row["complete_seconds"] > 0,
                    "invalid timing",
                )
        for run in (native, profile):
            build = run["native_build"]
            require(build["probe"]["source_identity"] == SOURCE, "source mismatch")
            require(build["library_sha256"] == LIBRARY, "library mismatch")
        require(
            native["environment"]["git"]
            == {
                "commit": REVISION,
                "dirty": False,
                "pending_generated_benchmark_artifacts": 0,
            },
            "measured source was changed",
        )
        require(
            native["qualification_policy"]
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
        require(
            native["native_schedule_settings"]["GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE"]
            == "indexed",
            "indexed request absent",
        )
        for row in reference["records"]:
            backend = row["reference_xc_backend"]
            require(backend["backend"] == "cuda-libxc", "reference XC fallback")
            require(
                all(c["on_gpu"] is True for c in backend["components"]),
                "reference component fallback",
            )

        errors = []
        for row in native["records"]:
            require(row["fock_builds"] is None, "unavailable work count was invented")
            for oracle in reference["records"]:
                if row["geometry"] != oracle["geometry"]:
                    continue
                pair = {
                    "energy": block_error(
                        [row["energy"]], [oracle["energy"]], atol=1e-8, rtol=0
                    ),
                    "force": block_error(
                        row["forces"], oracle["forces"], atol=1e-7, rtol=0
                    ),
                }
                require(
                    all(e["passed"] for e in pair.values()),
                    "independent E/F gate failed",
                )
                errors.append(pair)
            scf = row["native_scf_ao_work"]
            require(
                scf["requested"] and scf["selected"] and scf["xc_evaluations"] > 0,
                "SCF maps did not execute",
            )
            require(scf["cutoff"] == 1e-16, "SCF cutoff drift")
            require(
                0 < scf["point_ao_square_sum"] < scf["dense_point_ao_square_sum"],
                "SCF sparse work missing",
            )
            force = row["native_force_components"]
            require(
                force["stationary_integral_derivative_route"]
                == "prepared-native-complete",
                "force route drift",
            )
            require(
                force["wall_seconds"]["stationary_integral_derivatives"] > 0,
                "force timer missing",
            )
            grid = force["grid_work_plan"]
            batches = (grid["grid_points"] + grid["tile_points"] - 1) // grid[
                "tile_points"
            ]
            counts = force["work_counts"]["observed"]
            require(
                counts["geometry_batches"] == counts["phased_becke_batches"] == batches,
                "phased coverage incomplete",
            )
            ao = force["resident_ao_selection"]
            require(
                ao["mode"] == "explicit-sampled-jet-cutoff" and ao["cutoff"] == 1e-16,
                "force map cutoff drift",
            )
            require(
                ao["work"]["tile_count"] == batches, "force map coverage incomplete"
            )
            require(
                ao["work"]["point_ao_square_sum"]
                < ao["work"]["dense_point_ao_square_sum"],
                "force sparse work missing",
            )
        require(len(errors) == 72, "not all same-geometry comparisons were checked")
        summary["endpoint_comparisons"] += len(errors)
        phases = {}
        for phase in dict.fromkeys(PHASES):
            n = [r for r in native["records"] if r["phase"] == phase]
            r = [r for r in reference["records"] if r["phase"] == phase]
            nt, rt = ([x["complete_seconds"] for x in rows] for rows in (n, r))
            phases[phase] = {
                "native_seconds": nt,
                "reference_seconds": rt,
                "native_median_seconds": statistics.median(nt),
                "reference_median_seconds": statistics.median(rt),
                "ratio": statistics.median(nt) / statistics.median(rt),
                "native_iterations": [x["iterations"] for x in n],
                "reference_iterations": [x["iterations"] for x in r],
                "native_fock_builds": [x["fock_builds"] for x in n],
                "force_endpoint_median_seconds": statistics.median(
                    x["native_force_components"]["endpoint_seconds"] for x in n
                ),
                "non_force_endpoint_median_seconds": statistics.median(
                    x["complete_seconds"]
                    - x["native_force_components"]["endpoint_seconds"]
                    for x in n
                ),
                "force_component_medians_seconds": {
                    key: None
                    if n[0]["native_force_components"]["wall_seconds"][key] is None
                    else statistics.median(
                        x["native_force_components"]["wall_seconds"][key] for x in n
                    )
                    for key in n[0]["native_force_components"]["wall_seconds"]
                },
            }
        require(
            profile["status"] == "measured" and profile["clean_endpoint"] is False,
            "diagnostic scope drift",
        )
        require(profile["source_head"] == REVISION, "profile revision drift")
        require(
            [r["geometry"] for r in profile["records"]] == [0, 1],
            "profile geometry inventory",
        )
        components = []
        for row in profile["records"]:
            require(len(row["samples_seconds"]) == 5, "missing diagnostic repeats")
            for oracle in reference["records"]:
                if row["geometry"] == oracle["geometry"]:
                    require(
                        abs(row["energy"] - oracle["energy"]) <= 1e-8,
                        "profile independent energy gate",
                    )
            medians = {}
            for key in ("scf_fock_j", "scf_full_range_k", "semilocal_ao_grid_xc"):
                times = [s[key] for s in row["samples_seconds"]]
                require(
                    all(math.isfinite(t) and t > 0 for t in times),
                    "invalid component timing",
                )
                medians[key] = statistics.median(times)
                require(
                    medians[key] == row["median_seconds"][key], "profile median drift"
                )
            components.append(
                {
                    "geometry": row["geometry"],
                    "iterations": row["iterations"],
                    "median_seconds": medians,
                }
            )
            summary["component_samples"] += 5
        summary["cases"][str(atoms)] = {
            "inputs_hash": canonical_hash(protocol),
            "phases": phases,
            "max_energy_error": max(p["energy"]["max_absolute_error"] for p in errors),
            "max_force_error": max(p["force"]["max_absolute_error"] for p in errors),
            "diagnostic_components": components,
        }
    return summary


def verify() -> dict:
    manifest = read("publication.json")
    files = {e["path"]: (ROOT / e["path"]).read_bytes() for e in manifest["files"]}
    validate_publication(manifest, files)
    result = analyze()
    require(result == read("summary.json"), "retained summary differs from raw samples")
    require(
        result["endpoint_comparisons"] == 144 and result["component_samples"] == 20,
        "incomplete campaign",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
