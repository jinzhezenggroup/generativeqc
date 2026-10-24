"""Recheck retained independent observables and complete endpoint populations."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
from pathlib import Path

import numpy as np

from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent


def require(condition: object, context: object) -> None:
    """Keep evidence validation active under optimized Python as well."""
    if not condition:
        raise ValueError(context)


def statistics_of(values: list[float]) -> dict:
    """Keep every process observation; MAD is a variability floor, not a CI."""
    median = statistics.median(values)
    return {
        "values": values,
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "min": min(values),
        "max": max(values),
    }


def reproduce(data: dict) -> dict:
    """Validate all receipts before deriving the prespecified timing comparison."""
    references = ROOT.parent / "readme-final-20261022"
    oracles = json.loads(lzma.decompress((references / "oracles.json.xz").read_bytes()))
    reuse = json.loads((references / "reference-reuse.json").read_text())
    qualifications = {}
    resource_bounds = {}
    records = data["records"]
    # A rehashed bundle cannot silently omit a slow or unsuccessful population.
    expected = {
        f"final-benchmark/r{repeat}-{arm}-{method}.json"
        for repeat in range(3)
        for arm in ("master", "candidate", "gpu4pyscf")
        for method in ("pbe", "pbe0", "r2scan-3c")
    }
    expected.update(
        f"final-holdouts/{arm}-{case}.json"
        for arm in ("master", "candidate")
        for case in (
            "r2scan-direct-3-standard",
            "r2scan-direct-96-standard",
            "r2scan-3c-direct-48-standard",
            "r2scan-3c-df-96-standard",
            "m06-2x-direct-48-standard",
            "pbe0-direct-96-fast",
            "r2scan-3c-direct-96-fast",
            "wb97m-v-direct-3-standard",
        )
    )
    expected.update(
        f"pilot/{case}-{arm}.json"
        for case in ("r2scan-3c-96-standard", "m06-2x-48-standard", "r2scan-3c-96-fast")
        for arm in ("off", "on")
    )
    expected.update(f"profiles/{arm}-r2scan-3c.json" for arm in ("off", "default"))
    for cohort, rounds, arms in (
        ("small-followup", 3, ("master", "candidate")),
        ("small-isolation", 5, ("off", "default")),
        ("small-confirmation", 5, ("master", "candidate")),
    ):
        expected.update(
            f"{cohort}/r{repeat}-{arm}-r2scan.json"
            for repeat in range(rounds)
            for arm in arms
        )
    require(set(records) == expected, "missing or unexpected measurement population")
    for name, record in records.items():
        require(record["status"] == "complete", name)
        result = qualify([record], oracles, reference_reuse=reuse)[0]
        require(result["status"] == "qualified", (name, result))
        qualifications[name] = result
        if record["arm"] != "gpu4pyscf":
            require(
                record["source_revision"] == data["base_revision"],
                "invalid retained evidence",
            )
            expected = data["candidate_library_sha256"]
            if name.startswith("pilot/") or ("/r" in name and "-master-" in name):
                expected = data["master_library_sha256"]
            if name.startswith("final-holdouts/master-"):
                expected = data["master_library_sha256"]
            require(record["native_build"]["library_sha256"] == expected, name)
        for sample in record["samples"]:
            work = sample.get("force_work", {})
            for resource in ("device", "host"):
                key = f"additional_{resource}_peak_bound"
                if resource == "host":
                    key = "additional_host_numeric_bound"
                if key in work:
                    budget = work.get(f"additional_{resource}_budget")
                    # The small native WB force receipt exports a bound but
                    # no public ceiling. Keep that missing field explicit;
                    # it cannot establish a checked resource acceptance gate.
                    resource_bounds.setdefault(name, {})[
                        f"{sample['phase']}-{sample['repeat']}-{resource}"
                    ] = {"bound": work[key], "budget": budget}
                    if budget is not None:
                        require(work[key] <= budget, name)

    def sample(name: str, phase: str) -> dict:
        rows = [s for s in records[name]["samples"] if s["phase"] == phase]
        require(len(rows) == 1, name)
        return rows[0]

    timing = {}
    main = [name for name in records if name.startswith("final-benchmark/")]
    require(len(main) == 27, "invalid retained evidence")
    for method in ("pbe", "pbe0", "r2scan-3c"):
        timing[method] = {}
        for phase in ("cold", "warm"):
            arms = {}
            for arm in ("master", "candidate", "gpu4pyscf"):
                rows = [
                    sample(f"final-benchmark/r{r}-{arm}-{method}.json", phase)
                    for r in range(3)
                ]
                arms[arm] = {
                    **statistics_of([row["seconds"] for row in rows]),
                    "iterations": [row["iterations"] for row in rows],
                    "fock_builds": [row.get("fock_builds") for row in rows],
                }
            baseline, candidate, reference = (
                arms[arm] for arm in ("master", "candidate", "gpu4pyscf")
            )
            gain = 1 - candidate["median"] / baseline["median"]
            floor = max(
                0.02,
                2
                * (
                    baseline["mad"] / baseline["median"]
                    + candidate["mad"] / candidate["median"]
                ),
            )
            paired = []
            for repeat in range(3):
                paths = [
                    f"final-benchmark/r{repeat}-{arm}-{method}.json"
                    for arm in ("master", "candidate")
                ]
                left, right = (records[path] for path in paths)
                for key in (
                    "geometry_bohr",
                    "method",
                    "mode",
                    "orbital_basis",
                    "basis_representation",
                    "charge",
                    "spin",
                    "thread_controls",
                    "runner_sha256",
                    "protocol",
                ):
                    require(left[key] == right[key], (paths, key))
                left, right = (sample(path, phase) for path in paths)
                require(left["iterations"] == right["iterations"], paths)
                require(left["fock_builds"] == right["fock_builds"], paths)
                left_work, right_work = (
                    row["force_work"]["resident_ao_selection"]["work"]
                    for row in (left, right)
                )
                for key in (
                    "discovery_ao_jet_values",
                    "point_ao_visits",
                    "point_ao_square_sum",
                    "tile_count",
                    "discoveries",
                    "retained_map_bytes",
                    "dense_budget_tiles",
                ):
                    require(left_work[key] == right_work[key], (paths, key))
                paired.append(
                    {
                        "energy_abs": abs(left["energy"] - right["energy"]),
                        "force_max": float(
                            np.max(np.abs(np.asarray(left["forces"]) - right["forces"]))
                        ),
                    }
                )
            timing[method][phase] = {
                "arms": arms,
                "improvement_fraction": gain,
                "variability_floor": floor,
                "qualified_gain": gain > floor,
                "regresses": -gain > floor,
                "gpu4pyscf_gap_fraction": candidate["median"] / reference["median"] - 1,
                "paired_differences": paired,
            }
    # A separate small-system follow-up retains all three warm calls and uses
    # one median per fresh process, without pooling its earlier single pair.
    small = {}
    for phase in ("cold", "warm"):
        arms = {}
        for arm in ("master", "candidate"):
            values = []
            all_calls = []
            for repeat in range(3):
                record = records[f"small-followup/r{repeat}-{arm}-r2scan.json"]
                calls = [
                    row["seconds"] for row in record["samples"] if row["phase"] == phase
                ]
                require(len(calls) == (1 if phase == "cold" else 3), record["method"])
                all_calls.append(calls)
                values.append(statistics.median(calls))
            arms[arm] = {**statistics_of(values), "all_calls": all_calls}
        baseline, candidate = (arms[arm] for arm in ("master", "candidate"))
        gain = 1 - candidate["median"] / baseline["median"]
        floor = max(
            0.02,
            2
            * (
                baseline["mad"] / baseline["median"]
                + candidate["mad"] / candidate["median"]
            ),
        )
        small[phase] = {
            "arms": arms,
            "improvement_fraction": gain,
            "variability_floor": floor,
            "regresses": -gain > floor,
        }
    auxiliary_timing = {}
    for cohort, names in (
        ("small-isolation", ("off", "default")),
        ("small-confirmation", ("master", "candidate")),
    ):
        auxiliary_timing[cohort] = {}
        for phase in ("cold", "warm"):
            arms = {}
            for arm in names:
                values = [
                    sample(f"{cohort}/r{repeat}-{arm}-r2scan.json", phase)["seconds"]
                    for repeat in range(5)
                ]
                arms[arm] = statistics_of(values)
            baseline, candidate = (arms[arm] for arm in names)
            gain = 1 - candidate["median"] / baseline["median"]
            floor = max(
                0.02,
                2
                * (
                    baseline["mad"] / baseline["median"]
                    + candidate["mad"] / candidate["median"]
                ),
            )
            auxiliary_timing[cohort][phase] = {
                "arms": arms,
                "improvement_fraction": gain,
                "variability_floor": floor,
                "regresses": -gain > floor,
            }
    moved = data["moved"]
    require(moved["status"] == "complete", "invalid retained evidence")
    require(
        [row["phase"] for row in moved["samples"]] == ["cold", "warm", "moved"],
        "incomplete original/warm/displaced population",
    )
    require(
        moved["native_build"]["library_sha256"] == data["candidate_library_sha256"],
        "displaced native binary differs",
    )
    require(
        moved["gates"]
        == {"energy_per_atom": 2.5e-06, "force_max": 2.5e-05, "force_rms": 7.5e-06},
        "invalid retained evidence",
    )
    moved_errors = []
    for row in moved["samples"]:
        reference = moved["references"][row["geometry"]]
        difference = np.asarray(row["forces"]) - reference["forces"]
        errors = {
            "energy_per_atom": abs(row["energy"] - reference["energy"]) / 96,
            "force_max": float(np.max(np.abs(difference))),
            "force_rms": float(np.sqrt(np.mean(difference**2))),
        }
        require(
            all((value <= moved["gates"][key] for key, value in errors.items())),
            "invalid retained evidence",
        )
        moved_errors.append({"phase": row["phase"], "errors": errors})
    return {
        "qualification": qualifications,
        "resource_bounds": resource_bounds,
        "timing": timing,
        "moved": moved_errors,
        "small_followup": small,
        "auxiliary_timing": auxiliary_timing,
        "qualified_static_calls": sum(len(r["samples"]) for r in records.values()),
        "qualified_moved_calls": len(moved_errors),
    }


def main() -> None:
    """Authenticate the bundle, then recompute every reported observable gate."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    raw = (ROOT / "receipts.json.xz").read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == manifest["receipts_sha256"],
        "invalid retained evidence",
    )
    data = json.loads(lzma.decompress(raw))
    print(json.dumps(reproduce(data), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
