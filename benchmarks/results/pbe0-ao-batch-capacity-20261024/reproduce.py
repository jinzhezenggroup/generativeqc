"""Replay independent E/F gates, complete populations, work and timing evidence."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
from pathlib import Path

import numpy as np

from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent
HOLDOUTS = (
    "pbe0-direct-3-standard",
    "pbe0-direct-12-standard",
    "pbe0-direct-48-standard",
    "pbe-direct-3-standard",
    "b3lyp-direct-48-standard",
    "pbe0-df-96-standard",
    "pbe0-direct-96-fast",
    "r2scan-direct-96-standard",
    "r2scan-3c-direct-96-standard",
)


def require(condition: object, context: object) -> None:
    """Fail closed even when Python assertions are disabled."""
    if not condition:
        raise ValueError(context)


def describe(values: list[float]) -> dict:
    """Keep every sample; MAD defines a noise floor, not a confidence interval."""
    median = statistics.median(values)
    return {
        "values": values,
        "median": median,
        "mad": statistics.median(abs(x - median) for x in values),
        "min": min(values),
        "max": max(values),
    }


def comparison(arms: dict) -> dict:
    """Apply the prespecified floor to independently measured process medians."""
    left, right = arms["master"], arms["candidate"]
    gain = 1 - right["median"] / left["median"]
    floor = max(
        0.02, 2 * (left["mad"] / left["median"] + right["mad"] / right["median"])
    )
    return {
        "arms": arms,
        "improvement_fraction": gain,
        "variability_floor": floor,
        "qualified_gain": gain > floor,
        "regresses": -gain > floor,
        **(
            {
                "gpu4pyscf_gap_fraction": right["median"] / arms["gpu4pyscf"]["median"]
                - 1
            }
            if "gpu4pyscf" in arms
            else {}
        ),
    }


def reproduce(data: dict) -> dict:
    """Authenticate the full inventory before evaluating every retained call."""
    require(
        data["schema"] == "generativeqc.xc-live-panel-capacity-evidence.v1",
        "unsupported receipt schema",
    )
    expected = {
        f"final-benchmark/r{r}-{arm}-{method}.json"
        for r in range(3)
        for arm in ("master", "candidate", "gpu4pyscf")
        for method in ("pbe0", "pbe")
    }
    expected.update(
        f"holdouts/{arm}-{case}.json"
        for arm in ("master", "candidate")
        for case in HOLDOUTS
    )
    expected.update(
        f"batch-pilot/r{r}-bytes{size}.json"
        for r in range(3)
        for size in (33554432, 134217728)
    )
    expected.update(
        f"profiles/{arm}-pbe0.json"
        for arm in (
            "standard",
            "gpu4pyscf",
            "batch128",
            "candidate-off",
            "candidate-default",
        )
    )
    records = data["records"]
    require(set(records) == expected, "missing or unexpected measurement population")
    failed = data["failed_records"]
    failure_path = "packaging-failure/candidate-b3lyp-direct-48-standard.json"
    require(set(failed) == {failure_path}, "missing failed packaging attempt")
    failure = failed[failure_path]
    require(failure["status"] == "failed" and failure["samples"] == [], failure_path)
    require(
        "missing packaged stationary CUDA artifact for b3lyp_rks_spd"
        in failure["error"],
        failure_path,
    )
    reference_root = ROOT.parent / "readme-final-20261022"
    oracles = json.loads(
        lzma.decompress((reference_root / "oracles.json.xz").read_bytes())
    )
    reuse = json.loads((reference_root / "reference-reuse.json").read_text())
    qualifications, resources = {}, {}
    for name, record in records.items():
        require(record["status"] == "complete", name)
        require([s["phase"] for s in record["samples"]] == ["cold", "warm"], name)
        result = qualify([record], oracles, reference_reuse=reuse)[0]
        require(result["status"] == "qualified", (name, result))
        qualifications[name] = result
        if record["arm"] != "gpu4pyscf":
            candidate = "/candidate-" in name or "-candidate-" in name
            arm = "candidate" if candidate else "master"
            require(
                record["native_build"]["library_sha256"]
                == data["libraries"][arm]["sha256"],
                name,
            )
            require(
                record["source_revision"] == data["libraries"][arm]["revision"], name
            )
            require(
                record["native_build"]["source_identity"]
                == data["libraries"][arm]["source_identity"],
                name,
            )
        for sample in record["samples"]:
            work = sample.get("force_work", {})
            for kind, field in (
                ("device", "additional_device_peak_bound"),
                ("host", "additional_host_numeric_bound"),
            ):
                if field not in work:
                    continue
                cap = work.get(f"additional_{kind}_budget")
                resources[f"{name}/{sample['phase']}/{kind}"] = {
                    "bound": work[field],
                    "budget": cap,
                }
                if cap is not None:
                    require(work[field] <= cap, (name, field))

    def sample(name: str, phase: str) -> dict:
        rows = [s for s in records[name]["samples"] if s["phase"] == phase]
        require(len(rows) == 1, (name, phase))
        return rows[0]

    timing, paired = {}, {}
    for method in ("pbe0", "pbe"):
        timing[method] = {}
        for phase in ("cold", "warm"):
            arms = {}
            for arm in ("master", "candidate", "gpu4pyscf"):
                rows = [
                    sample(f"final-benchmark/r{r}-{arm}-{method}.json", phase)
                    for r in range(3)
                ]
                arms[arm] = {
                    **describe([s["seconds"] for s in rows]),
                    "iterations": [s["iterations"] for s in rows],
                    "fock_builds": [s.get("fock_builds") for s in rows],
                }
            timing[method][phase] = comparison(arms)
            for r in range(3):
                paths = [
                    f"final-benchmark/r{r}-{arm}-{method}.json"
                    for arm in ("master", "candidate")
                ]
                left, right = (records[p] for p in paths)
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
                left, right = (sample(p, phase) for p in paths)
                for key in ("settings", "iterations", "fock_builds"):
                    require(left[key] == right[key], (paths, key))
                lw, rw = (
                    s["force_work"]["resident_ao_selection"]["work"]
                    for s in (left, right)
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
                    require(lw[key] == rw[key], (paths, key))
                paired[f"{method}/{phase}/{r}"] = {
                    "energy_abs": abs(left["energy"] - right["energy"]),
                    "force_max": float(
                        np.max(np.abs(np.asarray(left["forces"]) - right["forces"]))
                    ),
                }
    # Held-out pairs have one cold/warm observation each. Keep their timings
    # visible, including small regressions, without inferring a noise floor.
    held_out = {}
    for case in HOLDOUTS:
        arms = {
            arm: records[f"holdouts/{arm}-{case}.json"]
            for arm in ("master", "candidate")
        }
        for key in ("geometry_bohr", "method", "mode", "orbital_basis", "protocol"):
            require(arms["master"][key] == arms["candidate"][key], (case, key))
        held_out[case] = {}
        for phase in ("cold", "warm"):
            rows = {
                arm: sample(f"holdouts/{arm}-{case}.json", phase) for arm in arms
            }
            for key in ("settings", "iterations", "fock_builds"):
                require(rows["master"][key] == rows["candidate"][key], (case, key))
            held_out[case][phase] = {
                arm: row["seconds"] for arm, row in rows.items()
            }
    for method in ("pbe0", "pbe"):
        require(timing[method]["cold"]["qualified_gain"], (method, "cold gain floor"))
    pilot = {}
    for phase in ("cold", "warm"):
        arms = {
            arm: describe(
                [
                    sample(f"batch-pilot/r{r}-bytes{size}.json", phase)["seconds"]
                    for r in range(3)
                ]
            )
            for arm, size in (("master", 33554432), ("candidate", 134217728))
        }
        pilot[phase] = comparison(arms)

    traced_work = {}
    for arm, cap in (("off", 33554432), ("default", 134217728)):
        rows = [r for r in data["traces"][arm] if r["operation"] == "cuda_ks_xc_shared"]
        profile = records[f"profiles/candidate-{arm}-pbe0.json"]
        require(
            len(rows) == sum(s["fock_builds"] for s in profile["samples"]),
            (arm, "incomplete XC trace"),
        )
        require(all(r["valid"] for r in rows), (arm, "invalid XC trace"))
        counters = [r["counters"] for r in rows]
        for counter in counters:
            require(
                0 < counter["xc_point_batch_bytes"] <= cap, (arm, "XC arena capacity")
            )
            require(
                1 < counter["xc_point_batch_tiles"] <= 32, (arm, "XC tile capacity")
            )
            require(counter["xc_point_ao_visits"] > 0, (arm, "missing AO work"))
        traced_work[arm] = counters
    require(
        [
            (c["xc_point_ao_visits"], c["xc_point_ao_square_sum"])
            for c in traced_work["off"]
        ]
        == [
            (c["xc_point_ao_visits"], c["xc_point_ao_square_sum"])
            for c in traced_work["default"]
        ],
        "capacity changed semantic AO work",
    )

    moved = data["moved"]
    require(moved["status"] == "complete", "moved endpoint incomplete")
    require(
        moved["native_build"]["library_sha256"]
        == data["libraries"]["candidate"]["sha256"],
        "moved binary mismatch",
    )
    require(
        [r["phase"] for r in moved["samples"]] == ["cold", "warm", "moved"],
        "missing moved population",
    )
    require(
        [r["geometry"] for r in moved["samples"]] == ["original", "original", "moved"],
        "moved sample geometry mismatch",
    )
    require(
        moved["gates"]
        == {"energy_per_atom": 2.5e-6, "force_max": 2.5e-5, "force_rms": 7.5e-6},
        "changed moved gates",
    )
    moved_errors = []
    for row in moved["samples"]:
        oracle = moved["references"]["moved" if row["phase"] == "moved" else "original"]
        forces, reference_forces = (
            np.asarray(row["forces"]),
            np.asarray(oracle["forces"]),
        )
        require(
            forces.shape == reference_forces.shape == (96, 3),
            "invalid moved force shape",
        )
        delta = forces - reference_forces
        errors = {
            "energy_per_atom": abs(row["energy"] - oracle["energy"]) / 96,
            "force_max": float(np.max(np.abs(delta))),
            "force_rms": float(np.sqrt(np.mean(delta**2))),
        }
        require(
            all(value <= moved["gates"][key] for key, value in errors.items()),
            (row["phase"], errors),
        )
        moved_errors.append({"phase": row["phase"], "errors": errors})
    return {
        "qualification": qualifications,
        "resources": resources,
        "timing": timing,
        "held_out": held_out,
        "failed_attempts": failed,
        "paired": paired,
        "pilot": pilot,
        "traced_work": traced_work,
        "moved": moved_errors,
        "qualified_static_calls": sum(len(r["samples"]) for r in records.values()),
        "qualified_moved_calls": len(moved_errors),
    }


def main() -> None:
    """Verify stored bytes before independently recomputing all acceptance gates."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest["reference_dependencies"].items():
        require(
            hashlib.sha256((ROOT.parents[2] / name).read_bytes()).hexdigest()
            == digest,
            (name, "reference dependency hash mismatch"),
        )
    raw = (ROOT / "receipts.json.xz").read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == manifest["receipts_sha256"],
        "receipt hash mismatch",
    )
    print(
        json.dumps(
            reproduce(json.loads(lzma.decompress(raw))), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
