"""Replay endpoint, reference and coverage gates without CUDA or PySCF."""

import argparse
import csv
import json
import lzma
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from prepare_inputs import validate
from protocol import KCAL_PER_HARTREE, RECIPES, REVISION, ROOT, load, sha

CHEMICAL_ACCURACY_KCAL_MOL = 1.0
ENERGY_REPLAY_ATOL = 1e-8
STRICT_LIMITS = {
    "energy_tolerance": 1e-12,
    "density_tolerance": 1e-11,
    "ccsd_energy_tolerance": 1e-12,
    "ccsd_residual_tolerance": 1e-10,
}


def require_close(
    actual: float, target: float, atol: float = ENERGY_REPLAY_ATOL
) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, target, abs_tol=atol, rel_tol=0
    ):
        raise ValueError(
            f"replay mismatch: observed {actual!r}, reconstructed {target!r}"
        )


def weights(recipe: str) -> tuple[tuple[float, float], tuple[float, float]]:
    low, high = RECIPES[recipe]
    x = math.exp(-1.63 * (high - low))
    y = (low / high) ** 3
    return ((-x / (1 - x), 1 / (1 - x)), (-y / (1 - y), 1 / (1 - y)))


def linear(terms: tuple[float, float], pair: tuple[float, float]) -> float:
    return math.fsum(term * weight for term, weight in zip(terms, pair, strict=True))


def checked_cuda(row: dict, recipe: str) -> float:
    result = row["result"]
    metadata = result["basis_metadata"]
    converged = result["converged"] and all(
        component["result"]["converged"]
        and component["result"]["executed_backend"] == "cuda"
        for component in result["components"]
    )
    if (
        not converged
        or result["executed_backend"] != "cuda"
        or result["forces"] is not None
        or metadata["spin_reference"] != "rhf"
        or metadata["hf_reference"] != "df"
        or metadata["frozen_core"] != 0
        or not metadata["density_fitting"]
        or metadata["multiplicity"] != 1
        or metadata["charge"] != 0
        or metadata["representation"] != "spherical"
        or metadata["hf_extrapolation"] != "exponential"
        or metadata["hf_exponent"] != 1.63
        or metadata["correlation_exponent"] != 3.0
        or result["convergence"]["arithmetic"] != "fp64"
        or result["convergence"]["preconvergence_selection"] != "fp64"
        or result["convergence"]["selected"] != "strict_fp64"
        or any(
            result["convergence"]["limits"][key] != value
            for key, value in STRICT_LIMITS.items()
        )
        or [item["cardinal"] for item in metadata["mp2"]] != list(RECIPES[recipe])
        or row["wall_seconds"] <= 0
    ):
        raise ValueError(
            f"invalid CUDA scientific or timing record: {row['system']} {recipe}"
        )
    orbital = row["orbital_input_ids"]
    auxiliary = row["auxiliary_input_ids"]
    if (
        [item["identity"] for item in metadata["mp2"]]
        != [orbital[str(n)] for n in RECIPES[recipe]]
        or metadata["correction"]["identity"] != orbital["2"]
        or metadata["auxiliary_basis_identities"]
        != {identity: auxiliary[n] for n, identity in orbital.items()}
    ):
        raise ValueError(f"basis record mismatch: {row['system']} {recipe}")
    by_role = {}
    for component in result["components"]:
        for role in component["roles"]:
            if role in by_role:
                raise ValueError(f"duplicated native component role {role}")
            by_role[role] = component["result"]["energy"]
    hf, mp2 = weights(recipe)
    for key, expected in (("hf_weights", hf), ("mp2_weights", mp2)):
        for actual, target in zip(result[key], expected, strict=True):
            require_close(actual, target, 1e-12)
    hf_energy = linear((by_role["hf_low"], by_role["hf_high"]), hf)
    corr = linear(
        (
            by_role["mp2_low"] - by_role["hf_low"],
            by_role["mp2_high"] - by_role["hf_high"],
        ),
        mp2,
    )
    delta = by_role["correction_ccsdt"] - by_role["correction_mp2"]
    for key, expected in (
        ("hf_energy", hf_energy),
        ("mp2_correlation_energy", corr),
        ("delta_ccsdt_energy", delta),
        ("energy", math.fsum((hf_energy, corr, delta))),
    ):
        require_close(result[key], expected)
    return result["energy"]


def checked_reference(row: dict) -> dict:
    if (
        not row["scf_converged"]
        or not row["ccsd_converged"]
        or row["wall_seconds"] <= 0
    ):
        raise ValueError(
            f"invalid conventional endpoint {row['system']} {row['cardinal']}"
        )
    require_close(
        row["ccsdt_hartree"],
        math.fsum(row[key] for key in ("hf_hartree", "ccsd_correlation", "triples")),
    )
    require_close(row["mp2_correlation"], row["mp2_correlation"])
    return row


def read_bundles(raw: Path) -> tuple[dict, dict, set]:
    gpu, cpu = defaultdict(list), defaultdict(list)
    libraries = set()
    acquisition = load(ROOT / "acquisition.json")
    if sha(ROOT / "inputs.json") != acquisition["scoped_inputs_sha256"]:
        raise ValueError("CRBH20 input digest mismatch")
    expected = acquisition["raw_bundles"]
    bundles = {f"raw/{path.name}": path for path in raw.glob("*.json.xz")}
    if bundles.keys() != expected.keys():
        raise ValueError(
            "raw bundle paths differ from the CRBH20 acquisition inventory"
        )
    for name, path in sorted(bundles.items()):
        receipt = expected[name]
        if sha(path) != receipt["sha256"]:
            raise ValueError(f"raw bundle digest mismatch: {path}")
        with lzma.open(path, "rt") as stream:
            bundle = json.load(stream)
        run, kind = bundle["provenance"], bundle["kind"]
        if run["dataset"] != "CRBH20" or any(
            run[field] != receipt[field]
            for field in ("inputs_sha256", "driver_sha256", "protocol_sha256")
        ):
            raise ValueError(f"input or driver identity mismatch: {path}")
        if kind == "gpu":
            if run["source_revision"] != REVISION or "H100" not in run["gpu"]["model"]:
                raise ValueError(f"GPU source or device mismatch: {path}")
            receipt_path = ROOT / "build-identity.json"
            build = load(receipt_path)
            if (
                run["build_receipt_sha256"] != sha(receipt_path)
                or any(
                    run[field] != build[field]
                    for field in (
                        "source_revision",
                        "native_source_identity",
                        "library_sha256",
                    )
                )
                or build["source_archive_sha256"]
                != load(ROOT / "source.json")["source_archive_sha256"]
            ):
                raise ValueError(f"unbound production build: {path}")
            libraries.add((run["native_source_identity"], run["library_sha256"]))
        else:
            if run["cpu_models"] != ["Intel(R) Xeon(R) CPU Max 9462"]:
                raise ValueError(f"CPU model mismatch: {path}")
        records = {}
        for journal in bundle["journals"].values():
            for row in journal:
                key = row["system"], row["recipe" if kind == "gpu" else "cardinal"]
                if row["system"] not in run["systems"] or not row["system"].startswith(
                    run["dataset"] + "/"
                ):
                    raise ValueError(f"unplanned endpoint {key} in {path}")
                if row["status"] == "started":
                    if key in records:
                        raise ValueError(f"duplicate start {key} in {path}")
                    records[key] = {
                        **row,
                        "status": "unfinished",
                        "classification": "unfinished",
                    }
                else:
                    if records.get(key, {}).get("status") != "unfinished":
                        raise ValueError(f"unmatched terminal row {key} in {path}")
                    records[key] = row
        for key, record in records.items():
            row = {**record, "bundle": path.name}
            if kind == "gpu":
                row["gpu_uuid"] = run["gpu"]["uuid"]
                if row["status"] == "ok":
                    row["energy_hartree"] = checked_cuda(row, key[1])
                gpu[key].append(row)
            else:
                if row["status"] == "ok":
                    checked_reference(row)
                cpu[key].append(row)
    if len(libraries) > 1:
        raise ValueError(
            f"expected one pinned production build, observed {len(libraries)}"
        )
    return gpu, cpu, libraries


def completed(attempts: list) -> dict | None:
    valid = [row for row in attempts if row["status"] == "ok"]
    return max(valid, key=lambda row: row["finished_utc_unix"], default=None)


def barrier(terms: dict, energies: dict) -> float | None:
    if not terms.keys() <= energies.keys():
        return None
    return (
        math.fsum(coef * energies[key] for key, coef in terms.items())
        * KCAL_PER_HARTREE
    )


def metric(errors: list[float], planned: int) -> dict:
    within = sum(abs(error) <= CHEMICAL_ACCURACY_KCAL_MOL for error in errors)
    return {
        "complete": len(errors),
        "planned": planned,
        "mae": statistics.fmean(map(abs, errors)) if errors else None,
        "max_abs": max(map(abs, errors), default=None),
        "within_1_kcal_mol": within,
        "complete_set_chemical_accuracy": len(errors) == planned == within,
    }


def cpu_energies(cpu: dict, recipe: str) -> tuple[dict, dict]:
    low, high = RECIPES[recipe]
    canonical, composite = {}, {}
    hf, mp2 = weights(recipe)
    for key in sorted({key for key, _ in cpu}):
        a, b, correction = (completed(cpu.get((key, n), [])) for n in (low, high, 2))
        if b:
            canonical[key] = b["ccsdt_hartree"]
        if a and b and correction:
            composite[key] = math.fsum(
                (
                    linear((a["hf_hartree"], b["hf_hartree"]), hf),
                    linear((a["mp2_correlation"], b["mp2_correlation"]), mp2),
                    correction["ccsdt_hartree"]
                    - correction["hf_hartree"]
                    - correction["mp2_correlation"],
                )
            )
    return canonical, composite


def summarize(data: dict, gpu: dict, cpu: dict, builds: set) -> tuple[dict, list, list]:
    measurements, comparisons = [], []
    summary = {"source_revision": REVISION, "native_builds": sorted(builds), "sets": {}}
    dataset = "CRBH20"
    rows = data["barriers"]
    systems = sorted({key for row in rows for key in row["stoichiometry"]})
    set_summary = summary["sets"][dataset] = {}
    by_recipe = {}
    for recipe in RECIPES:
        energies, histories = {}, {}
        times = []
        for key in systems:
            attempts = gpu.get((key, recipe), [])
            hit = completed(attempts)
            latest = hit or max(
                attempts,
                key=lambda row: row.get("finished_utc_unix", row.get("utc_unix")),
                default={},
            )
            measurements.append(
                {
                    "system": key,
                    "recipe": recipe,
                    "attempts": len(attempts),
                    "status": latest.get("classification", "not_run"),
                    "endpoint_seconds": latest.get("wall_seconds"),
                    "process_seconds": latest.get("process_wall_seconds"),
                    "gpu_uuid": latest.get("gpu_uuid"),
                    "energy_hartree": latest.get("energy_hartree"),
                    "raw_bundle": latest.get("bundle"),
                    "error": latest.get("error"),
                }
            )
            histories[key] = latest
            if hit:
                energies[key] = hit["energy_hartree"]
                times.append(hit["wall_seconds"])
        canonical, conventional = cpu_energies(cpu, recipe)
        barriers = by_recipe[recipe] = {}
        for row in rows:
            index, terms, ref = (
                row["index"],
                row["stoichiometry"],
                row["reference_kcal_mol"],
            )
            gpu_barrier = barrier(terms, energies)
            cpu_barrier = barrier(terms, canonical)
            conventional_barrier = barrier(terms, conventional)
            entry = {
                "dataset": dataset,
                "index": index,
                "recipe": recipe,
                "reference_kcal_mol": ref,
                "gpu_kcal_mol": gpu_barrier,
                "error_kcal_mol": gpu_barrier - ref
                if gpu_barrier is not None
                else None,
                "canonical_kcal_mol": cpu_barrier,
                "conventional_composite_kcal_mol": conventional_barrier,
                "blockers": " ".join(
                    f"{key}:{histories[key].get('classification', 'not_run')}"
                    for key in terms
                    if key not in energies
                ),
            }
            barriers[index] = entry
            comparisons.append(entry)
        errors = [
            row["error_kcal_mol"]
            for row in barriers.values()
            if row["gpu_kcal_mol"] is not None
        ]
        anchors = len(data["canonical_anchors"][dataset])
        set_summary[recipe] = {
            "barriers": metric(errors, len(rows)),
            "endpoint_status_counts": dict(
                Counter(
                    row["status"]
                    for row in measurements
                    if row["system"].startswith(dataset + "/")
                    and row["recipe"] == recipe
                )
            ),
            "successful_endpoint_seconds": {
                "median": statistics.median(times) if times else None,
                "max": max(times, default=None),
            },
            "vs_canonical": metric(
                [
                    row["gpu_kcal_mol"] - row["canonical_kcal_mol"]
                    for row in barriers.values()
                    if row["gpu_kcal_mol"] is not None
                    and row["canonical_kcal_mol"] is not None
                ],
                anchors,
            ),
            "conventional_composite_vs_canonical": metric(
                [
                    row["conventional_composite_kcal_mol"] - row["canonical_kcal_mol"]
                    for row in barriers.values()
                    if row["conventional_composite_kcal_mol"] is not None
                    and row["canonical_kcal_mol"] is not None
                ],
                anchors,
            ),
        }
    common = []
    for row in rows:
        index = row["index"]
        if any(by_recipe[recipe][index]["gpu_kcal_mol"] is None for recipe in RECIPES):
            continue
        if all(
            len({completed(gpu[key, recipe])["gpu_uuid"] for recipe in RECIPES}) == 1
            for key in row["stoichiometry"]
        ):
            common.append(index)
    set_summary["same_card_common"] = {
        recipe: metric(
            [by_recipe[recipe][i]["error_kcal_mol"] for i in common], len(rows)
        )
        for recipe in RECIPES
    }
    return summary, measurements, comparisons


def canonical_measurements(data: dict, cpu: dict) -> list[dict]:
    rows = []
    for barrier_row in data["barriers"]:
        dataset = barrier_row["set"]
        if barrier_row["index"] not in data["canonical_anchors"][dataset]:
            continue
        for key in barrier_row["stoichiometry"]:
            for cardinal in (2, 3, 4):
                attempts = cpu.get((key, cardinal), [])
                latest = completed(attempts) or max(
                    attempts,
                    key=lambda row: row.get("finished_utc_unix", row.get("utc_unix")),
                    default={},
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "index": barrier_row["index"],
                        "system": key,
                        "cardinal": cardinal,
                        "status": latest.get("classification", "not_run"),
                        "endpoint_seconds": latest.get("wall_seconds"),
                        "process_seconds": latest.get("process_wall_seconds"),
                        "ccsdt_hartree": latest.get("ccsdt_hartree"),
                        "raw_bundle": latest.get("bundle"),
                        "error": latest.get("error"),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw")
    options = parser.parse_args()
    data = load(ROOT / "inputs.json")
    validate(data["systems"], data["barriers"], data["canonical_anchors"])
    gpu, cpu, builds = read_bundles(options.raw)
    summary, endpoints, barriers = summarize(data, gpu, cpu, builds)
    anchors = canonical_measurements(data, cpu)
    options.output.mkdir(exist_ok=True)
    (options.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for filename, rows in (
        ("endpoints.csv", endpoints),
        ("barriers.csv", barriers),
        ("canonical-endpoints.csv", anchors),
    ):
        with (options.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
