"""Rebuild barriers, coverage and accuracy tables from the raw endpoint journals."""

import argparse
import csv
import hashlib
import json
import lzma
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median

from run_gpu import RECIPES, ROOT, SOURCE_REVISION, sha
from run_reference import PAIRS, selected_keys

HARTREE_TO_KCAL_MOL = 627.5094740631
CHEMICAL_ACCURACY = 1.0
ASSEMBLY_TOLERANCE_HARTREE = 1e-8

ROLES = (
    "hf_low",
    "hf_high",
    "mp2_low",
    "mp2_high",
    "correction_ccsdt",
    "correction_mp2",
)


def load_journal(
    paths: list[Path], key_field: str
) -> dict[tuple[str, str | int], dict]:
    rows = {}
    for path in paths:
        with lzma.open(path, "rt") as journal:
            for line in journal:
                row = json.loads(line)
                key = row["system"], row[key_field]
                if key in rows and rows[key]["status"] not in ("started", "planned"):
                    raise ValueError(f"duplicate terminal result: {key}")
                rows[key] = row
    return rows


def verify_provenance(raw: Path, data: dict) -> None:
    acquisition_bytes = lzma.decompress(
        (raw / "acquisition-inputs.json.xz").read_bytes()
    )
    acquisition = json.loads(acquisition_bytes)
    acquired = hashlib.sha256(acquisition_bytes).hexdigest()
    assert data["systems"] == acquisition["systems"]
    assert data["sources"] == acquisition["sources"]
    original = acquisition["barriers"]
    changes = [
        old["index"]
        for old, new in zip(original, data["barriers"], strict=True)
        if old != new
    ]
    assert changes == [27, 28]
    corrected = [
        row for row in original if row["set"] == "NHTBH38" and row["index"] in changes
    ]
    corrected[0]["side"], corrected[1]["side"] = (
        corrected[1]["side"],
        corrected[0]["side"],
    )
    assert data["barriers"] == original

    identity_path = raw / "build-identity.json"
    if identity_path.is_file():
        build = json.loads(identity_path.read_text())
        assert build["source_revision"] == SOURCE_REVISION
        assert build["build_driver_sha256"] == sha(ROOT / "build_qz.sh")
    for path in raw.glob("gpu-*-provenance.json"):
        receipt = json.loads(path.read_text())
        assert receipt["inputs_sha256"] == sha(ROOT / "inputs.json")
        assert (
            receipt["source_revision"]
            == receipt["library_build_revision"]
            == SOURCE_REVISION
        )
        assert (
            receipt["native_source_identity"]
            == receipt["reported_library_source_identity"]
            == build["native_source_identity"]
        )
        assert receipt["library_sha256"] == build["library_sha256"]
        assert receipt["driver_sha256"] == sha(ROOT / "run_gpu.py")
    old_helper = lzma.decompress((raw / "acquisition-gpu-driver.py.xz").read_bytes())
    for path in raw.glob("cpu-*-provenance.json"):
        receipt = json.loads(path.read_text())
        assert receipt["inputs_sha256"] == acquired
        assert receipt["driver_sha256"] == sha(ROOT / "run_reference.py")
        assert receipt["shared_driver_sha256"] == hashlib.sha256(old_helper).hexdigest()
        assert receipt["selected_pairs"] == {
            key: list(value) for key, value in PAIRS.items()
        }


def compose(
    hf: tuple[float, float], mp2: tuple[float, float], delta: float, low: int, high: int
) -> float:
    hf_low = -1.0 / math.expm1(1.63 * (high - low))
    mp2_low = low**3 / (low**3 - high**3)
    return math.fsum(
        (
            hf_low * hf[0],
            (1 - hf_low) * hf[1],
            mp2_low * mp2[0],
            (1 - mp2_low) * mp2[1],
            delta,
        )
    )


def verify_native(row: dict) -> float:
    result = row["result"]
    roles = {}
    for component in result["components"]:
        item = component["result"]
        assert item["converged"] and item["executed_backend"] == "cuda"
        for role in component["roles"]:
            assert role not in roles
            roles[role] = item["energy"]
    assert set(roles) == set(ROLES)
    assert result["converged"] and result["executed_backend"] == "cuda"
    assert result["forces"] is None
    assert result["basis_metadata"]["hf_reference"] == "df"
    assert result["convergence"]["arithmetic"] == "fp64"
    hf0, hf1, mp0, mp1, cc, mp = (roles[name] for name in ROLES)
    native_energy = result["energy"]
    reconstructed = compose(
        (hf0, hf1), (mp0 - hf0, mp1 - hf1), cc - mp, *RECIPES[row["recipe"]]
    )
    assert math.isfinite(native_energy)
    assert abs(native_energy - reconstructed) <= ASSEMBLY_TOLERANCE_HARTREE, row[
        "system"
    ]
    assert (
        0
        < result["endpoint_seconds"]
        <= row["wall_seconds"]
        <= row["process_wall_seconds"]
    )
    return native_energy


def barrier(energies: dict[str, float], keys: list[str]) -> float | None:
    if any(key not in energies for key in keys):
        return None
    return (
        math.fsum((energies[keys[0]], *(-energies[key] for key in keys[1:])))
        * HARTREE_TO_KCAL_MOL
    )


def metric(rows: list[dict], field: str, population: int) -> dict:
    errors = [(abs(row[field]), row) for row in rows if row[field] is not None]
    if not errors:
        return {"n": 0, "population": population}
    worst, row = max(errors, key=lambda pair: pair[0])
    return {
        "n": len(errors),
        "population": population,
        "mae_kcal_mol": fmean(value for value, _ in errors),
        "max_abs_kcal_mol": worst,
        "max_set": row["set"],
        "max_index": row["index"],
        "within_1_kcal_mol": sum(value <= CHEMICAL_ACCURACY for value, _ in errors),
        "all_complete_and_within_1": (
            len(errors) == population and worst <= CHEMICAL_ACCURACY
        ),
    }


def endpoint_stats(rows: list[dict], expected: int) -> dict:
    status_counts = Counter(row["status"] for row in rows)
    times = [row["wall_seconds"] for row in rows if row["status"] == "ok"]
    return {
        "expected": expected,
        "missing": expected - len(rows),
        "outcomes": dict(sorted(status_counts.items())),
        "wall_seconds": {
            "min": min(times),
            "median": median(times),
            "max": max(times),
            "sum": sum(times),
        }
        if times
        else None,
    }


def summarize(data: dict, gpu: dict, cpu: dict) -> tuple[list[dict], dict]:
    included = {row["set"] for row in data["barriers"]}
    native = {recipe: {} for recipe in RECIPES}
    for (_, recipe), row in gpu.items():
        assert row["system"] in data["systems"]
        if row["status"] == "ok":
            native[recipe][row["system"]] = verify_native(row)
    conventional = {n: {} for n in (2, 3, 4)}
    for (key, n), row in cpu.items():
        assert key in data["systems"]
        if row["status"] == "ok":
            assert row["scf_converged"] and (
                row["nelectron"] == 1 or row["ccsd_converged"]
            )
            assert math.isfinite(row["ccsdt_hartree"])
            conventional[n][key] = row["ccsdt_hartree"]

    rows = []
    for recipe, (lo, hi) in RECIPES.items():
        cpu_composite = {}
        for key in (
            conventional[2].keys() & conventional[lo].keys() & conventional[hi].keys()
        ):
            low, high, small = (cpu[key, n] for n in (lo, hi, 2))
            cpu_composite[key] = compose(
                (low["hf_hartree"], high["hf_hartree"]),
                (low["mp2_correlation"], high["mp2_correlation"]),
                small["ccsdt_hartree"] - small["hf_hartree"] - small["mp2_correlation"],
                lo,
                hi,
            )
        for source in data["barriers"]:
            keys = [source["transition"], *source["side"]]
            measurement = barrier(native[recipe], keys)
            canonical = barrier(conventional[hi], keys)
            composite = barrier(cpu_composite, keys)
            rows.append(
                {
                    "set": source["set"],
                    "index": source["index"],
                    "reaction": source["reaction"],
                    "direction": source["direction"],
                    "recipe": recipe,
                    "ref1_kcal_mol": source["ref1_kcal_mol"],
                    "ref2_kcal_mol": source["ref2_kcal_mol"],
                    "gpu_kcal_mol": measurement,
                    "cpu_canonical_kcal_mol": canonical,
                    "cpu_composite_kcal_mol": composite,
                    "gpu_ref1_error": difference(measurement, source["ref1_kcal_mol"]),
                    "gpu_canonical_delta": difference(measurement, canonical),
                    "cpu_composite_ref1_error": difference(
                        composite, source["ref1_kcal_mol"]
                    ),
                    "cpu_canonical_ref1_error": difference(
                        canonical, source["ref1_kcal_mol"]
                    ),
                    "cpu_composite_canonical_delta": difference(composite, canonical),
                    "gpu_cpu_composite_delta": difference(measurement, composite),
                    "missing_gpu_endpoints": " ".join(
                        key for key in keys if key not in native[recipe]
                    ),
                }
            )

    matched = {
        (r["set"], r["index"])
        for r in rows
        if r["recipe"] == "DT_D" and r["gpu_kcal_mol"] is not None
    } & {
        (r["set"], r["index"])
        for r in rows
        if r["recipe"] == "TQ_D" and r["gpu_kcal_mol"] is not None
    }
    stats = {}
    for recipe in RECIPES:
        stats[recipe] = {}
        for group in (*sorted(included), "BH76"):
            selected = [
                r for r in rows if r["recipe"] == recipe and group in (r["set"], "BH76")
            ]
            paired = [r for r in selected if (r["set"], r["index"]) in matched]
            anchors = [
                r
                for r in selected
                if ((r["index"] - 1) // 2) * 2 + 1 in PAIRS[r["set"]]
            ]
            group_systems = [
                s for s in data["systems"] if group in (s.split("/")[0], "BH76")
            ]
            observed = [
                gpu[key, recipe] for key in group_systems if (key, recipe) in gpu
            ]
            stats[recipe][group] = {
                "endpoint_stats": endpoint_stats(observed, len(group_systems)),
                "gpu_vs_ref1": metric(selected, "gpu_ref1_error", len(selected)),
                "gpu_vs_ref1_on_common_coverage": metric(
                    paired, "gpu_ref1_error", len(paired)
                ),
                "gpu_vs_canonical": metric(
                    anchors, "gpu_canonical_delta", len(anchors)
                ),
                "cpu_composite_vs_ref1": metric(
                    anchors, "cpu_composite_ref1_error", len(anchors)
                ),
                "cpu_canonical_vs_ref1": metric(
                    anchors, "cpu_canonical_ref1_error", len(anchors)
                ),
                "cpu_composite_vs_canonical": metric(
                    anchors, "cpu_composite_canonical_delta", len(anchors)
                ),
                "gpu_vs_cpu_composite": metric(
                    anchors, "gpu_cpu_composite_delta", len(anchors)
                ),
            }
    expected_cpu = selected_keys(data)
    stats["reference_endpoint_stats"] = {
        str(n): endpoint_stats(
            [cpu[key, n] for key in expected_cpu if (key, n) in cpu], len(expected_cpu)
        )
        for n in (2, 3, 4)
    }
    return rows, stats


def difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((ROOT / "inputs.json").read_text())
    verify_provenance(args.raw, data)
    gpu = load_journal(sorted(args.raw.glob("gpu-*.jsonl.xz")), "recipe")
    cpu = load_journal(sorted(args.raw.glob("cpu-*.jsonl.xz")), "cardinal")
    rows, summary = summarize(data, gpu, cpu)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "barriers.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
