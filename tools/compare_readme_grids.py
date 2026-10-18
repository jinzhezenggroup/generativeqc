"""Compare independent E+F quadrature convergence, not same-grid engine parity.

The baseline is an explicitly chosen denser GPU4PySCF quadrature, not an exact
integral. Native timings and engine-parity gates are reduced separately with
render_readme_dft; a same-grid pass alone never qualifies a smaller grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc import GridSpec

from benchmarks._readme_grid import grid_directory, grid_work, parse_grid
from benchmarks.readme_dft import METHODS, SCHEMA, specification
from benchmarks.readme_omol25 import SIZES, reference_baselines


def reference_population(
    root: Path, dimensions: tuple[int, int, int], method: str, atoms: int
) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]], dict[str, Any]]:
    """Retain provenance and valid geometry oracles even after a later timeout."""
    path = root / grid_directory(dimensions) / method / str(atoms) / "reference.json"
    receipt: dict[str, Any] = {"file": str(path), "status": "not_run"}
    outcome = path.with_suffix(".outcome")
    if outcome.exists():
        receipt["outcome"] = json.loads(outcome.read_text())
    if not path.exists():
        return None, {}, receipt
    payload = path.read_bytes()
    raw = json.loads(payload)
    if raw.get("schema") != SCHEMA or raw.get("engine") != "reference":
        raise ValueError(f"not an independent README reference: {path}")
    scientific = raw["protocol"]
    expected_grid = json.loads(json.dumps(asdict(GridSpec(*dimensions))))
    actual_grid = scientific.get("grid")
    if (
        scientific["atoms"] != atoms
        or scientific["method"]
        != ("RHF" if method == "hf" else f"{method.upper()}/RKS")
        or actual_grid is None
        or actual_grid != expected_grid
    ):
        raise ValueError(f"reference is filed under the wrong grid or case: {path}")
    receipt.update(status=raw["status"], sha256=hashlib.sha256(payload).hexdigest())
    return scientific, reference_baselines(raw, scientific, allow_missing=True), receipt


def compare(
    root: Path,
    dimensions: tuple[int, int, int],
    baseline: tuple[int, int, int],
    method: str,
    atoms: int,
    *,
    energy_tolerance: float,
    force_tolerance: float,
) -> dict[str, Any]:
    """Keep missing evidence unknown and reject all non-grid protocol changes."""
    for tolerance in (energy_tolerance, force_tolerance):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("convergence tolerances must be finite and positive")
    scientific, rows, receipt = reference_population(root, dimensions, method, atoms)
    dense, dense_rows, dense_receipt = reference_population(
        root, baseline, method, atoms
    )
    if (
        scientific is not None
        and dense is not None
        and {key: value for key, value in scientific.items() if key != "grid"}
        != {key: value for key, value in dense.items() if key != "grid"}
    ):
        raise ValueError("cross-grid protocols differ beyond quadrature")
    differences = []
    for geometry in (0, 1):
        row, oracle = rows.get(geometry), dense_rows.get(geometry)
        result: dict[str, Any] = {"geometry": geometry, "gate": None}
        if row is not None and oracle is not None:
            energy_error = abs(row["energy"] - oracle["energy"])
            force_error = float(
                np.max(np.abs(np.asarray(row["forces"]) - oracle["forces"]))
            )
            result.update(
                energy_error_hartree=energy_error,
                force_max_error_hartree_per_bohr=force_error,
                gate=energy_error <= energy_tolerance
                and force_error <= force_tolerance,
            )
        differences.append(result)
    gates = [row["gate"] for row in differences]
    return {
        "method": method,
        "density_fitting": (scientific or dense or {}).get("density_fitting"),
        "atoms": atoms,
        "grid_work": grid_work(
            atoms, GridSpec(*dimensions), has_vv10=specification(method).has_vv10
        ),
        "baseline_grid": list(baseline),
        "is_baseline": dimensions == baseline,
        "reference": receipt,
        "baseline_reference": dense_receipt,
        "geometries": differences,
        "convergence_gate": None
        if dimensions == baseline or None in gates
        else all(gates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--grids", type=parse_grid, nargs="+", required=True)
    parser.add_argument("--baseline-grid", type=parse_grid, required=True)
    parser.add_argument(
        "--methods", choices=METHODS[1:], nargs="+", default=["wb97m-v"]
    )
    parser.add_argument("--sizes", type=int, choices=SIZES, nargs="+", default=[3, 12])
    parser.add_argument("--energy-tolerance", type=float, required=True)
    parser.add_argument("--force-tolerance", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        compare(
            args.raw,
            dimensions,
            args.baseline_grid,
            method,
            atoms,
            energy_tolerance=args.energy_tolerance,
            force_tolerance=args.force_tolerance,
        )
        for method in args.methods
        for atoms in args.sizes
        for dimensions in args.grids
    ]
    summary = {
        "schema": "generativeqc.readme-grid-convergence.v1",
        "energy_tolerance_hartree": args.energy_tolerance,
        "force_tolerance_hartree_per_bohr": args.force_tolerance,
        "baseline_is_exact_integral": False,
        "comparisons": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grid-convergence.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Independent quadrature convergence",
        "",
        "Differences are against the selected GPU4PySCF grid, not exact integrals.",
        "Both original and moved geometries must pass. Engine parity is a separate gate.",
        "",
        "| Method | Mode | Atoms | Grid | Points/atom | Max ΔE (Eh) | Max ΔF (Eh/bohr) | Grid gate |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        work = result["grid_work"]
        geometries = result["geometries"]
        complete = all(row["gate"] is not None for row in geometries)
        energy = (
            f"{max(row['energy_error_hartree'] for row in geometries):.3e}"
            if complete
            else "—"
        )
        force = (
            f"{max(row['force_max_error_hartree_per_bohr'] for row in geometries):.3e}"
            if complete
            else "—"
        )
        gate = (
            "baseline"
            if result["is_baseline"]
            else {True: "pass", False: "fail", None: "unavailable"}[
                result["convergence_gate"]
            ]
        )
        dimensions = "×".join(str(value) for value in work["dimensions"])
        mode = {True: "DF", False: "Direct", None: "unknown"}[result["density_fitting"]]
        lines.append(
            f"| {result['method']} | {mode} | {result['atoms']} | {dimensions} | {work['points_per_atom']} | {energy} | {force} | {gate} |"
        )
    (args.output / "grid-convergence.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
