"""Prepare JK basis metadata, independent RHF audits, and all-repeat experiment gates.

PySCF is used explicitly for offline basis metadata and the independent oracle,
never to generate the density consumed by the native production endpoint.
Raw outputs belong under .artifacts/, not benchmarks/results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

from benchmarks._retention import raw_output_path

if TYPE_CHECKING:
    from pyscf.gto import Mole


def read_molecule(path: Path) -> Mole:
    """Reconstruct atom-specific raw orbital shells without a named-basis assumption."""
    from pyscf import gto

    tokens = iter(path.read_text().split())
    atom_count, shell_count, _, _ = (int(next(tokens)) for _ in range(4))
    atoms = []
    basis = {}
    for index in range(atom_count):
        element = gto.mole._symbol(int(next(tokens)))
        label = f"{element}{index}"
        atoms.append((label, tuple(float(next(tokens)) for _ in range(3))))
        basis[label] = []
    for _ in range(shell_count):
        atom, angular, primitives = (int(next(tokens)) for _ in range(3))
        shell = [angular]
        shell.extend(
            [float(next(tokens)), float(next(tokens))] for _ in range(primitives)
        )
        basis[atoms[atom][0]].append(shell)
    return gto.M(atom=atoms, basis=basis, unit="Bohr", verbose=0, max_memory=32000)


def prepare_case(case: str, destination: Path) -> None:
    """Export qualification geometry/raw basis metadata; never compute a guess or energy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if case in {"moved", "budget8"}:
        source = (
            Path(__file__).resolve().parent
            / "results/df-lambda-gemm-20261004/ethane230.input"
        )
        lines = source.read_text().splitlines()
        if case == "moved":
            coordinates = lines[2].split()
            coordinates[1] = f"{float(coordinates[1]) + 0.02:.17g}"
            lines[2] = " ".join(coordinates)
        else:
            dimensions = lines[0].split()
            dimensions[3] = str(8 << 30)
            lines[0] = " ".join(dimensions)
        destination.write_text("\n".join(lines) + "\n")
        return
    if case != "propane322":
        raise ValueError("unknown qualification case")
    from pyscf import df, gto

    atoms = [
        ("C", (-1.26, 0.88, 0)),
        ("C", (0, 0, 0)),
        ("C", (1.26, 0.88, 0)),
        ("H", (-2.18, 0.30, 0)),
        ("H", (-1.29, 1.52, 0.88)),
        ("H", (-1.29, 1.52, -0.88)),
        ("H", (0, -0.63, 0.89)),
        ("H", (0, -0.63, -0.89)),
        ("H", (2.18, 0.30, 0)),
        ("H", (1.29, 1.52, 0.88)),
        ("H", (1.29, 1.52, -0.88)),
    ]
    mol = gto.M(atom=atoms, basis="aug-cc-pvtz", unit="Angstrom", verbose=0)
    auxiliary = df.addons.make_auxmol(mol, "aug-cc-pvtz-ri")

    def shells(system: Mole) -> list:
        """Separate general contractions into the native input's one-contraction shells."""
        rows = []
        for atom in range(system.natm):
            for angular, *primitives in system._basis[system.atom_symbol(atom)]:
                for contraction in range(len(primitives[0]) - 1):
                    rows.append(
                        (
                            atom,
                            angular,
                            [
                                (entry[0], entry[contraction + 1])
                                for entry in primitives
                            ],
                        )
                    )
        return rows

    orbital, correlation = shells(mol), shells(auxiliary)
    with destination.open("w") as stream:
        stream.write(f"{mol.natm} {len(orbital)} {len(correlation)} {64 << 30}\n")
        for atom, position in enumerate(mol.atom_coords()):
            stream.write(
                f"{mol.atom_charge(atom)} "
                + " ".join(f"{value:.17g}" for value in position)
                + "\n"
            )
        for atom, angular, primitives in [*orbital, *correlation]:
            stream.write(f"{atom} {angular} {len(primitives)}\n")
            for exponent, coefficient in primitives:
                stream.write(f"{exponent:.17g} {coefficient:.17g}\n")


def prepare_jk(source: Path, destination: Path, auxiliary: str) -> None:
    """Export raw JK-fit contractions on exactly the input's atom order and geometry."""
    from pyscf import df

    mol = read_molecule(source)
    auxmol = df.addons.make_auxmol(mol, auxiliary)
    shells = []
    for atom in range(mol.natm):
        for shell in auxmol._basis[auxmol.atom_symbol(atom)]:
            angular, *rows = shell
            for contraction in range(len(rows[0]) - 1):
                shells.append(
                    (atom, angular, [(row[0], row[contraction + 1]) for row in rows])
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        stream.write(f"{len(shells)}\n")
        for atom, angular, rows in shells:
            stream.write(f"{atom} {angular} {len(rows)}\n")
            for exponent, coefficient in rows:
                stream.write(f"{exponent:.17g} {coefficient:.17g}\n")
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "jk_basis": auxiliary,
                "nbf": mol.nao_nr(),
                "naux": auxmol.nao_nr(),
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "jk_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "contract": "offline basis metadata only; no SCF/integral oracle",
            },
            indent=2,
        )
        + "\n"
    )


def oracle(source: Path, destination: Path, threads: int) -> None:
    """Run a fresh independent conventional RHF, retaining gauge-independent state."""
    import pyscf
    from pyscf import lib, scf

    lib.num_threads(threads)
    mol = read_molecule(source)
    reference = scf.RHF(mol)
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.direct_scf_tol = 0
    reference.max_cycle = 150
    start = perf_counter()
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("independent conventional RHF did not converge")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "pyscf_version": pyscf.__version__,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "reference_mode": "conventional FP64 RHF",
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reference_energy": float(reference.e_tot),
                "density": reference.make_rdm1().ravel().tolist(),
                "orbital_energies": reference.mo_energy.tolist(),
                "seconds": perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )


def maximum_difference(first: list[float], second: list[float]) -> float:
    """Compare all entries, rejecting truncated or incompatible payloads."""
    if (
        not first
        or len(first) != len(second)
        or not all(math.isfinite(value) for value in (*first, *second))
    ):
        raise ValueError("incompatible or empty physical arrays")
    return max(abs(left - right) for left, right in zip(first, second, strict=True))


def metric_median(rows: list[dict], name: str) -> float | None:
    """An uncounted retry invalidates the whole group census, not only that sample."""
    values = [row[name] for row in rows]
    return None if any(value is None for value in values) else median(values)


def correlation_audit(
    rows: list[dict], oracle_path: Path, differences_path: Path
) -> dict:
    """Audit same-Hamiltonian energy and two independent directional force differences.

    These retained PySCF references complement the fresh RHF oracle. They are
    not an independent analytic oracle for every Cartesian force component.
    """
    oracle_record = json.loads(oracle_path.read_text())
    finite_differences = json.loads(differences_path.read_text())
    if oracle_record["pyscf_version"] != "2.14.0":
        raise ValueError("requires the retained pinned PySCF 2.14.0 oracle")
    # The pinned oracle schema stores its central geometry in Angstrom.
    center_atoms = oracle_record.get("atoms_bohr")
    if center_atoms is None:
        atomic_numbers = {"H": 1, "C": 6}
        center_atoms = [
            (atomic_numbers[element], [value / 0.52917721092 for value in coordinates])
            for element, coordinates in oracle_record["atoms_angstrom"]
        ]
    force_rows = [row for row in rows if row["endpoint"] == "forces"]
    energy_rows = [row for row in rows if row["endpoint"] in {"energy", "forces"}]
    if not energy_rows:
        raise ValueError("correlation audit requires a correlation endpoint")
    if any(
        row["nbf"] != oracle_record["nbf"]
        or row["naux_correlation"] != oracle_record["naux"]
        for row in energy_rows
    ):
        raise ValueError("correlation oracle dimensions differ from native input")
    energy_error = max(
        abs(row["energy"] - oracle_record["total_energy"]) for row in energy_rows
    )
    directional = []
    for difference in finite_differences["rows"]:
        step = difference["step_bohr"]
        minus, plus = difference["minus"], difference["plus"]
        if minus["pyscf_version"] != "2.14.0" or plus["pyscf_version"] != "2.14.0":
            raise ValueError("finite-difference version mismatch")
        for (low_z, low), (high_z, high), (center_z, center) in zip(
            minus["atoms_bohr"],
            plus["atoms_bohr"],
            center_atoms,
            strict=True,
        ):
            if (
                low_z != high_z
                or low_z != center_z
                or any(
                    abs((lower + upper) / 2 - middle) > 1e-12
                    for lower, upper, middle in zip(low, high, center, strict=True)
                )
            ):
                raise ValueError(
                    "finite-difference center differs from correlation oracle"
                )
        direction = [
            (upper - lower) / (2 * step)
            for (_, low), (_, high) in zip(
                minus["atoms_bohr"], plus["atoms_bohr"], strict=True
            )
            for lower, upper in zip(low, high, strict=True)
        ]
        derivative = (plus["total_energy"] - minus["total_energy"]) / (2 * step)
        if force_rows:
            error = max(
                abs(
                    sum(
                        force * component
                        for force, component in zip(
                            row["forces"], direction, strict=True
                        )
                    )
                    + derivative
                )
                for row in force_rows
            )
            directional.append({"step_bohr": step, "absolute_error": error})
    if {item["step_bohr"] for item in finite_differences["rows"]} != {1e-4, 3e-5}:
        raise ValueError("both retained finite-difference steps are required")
    translation = (
        None
        if not force_rows
        else max(
            abs(sum(row["forces"][axis::3])) for row in force_rows for axis in range(3)
        )
    )
    residual = (
        None
        if not force_rows
        else max(
            abs(row[name])
            for row in force_rows
            for name in ("lambda_residual", "z_residual", "stationarity")
        )
    )
    passed = (
        math.isfinite(energy_error)
        and energy_error <= 3e-9
        and all(
            math.isfinite(item["absolute_error"]) and item["absolute_error"] <= 3e-7
            for item in directional
        )
        and (translation is None or translation <= 3e-8)
        and (residual is None or residual <= 1e-9)
    )
    return {
        "energy_maximum_error": energy_error,
        "directional_force_errors": directional,
        "translation_maximum": translation,
        "response_residual_maximum": residual,
        "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "finite_differences_sha256": hashlib.sha256(
            differences_path.read_bytes()
        ).hexdigest(),
        "gates_pass": passed,
    }


def summarize(paths: list[Path], independent: Path | None) -> dict:
    """Gate every repeat pair; never select accurate samples by iteration count."""
    rows = [json.loads(path.read_text()) for path in paths]
    audit_fields = (
        "reference_energy",
        "energy",
        "energy_change",
        "density_rms",
        "commutator_residual",
        "canonical_density_drift",
        "eigen_residual",
        "orthogonality_max",
    )
    if any(not math.isfinite(row[name]) for row in rows for name in audit_fields):
        raise ValueError("nonfinite final physical reference")
    reference = None if independent is None else json.loads(independent.read_text())
    result = {"groups": {}, "all_gates_pass": True}
    for endpoint in sorted({row["endpoint"] for row in rows}):
        controls = [
            row
            for row in rows
            if row["mode"] == "direct" and row["endpoint"] == endpoint
        ]
        candidates = [
            row
            for row in rows
            if row["mode"] == "df-direct" and row["endpoint"] == endpoint
        ]
        if not controls or not candidates:
            raise ValueError("both Direct and DF-to-Direct controls are required")
        for pre_tolerance in sorted({row["pre_tolerance"] for row in candidates}):
            selected = [
                row for row in candidates if row["pre_tolerance"] == pre_tolerance
            ]
            errors = {
                "reference_energy": max(
                    abs(control["reference_energy"] - candidate["reference_energy"])
                    for control in controls
                    for candidate in selected
                ),
                "energy": max(
                    abs(control["energy"] - candidate["energy"])
                    for control in controls
                    for candidate in selected
                ),
                "density": max(
                    maximum_difference(control["density"], candidate["density"])
                    for control in controls
                    for candidate in selected
                ),
                "orbital_energies": max(
                    maximum_difference(
                        control["orbital_energies"], candidate["orbital_energies"]
                    )
                    for control in controls
                    for candidate in selected
                ),
                "forces": None
                if endpoint != "forces"
                else max(
                    maximum_difference(control["forces"], candidate["forces"])
                    for control in controls
                    for candidate in selected
                ),
            }
            gates = {
                "reference_energy": 1e-10,
                "energy": 1e-9,
                "density": 1e-9,
                "orbital_energies": 1e-9,
                "forces": 3e-9,
            }
            passed = all(
                value is None or value <= gates[name] for name, value in errors.items()
            )
            audits = controls + selected
            passed &= all(
                row["converged"]
                and abs(row["energy_change"]) <= 1e-12
                and row["density_rms"] <= 1e-11
                and row["commutator_residual"] <= 1e-9
                and row["canonical_density_drift"] <= 1e-9
                and row["eigen_residual"] <= 1e-9
                and row["orthogonality_max"] <= 1e-9
                for row in audits
            )
            independent_errors = None
            if reference is not None:
                independent_errors = {
                    "energy": max(
                        abs(row["reference_energy"] - reference["reference_energy"])
                        for row in audits
                    ),
                    "density": max(
                        maximum_difference(row["density"], reference["density"])
                        for row in audits
                    ),
                    "orbital_energies": max(
                        maximum_difference(
                            row["orbital_energies"], reference["orbital_energies"]
                        )
                        for row in audits
                    ),
                }
                passed &= (
                    independent_errors["energy"] <= 1e-10
                    and independent_errors["density"] <= 1e-9
                    and independent_errors["orbital_energies"] <= 1e-9
                )
            key = f"{endpoint}/pre-{pre_tolerance:g}"
            result["groups"][key] = {
                "repeats": {"control": len(controls), "candidate": len(selected)},
                "control_median": {
                    name: metric_median(controls, name)
                    for name in (
                        "rhf_seconds",
                        "endpoint_seconds",
                        "direct_iterations",
                        "direct_fock_builds",
                    )
                },
                "candidate_median": {
                    name: metric_median(selected, name)
                    for name in (
                        "rhf_seconds",
                        "endpoint_seconds",
                        "pre_seconds",
                        "pre_iterations",
                        "direct_iterations",
                        "direct_fock_builds",
                    )
                },
                "maximum_errors": errors,
                "independent_maximum_errors": independent_errors,
                "gates": gates,
                "gates_pass": bool(passed),
                "fallbacks": sum(
                    row["pre_fallback"] or row["seed_fallback"] for row in selected
                ),
            }
            result["all_gates_pass"] &= bool(passed)
    return result


def main() -> None:
    """Separate preparation and oracle work from measured native executions."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    geometry = commands.add_parser("prepare-case")
    geometry.add_argument(
        "--case", choices=("moved", "budget8", "propane322"), required=True
    )
    geometry.add_argument("--output", type=raw_output_path, required=True)
    prepare = commands.add_parser("prepare-jk")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=raw_output_path, required=True)
    prepare.add_argument("--basis", default="aug-cc-pvtz-jkfit")
    independent = commands.add_parser("oracle")
    independent.add_argument("--input", type=Path, required=True)
    independent.add_argument("--output", type=raw_output_path, required=True)
    independent.add_argument("--threads", type=int, default=8)
    summary = commands.add_parser("summarize")
    summary.add_argument("inputs", nargs="+", type=Path)
    summary.add_argument("--oracle", type=Path)
    summary.add_argument("--correlation-oracle", type=Path)
    summary.add_argument("--finite-differences", type=Path)
    summary.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-case":
        prepare_case(args.case, args.output)
    elif args.command == "prepare-jk":
        prepare_jk(args.input, args.output, args.basis)
    elif args.command == "oracle":
        oracle(args.input, args.output, args.threads)
    else:
        record = summarize(args.inputs, args.oracle)
        if bool(args.correlation_oracle) != bool(args.finite_differences):
            parser.error(
                "correlation oracle and finite differences must be supplied together"
            )
        if args.correlation_oracle:
            record["correlation_audit"] = correlation_audit(
                [json.loads(path.read_text()) for path in args.inputs],
                args.correlation_oracle,
                args.finite_differences,
            )
            record["all_gates_pass"] &= record["correlation_audit"]["gates_pass"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        if not record["all_gates_pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
