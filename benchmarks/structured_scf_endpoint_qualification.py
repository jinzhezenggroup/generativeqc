"""Source-matched complete HF/KS A/B gates; PySCF is diagnostic-only.

Oracle mode is CPU-only. Device mode requires a finite Slurm allocation and a
test-hook build. Diagnostic matrix downloads are outside clean endpoint timing.
Failed rows survive in the receipt and cannot supply profitability evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

GATES = {"energy": 1e-8, "density": 1e-6, "fock": 1e-6, "forces": 3e-5}
MASKS = (0, 1, 2, 4, 120, 127)


def write_json(path: Path, value: dict) -> None:
    """Never serialize nonfinite values as permissive JSON NaN tokens."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def model(case: str, method: str, moved: bool = False) -> dict:
    """Small, through-f rectangular, and 96-AO domains without molecule selectors."""
    restricted = method in ("rhf", "rks")
    representation = "spherical" if case.endswith("spherical") else "cartesian"
    if case.startswith("through-f"):
        numbers, positions = [2, 1], [[0.0, 0.0, 0.0], [0.3, 0.2, 1.6]]
        basis = "diagnostic-spdf"
    else:
        numbers = [8, 1, 1]
        positions = [[0.0, 0.0, 0.0], [1.43, 0.0, 1.11], [-1.43, 0.0, 1.11]]
        basis = "sto-3g" if case == "small" else "def2-tzvp"
        if case.startswith("large"):
            numbers *= 2
            positions += [[0.3, 0.2, 5.4], [1.66, 0.3, 6.62], [-1.18, 0.1, 6.43]]
    if moved:
        positions[-1][0] += 0.012
    return {
        "case": case,
        "method": method,
        "atomic_numbers": numbers,
        "coordinates": positions,
        "charge": (1 if restricted else 0)
        if basis == "diagnostic-spdf"
        else (0 if restricted else 1),
        "multiplicity": 1 if restricted else 2,
        "basis": basis,
        "basis_representation": representation,
        "grid": {
            "version": 1,
            "radial_points": 32,
            "angular_polar": 8,
            "angular_azimuth": 16,
        },
    }


def basis_and_inputs(spec: dict) -> tuple:
    """Share only basis/quadrature data, never a production SCF equation."""
    from generativeqc import Atom
    from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
    from generativeqc.calculator import _named_basis_shells
    from generativeqc_compiler.dft import GridSpec, MolecularGrid

    atoms = tuple(
        Atom(number, tuple(position))
        for number, position in zip(
            spec["atomic_numbers"], spec["coordinates"], strict=True
        )
    )
    basis = spec["basis"]
    if basis == "diagnostic-spdf":
        elements = tuple(
            ElementBasis(
                number,
                tuple(
                    BasisShell(angular, (str(exponent),), (("1",),))
                    for angular, exponent in shell_data
                ),
            )
            for number, shell_data in (
                (1, ((0, 1.0), (0, 0.3))),
                (2, ((0, 2.0), (0, 0.8), (1, 1.0), (2, 0.7), (3, 0.5))),
            )
        )
        digest = hashlib.sha256(repr(elements).encode()).hexdigest()
        basis = BasisSet(
            "structured-spdf-diagnostic",
            elements,
            BasisProvenance(
                "original diagnostic coefficients", "1", "GPL-3.0-or-later", digest
            ),
            representation=spec["basis_representation"],
        )
        shells = basis.shells_for(atoms)
    else:
        shells = _named_basis_shells(basis, atoms)
    inputs = {
        **spec,
        "shells": [
            {
                "atom_index": shell.atom_index,
                "angular_momentum": shell.angular_momentum,
                "primitives": [
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ],
            }
            for shell in shells
        ],
    }
    grid = MolecularGrid(
        atoms,
        GridSpec(**spec["grid"]),
        charge=spec["charge"],
        multiplicity=spec["multiplicity"],
    ).explicit()
    return basis, atoms, inputs, grid


def reference_state(spec: dict) -> dict:
    """Independent Libcint/Libxc equations in the exact public AO convention."""
    import pyscf
    from pyscf import dft, scf

    from tools.generate_validation_references import pyscf_molecule

    if pyscf.__version__ != "2.14.0" or dft.libxc.__version__ != "7.0.0":
        raise ValueError("requires verified PySCF 2.14.0 / Libxc 7.0.0")
    _, _, inputs, grid = basis_and_inputs(spec)
    molecule, scale, _ = pyscf_molecule(inputs)
    mean_field = {"rhf": scf.RHF, "uhf": scf.UHF, "rks": dft.RKS, "uks": dft.UKS}[
        spec["method"]
    ](molecule)
    is_ks = spec["method"].endswith("ks")
    if is_ks:
        mean_field.xc = "GGA_X_PBE,GGA_C_PBE"
        mean_field.grids.coords = np.asarray(grid.points)
        mean_field.grids.weights = np.asarray(grid.weights)
        mean_field.grids.non0tab = None
        mean_field.small_rho_cutoff = 0.0
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 200
    mean_field.direct_scf_tol = 1e-14
    energy = float(mean_field.kernel())
    dimension = molecule.nao_nr()
    density = np.asarray(mean_field.make_rdm1()).reshape(-1, dimension, dimension)
    raw_density = density[0] if len(density) == 1 else density
    fock = np.asarray(mean_field.get_hcore())[None] + np.asarray(
        mean_field.get_veff(molecule, raw_density)
    ).reshape(density.shape)
    overlap = np.asarray(mean_field.get_ovlp())
    residual = fock @ density @ overlap - overlap @ density @ fock
    normalization = scale[:, None] * scale[None, :]
    return {
        "energy": energy,
        "density": (density / normalization).tolist(),
        "fock": (fock * normalization).tolist(),
        "overlap": (overlap * normalization).tolist(),
        "converged": bool(mean_field.converged),
        "nbf": dimension,
        "physical_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "forces": None if is_ks else (-mean_field.nuc_grad_method().kernel()).tolist(),
        "pyscf": pyscf.__version__,
        "libxc": dft.libxc.__version__,
    }


def oracle(spec: dict) -> dict:
    """KS forces reconverge moved basis and explicit Becke grid at two FD steps."""
    result = reference_state(spec)
    if spec["method"].endswith("ks"):
        forces = []
        converged = result["converged"]
        for step in (1e-3, 5e-4):
            gradient = np.empty((len(spec["coordinates"]), 3))
            for atom in range(len(gradient)):
                for axis in range(3):
                    energies = []
                    for sign in (-1, 1):
                        displaced = json.loads(json.dumps(spec))
                        displaced["coordinates"][atom][axis] += sign * step
                        state = reference_state(displaced)
                        converged &= state["converged"]
                        energies.append(state["energy"])
                    gradient[atom, axis] = -(energies[1] - energies[0]) / (2 * step)
            forces.append(gradient)
        result["forces"] = forces[-1].tolist()
        result["force_step_error"] = float(np.max(np.abs(forces[0] - forces[1])))
        result["oracle_accepted"] = bool(
            converged and result["force_step_error"] < 2e-6
        )
    else:
        result["oracle_accepted"] = result["converged"]
    result["oracle_accepted"] &= result["physical_residual_rms"] < 1e-8
    return result


def assess(actual: dict, reference: dict) -> dict:
    """Reject shape/nonfinite/final-state mismatches before numerical comparisons."""
    errors = {}
    for field in GATES:
        observed = np.asarray(actual.get(field), dtype=float)
        expected = np.asarray(reference[field], dtype=float)
        if observed.shape != expected.shape or not np.all(np.isfinite(observed)):
            errors[field] = None
        else:
            errors[field] = float(np.max(np.abs(observed - expected)))
    accepted = bool(
        actual.get("converged")
        and actual.get("status") == 0
        and actual.get("final_state_gates", {}).get("accepted", False)
        and reference["oracle_accepted"]
        and all(
            error is not None and error < GATES[field]
            for field, error in errors.items()
        )
    )
    return {"accepted": accepted, "errors": errors, "gates": GATES}


def hf_probe(spec: dict, folder: Path, driver: Path, mask: int, batch: int) -> list:
    """Complete CUDA HF with cached cold/warm/moved homogeneous batch owners."""
    _, _, inputs, _ = basis_and_inputs(spec)
    fixture = folder / "fixture.txt"
    fields = [
        (
            f"{len(spec['coordinates'])} {len(inputs['shells'])} "
            f"{int(spec['basis_representation'] == 'spherical')} "
            f"{int(spec['method'] == 'uhf')} {spec['charge']} {spec['multiplicity']}"
        )
    ]
    fields += [
        " ".join(map(str, (number, *position)))
        for number, position in zip(
            spec["atomic_numbers"], spec["coordinates"], strict=True
        )
    ]
    for shell in inputs["shells"]:
        fields.append(
            f"{shell['atom_index']} {shell['angular_momentum']} "
            f"{len(shell['primitives'])}"
        )
        fields += [
            f"{exponent:.17g} {coefficient:.17g}"
            for exponent, coefficient in shell["primitives"]
        ]
    fixture.write_text("\n".join(fields) + "\n")
    with (folder / "stderr.log").open("w") as stderr:
        completed = subprocess.run(
            [str(driver), str(fixture), str(mask), str(batch)],
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            check=True,
        )
    (folder / "native.json").write_text(completed.stdout)
    native = json.loads(completed.stdout)
    from generativeqc.autotune import source_identity

    if native["source_identity"] != source_identity(ROOT):
        raise ValueError("native HF driver loaded a different scientific source")
    if (
        Path(native["library_path"]).resolve()
        != Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    ):
        raise ValueError("native HF driver did not load the selected endpoint library")
    rows = []
    for row in native["rows"]:
        dimension = row["nbf"]
        spins = 1 if spec["method"] == "rhf" else 2
        fock = np.asarray(row["fock"]).reshape(batch, spins, dimension, dimension)
        overlap = np.asarray(row["overlap"]).reshape(batch, dimension, dimension)
        orthogonalizer = np.asarray(row["orthogonalizer"]).reshape(
            batch, dimension, dimension
        )
        coefficients = np.asarray(row["coefficients"]).reshape(
            batch, spins, dimension, dimension
        )
        eigenvalues = np.asarray(row["eigenvalues"]).reshape(batch, spins, dimension)
        for index, item in enumerate(row["items"]):
            if item["status"] == 0:
                density = np.asarray(item["density"]).reshape(
                    spins, dimension, dimension
                )
                residual = (
                    fock[index] @ density @ overlap[index]
                    - overlap[index] @ density @ fock[index]
                )
                metric = (
                    coefficients[index].swapaxes(-1, -2)
                    @ overlap[index]
                    @ coefficients[index]
                )
                inverse_root = (
                    orthogonalizer[index] @ overlap[index] @ orthogonalizer[index]
                )
                eigen_residual = (
                    fock[index] @ coefficients[index]
                    - (overlap[index] @ coefficients[index])
                    * eigenvalues[index][:, None, :]
                )
                final_gates = {
                    "physical_residual_rms": float(
                        np.sqrt(np.mean(residual * residual))
                    ),
                    "orthonormality_error": float(
                        np.max(np.abs(metric - np.eye(dimension)))
                    ),
                    "inverse_root_error": float(
                        np.max(np.abs(inverse_root - np.eye(dimension)))
                    ),
                    "generalized_eigen_error": float(np.max(np.abs(eigen_residual))),
                }
                final_gates["accepted"] = all(
                    np.isfinite(value) and value < 2e-7
                    for value in final_gates.values()
                )
            else:
                density = np.asarray([])
                final_gates = {"accepted": False}
            rows.append(
                {
                    **row,
                    **item,
                    "item": index,
                    "density": density.tolist(),
                    "fock": fock[index].tolist(),
                    "final_state_gates": final_gates,
                    "forces": np.asarray(item["forces"]).reshape(-1, 3).tolist(),
                }
            )
    return rows


def ks_probe(spec: dict, folder: Path, library: Path, mask: int, batch: int) -> list:
    """Public energy+analytic-force endpoint, followed by token-checked snapshots."""
    from generativeqc import Calculator
    from generativeqc._ks_snapshot import NativeKsSnapshot
    from generativeqc.ks import KsOptions
    from generativeqc_compiler.dft import GridSpec

    basis, atoms, _, _ = basis_and_inputs(spec)
    start = time.perf_counter()
    calculator = Calculator(
        method="pbe-" + spec["method"],
        basis=basis,
        device="cuda",
        basis_representation=spec["basis_representation"],
        precision="fp64",
        density_fitting="none",
        ks_options=KsOptions(grid=GridSpec(**spec["grid"])),
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    hook = calculator._library.tensor_structured_scf_qualification_for_test_v1
    hook.argtypes, hook.restype = [ctypes.c_int], ctypes.c_bool
    diagnostic = (
        calculator._library.tensor_structured_scf_preparation_diagnostic_for_test_v1
    )
    diagnostic.restype = ctypes.c_char_p
    if not hook(mask):
        raise ValueError("diagnostic route rejected")
    rows = []
    try:
        with calculator.prepare_batch(
            [atoms] * batch,
            charges=[spec["charge"]] * batch,
            multiplicities=[spec["multiplicity"]] * batch,
        ) as prepared:
            preparation = time.perf_counter() - start
            for phase in range(5):
                moved = model(spec["case"], spec["method"], phase >= 3)
                begin = time.perf_counter()
                result = prepared.execute(
                    [moved["coordinates"]] * batch if phase >= 3 else None,
                    properties=("energy", "forces"),
                    strict=False,
                )
                seconds = time.perf_counter() - begin
                for index, item in enumerate(result.items):
                    row = {
                        "phase": phase,
                        "item": index,
                        "status": item.status,
                        "converged": item.converged,
                        "backend": item.executed_backend,
                        "energy": item.energy if np.isfinite(item.energy) else None,
                        "forces": None if item.forces is None else item.forces.tolist(),
                        "iterations": item.iterations,
                        "complete_seconds": seconds,
                        "preparation_seconds": preparation if phase == 0 else 0.0,
                        "cold_total_seconds": preparation + seconds
                        if phase == 0
                        else None,
                        "resource_diagnostics": prepared.resource_diagnostics,
                        "diagnostic": None
                        if item.ks_diagnostic is None
                        else item.ks_diagnostic.to_payload(),
                    }
                    if item.succeeded:
                        from benchmarks.ks_matrix_provider_qualification import (
                            final_state_gates,
                        )

                        snapshot = NativeKsSnapshot(prepared, index)
                        try:
                            snapshot.check_current()
                            dimension, spins, natoms = map(int, snapshot.metadata[1:4])
                            offset = 2 + natoms * 4
                            entries = spins * dimension * dimension
                            for field in ("density", "fock"):
                                row[field] = (
                                    snapshot.values[offset : offset + entries]
                                    .reshape(spins, dimension, dimension)
                                    .tolist()
                                )
                                offset += entries
                            row["metadata"] = snapshot.metadata
                            row["final_state_gates"] = final_state_gates(
                                snapshot.metadata, snapshot.values, spec
                            )
                            row["bindings"] = json.loads(diagnostic().decode())
                        finally:
                            snapshot.close()
                    rows.append(row)
                    write_json(folder / "partial.json", {"rows": rows})
    finally:
        hook(-1)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("oracle", "device"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=(
            "small",
            "through-f-cartesian",
            "through-f-spherical",
            "large-cartesian",
            "large-spherical",
        ),
        required=True,
    )
    parser.add_argument("--method", choices=("rhf", "uhf", "rks", "uks"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--mask", type=int, choices=MASKS, default=127)
    parser.add_argument("--batch", type=int, choices=(1, 3), default=1)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = model(args.case, args.method)
    if args.mode == "oracle":
        for moved in (False, True):
            write_json(
                args.output / f"{int(moved)}.json",
                oracle(model(args.case, args.method, moved)),
            )
        return 0
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("device qualification requires scheduler-assigned visibility")
    if (
        not args.library
        or not args.reference
        or (args.method.endswith("hf") and not args.driver)
    ):
        parser.error("device requires --library, --reference, and HF --driver")
    os.environ["GENERATIVEQC_LIBRARY"] = str(args.library.resolve())
    os.environ["LD_LIBRARY_PATH"] = (
        str(args.library.resolve().parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["GENERATIVEQC_CUDA_KS_CHUNK"] = "2" if args.graph else "1"
    os.environ["GENERATIVEQC_CUDA_KS_REPLAY"] = "1" if args.graph else "0"
    from generativeqc.autotune import source_identity

    loaded = ctypes.CDLL(str(args.library.resolve()))
    loaded.generativeqc_get_source_identity.restype = ctypes.c_char_p
    if loaded.generativeqc_get_source_identity().decode() != source_identity(ROOT):
        raise ValueError("selected library does not match scientific source")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("qualification requires clean committed source")
    rows = (
        hf_probe(spec, args.output, args.driver.resolve(), args.mask, args.batch)
        if args.method.endswith("hf")
        else ks_probe(spec, args.output, args.library, args.mask, args.batch)
    )
    for row in rows:
        reference = json.loads(
            (args.reference / f"{int(row['phase'] >= 3)}.json").read_text()
        )
        row["assessment"] = assess(row, reference)
    receipt = {
        "status": "PASS"
        if all(row["assessment"]["accepted"] for row in rows)
        else "FAIL",
        "model": spec,
        "rows": rows,
        "mask": args.mask,
        "batch": args.batch,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "profitable": None,
        "default_promoted": False,
    }
    write_json(args.output / "receipt.json", receipt)
    return int(receipt["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
