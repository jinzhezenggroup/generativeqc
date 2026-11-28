"""Qualify fresh SCF plus a raw analytical water Hessian on one Slurm GPU.

Run `measure` once per engine in separate processes within the same allocation,
then run `compare` on the two receipts. An accuracy pass alone is not a win.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._support import raw_output_path

if TYPE_CHECKING:
    from generativeqc.basis import BasisSet

_GATE = 2e-8  # Ha/Bohr², applied before symmetrization or translation repair.


def water_cluster(waters: int) -> list[tuple[str, list[float]]]:
    """Perturb a 4×4×2 oxygen lattice and rotate neutral monomers in Bohr."""
    if not 1 <= waters <= 32:
        raise ValueError("water fixture supports 1–32 monomers")
    generator = np.random.default_rng(961128)
    angle = np.deg2rad(104.5 / 2)
    monomer = 1.81 * np.array(
        [
            [0.0, 0.0, 0.0],
            [np.sin(angle), 0, np.cos(angle)],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    atoms = []
    for index in range(waters):
        center = 6.2 * np.array([index % 4, (index // 4) % 4, index // 16])
        center += generator.uniform(-0.23, 0.23, size=3)
        rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        atoms.extend(
            zip(("O", "H", "H"), (center + monomer @ rotation).tolist(), strict=True)
        )
    return atoms


def explicit_primitives(basis: BasisSet) -> dict[str, list[list[Any]]]:
    """Export the owned H/O snapshot; a shared catalog name is not an identity."""
    if basis.representation != "cartesian":
        raise ValueError("water fixture requires a Cartesian basis")
    records = basis.by_element
    result = {}
    for symbol, number in (("H", 1), ("O", 8)):
        element = records[number]
        if (
            element.ecp_core_electrons
            or element.ecp_data
            or element.nuclear_charge != number
        ):
            raise ValueError(
                "water fixture requires unmodified all-electron primitives"
            )
        result[symbol] = [
            [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *[float(row[index]) for row in shell.coefficients],
                    ]
                    for index, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
    return result


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _reference_basis(mol: Any, expected: dict[str, list[list[Any]]]) -> str:
    # Inspect the molecule's parsed primitive input, not its original `basis` argument.
    # The parser may sort shells by angular momentum, preserving order within each l.
    actual = mol._basis
    if (
        any(
            actual.get(symbol) != sorted(shells, key=lambda shell: shell[0])
            for symbol, shells in expected.items()
        )
        or not mol.cart
    ):
        raise ValueError("GPU4PySCF did not build the required Cartesian primitives")
    return _digest(expected)


def measure(args: argparse.Namespace) -> None:
    """Measure one complete endpoint and authenticate its input and device."""
    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._model_resolution import _named_basis_record
    from generativeqc.response_solver import GMRESOptions
    from generativeqc_compiler.common.resources import ResourceBudget

    job = os.environ["SLURM_JOB_ID"]
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise ValueError("allocate exactly one GPU with Slurm")
    device = cp.cuda.Device(0).pci_bus_id
    query = ("nvidia-smi", "-i", device, "--format=csv,noheader,nounits")
    uuid, capacity = (
        subprocess.check_output(
            (*query, "--query-gpu=uuid,memory.total"), text=True, timeout=15
        )
        .strip()
        .split(", ")
    )
    out = raw_output_path(args.prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = out.with_suffix(".json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    atoms = water_cluster(args.waters)
    symbols = {symbol for symbol, _ in atoms}
    grid = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
    fitted = args.provider == "df"
    basis = _named_basis_record("sto-3g", "cartesian")
    primitives = {
        symbol: shells
        for symbol, shells in explicit_primitives(basis).items()
        if symbol in symbols
    }
    settings = {
        "method": args.method,
        "provider": args.provider,
        "waters": args.waters,
        "geometry_hash": _digest(atoms),
        "grid_hash": _digest(asdict(grid)),
        "grid_points": len(atoms) * 12 * 6 * 12,
        "auxbasis_response": 2 if fitted else None,
        "fp64": True,
    }
    with out.with_suffix(".csv").open("w") as csv:
        monitor = subprocess.Popen(
            (*query, "--query-gpu=memory.used", "-lms", "250"), stdout=csv
        )
        try:
            started = perf_counter()
            if args.engine == "native":
                calculator = Calculator(
                    method=f"{args.method}-rks",
                    device="cuda",
                    precision="fp64",
                    basis=basis,
                    basis_representation="cartesian",
                    density_fitting="cuda" if fitted else "none",
                    auxiliary_basis=basis if fitted else None,
                    ks_options=KsOptions(grid=grid),
                    resource_budget=ResourceBudget(
                        device_bytes=28 << 30, device_reserve_bytes=1 << 30
                    ),
                    max_iterations=400,
                    energy_tolerance=1e-13,
                    density_tolerance=1e-9,
                )
                result = calculator.hessian(
                    atoms,
                    block_size=1,
                    solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
                    grid_budget_bytes=8192 << 20,
                    integral_budget_bytes=6144 << 20,
                )
                matrix = result.matrix
                elapsed = perf_counter() - started
                diagnostics = {
                    key: value
                    for key, value in result.diagnostics.items()
                    if key != "resources"
                }
                if (diagnostics["df_workspace_bytes"] > 0) != fitted:
                    raise ValueError("native endpoint used the wrong provider")
                orbital_hash = _digest(primitives)
                auxiliary_hash = orbital_hash if fitted else None
            else:
                import gpu4pyscf

                from benchmarks.readme_wb97mv import reference_engine

                reference = reference_engine(
                    atoms, primitives, grid, xc=args.method.upper()
                )
                reference.mol.cart = True
                orbital_hash = _reference_basis(reference.mol, primitives)
                if fitted:
                    reference = reference.density_fit(auxbasis=primitives)
                # Use the same reference stationarity targets for Direct and
                # DF; a looser Direct SCF can obscure Hessian errors at 2e-8.
                reference.conv_tol = 1e-13
                reference.conv_tol_grad = 1e-9
                reference.conv_tol_cpscf = 1e-11
                reference.direct_scf_tol = 1e-14
                if not fitted:
                    # Incremental Direct J exhibited energy drift above the
                    # convergence threshold after the orbital gradient settled.
                    reference.direct_scf = False
                    reference.diis_space = 12
                reference.max_cycle = 400
                energy = float(reference.kernel())
                if not reference.converged or reference.mol.nao != 7 * args.waters:
                    raise RuntimeError(
                        "reference SCF failed its convergence/basis gate: "
                        f"converged={reference.converged}, nao={reference.mol.nao}, "
                        f"energy={energy:.16g}"
                    )
                cp.cuda.Stream.null.synchronize()
                scf_seconds = perf_counter() - started
                auxiliary_hash = (
                    _reference_basis(reference.with_df.auxmol, primitives)
                    if fitted
                    else None
                )
                hessian = reference.Hessian()
                hessian.grid_response = True
                if fitted:
                    hessian.auxbasis_response = 2
                matrix = (
                    cp.asnumpy(hessian.kernel())
                    .transpose(0, 2, 1, 3)
                    .reshape(3 * len(atoms), -1)
                    .copy()
                )
                elapsed = perf_counter() - started
                diagnostics = {
                    "energy": energy,
                    "scf_seconds": scf_seconds,
                    "scf_energy_tolerance": reference.conv_tol,
                    "scf_gradient_tolerance": reference.conv_tol_grad,
                    "version": gpu4pyscf.__version__,
                }
        finally:
            monitor.terminate()
            monitor.wait(timeout=15)
    samples = out.with_suffix(".csv").read_text().splitlines()
    peak = max(int(sample) for sample in samples if sample.strip().isdigit())
    matrix_path = out.with_suffix(".npy")
    np.save(matrix_path, matrix, allow_pickle=False)
    receipt = {
        **settings,
        "engine": args.engine,
        "job": job,
        "gpu_uuid": uuid,
        "capacity_mib": int(capacity),
        "peak_mib": peak,
        "endpoint_seconds": elapsed,
        "orbital_hash": orbital_hash,
        "auxiliary_hash": auxiliary_hash,
        "matrix_hash": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "diagnostics": diagnostics,
    }
    receipt_path.write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    print(json.dumps(receipt, allow_nan=False))


def compare(
    native_path: Path, reference_path: Path, accuracy_path: Path | None = None
) -> dict[str, Any]:
    """Compare same-GPU stock timing and, optionally, a separate audited Direct matrix."""
    control, reference = (
        json.loads(path.read_text()) for path in (native_path, reference_path)
    )
    if (control["engine"], reference["engine"]) != ("native", "gpu4pyscf"):
        raise ValueError("expected native and GPU4PySCF receipts in that order")
    input_keys = (
        "method",
        "provider",
        "waters",
        "geometry_hash",
        "grid_hash",
        "grid_points",
        "fp64",
        "orbital_hash",
        "auxiliary_hash",
        "auxbasis_response",
    )
    for key in ("job", "gpu_uuid", "capacity_mib", *input_keys):
        if control[key] != reference[key]:
            raise ValueError(f"mismatched {key}")
    if reference["diagnostics"].get("diagnostic_residual_correction"):
        raise ValueError("timing requires unmodified GPU4PySCF")
    audited = None
    if accuracy_path is not None:
        audited = json.loads(accuracy_path.read_text())
        for key in (*input_keys, "engine"):
            if audited[key] != reference[key]:
                raise ValueError(f"accuracy reference has mismatched {key}")
        audit = audited["diagnostics"]
        residual = audit.get("diagnostic_max_preconditioned_residual", float("inf"))
        if (
            control["provider"] != "direct"
            or audit.get("diagnostic_residual_correction") is not True
            or not 0 <= residual <= 1e-12
            or any(
                audit[key] != reference["diagnostics"][key]
                for key in ("version", "scf_energy_tolerance", "scf_gradient_tolerance")
            )
        ):
            raise ValueError("invalid residual-audited Direct accuracy reference")
    if not control["orbital_hash"] or (
        control["provider"] == "df"
        and (not control["auxiliary_hash"] or control["auxbasis_response"] != 2)
    ):
        raise ValueError("missing orbital or fitted model identity")
    if control["fp64"] is not True:
        raise ValueError("qualification requires fresh FP64 SCF")
    diagnostics = control["diagnostics"]
    coordinates = 9 * control["waters"]
    if (
        diagnostics["completed_directions"] != coordinates
        or diagnostics["cpks_physical_checks"] != coordinates
        or diagnostics["numerical_host_fallbacks"]
        or diagnostics["posthoc_symmetrization"] is not False
    ):
        raise ValueError(
            "qualification requires a complete raw native endpoint with no numerical host fallback"
        )

    def load_matrix(path: Path, row: dict[str, Any]) -> np.ndarray:
        matrix_path = path.with_suffix(".npy")
        if hashlib.sha256(matrix_path.read_bytes()).hexdigest() != row["matrix_hash"]:
            raise ValueError("Hessian artifact differs from its receipt")
        matrix = np.load(matrix_path, allow_pickle=False)
        if (
            matrix.shape != (coordinates, coordinates)
            or matrix.dtype != np.float64
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid raw Hessian")
        if not np.isfinite(row["endpoint_seconds"]) or row["endpoint_seconds"] <= 0:
            raise ValueError("invalid endpoint duration")
        if not 0 < row["peak_mib"] <= row["capacity_mib"]:
            raise ValueError("invalid peak device memory")
        return matrix

    native = load_matrix(native_path, control)
    stock = load_matrix(reference_path, reference)
    oracle = (
        load_matrix(accuracy_path, audited)
        if accuracy_path is not None and audited is not None
        else stock
    )
    raw_error = float(np.max(np.abs(native - oracle)))
    asymmetry = float(np.max(np.abs(native - native.T)))
    translation = float(
        np.max(np.abs(native @ np.tile(np.eye(3), (control["waters"] * 3, 1))))
    )
    return {
        "provider": control["provider"],
        "atoms": coordinates // 3,
        "native_seconds": control["endpoint_seconds"],
        "reference_seconds": reference["endpoint_seconds"],
        "native_peak_mib": control["peak_mib"],
        "reference_peak_mib": reference["peak_mib"],
        "capacity_mib": control["capacity_mib"],
        "raw_max_error": raw_error,
        "stock_raw_max_error": float(np.max(np.abs(native - stock))),
        "accuracy_reference_job": (audited or reference)["job"],
        "asymmetry": asymmetry,
        "translation_residual": translation,
        "accuracy_pass": max(raw_error, asymmetry, translation) <= _GATE,
        "speed_pass": control["endpoint_seconds"] < reference["endpoint_seconds"],
        "work": diagnostics,
    }


def main() -> None:
    """Parse standalone measurement and offline comparison commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    single = command.add_parser("measure")
    single.add_argument("engine", choices=("native", "gpu4pyscf"))
    single.add_argument("provider", choices=("direct", "df"))
    single.add_argument("waters", type=int)
    single.add_argument("prefix", type=Path)
    single.add_argument("--method", choices=("r2scan", "pbe"), default="r2scan")
    pair = command.add_parser("compare")
    pair.add_argument("native", type=Path)
    pair.add_argument("reference", type=Path)
    pair.add_argument("--accuracy-reference", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args)
    else:
        report = compare(args.native, args.reference, args.accuracy_reference)
        print(json.dumps(report, allow_nan=False, indent=2))
        if not (report["accuracy_pass"] and report["speed_pass"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
