"""Check local curvature at optimized geometries with independent CPU gradients."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pyscf
from pyscf import cc, gto, lib, scf
from pyscf.cc import ccsd_t, ccsd_t_lambda
from pyscf.grad import ccsd_t as triples_gradient
from run import ATOMIC_NUMBER, JsonObject, dump

STEPS = (0.002, 0.001)
CPU_CONTROLS = ("hf", "h2o", "co")
ENERGY_GATE = 3e-7
FORCE_GATE = 2.5e-5
CURVATURE_FLOOR = 1e-4
CURVATURE_STABILITY = 1e-4


def load_canonical_basis(path: Path) -> dict[str, list[list]]:
    """Translate the archived contraction matrices to PySCF spherical shells."""
    record = json.loads(path.read_text())
    if record["representation"] != "spherical":
        raise ValueError("curvature check requires a spherical AO basis")
    symbols = {atomic_number: symbol for symbol, atomic_number in ATOMIC_NUMBER.items()}
    basis = {}
    for atom in record["elements"]:
        if atom["ecp_core_electrons"]:
            raise ValueError("all-electron basis required")
        shells = []
        for shell in atom["shells"]:
            coefficients = zip(*shell["coefficients"], strict=True)
            primitives = [
                [float(exponent), *map(float, row)]
                for exponent, row in zip(shell["exponents"], coefficients, strict=True)
            ]
            shells.append([shell["angular_momentum"], *primitives])
        basis[symbols[atom["atomic_number"]]] = shells
    return basis


def gradient(
    case: JsonObject, xyz: np.ndarray, basis: dict[str, list[list]]
) -> JsonObject:
    """Solve a cold all-electron CCSD(T) energy and analytic gradient."""
    tick = time.monotonic()
    mol = gto.M(
        atom=list(zip(case["symbols"], xyz.tolist(), strict=True)),
        basis=basis,
        charge=case["charge"],
        spin=case["multiplicity"] - 1,
        cart=False,
        unit="Bohr",
        symmetry=False,
        verbose=0,
        max_memory=5000,
    )
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-13
    hf.conv_tol_grad = 1e-10
    hf.direct_scf_tol = 1e-15
    hf.max_cycle = 220
    hf.kernel()
    if not hf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    coupled = cc.CCSD(hf, frozen=0)
    coupled.conv_tol = 1e-13
    coupled.conv_tol_normt = 1e-10
    coupled.max_cycle = 160
    eris = coupled.ao2mo()
    coupled.kernel(eris=eris)
    if not coupled.converged:
        raise RuntimeError("PySCF CCSD did not converge")
    conv, l1, l2 = ccsd_t_lambda.kernel(
        coupled, eris, coupled.t1, coupled.t2, max_cycle=160, tol=1e-10
    )
    if not conv:
        raise RuntimeError("PySCF CCSD(T) Lambda did not converge")
    correction = ccsd_t.kernel(coupled, eris, coupled.t1, coupled.t2)
    grad = triples_gradient.Gradients(coupled).kernel(
        coupled.t1, coupled.t2, l1, l2, eris=eris
    )
    if not np.isfinite(grad).all():
        raise RuntimeError("PySCF returned a nonfinite gradient")
    return {
        "coordinates_bohr": xyz.tolist(),
        "energy_hartree": float(coupled.e_tot + correction),
        "gradient_hartree_per_bohr": grad.tolist(),
        "seconds": time.monotonic() - tick,
    }


def vibrational_basis(xyz: np.ndarray) -> tuple[int, np.ndarray]:
    translated = xyz - xyz.mean(axis=0)
    axes = np.eye(3)
    translations = [np.tile(axis, (len(xyz), 1)).ravel() for axis in axes]
    rotations = [np.cross(axis, translated).ravel() for axis in axes]
    u, s, _ = np.linalg.svd(
        np.column_stack(translations + rotations), full_matrices=True
    )
    rank = int(np.count_nonzero(s > s[0] * 1e-10))
    return rank, u[:, rank:]


def curvature(
    case: JsonObject,
    xyz: np.ndarray,
    basis: dict[str, list[list]],
    active: np.ndarray,
    step: float,
    dest: Path,
) -> np.ndarray:
    columns = []
    for index, direction in enumerate(active.T):
        delta = direction.reshape(xyz.shape) * step
        plus = gradient(case, xyz + delta, basis)
        minus = gradient(case, xyz - delta, basis)
        dump(dest / f"{step:g}-{index}.json", {"plus": plus, "minus": minus})
        gplus = np.array(plus["gradient_hartree_per_bohr"]).ravel()
        gminus = np.array(minus["gradient_hartree_per_bohr"]).ravel()
        columns.append(active.T @ (gplus - gminus) / (2 * step))
    return np.column_stack(columns)


def check(
    case: JsonObject, native: JsonObject, basis: dict[str, list[list]], dest: Path
) -> JsonObject:
    xyz = np.array(native["coordinates_bohr"])
    rank, active = vibrational_basis(xyz)
    expected_rank = 5 if len(xyz) == 2 else 6
    if rank != expected_rank:
        raise RuntimeError(
            f"unexpected rigid mode rank {rank}, expected {expected_rank}"
        )
    center = gradient(case, xyz, basis)
    dump(dest / "center.json", center)
    hessians = [curvature(case, xyz, basis, active, step, dest) for step in STEPS]
    eigenvalues = [np.linalg.eigvalsh((matrix + matrix.T) / 2) for matrix in hessians]
    antisymmetry = [float(np.max(np.abs(matrix - matrix.T))) for matrix in hessians]
    max_change = float(np.max(np.abs(eigenvalues[0] - eigenvalues[1])))
    max_force = float(np.linalg.norm(center["gradient_hartree_per_bohr"], axis=1).max())
    energy_delta = abs(center["energy_hartree"] - native["energy_hartree"])
    passed = (
        energy_delta <= ENERGY_GATE
        and max_force <= FORCE_GATE
        and min(v[0] for v in eigenvalues) > CURVATURE_FLOOR
        and max_change <= CURVATURE_STABILITY
        and max(antisymmetry) <= CURVATURE_STABILITY
    )
    return {
        "status": "minimum_verified" if passed else "gate_failed",
        "native_energy_hartree": native["energy_hartree"],
        "native_max_force_hartree_per_bohr": native["max_force_hartree_per_bohr"],
        "center_energy_difference_hartree": energy_delta,
        "center_max_force_hartree_per_bohr": max_force,
        "rigid_rank": rank,
        "mode_basis": active.tolist(),
        "curvature_steps_bohr": STEPS,
        "raw_projected_hessians_hartree_per_bohr2": [h.tolist() for h in hessians],
        "curvatures_hartree_per_bohr2": [v.tolist() for v in eigenvalues],
        "max_raw_antisymmetry": antisymmetry,
        "max_curvature_change_between_steps": max_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--molecule", action="append", choices=CPU_CONTROLS)
    args = parser.parse_args()
    args.out.mkdir()
    lib.num_threads(1)
    if pyscf.__version__ != "2.14.0":
        raise ValueError(f"expected PySCF 2.14.0, got {pyscf.__version__}")
    dump(
        args.out / "provenance.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "pyscf_sources": {
                module.__name__: hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (ccsd_t, ccsd_t_lambda, triples_gradient)
            },
            "basis_sha256": hashlib.sha256(
                (args.payload / "cc-pcvtz.json").read_bytes()
            ).hexdigest(),
            "gates": {
                "energy_hartree": ENERGY_GATE,
                "force_hartree_per_bohr": FORCE_GATE,
                "curvature_floor": CURVATURE_FLOOR,
                "curvature_stability": CURVATURE_STABILITY,
            },
        },
    )
    cohort = json.loads((args.payload / "cohort.json").read_text())
    cases = {case["id"]: case for case in cohort["molecules"]}
    basis = load_canonical_basis(args.payload / "cc-pcvtz.json")
    for name in args.molecule or CPU_CONTROLS:
        dest = args.out / name
        dest.mkdir()
        tick = time.monotonic()
        try:
            run = args.native / name / "cc-pcvtz"
            if json.loads((run / "receipt.json").read_text())["status"] != "converged":
                raise ValueError("native geometry has not passed its optimizer gate")
            source = (run / "stdout.json").read_bytes()
            result = check(cases[name], json.loads(source), basis, dest)
            result["native_receipt_sha256"] = hashlib.sha256(source).hexdigest()
        except (OSError, ValueError, RuntimeError) as error:
            result = {"status": "failed", "error": str(error)}
        result["seconds"] = time.monotonic() - tick
        dump(dest / "check.json", result)
        print(name, result["status"], flush=True)


if __name__ == "__main__":
    main()
