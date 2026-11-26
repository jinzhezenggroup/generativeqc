"""Check DF-CCSD(T) minima with independent PySCF energy differences."""

import argparse
import hashlib
import json
import platform
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pyscf
from check_minima import (
    CPU_CONTROLS,
    CURVATURE_FLOOR,
    CURVATURE_STABILITY,
    ENERGY_GATE,
    FORCE_GATE,
    load_canonical_basis,
    vibrational_basis,
)
from pyscf import cc, gto, lib, scf
from pyscf.cc import dfccsd
from run import JsonObject, dump

STEPS = (0.004, 0.002)


def energy(
    case: JsonObject, xyz: np.ndarray, orbital: dict, auxiliary: dict
) -> JsonObject:
    """Solve exact RHF with density fitting applied only to CC correlation."""
    tick = time.monotonic()
    mol = gto.M(
        atom=list(zip(case["symbols"], xyz.tolist(), strict=True)),
        basis=orbital,
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
        raise RuntimeError("PySCF exact RHF did not converge")
    coupled = cc.CCSD(hf, frozen=0).density_fit(auxbasis=auxiliary)
    if not isinstance(coupled, dfccsd.RCCSD):
        raise TypeError("PySCF did not select correlation DF-RCCSD")
    coupled.conv_tol = 1e-13
    coupled.conv_tol_normt = 1e-10
    coupled.max_cycle = 160
    eris = coupled.ao2mo()
    coupled.kernel(eris=eris)
    if not coupled.converged:
        raise RuntimeError("PySCF DF-RCCSD did not converge")
    correction = coupled.ccsd_t(eris=eris)
    total = float(coupled.e_tot + correction)
    if not np.isfinite(total):
        raise RuntimeError("PySCF returned a nonfinite energy")
    return {
        "coordinates_bohr": xyz.tolist(),
        "energy_hartree": total,
        "rhf_energy_hartree": float(hf.e_tot),
        "ccsd_correlation_hartree": float(coupled.e_corr),
        "triples_hartree": float(correction),
        "seconds": time.monotonic() - tick,
    }


def differences(
    case: JsonObject,
    xyz: np.ndarray,
    orbital: dict,
    auxiliary: dict,
    active: np.ndarray,
    step: float,
    center: float,
    destination: Path,
) -> tuple[np.ndarray, np.ndarray]:
    dim = active.shape[1]
    grad = np.zeros(dim)
    hess = np.zeros((dim, dim))
    directions = active.T.reshape((dim, *xyz.shape)) * step

    def displaced(delta: np.ndarray, stem: str) -> float:
        record = energy(case, xyz + delta, orbital, auxiliary)
        dump(destination / f"{step:g}-{stem}.json", record)
        return record["energy_hartree"] - center

    for i, direction in enumerate(directions):
        plus = displaced(direction, f"{i}+")
        minus = displaced(-direction, f"{i}-")
        grad[i] = (plus - minus) / (2 * step)
        hess[i, i] = (plus + minus) / step**2
    for i, j in combinations(range(dim), 2):
        numerator = 0.0
        for a, b in product((-1, 1), repeat=2):
            point = a * directions[i] + b * directions[j]
            numerator += a * b * displaced(point, f"{i}{a:+d}-{j}{b:+d}")
        hess[i, j] = hess[j, i] = numerator / (4 * step**2)
    return grad, hess


def check(
    case: JsonObject, native: JsonObject, orbital: dict, auxiliary: dict, dest: Path
) -> JsonObject:
    xyz = np.array(native["coordinates_bohr"])
    rank, active = vibrational_basis(xyz)
    expected_rank = 5 if len(xyz) == 2 else 6
    if rank != expected_rank:
        raise RuntimeError(
            f"unexpected rigid mode rank {rank}, expected {expected_rank}"
        )
    center = energy(case, xyz, orbital, auxiliary)
    dump(dest / "center.json", center)
    results = [
        differences(
            case,
            xyz,
            orbital,
            auxiliary,
            active,
            step,
            center["energy_hartree"],
            dest,
        )
        for step in STEPS
    ]
    gradients, hessians = zip(*results, strict=True)
    coefficients = (4 * gradients[1] - gradients[0]) / 3
    gradient = (active @ coefficients).reshape(xyz.shape)
    max_force = float(np.linalg.norm(gradient, axis=1).max())
    eigenvalues = [np.linalg.eigvalsh(matrix) for matrix in hessians]
    curvature_change = float(np.max(np.abs(eigenvalues[0] - eigenvalues[1])))
    energy_delta = abs(center["energy_hartree"] - native["energy_hartree"])
    passed = (
        energy_delta <= ENERGY_GATE
        and max_force <= FORCE_GATE
        and min(vector[0] for vector in eigenvalues) > CURVATURE_FLOOR
        and curvature_change <= CURVATURE_STABILITY
    )
    return {
        "status": "minimum_verified" if passed else "gate_failed",
        "center_energy_difference_hartree": energy_delta,
        "center_projected_max_force_hartree_per_bohr": max_force,
        "native_energy_hartree": native["energy_hartree"],
        "native_max_force_hartree_per_bohr": native["max_force_hartree_per_bohr"],
        "rigid_rank": rank,
        "mode_basis": active.tolist(),
        "steps_bohr": STEPS,
        "projected_gradients_hartree_per_bohr": [g.tolist() for g in gradients],
        "projected_hessians_hartree_per_bohr2": [h.tolist() for h in hessians],
        "curvatures_hartree_per_bohr2": [v.tolist() for v in eigenvalues],
        "max_curvature_change_between_steps": curvature_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--auxiliary", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    lib.num_threads(1)
    if pyscf.__version__ != "2.14.0":
        raise ValueError(f"expected PySCF 2.14.0, got {pyscf.__version__}")
    orbital_path = args.payload / "cc-pcvtz.json"
    dump(
        args.out / "provenance.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "pyscf_dfccsd_sha256": hashlib.sha256(
                Path(dfccsd.__file__).read_bytes()
            ).hexdigest(),
            "orbital_sha256": hashlib.sha256(orbital_path.read_bytes()).hexdigest(),
            "auxiliary_sha256": hashlib.sha256(args.auxiliary.read_bytes()).hexdigest(),
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
    orbital = load_canonical_basis(orbital_path)
    auxiliary = load_canonical_basis(args.auxiliary)
    for name in CPU_CONTROLS:
        dest = args.out / name
        dest.mkdir()
        tick = time.monotonic()
        try:
            run = args.native / name / "df-cc-pcvtz"
            if json.loads((run / "receipt.json").read_text())["status"] != "converged":
                raise ValueError("native geometry has not passed its optimizer gate")
            source = (run / "stdout.json").read_bytes()
            result = check(cases[name], json.loads(source), orbital, auxiliary, dest)
            result["native_receipt_sha256"] = hashlib.sha256(source).hexdigest()
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            result = {"status": "failed", "error": str(error)}
        result["seconds"] = time.monotonic() - tick
        dump(dest / "check.json", result)
        print(name, result["status"], flush=True)


if __name__ == "__main__":
    main()
