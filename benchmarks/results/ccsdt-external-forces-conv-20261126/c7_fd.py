"""Acquire a tight, conventional Psi4 energy for the C7 atom-4 z derivative."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from run import digest, protocol, write

if TYPE_CHECKING:
    from typing import Any

STEPS = {"h4": 0.004, "h2": 0.002, "h1": 0.001}
SETTINGS = {
    "tight": {"energy": 1e-13, "residual": 1e-11},
    "check": {"energy": 5e-14, "residual": 1e-12},
}
ATOM = 3
AXIS = 2


def energy(args: argparse.Namespace) -> dict[str, Any]:
    import psi4

    if psi4.__version__ != "1.10.2":
        raise ValueError(f"Unexpected Psi4 {psi4.__version__}")
    module = protocol(args.protocol)
    fixture = json.loads((args.protocol / "fixtures.json").read_text())["cases"]["C7"]
    atoms = fixture["atoms_bohr"]
    if atoms[ATOM][0] != 6 or fixture["nao"] != 178:
        raise ValueError("Unexpected C7 fixture")
    step = STEPS[args.step]
    atoms[ATOM][1][AXIS] += args.sign * step
    options = SETTINGS[args.convergence]
    basis = "issue3167orb"

    psi4.set_memory(args.memory_gib << 30)
    psi4.set_num_threads(args.threads)
    psi4.core.IOManager.shared_object().set_default_path(str(args.scratch))
    psi4.core.set_output_file(str(args.output / "psi4.txt"), False)
    module.psi4_basis(psi4, fixture, "orbital", basis, "BASIS")
    molecule = psi4.geometry(
        "0 1\n"
        + "\n".join(
            f"{module.ELEMENT[z]} " + " ".join(format(x, ".17g") for x in xyz)
            for z, xyz in atoms
        )
        + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n"
    )
    settings = {
        "basis": basis,
        "puream": True,
        "reference": "rhf",
        "freeze_core": False,
        "scf_type": "pk",
        "cc_type": "conv",
        "qc_module": "ccenergy",
        "ints_tolerance": 1e-14,
        "maxiter": 200,
        "e_convergence": options["energy"],
        "d_convergence": options["residual"],
        "ccenergy__e_convergence": options["energy"],
        "ccenergy__r_convergence": options["residual"],
    }
    psi4.set_options(settings)
    started = time.monotonic()
    value, wavefunction = psi4.energy("ccsd(t)", molecule=molecule, return_wfn=True)
    elapsed = time.monotonic() - started
    computed = wavefunction.molecule().geometry().to_array()
    if not math.isfinite(value) or any(
        abs(computed[i, a] - x) > 1e-12
        for i, (_, xyz) in enumerate(atoms)
        for a, x in enumerate(xyz)
    ):
        raise RuntimeError("Nonfinite energy or displaced geometry changed")
    if (
        wavefunction.basisset().nbf() != fixture["nao"]
        or wavefunction.nalpha() != 29
        or wavefunction.nbeta() != 29
    ):
        raise RuntimeError("Unexpected orbital or electron count")
    return {
        "status": "completed",
        "energy_hartree": float(value),
        "hf_energy_hartree": float(psi4.variable("SCF TOTAL ENERGY")),
        "endpoint_seconds": elapsed,
        "settings": settings,
        "psi4_version": psi4.__version__,
        "orbitals": fixture["nao"],
        "electrons": 58,
        "displaced_z_bohr": atoms[ATOM][1][AXIS],
        "basis_sha256": digest(module.basis_file(fixture, "orbital")),
        "fixtures_sha256": digest(args.protocol / "fixtures.json"),
        "cpu_model": module.cpu_model(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--memory-gib", type=int, required=True)
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--convergence", choices=SETTINGS, required=True)
    args = parser.parse_args()
    if args.threads <= 0 or args.memory_gib <= 0:
        parser.error("Threads and memory must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        record = energy(args)
    except Exception as exc:  # noqa: BLE001 - retain original worker failure
        traceback.print_exc()
        record = {"status": "failed", "exception": repr(exc)}
    write(
        args.output / "energy.json",
        {
            **record,
            "step_bohr": STEPS[args.step],
            "sign": args.sign,
            "atom_index": ATOM,
            "axis": AXIS,
            "convergence": args.convergence,
            "threads": args.threads,
            "psi4_memory_gib": args.memory_gib,
            "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "driver_sha256": digest(Path(__file__)),
        },
    )
    return int(record["status"] != "completed")


if __name__ == "__main__":
    sys.exit(main())
