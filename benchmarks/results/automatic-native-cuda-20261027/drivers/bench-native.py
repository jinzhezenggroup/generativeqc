"""Fresh-owner and retained-owner host-return E+F timings, no oracle in timed work."""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "tests/python")
from pathlib import Path

from generativeqc import Calculator, KsOptions
from test_libxc_native_cuda_forces import ATOMS, GRID, MOVED, PROPERTIES

library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
initial_digest = hashlib.sha256(library.read_bytes()).hexdigest()
rows = []
methods = [
    "pbe-rks",
    "pbe-uks",
    *[
        f"libxc-{spin}:{name}"
        for name in ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA")
        for spin in ("rks", "uks")
    ],
]


def calculator(method: str) -> Calculator:
    """Use identical strict scientific controls for both measured native libraries."""
    return Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        initial_guess=None,
        max_iterations=300,
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        screening_tolerance=1e-12,
        ks_options=KsOptions(grid=GRID, tile_points=37),
    )


# Symmetric process/driver/cache warmup, with an independent disposable owner.
with calculator("pbe-rks").prepare_batch([ATOMS]) as batch:
    batch.execute(properties=PROPERTIES, strict=True)
for method in methods:
    spins = 2 if "uks" in method else 1
    for repeat in range(2):
        start = time.perf_counter()
        calc = calculator(method)
        with calc.prepare_batch(
            [ATOMS], charges=[spins - 1], multiplicities=[spins], warm_start=True
        ) as batch:
            setup = time.perf_counter() - start
            for phase, atoms in [
                ("cold", ATOMS),
                ("warm", ATOMS),
                ("warm", ATOMS),
                ("warm", ATOMS),
                ("moved", MOVED),
            ]:
                start = time.perf_counter()
                result = batch.execute(
                    coordinates=([p for _, p in atoms],),
                    properties=PROPERTIES,
                    strict=True,
                ).items[0]
                seconds = time.perf_counter() - start
                d = result.ks_diagnostic.to_payload()
                rows.append(
                    {
                        "method": method,
                        "repeat": repeat,
                        "phase": phase,
                        "seconds": seconds + (setup if phase == "cold" else 0),
                        "energy": result.energy,
                        "executed_backend": result.executed_backend,
                        "forces": result.forces.tolist(),
                        "iterations": result.iterations,
                        "fock_builds": d["fock_builds"],
                        "coulomb_exchange_work": d["coulomb_exchange_work"],
                        "warm_used": result.warm_start_used,
                        "warm_fallback": result.warm_start_fallback,
                        "precision": result.precision,
                    }
                )
assert hashlib.sha256(library.read_bytes()).hexdigest() == initial_digest
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "library": str(library),
            "library_sha256": initial_digest,
            "library_bytes": library.stat().st_size,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "rows": rows,
        },
        indent=2,
    )
    + "\n"
)
