"""Explicitly refresh the pinned PySCF quadrature data (never a build dependency).

The production compiler consumes only the retained JSON. This maintenance tool
extracts numeric tables and Lebedev octahedral orbits, retaining the exact input
hashes so a package upgrade cannot silently change a scientific prescription.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    """Import PySCF 2.14.0 tables without performing electronic calculations."""
    import numpy as np
    import pyscf
    from pyscf.data import radii
    from pyscf.dft import LebedevGrid, gen_grid, radi

    if pyscf.__version__ != "2.14.0":
        raise ValueError("this import recipe pins PySCF 2.14.0")
    original = LebedevGrid.SphGenOh
    try:
        # Each angular constructor is a list of symmetry orbits. Retain those
        # coefficients rather than rounding an expanded point cloud to text.
        LebedevGrid.SphGenOh = lambda code, a, b, weight: np.array(
            [[code, a, b, weight]]
        )
        orbits = {
            str(count): getattr(LebedevGrid, f"MakeAngularGrid_{count}")().tolist()
            for count in LebedevGrid.LEBEDEV_NGRID
            if count > 1
        }
    finally:
        LebedevGrid.SphGenOh = original
    payload = {
        "schema": "generativeqc.treutler-lebedev-nwchem/v1",
        "upstream": "PySCF 2.14.0",
        "source_sha256": {
            str(
                Path(module.__file__).relative_to(Path(pyscf.__file__).parent)
            ): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (LebedevGrid, gen_grid, radi, radii)
        },
        "radial_counts": gen_grid.RAD_GRIDS.tolist(),
        "angular_orders": gen_grid.ANG_ORDER.tolist(),
        "lebedev_orders": LebedevGrid.LEBEDEV_ORDER,
        "treutler_xi": radi._treutler_ahlrichs_xi,
        "bragg_radii_bohr": radii.BRAGG[:119].tolist(),
        "orbits": orbits,
    }
    output = Path(__file__).resolve().parents[1] / (
        "python/generativeqc_compiler/dft/data/treutler-lebedev.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
