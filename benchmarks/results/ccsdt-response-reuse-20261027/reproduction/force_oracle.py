"""Independent original-equation PySCF energies and molecular force directions."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pyscf
from test_df_complete_force import independent

root = Path("/home/jzzeng/codes/qc-branch-audit-20260922/evidence-2516")
assert pyscf.__version__ == "2.14.0", pyscf.__version__
out = {"pyscf": pyscf.__version__, "records": []}
for size in (28, 56):
    m = json.loads((root / f"water{size}.metadata.json").read_text())
    start = time.monotonic()
    energy = independent(m, True)
    d = np.random.default_rng(2516 + size).normal(
        size=np.shape(m["inputs"]["coordinates"])
    )
    d /= np.linalg.norm(d)
    cases = []
    for h in (1e-4, 3e-5):
        e = []
        for sign in (-1, 1):
            changed = copy.deepcopy(m)
            changed["inputs"]["coordinates"] = (
                np.array(m["inputs"]["coordinates"]) + sign * h * d
            ).tolist()
            e.append(independent(changed, True)[0])
        cases.append(
            {
                "step": h,
                "minus": e[0],
                "plus": e[1],
                "directional_force": -(e[1] - e[0]) / (2 * h),
            }
        )
    out["records"].append(
        {
            "nbf": size,
            "energies": energy,
            "direction": d.tolist(),
            "differences": cases,
            "seconds": time.monotonic() - start,
        }
    )
    (root / "cluster-oracle.json").write_text(json.dumps(out, indent=2) + "\n")
    print(size, energy, cases, flush=True)
