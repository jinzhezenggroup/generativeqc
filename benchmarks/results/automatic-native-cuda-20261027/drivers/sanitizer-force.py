import json
import os
import sys

sys.path.insert(0, "tests/python")
import numpy as np
from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic
from test_libxc_native_cuda_forces import ATOMS, MOVED, PROPERTIES, _calculator

rows = []
for spins in (1, 2):
    with _calculator("MGGA_X_LTA", spins).prepare_batch(
        [ATOMS], charges=[spins - 1], multiplicities=[spins], warm_start=True
    ) as batch:
        for phase, atoms in [("cold", ATOMS), ("warm", ATOMS), ("moved", MOVED)]:
            r = batch.execute(
                coordinates=([p for _, p in atoms],), properties=PROPERTIES, strict=True
            ).items[0]
            assert r.converged and np.isfinite(r.forces).all()
            assert read_automatic_artifact_diagnostic(batch)["force_completed"]
            rows.append(
                {
                    "spins": spins,
                    "phase": phase,
                    "energy": r.energy,
                    "forces": np.asarray(r.forces).tolist(),
                    "iterations": r.iterations,
                }
            )
print(
    json.dumps({"slurm_job_id": os.environ["SLURM_JOB_ID"], "rows": rows}), flush=True
)
