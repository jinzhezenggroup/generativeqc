"""Re-execute final artifacts against the retained independent full-vector oracle.

The previous reconverged FD stencil is immutable reference data, not a new FD
measurement. Every endpoint and every force coordinate are evaluated anew.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/python")
import numpy as np
import pytest
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic
from test_dft_complete_cuda import no_cpu_derivatives
from test_libxc_automatic_cuda_ks import _physical_state, _reference
from test_libxc_native_cuda_forces import (
    ATOMS,
    MOVED,
    PROPERTIES,
    _calculator,
    _oracle_basis,
)
from test_stationary_aot_cuda import _forbid_aot_generation

root = Path.cwd()
review_path = root / "evidence/reference-reuse-source-review.json"
review = json.loads(review_path.read_text())
out = root / "evidence/rebound-forces"
out.mkdir(exist_ok=True)
paths = sorted((root / "evidence/qualified-forces").glob("*-h4.json")) + sorted(
    (root / "evidence/qualified-forces").glob("*-water.json")
)
for path in paths:
    source = json.loads(path.read_text())
    name = source["name"]
    spins = source["spins"]
    charge = source.get("charge", spins - 1)
    multiplicity = source.get("multiplicity", spins)
    water = path.name.endswith("-water.json")
    atoms = (
        [("O", (0.0, 0.0, 0.0)), ("H", (0.2, 1.5, 1.1)), ("H", (-0.3, -1.4, 1.0))]
        if water
        else ATOMS
    )
    moved = [*atoms[:2], ("H", (-0.27, -1.4, 1.0))] if water else MOVED
    start = time.perf_counter()
    calc = _calculator(name, spins)
    with calc.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=True
    ) as b:
        setup = time.perf_counter() - start
        rows = []
        for expected in source["rows"]:
            phase = expected["phase"]
            geometry = moved if phase == "moved" else atoms
            with pytest.MonkeyPatch.context() as guard, no_cpu_derivatives():
                _forbid_aot_generation(guard, b)
                start = time.perf_counter()
                r = b.execute(
                    coordinates=([p for _, p in geometry],),
                    properties=PROPERTIES,
                    strict=True,
                ).items[0]
                elapsed = time.perf_counter() - start
            receipt = read_automatic_artifact_diagnostic(b)
            for key in (
                "point_expression_identity",
                "point_source_identity",
                "force_plan_identity",
            ):
                assert receipt[key] == expected["artifact"][key], (name, key)
            transition = review["profiles"][receipt["force_profile"]]
            assert (
                expected["artifact"]["force_source_identity"]
                == transition["previous_identity"]
            )
            assert receipt["force_source_identity"] == transition["current_identity"]
            np.testing.assert_allclose(r.forces, expected["forces"], atol=3e-8, rtol=0)
            np.testing.assert_allclose(
                r.energy, sum(expected["scf"]["components"].values()), atol=3e-9, rtol=0
            )
            snapshot = NativeKsSnapshot(b, 0)
            try:
                oracle = _reference(
                    name,
                    spins,
                    geometry,
                    _physical_state(snapshot),
                    basis=_oracle_basis(),
                    charge=charge,
                    multiplicity=multiplicity,
                )[0]
            finally:
                snapshot.close()
            np.testing.assert_allclose(r.energy, oracle, atol=3e-9, rtol=0)
            rows.append(
                {
                    "phase": phase,
                    "reference_energy": oracle,
                    "energy": r.energy,
                    "forces": np.asarray(r.forces).tolist(),
                    "artifact": receipt,
                    "endpoint_seconds": elapsed,
                    "complete_endpoint_seconds": elapsed
                    + (setup if phase == "cold" else 0),
                    "scf": r.ks_diagnostic.to_payload(),
                    "force_parity_max": float(
                        np.max(np.abs(np.asarray(r.forces) - expected["forces"]))
                    ),
                }
            )
    (out / path.name).write_text(
        json.dumps(
            {
                "schema": "generativeqc.native-automatic-cuda-force-requalification.v1",
                "name": name,
                "spins": spins,
                "charge": charge,
                "multiplicity": multiplicity,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "reference_receipt": str(path),
                "reference_receipt_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "reference_mode": "retained independent reconverged FD; new complete endpoint and center SCF oracle",
                "source_reuse_review_sha256": hashlib.sha256(
                    review_path.read_bytes()
                ).hexdigest(),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(path.name, "passed", flush=True)
