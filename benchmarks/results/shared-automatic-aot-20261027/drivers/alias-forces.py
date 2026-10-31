"""Real VWN alias reuse with independent full-force-vector FD gates."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "tests/python")
from generativeqc_compiler.xc.automatic_cuda_package import automatic_cuda_package
from test_libxc_native_cuda_forces import _lifecycle

root = Path.cwd()
out = root / "evidence/alias-forces"
out.mkdir(exist_ok=True)
os.environ["GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR"] = str(out)
names = tuple(f"LDA_C_VWN_{index}" for index in range(1, 5))
for spins in (1, 2):
    for name in names:
        with pytest.MonkeyPatch.context() as guard:
            _lifecycle(name, spins, guard, full_vector=True, replay=False)
        print(name, spins, "independent full vector passed", flush=True)
records = [json.loads(path.read_text()) for path in sorted(out.glob("*.json"))]
receipts = [record["rows"][0]["artifact"] for record in records]
restricted = [r for r in receipts if r["spin_blocks"] == 1]
assert len({r["restricted_executable_identity"] for r in restricted}) == 1
assert len({r["polarized_executable_identity"] for r in restricted}) == 4
assert len({r["point_expression_identity"] for r in restricted}) == 4
assert len({r["point_source_identity"] for r in restricted}) == 4
workers, registrations = automatic_cuda_package(names)
assert len(workers) == 5 and len(registrations) == 4
(root / "evidence/alias-reuse.json").write_text(
    json.dumps(
        dict(
            schema="generativeqc.automatic-point-alias-reuse.v1",
            names=names,
            scientific_registration_count=len(registrations),
            executable_worker_count=len(workers),
            library_sha256=receipts[0]["point_executable_artifact"]["sha256"],
            restricted_executable_identities=[
                r["restricted_executable_identity"] for r in restricted
            ],
            polarized_executable_identities=[
                r["polarized_executable_identity"] for r in restricted
            ],
            scientific_identities=[r["point_expression_identity"] for r in restricted],
            endpoints=len(records),
            full_vector_reference=True,
            slurm_job_id=os.environ["SLURM_JOB_ID"],
        ),
        indent=2,
    )
    + "\n"
)
