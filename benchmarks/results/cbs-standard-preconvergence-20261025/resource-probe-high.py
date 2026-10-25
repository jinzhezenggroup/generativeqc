"""Retain both successful bounded fallbacks and complete resource refusals."""
import os
from pathlib import Path
from benchmarks.readme_cbs import checkpoint, digest, geometry, native_call, setup

assert os.environ.get("SLURM_JOB_ID")
root = Path("/data/jzzeng/cbs-standard-20261025")
atoms = geometry(Path("/data/jzzeng/readme-cbs-20261013/qm7.mat"), 2)
payload = {"source_artifact_sha256": digest(root / "candidate2.patch"),
    "library_sha256": digest(Path(os.environ["GENERATIVEQC_LIBRARY"])),
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "samples": [],
    "driver_source": Path(__file__).read_text()}
output = Path(".artifacts/cbs-standard/resource-probe-high.json")
for mib in (512, 1024, 2048):
    protocol, orbital, auxiliary = setup(atoms, 2, mib << 20, "df")
    for mode in ("fp64", "mixed"):
        row = {"budget_mib": mib, "preconvergence": mode, "protocol": protocol}
        payload["samples"].append(row)
        try:
            row.update(result=native_call(atoms, orbital, auxiliary, mib << 20,
                "df", "standard", mode), status="completed")
            work = next(c["result"]["cc_performance"] for c in row["result"]["components"]
                        if c["method"] == "df-ccsd(t)")
            print(mib, mode, "completed", work["ccsd_fp32_refused"], work["lambda_fp32_refused"], flush=True)
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(mib, mode, row["error"], flush=True)
        checkpoint(output, payload)
