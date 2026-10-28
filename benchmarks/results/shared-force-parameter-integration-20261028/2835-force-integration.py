"""Current integrated producer/consumer E+F qualification, without timing claims."""

import json
import os
import time
from pathlib import Path

import pytest
from test_libxc_native_cuda_forces import _lifecycle

out = Path(".artifacts/force-integration")
out.mkdir(parents=True, exist_ok=True)
os.environ["GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR"] = str(out)
names = (
    "MGGA_X_KTBM_0",
    "MGGA_X_KTBM_1",
    "LDA_C_VWN_1",
    "LDA_C_VWN_4",
    "GGA_X_APBE",
    "MGGA_X_LTA",
)
results = []
for name in names:
    for spins in (1, 2):
        start = time.perf_counter()
        try:
            with pytest.MonkeyPatch.context() as patch:
                _lifecycle(
                    name,
                    spins,
                    patch,
                    full_vector=name.startswith("MGGA_X_KTBM"),
                    replay=True,
                )
            row = {
                "name": name,
                "spins": spins,
                "status": "passed",
                "seconds": time.perf_counter() - start,
            }
        except Exception as exc:  # noqa: BLE001 -- retain every failed case, then fail the run
            import traceback

            traceback.print_exc()
            row = {
                "name": name,
                "spins": spins,
                "status": "failed",
                "seconds": time.perf_counter() - start,
                "error": repr(exc),
            }
        results.append(row)
        print(json.dumps(row), flush=True)
        (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
if any(r["status"] != "passed" for r in results):
    raise SystemExit(1)
