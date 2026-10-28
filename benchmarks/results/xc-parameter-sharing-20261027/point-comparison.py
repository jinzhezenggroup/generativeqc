"""Interleave literal and shared KTBM workers inside one finite Slurm job."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
    from generativeqc_compiler.xc.libxc_bulk import read_catalog

    root = Path("/home/jzzeng/tmp/qc-xc-parameter-reuse-20261027/evidence")
    names = [
        r["name"]
        for r in read_catalog()["registrations"]
        if r["entry"] == "maple/mgga_exc/mgga_x_ktbm.mpl"
    ]
    records = []
    for n, name in enumerate(names):
        for spins in (1, 2):
            for count in (16384, 262144):
                for tile in (0, 127):
                    for arm in (
                        ("baseline", "candidate")
                        if n % 2 == 0
                        else ("candidate", "baseline")
                    ):
                        cmd = [
                            str(root / f"point-benchmark-{arm}"),
                            str(automatic_functional_code(name)),
                            str(count),
                            str(spins),
                            str(tile),
                            "30",
                        ]
                        completed = subprocess.run(
                            cmd, text=True, capture_output=True, check=True, timeout=60
                        )
                        records.append(
                            {
                                "name": name,
                                "revision": arm,
                                **json.loads(completed.stdout),
                            }
                        )
        print(name, flush=True)
    (root / "point-comparison.json").write_text(
        json.dumps(
            {
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
