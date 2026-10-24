"""Same-binary full/LR Direct/GPU/GPU+RAM complete E+F experiment.

Run each tier in a fresh process within the same finite Slurm allocation. The
ordinary README driver owns geometry, fitted input identity, default scientific
controls, synchronization, full forces and frozen-density warm replay. Journal
entries describe submitted ERI work; their host durations are not kernel times.
Use a separate Nsight run for recurrence/DMA timing and overlap evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tier", choices=("direct", "gpu", "host", "auto", "disk", "tiered")
    )
    parser.add_argument("--source-operator", choices=("long", "full"), default="long")
    parser.add_argument(
        "--disable-incremental",
        action="store_true",
        help="qualify source reuse separately from the incumbent incremental Fock path",
    )
    parser.add_argument(
        "--gpu-mask", help="explicit common GPU mask for controlled partial-cache arms"
    )
    parser.add_argument("--gpu-bytes", type=int, default=16 << 30)
    parser.add_argument("--disk-bytes", type=int, default=64 << 30)
    parser.add_argument(
        "--tmp-dir", help="local directory; unset uses the native temporary directory"
    )
    parser.add_argument("--host-bytes", type=int, default=32 << 30)
    parser.add_argument("--tile-bytes", type=int, default=256 << 20)
    parser.add_argument("--host-mask", default="0xfffffff")
    parser.add_argument("--source-kind", choices=("commit", "tree"), default="commit")
    args, remaining = parser.parse_known_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("execute through srun with a finite --time")
    if (
        args.gpu_bytes < 0
        or args.host_bytes < 0
        or args.disk_bytes < 0
        or not 0 < args.tile_bytes <= 512 << 20
    ):
        parser.error("invalid cache or staging capacity")
    if args.tile_bytes % 8:
        parser.error("FP64 staging capacity must be divisible by eight")
    prefix = (
        "GENERATIVEQC_KS_RANGE_"
        if args.source_operator == "long"
        else "GENERATIVEQC_KS_"
    )
    controls = {
        f"{prefix}RESIDENT_VALUES": "0" if args.tier == "direct" else "1",
        "GENERATIVEQC_KS_RESIDENT_MAX_BYTES": str(args.gpu_bytes),
        f"{prefix}HOST_MAX_BYTES": str(
            args.host_bytes if args.tier in {"host", "tiered"} else 0
        ),
        f"{prefix}HOST_TILE_BYTES": str(args.tile_bytes),
        f"{prefix}HOST_BUCKET_MASK": args.host_mask,
        f"{prefix}DISK_MAX_BYTES": str(
            args.disk_bytes if args.tier in {"disk", "tiered"} else 0
        ),
        "GENERATIVEQC_CUDA_KS_CHUNK": "1",
    }
    if args.disable_incremental:
        controls["GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK"] = "0"
    # Freeze automatic GPU bucket fitting identically in B and C. An inherited
    # explicit mask would silently turn the experiment into a different policy.
    os.environ.pop("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", None)
    if args.gpu_mask is not None:
        controls["GENERATIVEQC_KS_RESIDENT_BUCKET_MASK"] = args.gpu_mask
    if args.tier == "auto":
        for name in (
            f"{prefix}RESIDENT_VALUES",
            f"{prefix}HOST_MAX_BYTES",
            f"{prefix}DISK_MAX_BYTES",
        ):
            controls.pop(name)
            os.environ.pop(name, None)
    if args.tmp_dir is not None:
        controls["GENERATIVEQC_ERI_TMPDIR"] = args.tmp_dir
    else:
        os.environ.pop("GENERATIVEQC_ERI_TMPDIR", None)
    os.environ.update(controls)
    from benchmarks import readme_defaults as driver
    from benchmarks import readme_hf_scaling
    from benchmarks._cases import benchmark_cases

    previous_cases = readme_hf_scaling.scaling_cases

    def cases() -> dict[str, Any]:
        return {
            **previous_cases(),
            "water-108": benchmark_cases()[
                "water-36mer-water27-derived-def2-svp-spherical"
            ],
        }

    readme_hf_scaling.scaling_cases = cases
    driver.SIZES = (*driver.SIZES, 108)
    original = driver.run_native

    def run(
        native_args: argparse.Namespace,
        atoms: Any,
        record: dict[str, Any],
        retain: Callable[[dict[str, Any]], None],
    ) -> None:
        supported = (
            native_args.method == "wb97m-v" and native_args.mode == "df"
            if args.source_operator == "long"
            else native_args.method in {"pbe0", "b3lyp"}
            and native_args.mode == "direct"
        )
        if not supported:
            raise ValueError(
                "choose WB97M-V DF for LR, or exact PBE0/B3LYP for full ERIs"
            )
        record["host_eri_experiment"] = {
            "tier": args.tier,
            "source_operator": args.source_operator,
            "source_identity_kind": args.source_kind,
            "controls": controls,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "trace": os.environ.get("GENERATIVEQC_DF_PROGRESS_TRACE"),
            "timing_qualification": "single complete observation; compare only source-matched populations",
        }

        def io_counts() -> dict[str, int]:
            # Process-wide physical I/O includes library/input traffic. Logical
            # ERI read bytes are recorded separately by source-action counters.
            path = Path("/proc/self/io")
            return {
                key.rstrip(":"): int(value)
                for key, value in (
                    line.split() for line in path.read_text().splitlines()
                )
            }

        before = io_counts()
        result = original(native_args, atoms, record, retain)
        record["host_eri_experiment"]["process_io"] = {
            "before": before,
            "after": io_counts(),
        }
        return result

    driver.run_native = run
    sys.argv = [sys.argv[0], "standard", *remaining]
    driver.main()


if __name__ == "__main__":
    main()
