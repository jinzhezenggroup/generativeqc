"""Reproduce strict-FP64 exact-Direct r2SCAN-3c complete E+F comparisons.

Run under finite Slurm with an authenticated CUDA core/AOT library selected by
GENERATIVEQC_LIBRARY. Each arm retains cold, five warm calls, moved geometry,
five moved-warm calls and the original 1800-second population limit. Independent
oracles are reconstructed from the existing pinned benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from benchmarks._retention import raw_output_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _worker(atoms: int, output: Path) -> None:
    """Retain returned physical Fock counts without changing the timed solver."""
    from benchmarks import readme_dft, readme_omol25
    from tools.generativeqc_validation.xc_diagonal_evidence import REFERENCE

    original = readme_omol25.native_endpoint_record

    def record(item: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(item, **kwargs)
        precision = item.precision or {}
        counters = (
            "mixed_stage_fock_builds",
            "strict_stage_fock_builds",
            "post_scf_fock_builds",
        )
        result["native_operator_work"] = {
            key: precision.get(key)
            for key in (
                *counters,
                "operator_work_counters_valid",
                "execution_retries",
                "skipped_final_fock_builds",
            )
        }
        if (
            result["fock_builds"] is None
            and precision.get("operator_work_counters_valid") is True
            and all(type(precision.get(name)) is int for name in counters)
        ):
            result["fock_builds"] = sum(precision[name] for name in counters)
            result["fock_builds_origin"] = "returned native physical-build counters"
        return result

    reference = output / "reference.json"
    reference.write_text(
        json.dumps(REFERENCE["load"](f"raw/reference-r2scan-3c-{atoms}.json.xz"))
    )
    readme_omol25.native_endpoint_record = record
    sys.argv = [
        "readme_dft",
        "--method",
        "r2scan-3c",
        "native",
        "--atoms",
        str(atoms),
        "--basis-file",
        "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json",
        "--density-fitting",
        "none",
        "--repeats",
        "5",
        "--reference",
        str(reference),
        "--output",
        str(output / "native.json"),
    ]
    readme_dft.main()


def _stop(child: subprocess.Popen) -> None:
    """Retire only this population's process group, including compiler children."""
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main(argv: Sequence[str] | None = None) -> None:
    """Run two explicit selectors without changing scheduler device visibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=raw_output_path)
    parser.add_argument("--atoms", type=int, choices=(48, 96), default=96)
    parser.add_argument(
        "--baseline",
        choices=("spatial-reuse", "relative-reuse"),
        default="spatial-reuse",
    )
    parser.add_argument(
        "--candidate",
        choices=("relative-reuse", "relative-density"),
        default="relative-reuse",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "GENERATIVEQC_LIBRARY"):
        if not os.environ.get(name):
            parser.error(f"{name} is required; use finite srun and a built library")
    output = args.output.resolve()
    if args.worker:
        _worker(args.atoms, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, GENERATIVEQC_CUDA_KS_OCCUPIED_XC="0")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "8"
    for arm, selector in (("baseline", args.baseline), ("candidate", args.candidate)):
        target = output / arm
        target.mkdir()
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.direct_force_reuse",
                str(target),
                "--atoms",
                str(args.atoms),
                "--worker",
            ],
            env=dict(environment, GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=selector),
            start_new_session=True,
        )
        try:
            status = child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            _stop(child)
            status = 124
        except BaseException:
            _stop(child)
            raise
        (target / "exit-status.txt").write_text(f"{status}\n")
        if status:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
