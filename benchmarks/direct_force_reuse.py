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
    """Claim an empty arm before setup; preserve evidence even after failed setup."""
    output = raw_output_path(output).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("worker output must not traverse a symlink")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"worker output is not empty: {output}")
    with (output / ".worker-started").open("x"):
        pass
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
    with reference.open("x") as stream:
        json.dump(REFERENCE["load"](f"raw/reference-r2scan-3c-{atoms}.json.xz"), stream)
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
    """Retire owned descendants even after their process-group leader exits."""
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, requested_signal)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if requested_signal == signal.SIGKILL:
                raise


def main(argv: Sequence[str] | None = None) -> None:
    """Compare explicit or unset selectors without changing device visibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=raw_output_path)
    parser.add_argument("--atoms", type=int, choices=(48, 96), default=96)
    parser.add_argument(
        "--baseline",
        choices=("spatial-reuse", "relative-reuse", "automatic"),
        default="spatial-reuse",
    )
    parser.add_argument(
        "--candidate",
        choices=("relative-reuse", "automatic"),
        default="relative-reuse",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "GENERATIVEQC_LIBRARY"):
        if not os.environ.get(name):
            parser.error(f"{name} is required; use finite srun and a built library")
    if args.worker:
        _worker(args.atoms, args.output)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, GENERATIVEQC_CUDA_KS_OCCUPIED_XC="0")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "8"
    for arm, selector in (("baseline", args.baseline), ("candidate", args.candidate)):
        target = output / arm
        target.mkdir()
        arm_environment = dict(environment)
        if selector == "automatic":
            arm_environment.pop("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", None)
        else:
            arm_environment["GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS"] = selector
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
            env=arm_environment,
            start_new_session=True,
        )
        failure = None
        try:
            status = child.wait(timeout=1800)
        except BaseException as error:  # noqa: BLE001 - re-raised after cleanup and receipt
            failure = error
            if isinstance(error, subprocess.TimeoutExpired):
                status = 124
            elif isinstance(error, KeyboardInterrupt):
                status = 130
            elif isinstance(error, SystemExit) and type(error.code) is int:
                status = error.code
            else:
                status = 1
        try:
            _stop(child)
        except BaseException as error:  # noqa: BLE001 - preserve the primary failure
            if failure is None and status == 0:
                failure, status = error, 1
            else:
                print(f"Population cleanup failed: {error!r}", file=sys.stderr)
        try:
            with (target / "exit-status.txt").open("x") as stream:
                stream.write(f"{status}\n")
        except BaseException as error:
            if failure is None and status == 0:
                raise
            print(f"Exit receipt failed: {error!r}", file=sys.stderr)
        if failure is not None and not isinstance(failure, subprocess.TimeoutExpired):
            raise failure
        if status:
            raise SystemExit(status) from failure


if __name__ == "__main__":
    main()
