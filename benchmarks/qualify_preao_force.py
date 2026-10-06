"""Matched complete PBE0 E+F with dense, sampled or the real guarded default.

Run inside a finite Slurm allocation. Experimental producers do not register a
policy profile; ``auto`` leaves policy resolution and producer dispatch intact.
Intrusive phase clocks belong to separate campaigns, never timing samples.
"""

import argparse
import hashlib
import json
import sys
import typing
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from generativeqc import _force_active_ao, _stationary_cuda
from generativeqc._stationary_composite_cuda import (
    PreparedCompositeStationaryCudaGradient,
)
from generativeqc.batch import PreparedBatch

from benchmarks.readme_omol25 import main
from benchmarks.readme_pbe0 import PBE0


def run() -> None:
    """Journal every public force census without timing the independent oracle."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--domain-producer",
        required=True,
        choices=(
            "dense",
            "sampled-jets",
            "pre-ao-envelope-native-csr",
            "auto",
        ),
    )
    parser.add_argument("--intrusive", action="store_true")
    arguments, remaining = parser.parse_known_args()
    original_force = PreparedBatch._public_dft_cuda_force
    original_diagnostic = _stationary_cuda.complete_rks_cuda_gradient_diagnostic
    original_composite = PreparedCompositeStationaryCudaGradient.execute
    work_records = []

    def diagnostic(*values: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if arguments.domain_producer != "auto":
            kwargs.update(
                resident_ao_cutoff=None
                if arguments.domain_producer == "dense"
                else 1e-16,
                resident_ao_cache_bytes=64 << 20,
                resident_ao_producer="sampled-jets"
                if arguments.domain_producer == "dense"
                else arguments.domain_producer,
                resident_ao_max_active_fraction=1.0,
            )
        kwargs["profile_device"] = arguments.intrusive
        return original_diagnostic(*values, **kwargs)

    def composite(
        self: typing.Any, *values: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        if arguments.intrusive:
            raise ValueError(
                "composite intrusive timers require a separate explicit harness"
            )
        if arguments.domain_producer != "auto":
            kwargs.update(
                active_ao_cutoff=None
                if arguments.domain_producer == "dense"
                else 1e-16,
                active_ao_cache_bytes=64 << 20,
                active_ao_producer="sampled-jets"
                if arguments.domain_producer == "dense"
                else arguments.domain_producer,
                active_ao_max_active_fraction=1.0,
            )
        return original_composite(self, *values, **kwargs)

    def observe_force(
        self: typing.Any, *values: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        forces, work = original_force(self, *values, **kwargs)
        if arguments.domain_producer == "auto":
            policy = work["force_active_ao_policy"]
            if policy["producer"] != "pre-ao-envelope-native-csr" or policy[
                "actual_mode"
            ] not in ("selected", "dense-identity", "dense-fallback"):
                raise RuntimeError(
                    "guarded default did not execute the qualified native CSR domain"
                )
        work_records.append(work)
        return forces, work

    def encode(value: typing.Any) -> typing.Any:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported receipt value: {type(value)}")

    output = Path(remaining[remaining.index("--output") + 1])
    with ExitStack() as stack:
        if arguments.domain_producer != "auto":
            stack.enter_context(
                patch.object(_force_active_ao, "QUALIFIED_FORCE_ACTIVE_AO_PROFILES", ())
            )
        stack.enter_context(
            patch.object(PreparedBatch, "_public_dft_cuda_force", observe_force)
        )
        stack.enter_context(
            patch.object(
                _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", diagnostic
            )
        )
        stack.enter_context(
            patch.object(PreparedCompositeStationaryCudaGradient, "execute", composite)
        )
        stack.enter_context(patch.object(sys, "argv", [sys.argv[0], *remaining]))
        try:
            main(PBE0)
        finally:
            if output.exists():
                record = json.loads(output.read_text())
                record["p0c_domain_producer"] = (
                    "pre-ao-envelope-native-csr"
                    if arguments.domain_producer == "auto"
                    else arguments.domain_producer
                )
                record["p0c_execution_mode"] = arguments.domain_producer
                record["p0c_intrusive_stage_profile"] = arguments.intrusive
                record["p0c_force_work"] = work_records
                root = Path(__file__).resolve().parents[1]
                for path in (
                    "python/generativeqc/_force_active_ao.py",
                    "python/generativeqc/_resident_ao_maps.py",
                    "python/generativeqc_compiler/dft/envelope_cuda.py",
                    "python/generativeqc_compiler/dft/cuda.py",
                    "src/dft/cuda_grid.cu",
                    "benchmarks/qualify_preao_force.py",
                ):
                    record["source_file_sha256"][path] = hashlib.sha256(
                        (root / path).read_bytes()
                    ).hexdigest()
                if record.get("status") == "measured" and len(work_records) != len(
                    record["records"]
                ):
                    raise RuntimeError(
                        "force census does not cover every public E+F call"
                    )
                output.write_text(
                    json.dumps(record, default=encode, indent=2, allow_nan=False) + "\n"
                )


if __name__ == "__main__":
    run()
