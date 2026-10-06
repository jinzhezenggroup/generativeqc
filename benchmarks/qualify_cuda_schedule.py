"""Run complete PBE0 endpoints on an exact CUDA schedule source trial.

Unlike AO-producer qualification, schedule trials may legitimately select a
different existing AO policy. Preserve that policy and its work receipt rather
than forcing a producer or assuming that a larger tile remains sparse. Intrusive
device timers belong in separate runs, never in the clean timing population.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from generativeqc import _stationary_cuda
from generativeqc.batch import PreparedBatch

from benchmarks.readme_omol25 import main
from benchmarks.readme_pbe0 import PBE0


def run() -> None:
    """Retain the standard scientific protocol and actual production routing."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--intrusive", action="store_true")
    arguments, remaining = parser.parse_known_args()
    diagnostic = _stationary_cuda.complete_rks_cuda_gradient_diagnostic
    original_force = PreparedBatch._public_dft_cuda_force
    work_records = []

    def profiled(*values: Any, **kwargs: Any) -> Any:
        kwargs["profile_device"] = arguments.intrusive
        return diagnostic(*values, **kwargs)

    def observed_force(self: Any, *values: Any, **kwargs: Any) -> Any:
        forces, work = original_force(self, *values, **kwargs)
        work_records.append(work)
        return forces, work

    def encode(value: Any) -> Any:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported schedule receipt value: {type(value)}")

    output = (
        Path(remaining[remaining.index("--output") + 1])
        if "--output" in remaining
        else None
    )
    with (
        patch.object(sys, "argv", [sys.argv[0], *remaining]),
        patch.object(
            _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", profiled
        ),
        patch.object(PreparedBatch, "_public_dft_cuda_force", observed_force),
    ):
        try:
            main(PBE0)
        finally:
            if output is not None and output.exists():
                record = json.loads(output.read_text())
                record["cuda_schedule_intrusive"] = arguments.intrusive
                record["cuda_schedule_force_work"] = work_records
                output.write_text(json.dumps(record, indent=2, default=encode) + "\n")


if __name__ == "__main__":
    run()
