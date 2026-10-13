"""Intrusive cold/first-warm attribution using the unchanged README endpoint.

Run as ``python -m benchmarks.profile_dft_endpoint --capture warm --method ...
native ... --reference ... --output ...`` under finite srun. All ordinary
README scientific options/gates apply. The deliberate stop after the first
frozen warm return is a diagnostic population, never a completed timing cohort.
Nsight may capture the warm call via cudaProfilerApi; ``--capture all`` includes
construction/discovery. CUDA event/census tracing is separately opt-in via the
native diagnostic environment variables. None of these runs supplies clean
endpoint speedup evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class WallLedger:
    """Nested host intervals with an exactly additive exclusive wall ledger.

    This measures synchronous host ownership, including waits already present
    in the endpoint. It adds no CUDA fences. GPU activities are a separate view
    and must not be added to these wall intervals.
    """

    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self.clock = clock
        self.rows: list[dict[str, Any]] = []
        self.stack: list[int] = []

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        parent = self.stack[-1] if self.stack else None
        index = len(self.rows)
        row = {"name": name, "parent": parent, "begin_ns": self.clock()}
        self.rows.append(row)
        self.stack.append(index)
        try:
            yield
        finally:
            row["end_ns"] = self.clock()
            if self.stack.pop() != index:
                raise RuntimeError("DFT profile scopes ended out of order")

    def snapshot(self, first: int = 0) -> dict[str, Any]:
        """Validate finished nesting and subtract immediate children exactly."""
        rows = self.rows[first:]
        totals: dict[str, int] = defaultdict(int)
        roots = 0
        for i, row in enumerate(rows, first):
            if "end_ns" not in row:
                raise ValueError("incomplete DFT wall interval")
            begin, end = row["begin_ns"], row["end_ns"]
            if end < begin:
                raise ValueError("DFT wall clock moved backwards")
            children = [child for child in rows if child["parent"] == i]
            last = begin
            child_ns = 0
            for child in children:
                if child["begin_ns"] < last or child["end_ns"] > end:
                    raise ValueError("overlapping or out-of-parent DFT wall interval")
                last = child["end_ns"]
                child_ns += child["end_ns"] - child["begin_ns"]
            totals[row["name"]] += end - begin - child_ns
            if row["parent"] is None or row["parent"] < first:
                roots += end - begin
        return {
            "clock": "perf_counter_ns",
            "exclusive_wall_seconds": {
                key: value / 1e9 for key, value in sorted(totals.items())
            },
            "root_seconds": roots / 1e9,
            "reconciliation_error_seconds": (roots - sum(totals.values())) / 1e9,
            "intervals": rows,
        }


class _NativeCall:
    """Observe a ctypes call while forwarding later ABI signature assignments."""

    def __init__(self, function: Any, observe: Callable[..., Any]) -> None:
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "observe", observe)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.function, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.function, name, value)

    def __call__(self, *args: Any) -> Any:
        return self.observe(self.function, *args)


class _DiagnosticComplete(Exception):
    """Internal deliberate endpoint bound, distinct from a numerical failure."""


def main() -> None:
    """Retain the real endpoint records and independently attributable scopes."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--capture", choices=("warm", "all", "none"), default="warm")
    parser.add_argument("--output", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    remaining += ["--output", str(known.output)]
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real-GPU profiling requires finite srun")

    import cupy as cp
    from generativeqc import Calculator, dispersion
    from generativeqc._dft_gradient import StationaryDerivativeContract
    from generativeqc._ks_snapshot import NativeKsSnapshot
    from generativeqc.batch import PreparedBatch

    from benchmarks import readme_omol25 as driver
    from benchmarks.readme_dft import main as endpoint_main

    ledger = WallLedger()
    patched_libraries: set[int] = set()
    calls = 0
    call_ledger_start = 0
    construction: list[dict[str, Any]] = []

    with ExitStack() as cleanup:

        def replace(owner: Any, name: str, function: Any) -> None:
            previous = getattr(owner, name)
            setattr(owner, name, function)
            cleanup.callback(setattr, owner, name, previous)

        @contextmanager
        def scope(name: str) -> Iterator[None]:
            cp.cuda.nvtx.RangePush(f"dft-profile/{name}")
            try:
                with ledger.scope(name):
                    yield
            finally:
                cp.cuda.nvtx.RangePop()

        def observe(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(function)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                with scope(name):
                    return function(*args, **kwargs)

            return wrapped

        def native_calls(library: Any) -> None:
            if id(library) in patched_libraries:
                return
            patched_libraries.add(id(library))
            for symbol, name in (
                ("generativeqc_batch_execute", "native_scf_and_publication"),
                (
                    "generativeqc_ks_batch_execute_selected_forces_v1",
                    "native_selected_execution",
                ),
            ):
                original = getattr(library, symbol, None)
                if original is None:
                    continue

                def invoke(function: Any, *args: Any, label: str = name) -> Any:
                    with scope(label):
                        return function(*args)

                replace(library, symbol, _NativeCall(original, invoke))

        original_init = Calculator.__init__

        def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
            first = len(ledger.rows)
            with scope("calculator_construction"):
                original_init(self, *args, **kwargs)
            native_calls(self._library)
            construction.append(ledger.snapshot(first))

        replace(Calculator, "__init__", initialize)
        original_prepare = Calculator.prepare_batch

        def prepare(*args: Any, **kwargs: Any) -> Any:
            first = len(ledger.rows)
            with scope("batch_preparation"):
                result = original_prepare(*args, **kwargs)
            construction.append(ledger.snapshot(first))
            return result

        replace(Calculator, "prepare_batch", prepare)
        for owner, method, label in (
            (
                PreparedBatch,
                "_public_dft_cuda_force",
                "stationary_force_and_publication",
            ),
            (NativeKsSnapshot, "__init__", "final_state_snapshot"),
            (NativeKsSnapshot, "decode", "final_state_decode"),
            (StationaryDerivativeContract, "validate", "final_state_validation"),
            (dispersion.D4CorrectionBatch, "execute", "d4"),
            (dispersion, "_evaluate_gcp_result", "native_cpu_gcp"),
        ):
            replace(owner, method, observe(label, getattr(owner, method)))

        original_execute = PreparedBatch.execute

        def execute(self: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal calls, call_ledger_start
            calls += 1
            call_ledger_start = len(ledger.rows)
            capture = known.capture == "warm" and calls == 2
            if capture:
                with scope("profiler_capture_start"):
                    cp.cuda.runtime.profilerStart()
            try:
                with scope("public_execute"):
                    return original_execute(self, *args, **kwargs)
            finally:
                if capture:
                    # The public result already owns complete host forces.
                    with scope("profiler_capture_stop"):
                        cp.cuda.runtime.profilerStop()

        replace(PreparedBatch, "execute", execute)
        original_retain = driver.retain_endpoint_record

        def retain(record: Any, row: Any, reference: Any, save: Any) -> None:
            wall = ledger.snapshot(call_ledger_start)
            # The unchanged driver also times result extraction and its final
            # null-stream drain outside execute(). Keep that residual visible;
            # nested construction diagnostics must not be added a second time.
            residual = row["seconds"] - wall["root_seconds"]
            if residual < -1e-6:
                raise ValueError("DFT observer exceeds the complete driver interval")
            wall["driver_seconds"] = row["seconds"]
            wall["outside_observer_seconds"] = residual
            complete = dict(wall["exclusive_wall_seconds"])
            complete["driver_return_and_final_sync"] = residual
            complete["endpoint_preparation"] = row["prepare_seconds"]
            wall["complete_exclusive_wall_seconds"] = complete
            wall["complete_seconds"] = row["complete_seconds"]
            wall["complete_reconciliation_error_seconds"] = row[
                "complete_seconds"
            ] - sum(complete.values())
            row["diagnostic_wall_ledger"] = wall
            if row["phase"] == "cold":
                row["diagnostic_construction_ledgers"] = construction
            record["diagnostic_only"] = True
            original_retain(record, row, reference, save)
            if row["phase"] == "warm":
                raise _DiagnosticComplete()

        replace(driver, "retain_endpoint_record", retain)
        if known.capture == "all":
            cp.cuda.runtime.profilerStart()
        try:
            # The README dispatcher reads argv itself; preserve its unchanged
            # argument parsing, scientific identity and per-call acceptance.
            import sys

            previous = sys.argv
            sys.argv = [previous[0], *remaining]
            cleanup.callback(setattr, sys, "argv", previous)
            try:
                endpoint_main()
            except _DiagnosticComplete:
                record = json.loads(known.output.read_text())
                record.update(
                    status="diagnostic_completed", stage="cold-and-first-warm-only"
                )
                record.pop("error", None)
                known.output.write_text(
                    json.dumps(record, indent=2, allow_nan=False) + "\n"
                )
        finally:
            if known.capture == "all":
                cp.cuda.runtime.profilerStop()


if __name__ == "__main__":
    main()
