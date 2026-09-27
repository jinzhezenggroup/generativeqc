"""Benchmark VibeQC's pure-Python integral compiler under threaded CPython.

This intentionally excludes native compilation and CUDA execution.  It measures
independent IntegralIR/symbolic-AD construction, where a free-threaded
interpreter can plausibly remove Python-bytecode serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import sysconfig
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vibeqc_compiler.integral import (
    FUSED_SHELL_SPEC_BY_NAME,
    KernelConsumer,
    build_integral_ir,
    build_shell_class_component_kernel,
)

_CASE_LIMITS = {
    "ppps": 27,
    "dpss": 18,
    "dppp": 12,
}


def _gil_enabled() -> bool | None:
    probe = getattr(sys, "_is_gil_enabled", None)
    if probe is None:
        return None
    return bool(probe())


def _work_items(repeats: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    items: list[tuple[str, tuple[str, ...]]] = []
    for _ in range(repeats):
        for name, limit in _CASE_LIMITS.items():
            spec = FUSED_SHELL_SPEC_BY_NAME[name]
            items.extend(
                (name, tuple(component)) for component in spec.components[:limit]
            )
    return tuple(items)


def _build_one(item: tuple[str, tuple[str, ...]]) -> str:
    name, component = item
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    integral = build_integral_ir(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    kernel = build_shell_class_component_kernel(
        spec,
        component,
        integral=integral,
    )
    identity = (
        name,
        component,
        len(kernel.graph.nodes),
        kernel.value.identifier,
        tuple(
            expression.identifier
            for center in kernel.gradients
            for expression in center
        ),
    )
    return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()


def _execute(
    items: tuple[tuple[str, tuple[str, ...]], ...],
    workers: int,
) -> tuple[float, str]:
    started = time.perf_counter()
    if workers == 1:
        results = tuple(map(_build_one, items))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = tuple(pool.map(_build_one, items))
    elapsed = time.perf_counter() - started
    fingerprint = hashlib.sha256("".join(results).encode("ascii")).hexdigest()
    return elapsed, fingerprint


def _benchmark(
    items: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    workers: tuple[int, ...],
    samples: int,
) -> dict[str, typing.Any]:
    # Prime imports and immutable module-level metadata outside measured work.
    _build_one(items[0])

    rows: dict[str, typing.Any] = {}
    expected_fingerprint: str | None = None
    for worker_count in workers:
        timings = []
        fingerprint = ""
        for _ in range(samples):
            elapsed, fingerprint = _execute(items, worker_count)
            timings.append(elapsed)
        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
        elif fingerprint != expected_fingerprint:
            raise RuntimeError("threaded compiler result differs from serial result")
        rows[str(worker_count)] = {
            "median_seconds": statistics.median(timings),
            "samples_seconds": timings,
        }

    baseline = rows["1"]["median_seconds"]
    for row in rows.values():
        row["speedup_vs_serial"] = baseline / row["median_seconds"]

    return {
        "workers": rows,
        "fingerprint": expected_fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if arguments.repeats < 1 or arguments.samples < 1:
        raise SystemExit("repeats and samples must be positive")
    worker_counts = tuple(arguments.workers)
    if (
        not worker_counts
        or worker_counts[0] != 1
        or any(value < 1 for value in worker_counts)
    ):
        raise SystemExit("workers must start with serial worker count 1")

    items = _work_items(arguments.repeats)
    result = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": sys.implementation.name,
            "py_gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
            "gil_enabled": _gil_enabled(),
        },
        "host": {
            "cpu_count": os.cpu_count(),
        },
        "workload": {
            "case_limits": _CASE_LIMITS,
            "repeats": arguments.repeats,
            "task_count": len(items),
        },
        **_benchmark(
            items,
            workers=worker_counts,
            samples=arguments.samples,
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
