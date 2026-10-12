"""CPU-only full RCCSD solver benchmark for the #762 native TensorIR cutover.

This times a supplied-MO RCCSD scientific solve (reference and MO blocks already
prepared), not a full HF+CCSD pipeline or a single contraction kernel.
Every observation runs in a fresh process: cold compilation and cached replay
reuse the same persistent cache but have independent process RSS high waters.

Run: PYTHONPATH=.:python python -m benchmarks.rccsd_cpu_compiled_endpoint
     --fixture h2o --threads 1 4 --output /tmp/rccsd-cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _worker(*, fixture: str, backend: str, cache: Path) -> dict[str, object]:
    import resource
    from types import SimpleNamespace

    import numpy as np

    from tools.cc_endpoint_fixtures import load, snapshot_from_fixture
    from tools.generativeqc_cc import SolverOptions, energy
    from tools.generativeqc_posthf.providers import BlockResult, ConventionalProvider

    class FixtureProvider(ConventionalProvider):
        def __init__(self, snapshot: object, g: np.ndarray) -> None:
            self.snapshot = snapshot
            self.g = g
            self.backend = "cpu"
            self.source = SimpleNamespace(_check_open=lambda: None)
            self.calls = 0

        def get(self, block: object) -> BlockResult:
            self.calls += 1
            return BlockResult(
                block,
                self.g[np.ix_(*block.slots)],
                self.snapshot.identity,
                self.snapshot.hamiltonian_id,
                {},
            )

    meta, arrays = load(fixture)
    source = SimpleNamespace(
        electron_count=int(arrays["occ"].sum()),
        geometry_hash="benchmark-" + fixture,
        basis_hash="benchmark-basis-" + fixture,
    )
    snapshot = snapshot_from_fixture(source, meta, arrays)
    provider = FixtureProvider(snapshot, arrays["g"])
    options = SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12)
    start = time.perf_counter()
    result = energy(
        snapshot,
        provider,
        options=options,
        backend=backend,
        cache=cache if backend == "cpu-native" else None,
    )
    elapsed = time.perf_counter() - start
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not result.converged:
        raise RuntimeError(f"{backend} {fixture} did not converge: {result.reason}")
    reference = float(meta["total_energy"])
    if abs(result.total_energy - reference) > 1e-8:
        raise RuntimeError(f"{backend} {fixture} fails pinned CPU endpoint oracle")
    return {
        "fixture": fixture,
        "backend": backend,
        "elapsed_seconds": elapsed,
        "process_peak_rss_kib_linux": rss_kib,
        "iterations": result.iterations,
        "energy": result.total_energy,
        "pinned_reference_energy": reference,
        "final_r1_max": result.final_r1_max,
        "final_r2_max": result.final_r2_max,
        "equation_hash": result.state.provenance["equation_hash"],
        "equation_backend": result.state.provenance["equation_backend"],
        "native_program_required_bytes": result.state.provenance[
            "native_program_required_bytes"
        ],
        "logical_required_bytes": result.state.provenance["logical_required_bytes"],
        "mo_block_reads": provider.calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", choices=["h2", "he", "h2o", "nh3", "ch4"], default="h2o"
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["cpu", "cpu-native"], default="cpu")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if any(thread < 1 for thread in args.threads):
        parser.error("--threads values must be positive")
    if args.worker:
        if args.cache is None:
            parser.error("worker requires --cache")
        print(
            json.dumps(
                _worker(fixture=args.fixture, backend=args.backend, cache=args.cache)
            )
        )
        return

    observations: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rccsd-cpu-762-") as root:
        for threads in args.threads:
            cache = Path(root) / f"threads-{threads}"
            env = dict(os.environ)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = str(threads)
            for label, backend in (
                ("reference", "cpu"),
                ("native_cold", "cpu-native"),
                ("native_warm", "cpu-native"),
            ):
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.rccsd_cpu_compiled_endpoint",
                    "--worker",
                    "--fixture",
                    args.fixture,
                    "--backend",
                    backend,
                    "--cache",
                    str(cache),
                ]
                process = subprocess.run(
                    command, env=env, capture_output=True, text=True, check=False
                )
                if process.returncode:
                    raise RuntimeError(
                        f"RCCSD {label} failed: {process.stderr.strip()}"
                    )
                observation = json.loads(process.stdout)
                observation["stage"] = label
                observation["blas_threads"] = threads
                observations.append(observation)
    record = {
        "schema": "generativeqc.rccsd.native-cpu-endpoint.v1",
        "scope": "complete supplied-MO RCCSD solver, excludes HF/AO generation",
        "peak_memory_scope": "isolated Linux process high-water RSS",
        "cold_warm_scope": "new artifact versus cached artifact in fresh processes",
        "no_cuda": True,
        "observations": observations,
    }
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
