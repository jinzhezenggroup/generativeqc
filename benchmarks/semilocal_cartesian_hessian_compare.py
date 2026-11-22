"""Compare native Cartesian Hessian/HVP endpoints and emit paired timing receipts."""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

PHASES = ("cold", "warm", "moved")
WORK_FIELDS = (
    "completed_directions",
    "response_iterations",
    "cpks_cached_actions",
    "cpks_physical_checks",
    "induced_fock_actions",
    "one_electron_primitive_components",
    "exact_primitive_components",
    "grid_mixed_contractions",
)


def compare(root: Path, pairs: int) -> None:
    """Reject missing phases, changed scientific work or a failed endpoint gate."""
    receipts = []
    methods = ("lda-rks", "pbe-rks")
    endpoints = {"dense": ("water", 9), "hvp": ("water-dimer", 1)}
    if pairs <= 0:
        raise ValueError("positive matched pair count required")
    for method, endpoint in itertools.product(methods, endpoints):
        fixture, directions = endpoints[endpoint]
        medians: dict[str, list[float]] = {phase: [] for phase in PHASES}
        for pair in range(1, pairs + 1):
            run = root / f"pair-{pair}"
            baseline = run / f"{method}-{endpoint}-baseline"
            candidate = run / f"{method}-{endpoint}-candidate"
            rows = [
                [
                    json.loads(line)
                    for line in path.with_suffix(".jsonl").read_text().splitlines()
                ]
                for path in (baseline, candidate)
            ]
            for phase, control, trial in zip(PHASES, *rows, strict=True):
                assert control["job"] == trial["job"] and control["job"]
                for arm in (control, trial):
                    assert arm["arm"] == "native"
                    assert (
                        arm["method"],
                        arm["endpoint"],
                        arm["fixture"],
                        arm["phase"],
                    ) == (method, endpoint, fixture, phase)
                    assert (
                        math.isfinite(arm["wall_seconds"]) and arm["wall_seconds"] > 0
                    )
                    audit = arm["diagnostics"]
                    assert audit["response_backend"] == "cuda"
                    assert audit["numerical_host_fallbacks"] == 0
                    assert audit["intermediate_panel_downloads"] == 0
                    assert audit["completed_directions"] == directions
                work = {name: trial["diagnostics"][name] for name in WORK_FIELDS}
                assert work == {
                    name: control["diagnostics"][name] for name in WORK_FIELDS
                }
                expected, actual = (
                    np.load(path / f"{phase}.npy", allow_pickle=False)
                    for path in (baseline, candidate)
                )
                assert expected.shape == actual.shape and np.isfinite(expected).all()
                np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)
                medians[phase].append(trial["wall_seconds"] / control["wall_seconds"])
                receipts.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "fixture": fixture,
                        "phase": phase,
                        "pair": pair,
                        "job": trial["job"],
                        "baseline_seconds": control["wall_seconds"],
                        "candidate_seconds": trial["wall_seconds"],
                        "max_abs_difference": float(np.max(np.abs(actual - expected))),
                        **work,
                    }
                )
        for phase, ratios in medians.items():
            ratio = statistics.median(ratios)
            assert ratio <= 1.05, (method, endpoint, phase, ratio)
    for receipt in receipts:
        print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pairs", type=int, default=9)
    arguments = parser.parse_args()
    compare(arguments.directory, arguments.pairs)
