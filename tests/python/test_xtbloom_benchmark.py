"""Protect accuracy gates independently from SCC/timing classification."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_xtbloom",
    Path(__file__).resolve().parents[2] / "benchmarks/compare_xtbloom.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def report() -> dict:
    """A tiny complete cold/warm/changed report for gate failure probes."""
    return {
        "settings": {"fresh_scc": True},
        "device": "cuda",
        "library_sha256": "test",
        "rows": [
            {
                "case": "h2",
                "geometry_sha256": "geometry",
                "samples": [
                    {
                        "mode": mode,
                        "repeat": 0,
                        "geometry_sha256": BENCHMARK.geometry_sha256(
                            ["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
                        ),
                        "iterations": 7,
                        "seconds": 1.0,
                        "energy": -1.0,
                        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    }
                    for mode in ("cold", "warm", "changed")
                ],
            }
        ],
    }


def test_inaccurate_repeat_cannot_hide_behind_iteration_mismatch() -> None:
    reference = report()
    candidate = copy.deepcopy(reference)
    candidate["rows"][0]["samples"][1].update(energy=-1.01, iterations=2, seconds=0.5)
    rows = BENCHMARK.compare_reports(reference, candidate)["rows"]
    assert not any(row["accuracy_passed"] for row in rows)
    assert not rows[1]["iterations_match"]
    assert rows[1]["speedup"] == 2.0


@pytest.mark.parametrize("field", ("energy", "forces"))
def test_comparison_rejects_nonfinite_results(field: str) -> None:
    reference = report()
    candidate = copy.deepcopy(reference)
    sample = candidate["rows"][0]["samples"][0]
    sample[field] = (
        float("nan")
        if field == "energy"
        else [[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="nonfinite"):
        BENCHMARK.compare_reports(reference, candidate)


@pytest.mark.parametrize("change", ("settings", "device", "geometry", "samples"))
def test_comparison_rejects_incomparable_work(change: str) -> None:
    reference = report()
    candidate = copy.deepcopy(reference)
    if change == "settings":
        candidate["settings"]["fresh_scc"] = False
    elif change == "device":
        candidate["device"] = "cpu"
    elif change == "geometry":
        candidate["rows"][0]["geometry_sha256"] = "changed"
    else:
        candidate["rows"][0]["samples"].pop()
    with pytest.raises(ValueError):
        BENCHMARK.compare_reports(reference, candidate)


@pytest.mark.parametrize("mode", ("cold", "warm", "changed"))
@pytest.mark.parametrize("change", ("symbols", "positions"))
def test_comparison_rejects_different_sample_geometry(mode: str, change: str) -> None:
    """Equal starting geometry and numerical outputs cannot hide changed inputs."""
    reference = report()
    candidate = copy.deepcopy(reference)
    symbols = ["H", "H"]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    if change == "symbols":
        symbols[1] = "He"
    else:
        positions[1][2] += 0.001
    sample = next(s for s in candidate["rows"][0]["samples"] if s["mode"] == mode)
    sample["geometry_sha256"] = BENCHMARK.geometry_sha256(symbols, positions)
    with pytest.raises(ValueError, match="sample input geometries differ"):
        BENCHMARK.compare_reports(reference, candidate)


@pytest.mark.parametrize("side", ("reference", "candidate", "both"))
def test_comparison_rejects_missing_sample_geometry(side: str) -> None:
    """Legacy receipts cannot establish which changed coordinates were measured."""
    reference = report()
    candidate = copy.deepcopy(reference)
    for name, measured in (("reference", reference), ("candidate", candidate)):
        if side in (name, "both"):
            del measured["rows"][0]["samples"][2]["geometry_sha256"]
    with pytest.raises(ValueError, match="sample geometry identities are missing"):
        BENCHMARK.compare_reports(reference, candidate)
