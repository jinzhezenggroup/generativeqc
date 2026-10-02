"""Protect accuracy gates independently from SCC/timing classification."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
        "timing_contract": BENCHMARK.TIMING_CONTRACT,
        "settings": {"fresh_scc": True},
        "device": "cuda",
        "library_sha256": "test",
        "rows": [
            {
                "case": "h2",
                "geometry_sha256": "geometry",
                "construction_seconds": 0.25,
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


def test_cold_total_cannot_hide_costly_constructor() -> None:
    reference = report()
    candidate = copy.deepcopy(reference)
    candidate["rows"][0]["construction_seconds"] = 2.0
    candidate["rows"][0]["samples"][0]["seconds"] = 0.5
    rows = {
        row["mode"]: row
        for row in BENCHMARK.compare_reports(reference, candidate)["rows"]
    }
    assert rows["cold"]["speedup"] == 2.0
    assert rows["cold_total"]["speedup"] == 0.5
    assert rows["warm"]["speedup"] == 1.0
    assert rows["changed"]["speedup"] == 1.0
    assert all(
        row["accuracy_passed"] and row["iterations_match"] for row in rows.values()
    )


def test_comparison_rejects_mixed_cleanup_contracts() -> None:
    reference, candidate = report(), report()
    del reference["timing_contract"]
    with pytest.raises(ValueError, match="timing contracts differ"):
        BENCHMARK.compare_reports(reference, candidate)
    del candidate["timing_contract"]
    assert BENCHMARK.compare_reports(reference, candidate)["timing_contract"] == (
        "legacy-mixed-cleanup"
    )


@pytest.mark.parametrize("engine", ("generativeqc", "xtbloom"))
def test_measurement_separates_previous_calculator_cleanup(
    engine: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately expensive finalizer must not inflate the next constructor."""
    import generativeqc

    library = tmp_path / "native.so"
    library.touch()
    destination = tmp_path / "measurement.json"
    elapsed = [0.0]

    class Calculator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            elapsed[0] += 1.0
            self._library = SimpleNamespace(_name=str(library))

        def singlepoint(self, *args: object, **kwargs: object) -> SimpleNamespace:
            elapsed[0] += 2.0
            return SimpleNamespace(
                converged=True,
                iterations=3,
                scc_converged=True,
                scc_iterations=3,
                energy=-1.0,
                forces=[[0.0, 0.0, 0.0]],
            )

        def update(self, **kwargs: object) -> None:
            pass

        def __del__(self) -> None:
            elapsed[0] += 100.0

    monkeypatch.setattr(generativeqc, "Calculator", Calculator)
    monkeypatch.setitem(
        sys.modules,
        "xtbloom",
        SimpleNamespace(Calculator=Calculator, __file__=str(library)),
    )
    monkeypatch.setitem(
        sys.modules,
        "xtbloom.library",
        SimpleNamespace(load_library=lambda: SimpleNamespace(_name=str(library))),
    )
    monkeypatch.setattr(BENCHMARK.time, "perf_counter", lambda: elapsed[0])
    monkeypatch.setattr(BENCHMARK, "source_revision", lambda _: "fixture")
    monkeypatch.setattr(
        BENCHMARK,
        "cases",
        lambda _: [
            {"name": name, "symbols": ["H"], "positions": [[0.0, 0.0, 0.0]]}
            for name in ("first", "second")
        ],
    )
    monkeypatch.setenv(f"{engine.upper()}_LIBRARY", str(library))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--engine",
            engine,
            "--device",
            "cpu",
            "--repeat",
            "1",
            "--output",
            str(destination),
        ],
    )
    BENCHMARK.main()
    measured = json.loads(destination.read_text())
    assert measured["timing_contract"] == BENCHMARK.TIMING_CONTRACT
    assert len(measured["rows"]) == 2
    for row in measured["rows"]:
        assert row["construction_seconds"] == 1.0
        assert row["cleanup_seconds"] == 100.0
        assert [sample["seconds"] for sample in row["samples"]] == [2.0] * 3


@pytest.mark.parametrize("construction", [None, -0.1, float("nan"), float("inf")])
def test_missing_or_invalid_setup_cost_fails_closed(construction: float | None) -> None:
    reference = report()
    candidate = copy.deepcopy(reference)
    if construction is None:
        del candidate["rows"][0]["construction_seconds"]
    else:
        candidate["rows"][0]["construction_seconds"] = construction
    with pytest.raises(ValueError, match="construction timing"):
        BENCHMARK.compare_reports(reference, candidate)


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
