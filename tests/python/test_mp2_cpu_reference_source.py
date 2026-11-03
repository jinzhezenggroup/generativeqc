"""Complete CPU MP2 observables and bounded prepared-source lifetime coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator, load_basis

if TYPE_CHECKING:
    from generativeqc import BatchItemResult


@pytest.fixture(scope="module")
def physical_h2() -> dict[str, Any]:
    """Use the independently pinned physical DZ component, not route parity alone."""
    directory = Path(__file__).parents[1] / "reference_data/cbs"
    record = json.loads((directory / "h2-dz-tz.json").read_text())
    return {**record, "basis": load_basis(directory / record["low_basis"])}


def _calculator(record: dict[str, Any], budget: int) -> Calculator:
    return Calculator(
        method="mp2",
        basis=record["basis"],
        device="cpu",
        precision="fp64",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        correlation_memory_budget_bytes=budget,
    )


def _assert_observables(
    result: BatchItemResult, record: dict[str, Any], budget: int
) -> None:
    expected = record["reference"]["components"]["mp2_low"]
    assert result.converged and result.executed_backend == "cpu_reference"
    assert abs(result.energy - expected["energy"]) <= 1e-9
    np.testing.assert_allclose(result.forces, expected["forces"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=2e-8, rtol=0)
    assert result.correlation.force_provenance_flags == 0x7
    assert result.correlation.response_absolute_residual <= 1e-9
    assert 0 < result.correlation.planned_endpoint_peak_bytes <= budget
    assert (
        result.correlation.planned_endpoint_peak_bytes
        <= result.correlation.numeric_capacity_bytes
        <= budget
    )
    # Retained warm/reference owners are outside the force allocator's ledger.
    assert result.correlation.measured_endpoint_peak_bytes == 0


def _execute_twice(record: dict[str, Any], budget: int) -> list[BatchItemResult]:
    """Exercise owner reuse and last-good warm state across complete publications."""
    results = []
    with _calculator(record, budget).prepare_batch(
        [record["atoms"]], charges=[record["charge"]], warm_start=True
    ) as batch:
        for iteration in range(2):
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            _assert_observables(result, record, budget)
            assert result.warm_start_used == bool(iteration)
            results.append(result)
    return results


def _trace_values(journal: Path, key: str) -> list[Any]:
    return [
        event["value"]
        for line in journal.read_text().splitlines()
        if (event := json.loads(line)).get("key") == key
        and event["event"] == "VALUE"
        and event["name"] == "mp2_endpoint"
    ]


@pytest.mark.parametrize(
    "selection,tiling", [(None, None), ("0", None), ("1", None), ("1", "0")]
)
def test_cpu_mp2_prepared_source_complete_endpoint(
    physical_h2: dict[str, Any],
    selection: str | None,
    tiling: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default, explicit resident and streaming routes share independent E/F gates."""
    for name, value in (
        ("GENERATIVEQC_MP2_REFERENCE_SOURCE", selection),
        ("GENERATIVEQC_MP2_FORCE_SOURCE_TILING", tiling),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    journal = tmp_path / "endpoint.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
    _execute_twice(physical_h2, 16 << 20)
    routes = _trace_values(journal, "mp2_force_cpu_prepared_source")
    assert routes and all(route == int(selection != "0") for route in routes)
    assert not _trace_values(journal, "mp2_cpu_reference_source_fallback")
    retained = _trace_values(journal, "mp2_force_source_retained_bytes")
    assert retained and all(0 < capacity < 16 << 20 for capacity in retained)
    tiles = _trace_values(journal, "mp2_force_source_axis_tile")
    assert tiles and all(tile > 0 for tile in tiles)
    if tiling is None:
        assert all(tile > 1 for tile in tiles)


def test_cpu_mp2_optional_source_preserves_reported_budget(
    physical_h2: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use native admission on this build and require real resident eviction/retry.

    Reuse the streaming endpoint's reported peak as its tighter allowance. This
    avoids hard-coded platform byte thresholds and an expensive search through
    minimum-size source tiles. Both warm publications must fit that allowance;
    the resident source must actually retire rather than bypassing admission.
    """
    monkeypatch.setenv("GENERATIVEQC_MP2_REFERENCE_SOURCE", "0")
    monkeypatch.delenv("GENERATIVEQC_MP2_FORCE_SOURCE_TILING", raising=False)
    journal = tmp_path / "admission.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
    roomy = _execute_twice(physical_h2, 16 << 20)
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 7:"):
        _execute_twice(physical_h2, 8 << 20)
    budget = max(result.correlation.numeric_capacity_bytes for result in roomy)
    expected = _execute_twice(physical_h2, budget)
    resident_journal = tmp_path / "resident-retry.jsonl"
    monkeypatch.setenv("GENERATIVEQC_MP2_REFERENCE_SOURCE", "1")
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(resident_journal))
    actual = _execute_twice(physical_h2, budget)
    for streaming, resident in zip(expected, actual, strict=True):
        np.testing.assert_allclose(
            resident.energy, streaming.energy, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(resident.forces, streaming.forces, atol=2e-8, rtol=0)
    assert _trace_values(resident_journal, "mp2_cpu_reference_source_fallback")
    routes = _trace_values(resident_journal, "mp2_force_cpu_prepared_source")
    assert 0 in routes, "the admitted endpoint must retry without optional residency"


def test_cpu_mp2_scientific_failure_does_not_retire_optional_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An invalid denominator is not recoverable source-memory pressure."""
    journal = tmp_path / "scientific-error.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
    monkeypatch.setenv("GENERATIVEQC_MP2_REFERENCE_SOURCE", "1")
    calculator = Calculator(method="mp2", device="cpu", mp2_denominator_threshold=1e6)
    with pytest.raises(RuntimeError, match="near-zero MP2 denominator"):
        calculator.singlepoint(
            [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
            properties=("energy", "forces"),
        )
    assert not _trace_values(journal, "mp2_reference_source_fallback")
    assert not _trace_values(journal, "mp2_cpu_reference_source_fallback")
