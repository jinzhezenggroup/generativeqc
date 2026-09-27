"""Unit tests for the cross-functional DFT force component evidence schema."""

from __future__ import annotations

import math

import pytest

from benchmarks.dft_force_components import (
    COMPONENTS,
    normalize_force_work,
    select_force_work,
)


def test_stationary_timeline_normalizes_without_zero_filling_missing_components() -> (
    None
):
    work = {
        "endpoint_seconds": 2.0,
        "timeline": {
            "endpoint_seconds": 2.0,
            "exclusive_wall_seconds": {
                "preparation": 0.2,
                "python_packing": 0.4,
                "primitive_derivative_reduction_sync": 0.5,
                "xc_geometry_and_sync": 0.3,
                "final_reduction": 0.1,
            },
        },
        "device_phase_ms": {
            "primitive_derivative_kernel": 4.0,
            "primitive_reduction": 1.0,
            "geometry_kernel": 2.0,
            "geometry_reduction": 0.5,
            "primitive_h2d": 0.7,
            "geometry_h2d": 0.3,
            "synchronization_wait_wall": 1.5,
        },
        "transfer_work": {
            "source_h2d_bytes": 100,
            "source_d2h_bytes": 20,
            "source_h2d_calls": 4,
            "source_d2h_calls": 2,
            "tensor_h2d_numeric_bytes": 12,
            "tensor_d2h_bytes": 8,
        },
        "synchronizations": 3,
    }

    record = normalize_force_work(work, state_export_seconds=0.1)

    assert record["schema"] == "vibeqc.dft-force-components.v1"
    assert record["source_route"] == "stationary-exclusive-wall"
    assert record["endpoint_seconds"] == pytest.approx(2.1)
    assert record["wall_seconds"]["host_packing"] == pytest.approx(0.5)
    assert record["wall_seconds"]["stationary_integral_derivatives"] == pytest.approx(
        0.5
    )
    assert record["wall_seconds"]["semilocal_geometry_response"] == pytest.approx(0.3)
    assert record["profiled_ms"]["stationary_integral_derivatives"] == pytest.approx(
        5.0
    )
    assert record["profiled_ms"]["h2d_d2h"] == pytest.approx(1.0)
    assert record["profiled_ms"]["synchronization_fences"] == pytest.approx(1.5)
    assert record["traffic"]["source_h2d_bytes"] == 100
    assert record["traffic"]["source_synchronizations"] == 3
    assert record["wall_seconds"]["scf_fock_j"] is None
    assert "scf_fock_j" in record["coverage"]["missing_wall_seconds"]
    assert set(record["wall_seconds"]) == set(COMPONENTS)


def test_wb97mv_component_seconds_map_to_same_schema() -> None:
    work = {
        "execution": "cuda-complete-wb97mv",
        "endpoint_seconds": 5.0,
        "component_seconds": {
            "prepare": 0.5,
            "integral_derivatives": 2.0,
            "semilocal_geometry_and_features": 1.0,
            "vv10_pairs": 0.4,
            "nonlocal_geometry": 0.5,
            "reduction_and_validation": 0.6,
        },
        "native_integral_resources": {
            "one_electron_h2d_bytes": 12,
            "one_electron_d2h_bytes": 24,
            "final_state_export_d2h_bytes": 36,
            "final_state_export_synchronizations": 2,
        },
        "snapshot_export_work": {"d2h_bytes": 48, "synchronizations": 1},
    }

    record = normalize_force_work(work)

    assert record["source_route"] == "wb97mv-component-seconds"
    assert record["wall_seconds"]["stationary_integral_derivatives"] == pytest.approx(
        2.0
    )
    assert record["wall_seconds"]["semilocal_geometry_response"] == pytest.approx(1.0)
    assert record["wall_seconds"]["vv10_rvv10"] == pytest.approx(0.9)
    assert record["attributed_wall_seconds"] == pytest.approx(5.0)
    assert record["unattributed_wall_seconds"] == pytest.approx(0.0)
    assert record["traffic"]["final_state_export_d2h_bytes"] == 36
    assert record["traffic"]["snapshot_export_synchronizations"] == 1
    assert record["wall_seconds"]["scf_short_range_k"] is None


def test_public_generated_force_record_selects_requested_item() -> None:
    raw = [
        {"index": 1, "work": {"endpoint_seconds": 2.0}},
        {"index": 0, "work": {"endpoint_seconds": 1.0}},
    ]
    assert select_force_work(raw, index=0)["endpoint_seconds"] == 1.0
    assert select_force_work(raw, index=1)["endpoint_seconds"] == 2.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_timing_never_becomes_component_evidence(bad: float) -> None:
    work = {
        "endpoint_seconds": 1.0,
        "timeline": {
            "endpoint_seconds": 1.0,
            "exclusive_wall_seconds": {"python_packing": bad},
        },
    }
    with pytest.raises(ValueError):
        normalize_force_work(work)


def test_missing_profiled_values_are_null_not_nan_or_zero() -> None:
    record = normalize_force_work(
        {
            "endpoint_seconds": 1.0,
            "timeline": {
                "endpoint_seconds": 1.0,
                "exclusive_wall_seconds": {"final_reduction": 0.1},
            },
        }
    )
    assert record["profiled_ms"]["h2d_d2h"] is None
    assert record["wall_seconds"]["scf_full_range_k"] is None
    assert math.isfinite(record["unattributed_wall_seconds"])


def test_stationary_split_geometry_phases_preserve_component_attribution() -> None:
    work = {
        "endpoint_seconds": 1.0,
        "timeline": {
            "endpoint_seconds": 1.0,
            "exclusive_wall_seconds": {
                "xc_geometry_enqueue": 0.12,
                "xc_geometry_drain": 0.08,
            },
        },
    }
    record = normalize_force_work(work)
    assert record["wall_seconds"]["semilocal_geometry_response"] == pytest.approx(0.20)
    assert "semilocal_geometry_response" in record["coverage"]["wall_seconds"]

