"""Tests for SCF trace attribution in the shared DFT force evidence schema."""

from __future__ import annotations

import pytest

from benchmarks.dft_force_components import (
    expected_scf_components,
    merge_scf_profile,
    normalize_force_work,
    normalize_scf_trace,
)


def _trace(operation: str, gpu_ms: float, identifier: int) -> dict:
    return {
        "schema": "generativeqc.df_trace",
        "version": 1,
        "id": identifier,
        "operation": operation,
        "execution": "stream",
        "valid": True,
        "cuda_error": 0,
        "nvtx": False,
        "systems": 1,
        "system_offset": 0,
        "nbf": 2,
        "naux": 3,
        "source_backed": True,
        "streamed": False,
        "final_synchronization_ms": 0.0,
        "host_completion_ms": 1.0,
        "profiler_event_count": 2,
        "dropped_regions": 0,
        "dropped_tiles": 0,
        "regions": [
            {
                "name": operation,
                "parent": -1,
                "host_ms": 1.0,
                "gpu_ms": gpu_ms,
            }
        ],
        "counters": {},
        "tiles": [],
    }


def test_expected_scf_components_follow_method_graph_operators() -> None:
    assert expected_scf_components(()) == ("scf_fock_j", "semilocal_ao_grid_xc")
    assert expected_scf_components(("full-range",)) == (
        "scf_fock_j",
        "semilocal_ao_grid_xc",
        "scf_full_range_k",
    )
    assert expected_scf_components(("short-range", "long-range")) == (
        "scf_fock_j",
        "semilocal_ao_grid_xc",
        "scf_short_range_k",
        "scf_long_range_k",
    )
    assert expected_scf_components(
        ("short-range", "long-range"), nonlocal_correlation=True
    ) == (
        "scf_fock_j",
        "semilocal_ao_grid_xc",
        "vv10_rvv10",
        "scf_short_range_k",
        "scf_long_range_k",
    )
    with pytest.raises(ValueError, match="unsupported exchange operator"):
        expected_scf_components(("made-up-range",))


def test_scf_trace_maps_separate_j_and_full_range_k_without_wall_inference() -> None:
    profile = normalize_scf_trace(
        [_trace("ri_j", 2.0, 1), _trace("ri_k", 3.0, 2)],
        exchange_operators=("full-range",),
    )

    assert profile["profiled_ms"]["scf_fock_j"] == pytest.approx(2.0)
    assert profile["profiled_ms"]["scf_full_range_k"] == pytest.approx(3.0)
    assert profile["profiled_ms"]["semilocal_ao_grid_xc"] is None
    assert profile["missing_expected_components"] == ["semilocal_ao_grid_xc"]
    assert profile["ambiguous_profiled_ms"] == {}


def test_shared_jk_trace_stays_ambiguous_instead_of_double_counting() -> None:
    profile = normalize_scf_trace(
        [_trace("ri_jk_shared", 5.0, 1)],
        exchange_operators=("full-range",),
    )

    assert profile["profiled_ms"]["scf_fock_j"] is None
    assert profile["profiled_ms"]["scf_full_range_k"] is None
    assert profile["ambiguous_profiled_ms"]["scf_shared_jk"] == pytest.approx(5.0)
    assert set(profile["missing_expected_components"]) == {
        "scf_fock_j",
        "semilocal_ao_grid_xc",
        "scf_full_range_k",
    }


def test_scf_trace_can_be_merged_with_force_telemetry_without_mixing_clocks() -> None:
    force = normalize_force_work(
        {
            "endpoint_seconds": 1.0,
            "timeline": {
                "endpoint_seconds": 1.0,
                "exclusive_wall_seconds": {
                    "primitive_derivative_reduction_sync": 0.4,
                    "xc_geometry_and_sync": 0.3,
                    "final_reduction": 0.1,
                },
            },
        }
    )
    profile = normalize_scf_trace(
        [_trace("ri_j", 2.0, 1)],
        exchange_operators=(),
    )

    merged = merge_scf_profile(force, profile)

    assert merged["wall_seconds"]["stationary_integral_derivatives"] == pytest.approx(
        0.4
    )
    assert merged["profiled_ms"]["scf_fock_j"] == pytest.approx(2.0)
    assert merged["wall_seconds"]["scf_fock_j"] is None
    assert "scf_fock_j" in merged["coverage"]["profiled_ms"]
