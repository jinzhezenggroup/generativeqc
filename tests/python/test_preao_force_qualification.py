"""Host-only gates for complete force work and unmodified default dispatch."""

import copy
import gzip
import json
import typing
from pathlib import Path

import pytest

from benchmarks.verify_preao_force import (
    load_report,
    verify_census,
    verify_public_default,
)


def force_work(*, selected: bool) -> dict[str, typing.Any]:
    """Two uniform tiles: one identity span and optionally one selected span."""
    visits, squares = (32, 272) if selected else (40, 400)
    grid = {
        "evaluation_tiles": 2,
        "feature_passes": 2,
        "evaluation_points": 4,
        "active_point_ao": visits,
        "deriv2_point_ao": visits,
        "ao_jet_values": 10 * visits,
        "evaluation_passes": 2,
        "deriv2_passes": 2,
        "deriv0_passes": 0,
        "deriv1_passes": 0,
        "deriv3_passes": 0,
        "dense_point_ao": 40,
        "projection_matrices": 8,
        "projection_passes": 2,
        "projection_fma_pairs": 4 * squares,
        "projection_output_values": 4 * visits,
        "identical_spin_copy_bytes": 32 * visits,
        "density_gather_passes": int(selected),
        "density_gather_elements": 36 if selected else 0,
        "ao_map_h2d_bytes": 0,
        "discovery_point_ao": 0,
        "discovery_ao_jet_values": 0,
    }
    return {
        "grid_metrics": {"ao_grid_work": grid},
        "resident_ao_selection": {
            "derivative_order": 2,
            "full_ao_capacity": 10,
            "work": None,
        },
        "grid_work_plan": {"grid_points": 4, "tile_count": 2, "tile_points": 2},
    }


def native_map_work(*, selected: bool) -> dict[str, typing.Any]:
    """A declined map changes scheduling but retains the full AO inventory."""
    work = force_work(selected=selected)
    work["resident_ao_selection"]["work"] = {
        "native_csr_ready": selected,
        "occupancy_declined": not selected,
        "tile_count": 2,
        "point_ao_visits": 32 if selected else 40,
        "point_ao_square_sum": 272 if selected else 400,
        "empty_tile_count": 0,
        "dense_point_ao_square_sum": 400,
        "dense_occupancy_tiles": 0 if selected else 2,
        "dense_budget_tiles": 0,
        "dense_capability_tiles": 0,
        "dense_allocation_tiles": 0,
        "host_ao_label_lookups": 0,
        "ao_map_h2d_bytes": 0,
        "numeric_peak_bound_bytes": 128,
        "budget_bytes": 1024,
        "discovery_seconds": 0,
        "discovery_d2h_bytes": 0,
        "discovery_offsets_h2d_bytes": 0,
        "discoveries": 0,
    }
    return work


@pytest.mark.parametrize("selected", [False, True])
def test_native_census_checks_selected_and_dense_occupancy_domains(
    selected: bool,
) -> None:
    work = native_map_work(selected=selected)
    assert verify_census({"phase": "warm"}, work, "pre-ao-envelope-native-csr")


def test_dense_census_checks_identity_density_alias() -> None:
    assert verify_census({"phase": "warm"}, force_work(selected=False), "dense")


@pytest.mark.parametrize(
    "field,value",
    [
        ("ao_jet_values", 319),
        ("projection_fma_pairs", 1087),
        ("density_gather_elements", 136),
        ("ao_map_h2d_bytes", 8),
        ("discovery_point_ao", 1),
        ("deriv1_passes", 1),
    ],
)
def test_census_rejects_forged_or_dense_discovery_work(field: str, value: int) -> None:
    work = native_map_work(selected=True)
    work["grid_metrics"]["ao_grid_work"][field] = value
    with pytest.raises(AssertionError):
        verify_census({"phase": "warm"}, work, "pre-ao-envelope-native-csr")


def test_warm_census_rejects_repeated_discovery_transfer() -> None:
    work = native_map_work(selected=True)
    work["resident_ao_selection"]["work"]["discovery_d2h_bytes"] = 8
    with pytest.raises(AssertionError):
        verify_census({"phase": "warm"}, work, "pre-ao-envelope-native-csr")


def test_public_default_receipt_cannot_be_an_experimental_override() -> None:
    report = {
        "p0c_execution_mode": "auto",
        "p0c_force_work": [
            {
                "force_active_ao_policy": {
                    "decision": "selected",
                    "producer": "pre-ao-envelope-native-csr",
                    "actual_mode": "selected",
                }
            }
        ],
    }
    verify_public_default(report)
    overridden = copy.deepcopy(report)
    overridden["p0c_execution_mode"] = "pre-ao-envelope-native-csr"
    with pytest.raises(AssertionError):
        verify_public_default(overridden)
    overridden = copy.deepcopy(report)
    overridden["p0c_force_work"][0]["force_active_ao_policy"]["decision"] = "dense"
    with pytest.raises(AssertionError):
        verify_public_default(overridden)


def test_published_gzip_report_keeps_the_original_values(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = {"status": "measured", "seconds": [0.123456789, 123.456789]}
    path.with_suffix(".json.gz").write_bytes(
        gzip.compress(json.dumps(report).encode(), mtime=0)
    )
    assert load_report(path) == report
