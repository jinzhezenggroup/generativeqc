"""Tests for the unified DFT force component evidence aggregator."""

from __future__ import annotations

import pytest

from tools.benchmark_dft_force_components import _coverage, extract_records


@pytest.mark.parametrize(
    "supported,enabled,batches",
    [(False, True, 4), (True, False, 4), (True, True, 0), (True, True, 4)],
)
def test_becke_phase_observations_are_not_clean_wall_components(
    supported: bool, enabled: bool, batches: int
) -> None:
    from benchmarks.dft_force_components import BECKE_PHASES, normalize_force_work

    work = {
        "endpoint_seconds": 1.0,
        "becke_primitive_requested": 1,
        "becke_primitive_selected": 0,
        "becke_reverse_pair_visits": 64,
        "becke_reverse_pair_panel_write_bytes": 2048,
        "becke_profile_batches": batches,
        "becke_phase_profile_supported": supported,
        "becke_phase_profile_enabled": enabled,
        "becke_phase_ms": dict.fromkeys(BECKE_PHASES, 2000.0),
        "becke_work_counter_semantics": "launched dense domains",
        "becke_traffic_model": "logical values, not hardware traffic",
    }
    result = normalize_force_work(work)
    owner = result["becke_owners"]["stationary"]
    assert owner["selection"]["becke_primitive_requested"] == 1
    assert owner["selection"]["becke_primitive_selected"] == 0
    assert owner["work_counters"]["becke_reverse_pair_visits"] == 64
    assert owner["logical_traffic_bytes"] == {
        "becke_reverse_pair_panel_write_bytes": 2048
    }
    assert owner["profiled_ms"] == dict.fromkeys(
        BECKE_PHASES, 2000.0 if supported and enabled and batches else None
    )
    assert owner["profile_intrusive"] == bool(supported and enabled and batches)
    assert result["attributed_wall_seconds"] == 0.0
    assert result["endpoint_seconds"] == 1.0


def test_composite_becke_owners_remain_separate_and_old_abi_is_unmeasured() -> None:
    from benchmarks.dft_force_components import BECKE_PHASES, normalize_force_work

    work = {
        "execution": "cuda-complete-composite",
        "component_seconds": {},
        "stationary_source_work": {
            "semilocal": {
                "becke_primitive_selected": 1,
                "becke_reverse_pair_visits": 16,
            },
            "nonlocal": {
                "becke_primitive_selected": 0,
                "becke_reverse_pair_visits": 32,
            },
        },
    }
    result = normalize_force_work(work)
    owners = result["becke_owners"]
    assert set(owners) == {"semilocal", "nonlocal"}
    for name, selected, visits in (("semilocal", 1, 16), ("nonlocal", 0, 32)):
        assert owners[name]["selection"]["becke_primitive_selected"] == selected
        assert owners[name]["work_counters"]["becke_reverse_pair_visits"] == visits
        assert owners[name]["profiled_ms"] == dict.fromkeys(BECKE_PHASES)


def test_extract_stationary_record_uses_normalized_component_schema() -> None:
    payload = {
        "schema": "generativeqc.stationary-cuda-force-benchmark.v1",
        "records": [
            {
                "status": "ok",
                "system": "water",
                "method": "pbe-rks",
                "scenario": "same_state_warm",
                "repeat": 0,
                "timeline": {
                    "exclusive_wall_seconds": {"state_export": 0.0},
                },
                "work": {
                    "endpoint_seconds": 1.0,
                    "timeline": {
                        "endpoint_seconds": 1.0,
                        "exclusive_wall_seconds": {
                            "python_packing": 0.4,
                            "primitive_derivative_reduction_sync": 0.3,
                            "xc_geometry_and_sync": 0.2,
                            "final_reduction": 0.1,
                        },
                    },
                },
            },
            {
                "status": "unsupported",
                "system": "water",
                "method": "other-rks",
            },
        ],
    }

    rows = extract_records(payload)

    assert len(rows) == 2
    assert rows[0]["metadata"]["method"] == "pbe-rks"
    components = rows[0]["components"]
    assert components["schema"] == "generativeqc.dft-force-components.v1"
    assert components["wall_seconds"]["host_packing"] == 0.4
    assert components["endpoint_seconds"] == 1.0
    assert rows[1]["status"] == "unsupported"
    assert rows[1]["metadata"]["method"] == "other-rks"


def test_extract_wb97mv_uses_latest_cumulative_force_work() -> None:
    first = {
        "execution": "cuda-complete-wb97mv",
        "endpoint_seconds": 9.0,
        "component_seconds": {
            "prepare": 1.0,
            "integral_derivatives": 4.0,
            "semilocal_geometry_and_features": 2.0,
            "vv10_pairs": 0.5,
            "nonlocal_geometry": 0.5,
            "reduction_and_validation": 1.0,
        },
    }
    second = {
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
    }
    payload = {
        "schema": "generativeqc.readme-wb97mv.v1",
        "method": "WB97M-V/RKS",
        "atoms": 3,
        "native_force_work": [
            {"index": 0, "work": first},
            {"index": 0, "work": second},
        ],
    }

    rows = extract_records(payload)

    assert len(rows) == 1
    assert rows[0]["metadata"]["atoms"] == 3
    assert rows[0]["components"]["endpoint_seconds"] == 5.0
    assert rows[0]["components"]["wall_seconds"]["vv10_rvv10"] == 0.9
    assert rows[0]["comparison"]["boundary"] == "scf_energy_plus_force"
    assert (
        "does not expose a compatible"
        in rows[0]["comparison"]["reference_component_attribution"]
    )


def test_extract_cross_functional_matrix_keeps_scf_profile_separate() -> None:
    force_components = {
        "schema": "generativeqc.dft-force-components.v1",
        "source_route": "stationary-exclusive-wall",
        "wall_seconds": {"stationary_integral_derivatives": 0.4},
        "profiled_ms": {},
        "coverage": {
            "wall_seconds": ["stationary_integral_derivatives"],
            "profiled_ms": [],
            "missing_wall_seconds": ["scf_fock_j"],
        },
    }
    scf_profile = {
        "schema": "generativeqc.dft-scf-components.v1",
        "profiled_ms": {"scf_fock_j": 2.0},
        "expected_components": ["scf_fock_j", "semilocal_ao_grid_xc"],
        "missing_expected_components": ["semilocal_ao_grid_xc"],
    }
    payload = {
        "schema": "generativeqc.dft-force-matrix.v1",
        "records": [
            {
                "status": "measured",
                "method": "pbe-rks",
                "selector": "pbe-rks",
                "system": "water-3",
                "atoms": 3,
                "basis": "def2-svp",
                "density_fitting": "none",
                "method_identity": "method-identity",
                "ks_options_identity": "ks-identity",
                "library_sha256": "library-sha",
                "cold": {
                    "scenario": "cold",
                    "force_components": force_components,
                },
                "warm": [],
                "scf_profile": {
                    "status": "measured",
                    "profile": scf_profile,
                    "trace": {"path": "trace.jsonl", "sha256": "abc"},
                },
            }
        ],
    }

    rows = extract_records(payload)

    assert len(rows) == 2
    assert rows[0]["metadata"]["scenario"] == "cold"
    assert rows[0]["metadata"]["method_identity"] == "method-identity"
    assert rows[0]["metadata"]["ks_options_identity"] == "ks-identity"
    assert rows[0]["metadata"]["library_sha256"] == "library-sha"
    assert rows[0]["components"]["schema"] == "generativeqc.dft-force-components.v1"
    assert rows[1]["metadata"]["scenario"] == "diagnostic_scf_profile"
    assert rows[1]["scf_profile"]["profiled_ms"]["scf_fock_j"] == 2.0
    assert "components" not in rows[1]


def test_extract_matrix_retains_case_and_force_negative_evidence() -> None:
    payload = {
        "schema": "generativeqc.dft-force-matrix.v1",
        "records": [
            {
                "status": "unsupported",
                "method": "cam-b3lyp-rks",
                "system": "water-3",
                "basis": "def2-svp",
                "density_fitting": "none",
                "error_type": "NotImplementedError",
                "error": "generic RSH force owner missing",
            },
            {
                "status": "measured",
                "method": "pbe0-rks",
                "selector": "pbe0-rks",
                "system": "water-3",
                "atoms": 3,
                "basis": "def2-svp",
                "density_fitting": "none",
                "cold": {
                    "scenario": "cold",
                    "force_status": "unsupported",
                    "force_error": "force route unavailable",
                },
                "warm": [],
                "scf_profile": {
                    "status": "unavailable",
                    "reason": "selected SCF provider emitted no trace roots",
                },
            },
        ],
    }

    rows = extract_records(payload)

    assert len(rows) == 3
    assert rows[0]["status"] == "unsupported"
    assert rows[0]["metadata"]["scenario"] == "case"
    assert rows[1]["status"] == "unsupported"
    assert rows[1]["metadata"]["scenario"] == "cold"
    assert rows[2]["status"] == "unavailable"
    assert rows[2]["metadata"]["scenario"] == "diagnostic_scf_profile"


def test_report_coverage_counts_negative_outcomes() -> None:
    coverage = _coverage(
        [
            {"status": "measured"},
            {"status": "measured"},
            {"status": "unsupported"},
            {"status": "failed"},
            {"status": "unavailable"},
        ]
    )

    assert coverage["outcomes"] == {
        "failed": 1,
        "measured": 2,
        "unavailable": 1,
        "unsupported": 1,
    }


def test_extract_readme_dft_endpoint_keeps_reference_boundary_coarse() -> None:
    payload = {
        "schema": "generativeqc.readme-endpoint.v1",
        "status": "measured",
        "method": "pbe0-rks",
        "atoms": 3,
        "aos": 24,
        "basis": "def2-SVP spherical",
        "mode": "direct",
        "endpoint": "SCF energy",
        "native_prepare_seconds": 0.25,
        "native_build": {"library_sha256": "native-sha"},
        "environment": {"packages": {"gpu4pyscf": "2.0"}},
        "native_cold": {"seconds": 1.5},
        "reference_cold": {"seconds": 1.1},
        "priming": {
            "native": {"seconds": 0.8},
            "reference": {"seconds": 0.7},
        },
        "native_samples": [{"seconds": 0.6}, {"seconds": 0.5}],
        "reference_samples": [{"seconds": 0.4}, {"seconds": 0.45}],
        "accuracy": {"maximum_energy_error_hartree": 1.0e-10},
    }

    rows = extract_records(payload)

    assert len(rows) == 1
    row = rows[0]
    assert row["metadata"]["method"] == "pbe0-rks"
    assert row["metadata"]["mode"] == "direct"
    assert row["comparison"]["boundary"] == "scf_energy"
    assert row["comparison"]["native"]["prepare_seconds"] == 0.25
    assert row["comparison"]["native_build"]["library_sha256"] == "native-sha"
    assert row["comparison"]["environment"]["packages"]["gpu4pyscf"] == "2.0"
    assert row["comparison"]["native"]["warm_seconds"] == [0.6, 0.5]
    assert row["comparison"]["reference"]["warm_seconds"] == [0.4, 0.45]
    assert "components" not in row


def test_report_coverage_tracks_external_methods_without_fake_components() -> None:
    coverage = _coverage(
        [
            {
                "status": "measured",
                "metadata": {"method": "pbe-rks"},
                "comparison": {"boundary": "scf_energy"},
            },
            {
                "status": "measured",
                "metadata": {"method": "WB97M-V/RKS"},
                "comparison": {"boundary": "scf_energy_plus_force"},
            },
        ]
    )

    assert coverage["external_comparison_records"] == 2
    assert coverage["external_comparison_methods"] == ["WB97M-V/RKS", "pbe-rks"]
    assert coverage["external_comparison_boundaries"] == [
        "scf_energy",
        "scf_energy_plus_force",
    ]
    assert coverage["wall_components_observed"] == []


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_external_comparison_rejects_invalid_duration(seconds: float) -> None:
    payload = {
        "schema": "generativeqc.readme-endpoint.v1",
        "status": "measured",
        "method": "pbe-rks",
        "endpoint": "SCF energy",
        "native_cold": {"seconds": seconds},
    }

    with pytest.raises(ValueError, match="finite and nonnegative"):
        extract_records(payload)


def test_extract_matrix_retains_fixed_final_state_and_fixed_density_gap() -> None:
    components = {
        "schema": "generativeqc.dft-force-components.v1",
        "source_route": "stationary-exclusive-wall",
        "wall_seconds": {},
        "profiled_ms": {},
        "coverage": {
            "wall_seconds": [],
            "profiled_ms": [],
            "missing_wall_seconds": [],
        },
        "work_counts": {
            "generated": {"coulomb_public_ao_quartets": 16},
            "screened": {},
            "compacted": {},
            "executed": {"semilocal_geometry_points": 40},
            "capacity": {"ordered_quartets": 16},
            "observed": {},
        },
    }
    payload = {
        "schema": "generativeqc.dft-force-matrix.v1",
        "records": [
            {
                "status": "measured",
                "method": "pbe-rks",
                "system": "water-3",
                "fixed_final_state": [
                    {
                        "scenario": "fixed_final_state_0",
                        "measurement_boundary": "fixed_final_state_force",
                        "scf_replayed": False,
                        "force_status": "ok",
                        "force_components": components,
                    }
                ],
                "fixed_density_scf_profile": {
                    "status": "measured",
                    "measurement_boundary": "fixed_density_scf_components",
                    "fixed_density": True,
                    "scf_replayed": False,
                    "expected_components": ["scf_fock_j", "semilocal_ao_grid_xc"],
                    "profile": {
                        "profiled_ms": {
                            "scf_fock_j": 0.7,
                            "semilocal_ao_grid_xc": 1.2,
                        },
                        "missing_expected_components": [],
                    },
                },
            }
        ],
    }

    rows = extract_records(payload)
    fixed = next(
        row for row in rows if row["metadata"]["scenario"] == "fixed_final_state_0"
    )
    missing = next(
        row
        for row in rows
        if row["metadata"]["scenario"] == "diagnostic_fixed_density_scf_profile"
    )
    coverage = _coverage(rows)

    assert fixed["metadata"]["measurement_boundary"] == "fixed_final_state_force"
    assert fixed["metadata"]["scf_replayed"] is False
    assert missing["status"] == "measured"
    assert missing["expected_components"] == ["scf_fock_j", "semilocal_ao_grid_xc"]
    assert missing["scf_profile"]["profiled_ms"]["scf_fock_j"] == 0.7
    assert coverage["fixed_final_state_records"] == 1
    assert coverage["work_count_stages_observed"] == ["executed", "generated"]
    assert coverage["work_capacity_metrics_observed"] == ["ordered_quartets"]
    assert coverage["fixed_density_scf_expected_components_missing"] == []
    assert "scf_fock_j" in coverage["scf_profiled_components_observed"]
    assert "semilocal_ao_grid_xc" in coverage["scf_profiled_components_observed"]


def test_stationary_normalizer_retains_complete_grid_plan_and_native_route() -> None:
    from benchmarks.dft_force_components import normalize_force_work

    plan = {
        "schema": "generativeqc.stationary-grid-work.v1",
        "grid_points": 2359296,
        "grid_pair_visits": 21516784080,
        "chunk_count": 220,
        "chunk_pair_visits": 98058240,
    }
    result = normalize_force_work(
        {
            "timeline": {"exclusive_wall_seconds": {}},
            "grid_work_plan": plan,
            "native_integrals_required": True,
            "stationary_integral_derivative_route": "prepared-native-complete",
        }
    )
    assert result["grid_work_plan"] == plan
    assert result["native_integrals_required"] is True
    assert result["stationary_integral_derivative_route"] == "prepared-native-complete"
    assert result["work_counts"]["executed"] == {}
