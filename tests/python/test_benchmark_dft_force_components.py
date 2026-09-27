"""Tests for the unified DFT force component evidence aggregator."""

from __future__ import annotations

from tools.benchmark_dft_force_components import _coverage, extract_records


def test_extract_stationary_record_uses_normalized_component_schema() -> None:
    payload = {
        "schema": "vibeqc.stationary-cuda-force-benchmark.v1",
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
    assert components["schema"] == "vibeqc.dft-force-components.v1"
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
        "schema": "vibeqc.readme-wb97mv.v1",
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


def test_extract_cross_functional_matrix_keeps_scf_profile_separate() -> None:
    force_components = {
        "schema": "vibeqc.dft-force-components.v1",
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
        "schema": "vibeqc.dft-scf-components.v1",
        "profiled_ms": {"scf_fock_j": 2.0},
        "expected_components": ["scf_fock_j", "semilocal_ao_grid_xc"],
        "missing_expected_components": ["semilocal_ao_grid_xc"],
    }
    payload = {
        "schema": "vibeqc.dft-force-matrix.v1",
        "records": [
            {
                "status": "measured",
                "method": "pbe-rks",
                "selector": "pbe-rks",
                "system": "water-3",
                "atoms": 3,
                "basis": "def2-svp",
                "density_fitting": "none",
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
    assert rows[0]["components"]["schema"] == "vibeqc.dft-force-components.v1"
    assert rows[1]["metadata"]["scenario"] == "diagnostic_scf_profile"
    assert rows[1]["scf_profile"]["profiled_ms"]["scf_fock_j"] == 2.0
    assert "components" not in rows[1]


def test_extract_matrix_retains_case_and_force_negative_evidence() -> None:
    payload = {
        "schema": "vibeqc.dft-force-matrix.v1",
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
