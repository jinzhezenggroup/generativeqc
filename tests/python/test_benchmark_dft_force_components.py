"""Tests for the unified DFT force component evidence aggregator."""

from __future__ import annotations

from tools.benchmark_dft_force_components import extract_records


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

    assert len(rows) == 1
    assert rows[0]["metadata"]["method"] == "pbe-rks"
    components = rows[0]["components"]
    assert components["schema"] == "vibeqc.dft-force-components.v1"
    assert components["wall_seconds"]["host_packing"] == 0.4
    assert components["endpoint_seconds"] == 1.0


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
