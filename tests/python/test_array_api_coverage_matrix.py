"""Machine-readable operation coverage must not imply native Array API support."""

from generativeqc_compiler.array_api.capabilities import (
    SUPPORTED_FUNCTIONS,
    capabilities,
)


def test_coverage_matrix_matches_the_public_preview_inventory() -> None:
    report = capabilities()
    matrix = report["operation_matrix"]
    assert set(matrix) == SUPPORTED_FUNCTIONS == set(report["functions"])
    assert report["target_standard_for_gap_analysis"] == "2025.12"
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False


def test_coverage_distinguishes_boolean_data_and_reference_execution() -> None:
    matrix = capabilities()["operation_matrix"]
    assert matrix["equal"]["dtypes"] == "float32|float64 -> bool"
    assert matrix["equal"]["ad"] == "non-differentiable"
    assert matrix["sum"]["audited_keywords"]["keepdims"] == "generic-arrays-only"
    assert matrix["astype"]["audited_keywords"]["device"] == "None-only"
    assert "where" not in matrix
    for operation in matrix.values():
        assert operation["eager"] == "numpy-reference"
        assert operation["capture"] == "tensorir-reference"
        assert operation["native_cpu"] is False
        assert operation["native_cuda"] is False


def test_coverage_report_is_detached_from_internal_keyword_metadata() -> None:
    first = capabilities()["operation_matrix"]
    first["sum"]["audited_keywords"]["keepdims"] = "changed-by-caller"
    second = capabilities()["operation_matrix"]
    assert second["sum"]["audited_keywords"]["keepdims"] == "generic-arrays-only"
