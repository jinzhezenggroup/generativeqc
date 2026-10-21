"""Compatibility and detached-value checks for optional CC precision telemetry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from generativeqc import _native
from generativeqc._result_translation import read_cc_performance_result


@pytest.mark.parametrize("available", (False, True))
@pytest.mark.parametrize("index", (None, 0))
def test_old_library_and_batch_precision_stay_unknown(
    available: bool, index: int | None
) -> None:
    """Unknown precision must not be fabricated as a measured FP64 zero."""

    def unavailable(*args: object) -> int:
        assert index is None, "a batch must not call the singlepoint getter"
        return _native.STATUS_NOT_IMPLEMENTED

    library = SimpleNamespace(
        generativeqc_calculation_get_cc_performance_diagnostic=lambda *args: 0,
        generativeqc_batch_get_cc_performance_diagnostic=lambda *args: 0,
    )
    if available:
        library.generativeqc_calculation_get_cc_precision_v1 = unavailable
        library.generativeqc_calculation_get_df_response_v1 = unavailable
    result = read_cc_performance_result(library, None, index=index)
    assert result is not None
    assert result.ccsd_fp32_requested is None
    assert result.ccsd_fp32_executed is None
    assert result.ccsd_fp32_gemm_calls is None
    assert result.lambda_fp32_requested is None
    assert result.lambda_fp32_gemm_calls is None
    assert result.lambda_seconds is None


def test_precision_values_are_copied_before_native_owner_cleanup() -> None:
    retained = []

    def precision(owner: object, counters: object, count: int) -> int:
        assert count == 6
        counters[:] = [1, 1, 0, 3, 17, 1234]
        retained.append(counters)
        return 0

    library = SimpleNamespace(
        generativeqc_calculation_get_cc_performance_diagnostic=lambda *args: 0,
        generativeqc_calculation_get_cc_precision_v1=precision,
    )
    result = read_cc_performance_result(library, None)
    retained[0][:] = [0] * 6
    assert result.ccsd_fp32_requested is True
    assert result.ccsd_fp32_executed is True
    assert result.ccsd_fp32_refused is False
    assert result.ccsd_fp32_evaluations == 3
    assert result.ccsd_fp32_gemm_calls == 17
    assert result.ccsd_fp32_cast_elements == 1234


def test_response_values_are_copied_before_native_owner_cleanup() -> None:
    """Separate response spans retain observed zeros and detached phase/work data."""
    retained = []

    def response(
        owner: object, counters: object, count: int, seconds: object, phases: int
    ) -> int:
        assert (count, phases) == (12, 3)
        counters[:] = [1, 1, 0, 8, 3, 12, 7, 100, 2000, 32, 4096, 2048]
        seconds[:] = [2.0, 0.5, 0.25]
        retained.extend((counters, seconds))
        return 0

    library = SimpleNamespace(
        generativeqc_calculation_get_cc_performance_diagnostic=lambda *args: 0,
        generativeqc_calculation_get_df_response_v1=response,
    )
    result = read_cc_performance_result(library, None)
    retained[0][:] = [0] * 12
    retained[1][:] = [0] * 3
    assert result.lambda_fp32_requested is True
    assert result.lambda_fp32_executed is True
    assert result.lambda_fp32_refused is False
    assert (result.lambda_fp32_iterations, result.lambda_fp64_iterations) == (8, 3)
    assert (result.lambda_fp32_actions, result.lambda_fp64_actions) == (12, 7)
    assert (result.lambda_fp32_gemm_calls, result.lambda_fp32_cast_elements) == (
        100,
        2000,
    )
    assert result.lambda_auxiliary_batch_size == 32
    assert (result.lambda_numeric_capacity_bytes, result.lambda_owned_device_bytes) == (
        4096,
        2048,
    )
    assert (
        result.lambda_seconds,
        result.source_response_seconds,
        result.orbital_seconds,
    ) == (2.0, 0.5, 0.25)
