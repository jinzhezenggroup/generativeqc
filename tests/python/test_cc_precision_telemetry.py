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
    result = read_cc_performance_result(library, None, index=index)
    assert result is not None
    assert result.ccsd_fp32_requested is None
    assert result.ccsd_fp32_executed is None
    assert result.ccsd_fp32_gemm_calls is None


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
