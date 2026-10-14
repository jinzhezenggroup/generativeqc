"""Host-only checks for the versioned precision-work decoder."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc._precision_work import (
    merge_precision_work,
    query_gradient_precision,
    query_precision_work,
)


def test_gradient_decoder_reports_attempts_and_retry_without_fabricating_counts() -> (
    None
):
    observed = []

    def getter(handle: object, index: int, output: Any, count: int) -> int:
        observed.append(index)
        assert count == 5
        output[:] = [1, 1, 500, 400, 1]
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(generativeqc_batch_gradient_precision_v1=getter)
    record = query_gradient_precision(library, None, 3)
    assert observed == [3]
    assert record["returned_mode"] == "fp64"
    assert record["fp64_gradient_retries"] == 1
    assert record["attempted_fp32_geometry_points"] == 500
    assert record["attempted_fp32_grid_points"] == 400
    assert record["weighted_task_retries"] is None
    assert not record["weighted_task_counters_collected"]
    assert query_gradient_precision(SimpleNamespace(), None) is None


@pytest.mark.parametrize(
    "qualification,expected",
    ((1, "pbe0"), (2, "pbe"), (3, "b3lyp"), (4, "r2scan"), (6, "m06-2x"), (99, None)),
)
def test_gradient_v2_does_not_infer_a_model_from_scf(
    qualification: int, expected: str | None
) -> None:
    def getter(handle: object, output: Any, count: int) -> int:
        assert count == 6
        output[:] = [1, 1, 700, 600, 0, qualification]
        return _native.STATUS_SUCCESS

    record = query_gradient_precision(
        SimpleNamespace(generativeqc_calculation_gradient_precision_v2=getter), None
    )
    assert record["schema_version"] == 2
    assert record["qualification"] == (
        f"{expected}-rks-budget-fp32-v1" if expected else "unknown"
    )
    assert record["returned_mode"] == "mixed"
    assert record["attempted_fp32_geometry_points"] == 700
    assert ("xc_cartesian_conversion" in record["fp64_components"]) == (
        qualification in (3, 4, 6)
    )


@pytest.mark.parametrize("collected,retries", ((False, 0), (True, 2), (True, 12)))
def test_gradient_v3_distinguishes_per_item_counts_from_unavailable(
    collected: bool, retries: int
) -> None:
    def getter(handle: object, index: int, output: Any, count: int) -> int:
        assert index == 2 and count == 10
        output[:] = [1, 1, 0, 0, 0, 5, int(collected), 12, 96, retries]
        return _native.STATUS_SUCCESS

    record = query_gradient_precision(
        SimpleNamespace(generativeqc_batch_gradient_precision_v3=getter), None, 2
    )
    assert record["qualification"] == "rhf-budget-fp32-v1"
    assert record["weighted_task_counters_collected"] is collected
    assert record["weighted_task_retries"] == (retries if collected else None)
    assert record["attempted_fp32_weighted_tasks"] == (12 if collected else None)
    assert record["attempted_fp32_primitive_sources"] == (96 if collected else None)
    assert record["returned_mode"] == (
        "fp64" if collected and retries == 12 else "mixed"
    )
    assert record["attempted_fp32_geometry_points"] == 0
    assert record["attempted_fp32_grid_points"] == 0


@pytest.mark.parametrize("retries", (0, 4, 12))
@pytest.mark.parametrize("dispatch", (1, 2))
def test_df_shell_work_is_not_direct_hf_or_an_xc_census(
    retries: int, dispatch: int
) -> None:
    """DF has its own consumer identity and can return a wholly strict replay."""

    def getter(handle: object, output: Any, count: int) -> int:
        assert count == 10
        output[:] = [1, dispatch, 0, 0, 0, 7, 1, 12, 96, retries]
        return _native.STATUS_SUCCESS

    record = query_gradient_precision(
        SimpleNamespace(generativeqc_calculation_gradient_precision_v3=getter), None
    )
    assert record["qualification"] == "rhf-df-shell-budget-fp32-v1"
    assert record["weighted_fp32_orders"] == []
    assert record["weighted_fp32_consumer"] == "df-shell-polynomial-and-rys"
    assert record["returned_mode"] == ("fp64" if retries == 12 else "mixed")
    assert record["attempted_fp32_primitive_sources"] == 96
    assert record["weighted_task_retries"] == retries
    assert "df_metric_and_response_weights" in record["fp64_components"]
    assert "generic_df_derivatives" in record["fp64_components"]
    assert ("df_shell_sp_bra_contractions" in record["fp64_components"]) == (
        dispatch == 2
    )
    assert "weighted_orders_above_5" not in record["fp64_components"]


def _library(*, unknown: bool = False, unavailable: bool = False) -> SimpleNamespace:
    def work(
        handle: object,
        version: int,
        pointer: object,
        events: Any,
        event_capacity: int,
        operators: Any,
        operator_capacity: int,
    ) -> int:
        del handle
        if unavailable:
            return _native.STATUS_PRECISION_UNAVAILABLE
        summary = ctypes.cast(
            pointer, ctypes.POINTER(_native.PrecisionWorkDetail)
        ).contents
        summary.detail_version = version
        summary.owner_id = 9
        summary.returned_solve_epoch = 8
        summary.returned_state_generation = 7
        summary.event_count = summary.operator_count = 1
        summary.complete = summary.operator_inventory_complete = 0
        if events is not None:
            assert event_capacity == 1
            events[0].kind = _native.PRECISION_EVENT_STRICT_FOCK
            events[0].phase = _native.PRECISION_PHASE_SCF
            events[0].sequence = 0
            events[0].iteration = 13
            events[0].owner_id = 9
            events[0].solve_epoch = 8
            events[0].state_generation = 7
        if operators is not None:
            assert operator_capacity == 1
            operators[0].kind = (
                777 if unknown else _native.PRECISION_OPERATOR_EXCHANGE_K
            )
            operators[0].storage_dtype = _native.PRECISION_DTYPE_FP64
            operators[0].compute_dtype = (
                777 if unknown else _native.PRECISION_DTYPE_FP32
            )
            operators[0].accumulation_dtype = _native.PRECISION_DTYPE_FP64
            operators[0].reduction_dtype = _native.PRECISION_DTYPE_FP64
            operators[0].arithmetic_mode = (
                777 if unknown else _native.PRECISION_ARITHMETIC_MIXED
            )
            operators[0].count = 23
        return _native.STATUS_SUCCESS

    return SimpleNamespace(generativeqc_calculation_get_precision_work=work)


def test_decoder_preserves_execution_rows_without_certifying_them() -> None:
    precision = query_precision_work(_library(), ctypes.c_void_p())
    assert precision is not None
    assert precision["detail_version"] == 1
    assert precision["complete"] is False
    assert precision["operator_inventory_complete"] is False
    assert precision["returned_state_identity"] == "cuda-ks:9:8:7"
    assert precision["scf_fock_timeline"] == [
        {
            "kind": "strict_fock",
            "sequence": 0,
            "iteration": 13,
            "state": "cuda-ks:9:8:7",
            "phase": "scf",
            "count": 1,
        }
    ]
    assert precision["operators"] == [
        {
            "name": "exchange_k",
            "storage": "fp64",
            "compute": "fp32",
            "accumulation": "fp64",
            "reduction": "fp64",
            "arithmetic_mode": "mixed",
            "count": 23,
        }
    ]


def test_unknown_values_old_library_and_unavailable_run_fail_closed() -> None:
    unknown = query_precision_work(_library(unknown=True), ctypes.c_void_p())
    assert unknown is not None
    assert unknown["operators"][0]["name"] == "unknown:777"
    assert unknown["operators"][0]["compute"] == "unknown:777"
    assert unknown["operators"][0]["arithmetic_mode"] == "unknown:777"

    old = query_precision_work(SimpleNamespace(), ctypes.c_void_p())
    assert old is not None
    assert old["detail_version"] is None
    assert old["complete"] is False
    assert old["operators"] == old["scf_fock_timeline"] == []

    assert query_precision_work(_library(unavailable=True), ctypes.c_void_p()) is None


def test_batch_decoder_preserves_original_index() -> None:
    observed: list[int] = []

    def work(
        handle: object,
        index: int,
        version: int,
        pointer: object,
        events: Any,
        event_capacity: int,
        operators: Any,
        operator_capacity: int,
    ) -> int:
        del handle, events, event_capacity, operators, operator_capacity
        observed.append(index)
        summary = ctypes.cast(
            pointer, ctypes.POINTER(_native.PrecisionWorkDetail)
        ).contents
        summary.detail_version = version
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(generativeqc_batch_get_precision_work=work)
    precision = query_precision_work(library, ctypes.c_void_p(), index=7)
    assert precision is not None and precision["detail_version"] == 1
    assert observed == [7, 7]


def test_merge_preserves_native_aggregate_and_partial_detail() -> None:
    aggregate = {
        "requested_mode": "auto",
        "mixed_stage_fock_builds": 2,
        "strict_stage_fock_builds": 1,
        "operator_work_counters_valid": False,
    }
    detail = {
        "detail_version": 1,
        "complete": False,
        "operator_inventory_complete": False,
        "returned_state_identity": "cuda-ks:9:8:7",
        "scf_fock_timeline": [{"kind": "mixed_fock"}],
        "operators": [{"name": "coulomb_recurrence", "count": 23}],
    }

    merged = merge_precision_work(aggregate, detail)
    assert merged is not None
    assert merged["requested_mode"] == "auto"
    assert merged["complete"] is False
    assert merged["operators"] == detail["operators"]
    assert merged["native_provenance"] == aggregate
    assert merged["native_provenance"] is not aggregate

    assert merge_precision_work(aggregate, None) is aggregate
    assert merge_precision_work(None, detail) is detail
