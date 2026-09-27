"""Host-only checks for the versioned precision-work decoder."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

from vibeqc import _native
from vibeqc._precision_work import query_precision_work


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

    return SimpleNamespace(vibeqc_calculation_get_precision_work=work)


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

    library = SimpleNamespace(vibeqc_batch_get_precision_work=work)
    precision = query_precision_work(library, ctypes.c_void_p(), index=7)
    assert precision is not None and precision["detail_version"] == 1
    assert observed == [7, 7]
