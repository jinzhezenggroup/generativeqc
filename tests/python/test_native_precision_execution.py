"""Real native precision stage order, bounded retries and exception isolation."""

from __future__ import annotations

import ctypes
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc._precision_execution import (
    PrecisionEvent,
    PrecisionOutcome,
    _Callback,
    _Context,
    _Receipt,
    run_precision_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the production C adapter/core without any electronic stand-in."""
    directory = tmp_path_factory.mktemp("precision-execution")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_precision_admission.cpp"],
        directory / "execution.so",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    selected = ctypes.CDLL(str(path))
    selected.generativeqc_status_message = lambda status: b"precision execution failed"
    return selected


def test_public_sdk_exception_and_execution_contract(
    required_native_cxx: Any, tmp_path: Path
) -> None:
    executable = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_precision_execution.cpp",
            ROOT / "src/api/c_api_precision_admission.cpp",
        ],
        tmp_path / "precision-execution",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


@pytest.mark.parametrize("retry", (False, True))
@pytest.mark.parametrize(
    "qualified,audit,verification",
    (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ),
)
def test_native_bounded_disposition_and_release_order(
    library: Any, retry: bool, qualified: bool, audit: bool, verification: bool
) -> None:
    seen = []

    def operation(event: PrecisionEvent) -> int:
        seen.append(event)
        if event == PrecisionEvent.CANDIDATE and retry:
            raise RuntimeError("eligible candidate failure")
        return int(
            {
                PrecisionEvent.CLASSIFY: qualified,
                PrecisionEvent.AUDIT: audit,
                PrecisionEvent.VERIFY: verification,
            }.get(event, False)
        )

    outcome = run_precision_schedule(library, operation, retry_candidate=retry)
    expected = [PrecisionEvent.CANDIDATE]
    if retry:
        expected.append(PrecisionEvent.FP64_FALLBACK)
    expected.append(PrecisionEvent.CLASSIFY)
    if not qualified:
        expected.extend((PrecisionEvent.RELEASE, PrecisionEvent.AUDIT))
        if not audit:
            expected.append(PrecisionEvent.VERIFY)
    assert seen == expected
    assert outcome == (
        PrecisionOutcome.QUALIFIED
        if qualified
        else PrecisionOutcome.AUDITED
        if audit
        else PrecisionOutcome.VERIFIED
        if verification
        else PrecisionOutcome.REJECTED
    )


@pytest.mark.parametrize("failure", list(PrecisionEvent))
@pytest.mark.parametrize(
    "error_type", (ValueError, MemoryError, KeyboardInterrupt, SystemExit)
)
def test_every_callback_failure_stops_work_and_preserves_exception(
    library: Any, failure: PrecisionEvent, error_type: type[BaseException]
) -> None:
    original = error_type("original callback failure")
    seen = []

    def operation(event: PrecisionEvent) -> int:
        seen.append(event)
        if event == failure:
            raise original
        if (
            event == PrecisionEvent.CANDIDATE
            and failure == PrecisionEvent.FP64_FALLBACK
        ):
            raise RuntimeError("recoverable first attempt")
        return 0

    with pytest.raises(error_type) as raised:
        run_precision_schedule(library, operation, retry_candidate=True)
    assert raised.value is original
    assert seen[-1] == failure
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize(
    "failure",
    (
        PrecisionEvent.CANDIDATE,
        PrecisionEvent.FP64_FALLBACK,
        PrecisionEvent.CLASSIFY,
        PrecisionEvent.RELEASE,
        PrecisionEvent.AUDIT,
        PrecisionEvent.VERIFY,
    ),
)
def test_numerical_failure_is_retried_only_for_the_initial_candidate(
    library: Any, failure: PrecisionEvent
) -> None:
    original = RuntimeError("scientific failure")
    seen = []

    def operation(event: PrecisionEvent) -> int:
        seen.append(event)
        if event == failure:
            raise original
        if (
            event == PrecisionEvent.CANDIDATE
            and failure == PrecisionEvent.FP64_FALLBACK
        ):
            raise NotImplementedError("eligible first attempt")
        return 0

    with pytest.raises(RuntimeError) as raised:
        run_precision_schedule(
            library, operation, retry_candidate=failure != PrecisionEvent.CANDIDATE
        )
    assert raised.value is original
    assert seen[-1] == failure


def test_missing_refused_or_forged_native_controller_never_runs_portable_work() -> None:
    seen = []

    def operation(event: PrecisionEvent) -> int:
        seen.append(event)
        return 1

    for selected in (
        SimpleNamespace(),
        SimpleNamespace(
            generativeqc_precision_execution_run_v1=lambda *args: (
                _native.STATUS_INVALID_ARGUMENT
            ),
            generativeqc_status_message=lambda status: b"native refusal",
        ),
        SimpleNamespace(
            generativeqc_precision_execution_run_v1=lambda *args: _native.STATUS_SUCCESS
        ),
    ):
        with pytest.raises((RuntimeError, NotImplementedError)):
            run_precision_schedule(selected, operation, retry_candidate=True)
    assert seen == []


def test_inconsistent_receipt_cannot_publish_a_failed_audit(library: Any) -> None:
    real_query = library.generativeqc_precision_execution_run_v1

    def forged(context: Any, callback: Any, state: Any, output: Any) -> int:
        status = real_query(context, callback, state, output)
        ctypes.cast(
            output, ctypes.POINTER(_Receipt)
        ).contents.outcome = PrecisionOutcome.VERIFIED
        return status

    selected = SimpleNamespace(generativeqc_precision_execution_run_v1=forged)
    with pytest.raises(RuntimeError, match="inconsistent receipt"):
        run_precision_schedule(selected, lambda event: 0, retry_candidate=False)


def test_duplicate_native_operation_is_refused_before_duplicate_work() -> None:
    seen = []

    def repeated(context: Any, callback: Any, state: Any, output: Any) -> int:
        decision = ctypes.c_uint32()
        callback(None, PrecisionEvent.CANDIDATE, ctypes.byref(decision))
        callback(None, PrecisionEvent.CANDIDATE, ctypes.byref(decision))
        return _native.STATUS_SUCCESS

    def operation(event: PrecisionEvent) -> int:
        seen.append(event)
        return 0

    with pytest.raises(RuntimeError, match="repeated an operation"):
        run_precision_schedule(
            SimpleNamespace(generativeqc_precision_execution_run_v1=repeated),
            operation,
            retry_candidate=False,
        )
    assert seen == [PrecisionEvent.CANDIDATE]


def raw_query(library: Any) -> Any:
    query = library.generativeqc_precision_execution_run_v1
    query.argtypes = [
        ctypes.POINTER(_Context),
        _Callback,
        ctypes.c_void_p,
        ctypes.POINTER(_Receipt),
    ]
    query.restype = ctypes.c_int
    return query


@pytest.mark.parametrize(
    "field,value,status",
    (
        ("struct_size", 0, _native.STATUS_ABI_MISMATCH),
        ("abi_version", _native.ABI_VERSION + 1, _native.STATUS_ABI_MISMATCH),
        ("retry_candidate", 2, _native.STATUS_INVALID_ARGUMENT),
        ("reserved", 1, _native.STATUS_INVALID_ARGUMENT),
    ),
)
def test_malformed_descriptor_is_atomic_and_invokes_no_callback(
    library: Any, field: str, value: int, status: int
) -> None:
    context = _Context(ctypes.sizeof(_Context), _native.ABI_VERSION, 0, 0)
    setattr(context, field, value)
    receipt = _Receipt.from_buffer_copy(b"\xa5" * ctypes.sizeof(_Receipt))
    receipt.struct_size = ctypes.sizeof(_Receipt)
    receipt.abi_version = _native.ABI_VERSION
    original = bytes(receipt)
    seen = []

    @_Callback
    def operation(state: Any, event: int, output: Any) -> int:
        seen.append(event)
        return 0

    assert (
        raw_query(library)(
            ctypes.byref(context), operation, None, ctypes.byref(receipt)
        )
        == status
    )
    assert bytes(receipt) == original
    assert seen == []


def test_overlapping_buffers_copy_inputs_and_bound_outputs(library: Any) -> None:
    storage = (ctypes.c_uint64 * 4)()
    context = _Context.from_buffer(storage)
    receipt = _Receipt.from_buffer(storage)
    context.struct_size = ctypes.sizeof(_Receipt)
    context.abi_version = _native.ABI_VERSION
    context.retry_candidate = 1
    context.reserved = 0
    storage[-1] = 97
    seen = []

    @_Callback
    def operation(state: Any, event: int, output: Any) -> int:
        seen.append(event)
        output[0] = int(event == PrecisionEvent.CLASSIFY)
        if event == PrecisionEvent.CANDIDATE:
            context.retry_candidate = 0
            context.reserved = 9
            return 1
        return 0

    assert (
        raw_query(library)(
            ctypes.byref(context), operation, None, ctypes.byref(receipt)
        )
        == _native.STATUS_SUCCESS
    )
    assert seen == [
        PrecisionEvent.CANDIDATE,
        PrecisionEvent.FP64_FALLBACK,
        PrecisionEvent.CLASSIFY,
    ]
    assert receipt.outcome == PrecisionOutcome.QUALIFIED
    assert storage[-1] == 97


def test_null_inputs_and_invalid_output_headers_do_not_start_work(library: Any) -> None:
    query = raw_query(library)
    context = _Context(ctypes.sizeof(_Context), _native.ABI_VERSION, 0, 0)
    receipt = _Receipt(ctypes.sizeof(_Receipt), _native.ABI_VERSION)
    seen = []

    @_Callback
    def operation(state: Any, event: int, output: Any) -> int:
        seen.append(event)
        return 0

    original = bytes(receipt)
    assert (
        query(None, operation, None, ctypes.byref(receipt))
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert (
        query(ctypes.byref(context), _Callback(), None, ctypes.byref(receipt))
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert (
        query(ctypes.byref(context), operation, None, None)
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert bytes(receipt) == original
    for field, value in (("struct_size", 0), ("abi_version", _native.ABI_VERSION + 1)):
        malformed = _Receipt.from_buffer_copy(original)
        setattr(malformed, field, value)
        before = bytes(malformed)
        assert (
            query(ctypes.byref(context), operation, None, ctypes.byref(malformed))
            == _native.STATUS_ABI_MISMATCH
        )
        assert bytes(malformed) == before
    assert seen == []


@pytest.mark.parametrize(
    "failure",
    (
        PrecisionEvent.CANDIDATE,
        PrecisionEvent.CLASSIFY,
        PrecisionEvent.RELEASE,
        PrecisionEvent.AUDIT,
        PrecisionEvent.VERIFY,
    ),
)
def test_invalid_callback_decisions_fail_without_later_work(
    library: Any, failure: PrecisionEvent
) -> None:
    context = _Context(ctypes.sizeof(_Context), _native.ABI_VERSION, 1, 0)
    receipt = _Receipt(ctypes.sizeof(_Receipt), _native.ABI_VERSION)
    seen = []

    @_Callback
    def operation(state: Any, event: int, output: Any) -> int:
        seen.append(event)
        output[0] = 2 if event == failure else 0
        return 0

    assert (
        raw_query(library)(
            ctypes.byref(context), operation, None, ctypes.byref(receipt)
        )
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert seen[-1] == failure
    assert receipt.outcome == PrecisionOutcome.FAILED
    assert receipt.attempted_events & (1 << failure)
    assert not receipt.completed_events & (1 << failure)


def test_reentrant_parallel_calls_have_no_shared_schedule_state(library: Any) -> None:
    def run(index: int) -> PrecisionOutcome:
        return run_precision_schedule(
            library,
            lambda event: int(event == PrecisionEvent.AUDIT),
            retry_candidate=bool(index % 2),
        )

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert set(workers.map(run, range(64))) == {PrecisionOutcome.AUDITED}
