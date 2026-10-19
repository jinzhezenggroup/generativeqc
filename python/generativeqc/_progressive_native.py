"""Synchronous transport for the native bounded progressive-HF schedule.

Only an already selected target library is consulted. Operations retain their
own scientific owners; callback exceptions are caught before the C boundary,
then re-raised with their original traceback after native-directed cleanup.
"""

from __future__ import annotations

import ctypes
import sys
import typing
from enum import IntEnum

from . import _native


class HFEvent(IntEnum):
    """Version-one operations; callback results are operation-specific flags."""

    PREPARE_SOURCE = 0
    EXECUTE_SOURCE = 1
    PREPARE_TARGET = 2
    PROJECT = 3
    EXECUTE_TARGET = 4
    RETAIN_TARGET = 5
    CLOSE_TARGET = 6
    CLOSE_SOURCE = 7
    AUDIT_TARGET = 8


def _portable_run(invoke: typing.Callable[[HFEvent], int]) -> None:
    """Keep the old-library schedule explicit, without discovering a library."""
    invoke(HFEvent.PREPARE_SOURCE)
    try:
        source = invoke(HFEvent.EXECUTE_SOURCE)
        invoke(HFEvent.PREPARE_TARGET)
        try:
            if source:
                invoke(HFEvent.PROJECT)
            target = invoke(HFEvent.EXECUTE_TARGET)
            retained = invoke(HFEvent.RETAIN_TARGET) if target == 3 else 0
        finally:
            invoke(HFEvent.CLOSE_TARGET)
    finally:
        invoke(HFEvent.CLOSE_SOURCE)
    if retained:
        invoke(HFEvent.AUDIT_TARGET)


def run_hf_schedule(library: typing.Any, operation: typing.Callable) -> str:
    """Execute native policy or missing-symbol compatibility, never on refusal.

    ``operation(event, exception_info)`` returns flags and must close its owner
    when asked, including when an earlier operation failed. Cleanup receives the
    first exception's type, value and traceback, as a context manager would.
    """
    pending = None

    def invoke(event: HFEvent) -> int:
        nonlocal pending
        try:
            return operation(event, pending or (None, None, None))
        except BaseException:
            if pending is None:
                pending = sys.exc_info()
            raise

    query = getattr(library, "generativeqc_progressive_hf_run_v1", None)
    if query is None:
        try:
            _portable_run(invoke)
        except BaseException:
            if pending is not None:
                raise pending[1].with_traceback(pending[2]) from None
            raise
        return "portable-compatibility"

    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    )

    @callback_type
    def callback(_state: int | None, event: int, output: typing.Any) -> int:
        try:
            output[0] = invoke(HFEvent(event))
            return 0
        except BaseException:  # noqa: BLE001 - Exceptions must never unwind through libffi.
            return 1

    query.argtypes = [
        callback_type,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    receipt = (ctypes.c_uint64 * 2)()
    status = query(callback, None, receipt, len(receipt))
    if pending is not None:
        raise pending[1].with_traceback(pending[2])
    _native.check(library, status)
    mandatory = sum(
        1 << event
        for event in (
            HFEvent.PREPARE_SOURCE,
            HFEvent.EXECUTE_SOURCE,
            HFEvent.PREPARE_TARGET,
            HFEvent.EXECUTE_TARGET,
            HFEvent.CLOSE_TARGET,
            HFEvent.CLOSE_SOURCE,
        )
    )
    completed, flags = receipt
    if (
        completed & mandatory != mandatory
        or completed >> len(HFEvent)
        or flags & ~15
        or bool(completed & (1 << HFEvent.PROJECT)) != bool(flags & 1)
        or (bool(flags & 4) and not flags & 2)
        or bool(completed & (1 << HFEvent.RETAIN_TARGET)) != bool(flags & 4)
        or (bool(flags & 8) and not flags & 4)
        or bool(completed & (1 << HFEvent.AUDIT_TARGET)) != bool(flags & 8)
    ):
        raise RuntimeError(
            "native progressive HF controller returned an inconsistent receipt"
        )
    return "native-v1"
