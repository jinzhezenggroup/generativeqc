"""Selected-library transport for bounded native precision execution order.

Callbacks adapt existing endpoint owners and preserve original exceptions. They
do not choose stage order or retry a missing/refused native execution protocol.
"""

from __future__ import annotations

import ctypes
import sys
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from . import _native

if TYPE_CHECKING:
    from collections.abc import Callable


class PrecisionEvent(IntEnum):
    """Synchronous operation identifiers shared with the public native ABI."""

    CANDIDATE = 0
    FP64_FALLBACK = 1
    CLASSIFY = 2
    RELEASE = 3
    AUDIT = 4
    VERIFY = 5


class PrecisionOutcome(IntEnum):
    """Completed execution disposition; REJECTED is not a successful audit."""

    QUALIFIED = 0
    AUDITED = 1
    VERIFIED = 2
    REJECTED = 3
    FAILED = 4


class _Context(ctypes.Structure):
    """Only numerical retry eligibility is supplied by the existing caller."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("struct_size", "abi_version", "retry_candidate", "reserved")
    ]


class _Receipt(ctypes.Structure):
    """Progress masks distinguish failed attempts from completed operations."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "policy_version",
            "attempted_events",
            "completed_events",
            "outcome",
        )
    ]


_Callback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
)


def require_precision_schedule(library: Any) -> Any:
    """Reject an incompatible selected owner before any fleet or endpoint work."""
    query = getattr(library, "generativeqc_precision_execution_run_v1", None)
    if query is None:
        raise NotImplementedError(
            "native library lacks required precision execution controller; "
            "rebuild or install a current library"
        )
    return query


def run_precision_schedule(
    library: Any,
    operation: Callable[[PrecisionEvent], int],
    *,
    retry_candidate: bool,
) -> PrecisionOutcome:
    """Require native control before work, retaining bounded exception semantics.

    Only RuntimeError/NotImplementedError from the first candidate may request
    its incumbent FP64 retry. A selected retry consumes that original exception;
    later failures keep their own traceback. Cancellation, allocation, input and
    audit errors are never translated into numerical retry eligibility.
    """
    query = require_precision_schedule(library)
    pending = None
    retry_error = None
    attempted = 0
    completed = 0
    decisions = {}

    @_Callback
    def callback(state: Any, event_value: int, output: Any) -> int:
        nonlocal pending, retry_error, attempted, completed
        event = None
        try:
            event = PrecisionEvent(event_value)
            if attempted & (1 << event):
                raise RuntimeError("native precision controller repeated an operation")
            attempted |= 1 << event
            if event == PrecisionEvent.FP64_FALLBACK:
                retry_error = None
            decision = operation(event)
            if not isinstance(decision, int) or decision < 0 or decision > 2**32 - 1:
                raise ValueError("precision callback returned an invalid decision")
            output[0] = decision
            decisions[event] = decision
            completed |= 1 << event
            return 0
        except BaseException as error:  # noqa: BLE001 - Never unwind through libffi.
            if (
                event == PrecisionEvent.CANDIDATE
                and retry_candidate
                and isinstance(error, (RuntimeError, NotImplementedError))
            ):
                retry_error = sys.exc_info()
                return 1
            if pending is None:
                pending = sys.exc_info()
            return 2

    query.argtypes = [
        ctypes.POINTER(_Context),
        _Callback,
        ctypes.c_void_p,
        ctypes.POINTER(_Receipt),
    ]
    query.restype = ctypes.c_int
    context = _Context(
        ctypes.sizeof(_Context), _native.ABI_VERSION, int(retry_candidate), 0
    )
    receipt = _Receipt(ctypes.sizeof(_Receipt), _native.ABI_VERSION)
    status = query(ctypes.byref(context), callback, None, ctypes.byref(receipt))
    failure = pending or retry_error
    if failure is not None:
        raise failure[1].with_traceback(failure[2])
    _native.check(library, status)
    candidate = 1 << PrecisionEvent.CANDIDATE
    fallback = 1 << PrecisionEvent.FP64_FALLBACK
    base = candidate | (1 << PrecisionEvent.CLASSIFY)
    audit = base | (1 << PrecisionEvent.RELEASE) | (1 << PrecisionEvent.AUDIT)
    verification = audit | (1 << PrecisionEvent.VERIFY)
    try:
        outcome = PrecisionOutcome(receipt.outcome)
    except ValueError as error:
        raise RuntimeError(
            "native precision execution returned an invalid receipt"
        ) from error
    required = {
        PrecisionOutcome.QUALIFIED: base,
        PrecisionOutcome.AUDITED: audit,
        PrecisionOutcome.VERIFIED: verification,
        PrecisionOutcome.REJECTED: verification,
    }.get(outcome)
    retried = bool(receipt.attempted_events & fallback)
    expected_attempts = (
        (required | fallback) if required is not None and retried else required
    )
    expected_completed = (
        (expected_attempts & ~candidate) if retried else expected_attempts
    )
    if (
        receipt.struct_size != ctypes.sizeof(_Receipt)
        or receipt.abi_version != _native.ABI_VERSION
        or receipt.policy_version != 1
        or required is None
        or (retried and not retry_candidate)
        or receipt.attempted_events != attempted
        or receipt.completed_events != completed
        or attempted != expected_attempts
        or completed != expected_completed
        or decisions.get(PrecisionEvent.CLASSIFY)
        != int(outcome == PrecisionOutcome.QUALIFIED)
        or (
            outcome != PrecisionOutcome.QUALIFIED
            and decisions.get(PrecisionEvent.AUDIT)
            != int(outcome == PrecisionOutcome.AUDITED)
        )
        or (
            outcome in (PrecisionOutcome.VERIFIED, PrecisionOutcome.REJECTED)
            and decisions.get(PrecisionEvent.VERIFY)
            != int(outcome == PrecisionOutcome.VERIFIED)
        )
    ):
        raise RuntimeError(
            "native precision execution returned an inconsistent receipt"
        )
    return outcome
