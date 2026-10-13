"""Executable transport, status and lifecycle gates for the native solve ABI."""

import ctypes as ct
import threading
import typing
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._resident_cpks import solve_resident
from generativeqc.response_problem import (
    ResponseSolveError,
    ResponseUnsupported,
    RotationLayout,
)
from generativeqc.response_solver import GMRESOptions


@pytest.fixture
def operator() -> typing.Any:
    state = SimpleNamespace(closed=False, called=False, reason=1, status=0)

    def validate() -> None:
        if state.closed:
            raise ValueError("stale response owner")

    def native(*args: typing.Any) -> int:
        state.called = True
        if state.status:
            return state.status
        solution = np.ctypeslib.as_array(args[10], shape=(args[11],))
        if state.reason != 2:
            solution[:] = np.ctypeslib.as_array(args[5], shape=(args[6],)) * 0.5
        residual = np.ctypeslib.as_array(args[12], shape=(args[13],))
        residual[:] = (0, 0) if state.reason < 2 else (1, 1)
        work = np.ctypeslib.as_array(args[14], shape=(args[15],))
        work[:] = (state.reason, 2, 0, 3, 256, 1024, 64, 24, 4, 7)
        return 0

    library = SimpleNamespace(generativeqc_ks_response_solve_v1=native)
    source = SimpleNamespace(
        backend="cuda",
        _library=library,
        _handle=ct.c_void_p(1),
        _batch=SimpleNamespace(_batch=ct.c_void_p(1), _context=None),
    )
    cuda = SimpleNamespace(_lock=threading.RLock(), _handle=ct.c_void_p(2))
    fock = SimpleNamespace(_lock=threading.RLock(), _handle=ct.c_void_p(3))
    return SimpleNamespace(
        validate_current=validate,
        state=SimpleNamespace(_source=source),
        problem=SimpleNamespace(layout=RotationLayout((0,), (1, 2))),
        xc_kernel=SimpleNamespace(_cuda=cuda),
        backend=SimpleNamespace(_plan=fock),
        _device_budget_bytes=4096,
        _last_resident_solve=None,
        test_state=state,
    )


def test_native_result_and_diagnostics_are_published(operator: typing.Any) -> None:
    result = solve_resident(operator, np.array([2.0, 4.0]))
    assert result.converged and result.reason == "converged"
    np.testing.assert_array_equal(result.solution, [1.0, 2.0])
    assert not result.solution.flags.writeable
    assert operator._last_resident_solve["owned_device_bytes"] == 1024


@pytest.mark.parametrize("reason", (2, 3, 4, 5, 6, 7, 8))
def test_native_nonconvergence_stays_explicit(
    operator: typing.Any, reason: int
) -> None:
    operator.test_state.reason = reason
    result = solve_resident(operator, np.ones(2))
    assert not result.converged
    with pytest.raises(ResponseSolveError):
        solve_resident(operator, np.ones(2), raise_on_failure=True)


def test_native_capacity_failure_does_not_publish(operator: typing.Any) -> None:
    operator.test_state.status = 7
    with pytest.raises(MemoryError, match="capacity"):
        solve_resident(operator, np.ones(2))
    assert operator._last_resident_solve is None


def test_stale_zero_rhs_does_not_reach_native(operator: typing.Any) -> None:
    operator.test_state.closed = True
    with pytest.raises(ValueError, match="stale"):
        solve_resident(operator, np.zeros(2))
    assert not operator.test_state.called


def test_old_library_is_not_a_silent_host_retry(operator: typing.Any) -> None:
    operator.state._source._library.generativeqc_ks_response_solve_v1 = None
    with pytest.raises(ResponseUnsupported, match="no host retry"):
        solve_resident(operator, np.ones(2))
    assert not operator.test_state.called


@pytest.mark.parametrize(
    "values", (np.ones(3), np.ones(2, dtype=complex), [float("nan"), 1])
)
def test_bad_rhs_is_rejected_before_native(
    operator: typing.Any, values: typing.Any
) -> None:
    with pytest.raises(ValueError):
        solve_resident(operator, values)
    assert not operator.test_state.called


def test_controls_cannot_wrap_uint64(operator: typing.Any) -> None:
    with pytest.raises(ValueError, match="uint64"):
        solve_resident(operator, np.ones(2), options=GMRESOptions(restart=2**70))
    assert not operator.test_state.called
