"""Synchronous host-callback adapters; the shared C++ owner performs GMRES arithmetic."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from . import _native
from .elements import checked_integer

_REASONS = {
    1: "initial_residual",
    2: "converged",
    3: "workspace_limit",
    4: "max_iterations",
    5: "breakdown",
    6: "stagnation",
    7: "nonfinite_input",
    8: "nonfinite_operator",
    9: "nonfinite_preconditioner",
}


def _descriptor(dimension: int, options: Any) -> _native.ResponseGmresDescriptor:
    """Marshal validated control values without vectors, callbacks or allocations by dimension."""
    return _native.ResponseGmresDescriptor(
        struct_size=ctypes.sizeof(_native.ResponseGmresDescriptor),
        abi_version=_native.ABI_VERSION,
        dimension=checked_integer(
            dimension, "response dimension", low=1, high=2**64 - 1
        ),
        rtol=options.rtol,
        atol=options.atol,
        restart=checked_integer(
            options.restart, "response restart", low=1, high=2**64 - 1
        ),
        max_iterations=checked_integer(
            options.max_iterations, "response iterations", low=1, high=2**64 - 1
        ),
        max_workspace_bytes=checked_integer(
            options.max_workspace_bytes, "response workspace", low=1, high=2**64 - 1
        ),
        reorthogonalize=checked_integer(
            options.reorthogonalize,
            "response reorthogonalization",
            low=1,
            high=2**32 - 1,
        ),
        breakdown_tolerance=options.breakdown_tolerance,
        true_residual_every=checked_integer(
            options.true_residual_every,
            "response checkpoint interval",
            low=1,
            high=2**64 - 1,
        ),
        stagnation_window=checked_integer(
            options.stagnation_window,
            "response stagnation window",
            low=1,
            high=2**64 - 1,
        ),
        stagnation_tolerance=options.stagnation_tolerance,
    )


def workspace_bytes(library: Any, dimension: int, options: Any) -> int:
    """Query the same owner that admits the solve before creating dimension-sized buffers."""
    descriptor = _descriptor(dimension, options)
    output = ctypes.c_uint64()
    _native.check(
        library,
        library.generativeqc_response_gmres_workspace_v1(
            ctypes.byref(descriptor), ctypes.byref(output)
        ),
    )
    return output.value


def solve_host(
    library: Any,
    operator: Any,
    rhs: Any,
    options: Any,
    *,
    initial_guess: Any,
    preconditioner: Any,
    collect_basis: bool,
) -> Any:
    """Copy native publication before cleanup, and rethrow original callback failures.

    No callback or NumPy view outlives the synchronous call. The operator retains
    its own physical equations and current-lease validation. A native refusal is
    a failure, not permission to retry Python arithmetic.
    """
    from .response_solver import SolveResult, _HostKrylovEngine

    dimension = checked_integer(
        operator.dimension, "response dimension", low=1, high=2**64 - 1
    )
    values = np.array(rhs, dtype=np.float64, order="C", copy=True)
    guess = (
        None
        if initial_guess is None
        else np.array(initial_guess, dtype=np.float64, order="C", copy=True)
    )
    engine = _HostKrylovEngine(dimension)
    failures: list[BaseException] = []

    def callback(action: Any, *, precondition: bool = False) -> Any:
        def invoke(user_data: Any, source: Any, destination: Any, count: int) -> int:
            try:
                # Tracebacks and user callbacks can retain this Python array.
                # Never let such retention turn a completed native lease into a dangling view.
                vector = np.ctypeslib.as_array(source, shape=(count,)).copy()
                vector.flags.writeable = False
                output = (
                    engine.precondition(action, vector)
                    if precondition
                    else engine.apply(action, vector)
                )
                if not np.isfinite(output).all():
                    raise ValueError("response action must return finite real data")
                np.ctypeslib.as_array(destination, shape=(count,))[:] = output
                return _native.STATUS_SUCCESS
            except BaseException as error:  # noqa: BLE001 - a ctypes callback cannot unwind through C
                failures.append(error)
                return _native.STATUS_INTERNAL_ERROR

        return _native.ResponseAction(invoke)

    apply = callback(operator)
    condition = (
        _native.ResponseAction()
        if preconditioner is None
        else callback(preconditioner, precondition=True)
    )
    descriptor = _descriptor(dimension, options)
    descriptor.rhs = values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    descriptor.initial_guess = (
        None if guess is None else guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    descriptor.apply = apply
    descriptor.precondition = condition
    descriptor.flags = 1 | 4 | (2 if collect_basis else 0)
    owner = ctypes.c_void_p()
    try:
        status = library.generativeqc_response_solve_v1(
            ctypes.byref(descriptor), ctypes.byref(owner)
        )
        if failures:
            raise failures[0]
        _native.check(library, status)
        view = _native.ResponseResultView(
            struct_size=ctypes.sizeof(_native.ResponseResultView),
            abi_version=_native.ABI_VERSION,
        )
        _native.check(
            library,
            library.generativeqc_response_result_view_v1_get(owner, ctypes.byref(view)),
        )
        if view.reason == 7:
            raise ValueError("GMRES vector norm overflows FP64")

        def copied(pointer: Any, count: int) -> np.ndarray:
            return (
                immutable(np.ctypeslib.as_array(pointer, shape=(count,)))
                if count
                else immutable(np.empty(0))
            )

        solution = copied(view.solution, view.solution_count)
        if view.reason == 3:
            solution = immutable(np.zeros(dimension) if guess is None else guess)
        basis = (
            copied(view.basis, dimension * view.basis_columns)
            .reshape((view.basis_columns, dimension))
            .T
        )
        return SolveResult(
            solution,
            bool(view.converged),
            view.residual_norm,
            view.relative_residual,
            view.iterations,
            _REASONS[view.reason],
            tuple(copied(view.history, view.history_count)),
            view.operator_actions,
            view.preconditioner_actions,
            view.orthogonalization_seconds,
            view.operator_seconds,
            view.workspace_bytes,
            basis=basis,
        )
    finally:
        if owner:
            library.generativeqc_response_result_destroy(owner)
