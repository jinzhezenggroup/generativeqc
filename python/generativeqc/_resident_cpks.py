"""Transport-only adapter to the native, current-state resident CPKS solver."""

from __future__ import annotations

import ctypes as ct
import typing
from time import perf_counter

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from . import _native
from .response_problem import ResponseUnsupported
from .response_solver import GMRESOptions, SolveResult


class _Controls(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint64),
        ("rtol", ct.c_double),
        ("atol", ct.c_double),
        ("breakdown_tolerance", ct.c_double),
        ("stagnation_tolerance", ct.c_double),
        ("restart", ct.c_uint64),
        ("max_iterations", ct.c_uint64),
        ("max_workspace_bytes", ct.c_uint64),
        ("reorthogonalize", ct.c_uint64),
        ("true_residual_every", ct.c_uint64),
        ("stagnation_window", ct.c_uint64),
    ]


_REASONS = (
    "initial_residual",
    "converged",
    "workspace_limit",
    "max_iterations",
    "breakdown",
    "stagnation",
    "nonfinite_input",
    "nonfinite_operator",
    "nonfinite_preconditioner",
)


def solve_resident(
    operator: typing.Any,
    rhs: typing.Any,
    *,
    options: GMRESOptions | None = None,
    initial_guess: typing.Any = None,
    raise_on_failure: bool = False,
) -> SolveResult:
    """Solve without Python orbital transforms, actions or Krylov bookkeeping.

    The source snapshot, XC owner and Fock owner remain locked until native
    publication. Setup uploads C/epsilon once; iteration transfers only scalar
    reductions/status, and downloads the final response vector once. This does
    not supply a molecular Hessian, block Arnoldi or a recycling algorithm.
    """
    operator.validate_current()
    if operator.state._source.backend != "cuda":
        raise ResponseUnsupported("resident CPKS requires a native CUDA state")
    options = GMRESOptions() if options is None else options
    if not isinstance(options, GMRESOptions):
        raise TypeError("resident CPKS requires GMRESOptions")
    controls = _Controls()
    controls.struct_size = ct.sizeof(controls)
    for name, _ in _Controls._fields_[1:]:
        value = getattr(options, name)
        if (
            name
            in {
                "restart",
                "max_iterations",
                "max_workspace_bytes",
                "reorthogonalize",
                "true_residual_every",
                "stagnation_window",
            }
            and not 0 < value < 2**64
        ):
            raise ValueError(f"resident CPKS {name} must fit a positive uint64")
        setattr(controls, name, value)
    values = np.array(
        operator.problem.layout.validate_vector(rhs), dtype=np.float64, copy=True
    )
    guess = (
        np.empty(0, dtype=np.float64)
        if initial_guess is None
        else np.array(
            operator.problem.layout.validate_vector(initial_guess),
            dtype=np.float64,
            copy=True,
        )
    )
    source = operator.state._source
    cuda = operator.xc_kernel._cuda
    fock = operator.backend._plan
    library = source._library
    function = getattr(library, "generativeqc_ks_response_solve_v1", None)
    if function is None:
        raise ResponseUnsupported(
            "native library lacks the resident CPKS solver; no host retry"
        )
    pointer = ct.POINTER(ct.c_double)
    function.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(_Controls),
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_uint64),
        ct.c_size_t,
    ]
    function.restype = ct.c_int
    solution = np.zeros(values.size, dtype=np.float64)
    residual = np.empty(2, dtype=np.float64)
    work = np.empty(10, dtype=np.uint64)
    started = perf_counter()
    with cuda._lock, fock._lock:
        operator.validate_current()
        status = function(
            source._batch._batch,
            source._handle,
            cuda._handle,
            fock._handle,
            ct.byref(controls),
            values.ctypes.data_as(pointer),
            values.size,
            guess.ctypes.data_as(pointer),
            guess.size,
            operator._device_budget_bytes,
            solution.ctypes.data_as(pointer),
            solution.size,
            residual.ctypes.data_as(pointer),
            residual.size,
            work.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            work.size,
        )
        if status == _native.STATUS_OUT_OF_MEMORY:
            raise MemoryError("native resident CPKS capacity admission failed")
        _native.check(library, status, context=source._batch._context)
        operator.validate_current()
    if int(work[0]) >= len(_REASONS):
        raise RuntimeError("native resident CPKS returned an unknown solver status")
    reason = _REASONS[int(work[0])]
    operator._last_resident_solve = dict(
        zip(
            (
                "status",
                "iterations",
                "restarts",
                "operator_actions",
                "host_workspace_bytes",
                "owned_device_bytes",
                "h2d_bytes",
                "d2h_bytes",
                "synchronizations",
                "blas_calls",
            ),
            map(int, work),
            strict=True,
        )
    )
    operator._last_resident_solve["seconds"] = perf_counter() - started
    operator._last_execution = "native-cuda"
    result = SolveResult(
        solution=immutable(solution),
        converged=int(work[0]) in (0, 1),
        residual_norm=float(residual[0]),
        relative_residual=float(residual[1]),
        iterations=int(work[1]),
        reason=reason,
        history=(),
        operator_actions=int(work[3]),
        preconditioner_actions=0,
        orthogonalization_seconds=0.0,
        operator_seconds=0.0,
        workspace_bytes=int(work[4]),
    )
    return result.require_converged() if raise_on_failure else result
