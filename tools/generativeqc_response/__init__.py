"""Shared native response-solver infrastructure for HF/DFT orbital response.

The package intentionally separates three concerns:

* :mod:`generativeqc.response_problem` owns immutable scientific snapshots.
* :mod:`generativeqc.response_operator` owns matrix-free JVP/VJP actions.
* :mod:`generativeqc.response_solver` owns bounded linear-solver and recycling state.

CC-specific right-hand sides and weights remain in their consuming issue.  The
shared contract here is only the orbital-response problem, operator action and
linear solve.
"""

from generativeqc.response_operator import (
    CPKSResponseOperator,
    DenseMatrixResponseOperator,
    RHFResponseOperator,
)
from generativeqc.response_problem import (
    ResponseCompatibilityError,
    ResponseProblem,
    ResponseSolveError,
    ResponseUnsupported,
    RotationLayout,
)
from generativeqc.response_solver import (
    DiagonalPreconditioner,
    GMRESOptions,
    KrylovRecycleSpace,
    MultiRHSResult,
    SolveResult,
    resident_vector_slots,
    solve,
    solve_many,
)
from generativeqc.response_xc import FixedDensityXCDerivativeKernel

from .backends import CudaDFJKBackend, DenseAOResponseBackend, NativeJKBackend
from .direct_cuda import CudaDirectJKBackend
from .native_ks import NativeRKSResponse, NativeUKSResponse
from .oracle import explicit_rhf_response_matrix, finite_rotation_jvp
from .resident_cuda import CudaResidentRHFResponse
from .resident_uhf_cuda import CudaResidentUHFResponse
from .spin_cuda import CudaSpinJKBackend
from .uhf import (
    UHFReferenceSnapshot,
    UHFResponseOperator,
    UHFResponseProblem,
    UHFSpinRotationLayout,
    UKSResponseOperator,
)

__all__ = [
    "CPKSResponseOperator",
    "CudaDFJKBackend",
    "CudaDirectJKBackend",
    "CudaResidentRHFResponse",
    "CudaResidentUHFResponse",
    "CudaSpinJKBackend",
    "DenseAOResponseBackend",
    "DenseMatrixResponseOperator",
    "DiagonalPreconditioner",
    "FixedDensityXCDerivativeKernel",
    "GMRESOptions",
    "KrylovRecycleSpace",
    "MultiRHSResult",
    "NativeJKBackend",
    "NativeRKSResponse",
    "NativeUKSResponse",
    "RHFResponseOperator",
    "ResponseCompatibilityError",
    "ResponseProblem",
    "ResponseSolveError",
    "ResponseUnsupported",
    "RotationLayout",
    "SolveResult",
    "UHFReferenceSnapshot",
    "UHFResponseOperator",
    "UHFResponseProblem",
    "UHFSpinRotationLayout",
    "UKSResponseOperator",
    "explicit_rhf_response_matrix",
    "finite_rotation_jvp",
    "resident_vector_slots",
    "solve",
    "solve_many",
]
