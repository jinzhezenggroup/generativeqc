"""Shared library-operation execution vocabulary; this is not scientific IR."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LibraryRequest:
    """Explicit FP64 library call convention, including workspace and residual gates.

    GEMM uses A(m,k), B(k,n), C(m,n); symmetric eigensolve and Cholesky use
    square A(m,m). Layout applies to every operand. Providers must report errors
    and required workspace before enqueueing, and eigensolves/factorizations
    must expose a residual rather than assuming success from a status code.
    A caller may explicitly retain an independently qualified fixed algorithm
    with ``residual_policy="qualification"``. That mode promises qualification
    evidence only, not a per-execution residual or a runtime tolerance gate.
    """

    operation: str
    shape: tuple[int, ...]
    layout: str = "row_major"
    dtype: str = "fp64"
    workspace_limit_bytes: int = 0
    residual_tolerance: float = 1e-11
    residual_policy: str = field(default="runtime", kw_only=True)

    def __post_init__(self) -> None:
        import math

        if self.residual_policy not in ("runtime", "qualification"):
            raise ValueError("unknown residual policy")
        rank = 3 if self.operation == "gemm" else 1
        if self.operation not in ("gemm", "symmetric_eigh", "cholesky"):
            raise ValueError("unknown library operation")
        if len(self.shape) != rank or any(
            type(n) is not int or n < 1 for n in self.shape
        ):
            raise ValueError("invalid library operation dimensions")
        if self.layout not in ("row_major", "column_major") or self.dtype != "fp64":
            raise ValueError("explicit supported layout and FP64 dtype required")
        if (
            type(self.workspace_limit_bytes) is not int
            or self.workspace_limit_bytes < 0
        ):
            raise ValueError("workspace limit must be nonnegative")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0:
            raise ValueError("positive finite residual tolerance required")
