"""Runtime-owned buffers, FFI lifetime and publication for bounded XC programs."""

from __future__ import annotations

import ctypes as ct
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.xc.user_expression_program import (
    MAX_POINTS,
    UserExpressionProgram,
)

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter


def _validate_inputs(features: np.ndarray) -> np.ndarray:
    """Own a bounded FP64 snapshot, including both unused declared inputs."""
    if not isinstance(features, np.ndarray) or features.dtype != np.float64:
        raise TypeError("expression features require a float64 ndarray")
    if features.ndim != 2 or features.shape[0] != 2:
        raise ValueError("expression features require shape [2, point]")
    if features.shape[1] > MAX_POINTS:
        raise ValueError("expression batch exceeds point budget")
    x = np.array(features, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("rho inputs require finite nonnegative values")
    return x


class NativeUserExpression:
    """Runtime owner of an explicitly compiled CPU polynomial artifact.

    The native ABI fills private scratch. Only successful, finite results become
    immutable detached byte storage; no caller output/donation API is provided.
    Compilation and calls confer no SCF, force, response or CUDA admission.
    """

    def __init__(
        self,
        program: UserExpressionProgram,
        *,
        compiler: CppCompilerAdapter,
        cache: str | Path,
    ) -> None:
        """Compile the sealed program and retain the loaded CPU ABI lifetime."""
        if type(program) is not UserExpressionProgram:
            raise TypeError("native expression requires UserExpressionProgram")
        self._program = program
        self.artifact = program.compile(compiler=compiler, cache=cache)
        self._library = ct.CDLL(str(self.artifact.library))
        self._function = self._library.xc_user_expression
        self._function.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
        ]
        self._function.restype = ct.c_int

    @property
    def program(self) -> UserExpressionProgram:
        """Return the immutable mathematical recipe used by this runtime."""
        return self._program

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """Execute [2, point] float64 rho inputs and publish [output, point].

        Both rho rows must be finite and nonnegative, including constant formulas.
        Vacuum and empty batches are allowed. Excessive size or invalid dtype/
        shape fails before execution; nonfinite native results raise without
        publishing scratch. Successful results own immutable detached storage.
        """
        x = _validate_inputs(features)
        result = np.empty((len(self.program.outputs), x.shape[1]), dtype=np.float64)
        status = self._function(
            x.ctypes.data_as(ct.POINTER(ct.c_double)),
            x.size,
            result.ctypes.data_as(ct.POINTER(ct.c_double)),
            result.size,
            x.shape[1],
        )
        if status:
            raise ArithmeticError(
                f"native expression failed (status {status}); no result published"
            )
        return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)

    def point(self, *, rho_a: float, rho_b: float) -> np.ndarray:
        """Execute one finite nonnegative Python-float/integer density pair.

        Return one immutable vector in program output order. Invalid inputs and
        nonfinite outputs fail under the same atomic-publication policy as batch
        evaluation. This is a feature endpoint, not SCF or nuclear-force access.
        """
        if any(type(v) not in (int, float) for v in (rho_a, rho_b)):
            raise TypeError("point inputs require Python FP64 floats or integers")
        try:
            x = np.array([[rho_a], [rho_b]], dtype=np.float64)
        except OverflowError as exc:
            raise ValueError("rho inputs exceed FP64") from exc
        return self.evaluate(x)[:, 0]
