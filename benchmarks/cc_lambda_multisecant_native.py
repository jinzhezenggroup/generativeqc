"""Synchronous ctypes adapter to the opt-in native screening GMRES bridge."""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class NativeGMRES:
    """Borrow a prebuilt, source-matched bridge; never compile during timing."""

    def __init__(self, library: Path) -> None:
        self.library = ct.CDLL(str(library.resolve()))
        self.pointer = ct.POINTER(ct.c_double)
        self.callback = ct.CFUNCTYPE(ct.c_int, self.pointer, self.pointer)
        self.entry = self.library.multisecant_gmres
        self.entry.argtypes = [
            ct.c_size_t,
            self.callback,
            self.callback,
            self.pointer,
            self.pointer,
            ct.POINTER(ct.c_size_t),
            self.pointer,
            ct.c_char_p,
            ct.c_size_t,
        ]
        self.entry.restype = ct.c_int

    def solve(
        self,
        apply: Callable[[np.ndarray], np.ndarray],
        inverse: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
    ) -> SimpleNamespace:
        """Run native GMRES, propagating Python errors across the C boundary."""
        rhs = np.asarray(rhs)
        if rhs.ndim != 1 or rhs.dtype != np.float64 or not np.isfinite(rhs).all():
            raise ValueError("screening RHS must be a finite FP64 vector")
        rhs = np.ascontiguousarray(rhs)
        n = rhs.size
        errors: list[Exception] = []

        def bind(action: Callable[[np.ndarray], np.ndarray]) -> ct._CFuncPtr:
            def callback(source: ct._Pointer, target: ct._Pointer) -> int:
                try:
                    borrowed = np.ctypeslib.as_array(source, shape=(n,))
                    borrowed.setflags(write=False)
                    values = np.asarray(action(borrowed))
                    if (
                        values.shape != (n,)
                        or values.dtype != np.float64
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError("invalid screening callback output")
                    np.ctypeslib.as_array(target, shape=(n,))[:] = values
                    return 0
                except Exception as error:  # noqa: BLE001 -- ctypes cannot propagate callback exceptions.
                    errors.append(error)
                    return 1

            return self.callback(callback)

        output = np.empty(n)
        counts = (ct.c_size_t * 3)()
        residual = ct.c_double()
        error = ct.create_string_buffer(1024)
        # Local callback owners and ndarray buffers live until the synchronous
        # native call has drained, including a translated callback exception.
        transpose_callback, inverse_callback = bind(apply), bind(inverse)
        status = self.entry(
            n,
            transpose_callback,
            inverse_callback,
            rhs.ctypes.data_as(self.pointer),
            output.ctypes.data_as(self.pointer),
            counts,
            ct.byref(residual),
            error,
            len(error),
        )
        if errors:
            raise RuntimeError("screening callback failed") from errors[0]
        if status:
            raise RuntimeError(error.value.decode())
        return SimpleNamespace(
            solution=output,
            iterations=counts[0],
            operator_actions=counts[1],
            preconditioner_actions=counts[2],
            residual_norm=residual.value,
        )
