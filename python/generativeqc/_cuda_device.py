"""Lazy CUDA driver identity queries for runtime target selection.

Keep device discovery out of compiler/source generation and avoid requiring
CuPy or the native chemistry library for standalone TensorIR execution.
"""

from __future__ import annotations

import ctypes
import sys


def device_architecture(device: int = 0) -> str:
    """Return the compute capability of a CUDA-visible device as ``sm_XX``.

    The driver interprets the ordinal after CUDA_VISIBLE_DEVICES filtering.
    This query does not create/switch a context or modify device visibility.
    Missing drivers/devices fail explicitly instead of choosing a guessed GPU.
    Call only inside the caller's GPU allocation.
    """
    if type(device) is not int or not 0 <= device <= 2**31 - 1:
        raise ValueError("CUDA device must be a nonnegative int32 ordinal")
    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        driver = ctypes.CDLL(name)
    except OSError as error:
        raise RuntimeError(
            "CUDA target detection requires the NVIDIA driver"
        ) from error

    def call(symbol: str, types: list[object], *arguments: object) -> None:
        function = getattr(driver, symbol)
        function.argtypes = types
        function.restype = ctypes.c_int
        status = function(*arguments)
        if status:
            raise RuntimeError(
                f"CUDA target detection failed for device {device}: {symbol} (status {status})"
            )

    integer = ctypes.c_int
    pointer = ctypes.POINTER(integer)
    call("cuInit", [ctypes.c_uint], 0)
    handle, major, minor = integer(), integer(), integer()
    call("cuDeviceGet", [pointer, integer], ctypes.byref(handle), device)
    call(
        "cuDeviceComputeCapability",
        [pointer, pointer, integer],
        ctypes.byref(major),
        ctypes.byref(minor),
        handle,
    )
    if major.value < 1 or not 0 <= minor.value <= 9:
        raise RuntimeError("CUDA driver returned an invalid compute capability")
    return f"sm_{major.value}{minor.value}"
