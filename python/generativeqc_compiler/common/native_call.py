"""Small ctypes helpers for status-returning native ABIs."""

from __future__ import annotations

import ctypes
import typing


def checked_native_call(
    function: typing.Callable[..., int],
    *args: typing.Any,
    error_type: type[Exception] = RuntimeError,
    status_error_types: typing.Mapping[int, type[Exception]] | None = None,
    buffer_bytes: int = 2048,
) -> None:
    """Raise a status-specific exception when the caller's ABI distinguishes it."""
    if type(buffer_bytes) is not int or buffer_bytes <= 0:
        raise ValueError("native error buffer size must be a positive integer")
    error = ctypes.create_string_buffer(buffer_bytes)
    status = function(*args, error, len(error))
    if status:
        selected_type = (status_error_types or {}).get(status, error_type)
        raise selected_type(error.value.decode())
