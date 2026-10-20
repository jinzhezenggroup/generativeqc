"""CUPTI caller-clock envelope; distinct from supervised monotonic wall time."""

from __future__ import annotations

import ctypes
import hashlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class CallerClock:
    """Use a pinned CUPTI file's normalized clock without enabling activities."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("CUPTI clock library identity mismatch")
        self.library = ctypes.CDLL(str(path))
        self.version = self.library.cuptiGetVersion
        self.version.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.version.restype = ctypes.c_int
        version = ctypes.c_uint32()
        if self.version(ctypes.byref(version)) != 0 or version.value != 28:
            raise ValueError("unsupported CUPTI clock API")
        self.timestamp = self.library.cuptiGetTimestamp
        self.timestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.timestamp.restype = ctypes.c_int
        self.sha256 = expected_sha256

    def sample(self) -> dict[str, object]:
        """Retain actual CPU brackets, including any first-call initialization."""
        value = ctypes.c_uint64()
        monotonic_before = time.monotonic_ns()
        realtime_before = time.time_ns()
        status = self.timestamp(ctypes.byref(value))
        realtime_after = time.time_ns()
        monotonic_after = time.monotonic_ns()
        if status != 0 or value.value == 0:
            raise ValueError("CUPTI clock query failed")
        return {
            "clock": "cupti28-normalized-default-ns",
            "cupti_ns": value.value,
            "realtime_bracket_ns": [realtime_before, realtime_after],
            "monotonic_bracket_ns": [monotonic_before, monotonic_after],
            "library_sha256": self.sha256,
            "activity_enable": False,
        }
