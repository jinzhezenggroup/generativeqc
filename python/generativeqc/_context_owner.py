"""Single-entry native context ownership for reusable singlepoint execution."""

from __future__ import annotations

import ctypes
import threading
import typing
from contextlib import suppress

from . import _native


class ContextOwner:
    """Retain one context and serialize its complete Python result transaction.

    The native context mutex protects individual ABI calls. This lock also
    protects the gap between preparation, execution and diagnostic reads, plus
    explicit cleanup. Each calculator owns its own instance. No result or
    converged SCC state is cached at the Python layer.
    """

    def __init__(self, library: typing.Any) -> None:
        self.lock = threading.RLock()
        self._library = library
        self._context = ctypes.c_void_p()
        self._identity: tuple[int, int] | None = None

    def get(self, descriptor: _native.ContextDescriptor) -> ctypes.c_void_p:
        """Return the sole context; caller holds ``lock`` through its last read.

        Backend/device changes retire the previous context before allocation.
        Creation failures leave an empty owner, so the next call can retry.
        """
        identity = (descriptor.backend, descriptor.device_id)
        if self._identity != identity:
            self.clear()
        if not self._context.value:
            candidate = ctypes.c_void_p()
            _native.check(
                self._library,
                self._library.generativeqc_context_create(
                    ctypes.byref(descriptor), ctypes.byref(candidate)
                ),
            )
            self._context = candidate
            self._identity = identity
        return self._context

    def clear(self) -> None:
        """Release resident work under the transaction lock; later calls rebuild."""
        with self.lock:
            if self._context.value:
                self._library.generativeqc_context_destroy(self._context)
                self._context = ctypes.c_void_p()
            self._identity = None

    def __del__(self) -> None:
        # Partial construction/interpreter shutdown must not mask user errors.
        with suppress(Exception):
            self.clear()
