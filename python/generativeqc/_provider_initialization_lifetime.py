"""One reachable transaction for this interpreter's managed optional SDK path.

This module is shared by ordinary imports across CDLL instances. It does not
claim a process-wide native/multiple-interpreter singleton. No finalizer attempts
CUDA cleanup: only explicit status-bearing close can clear the slot.
"""

from __future__ import annotations

import atexit
import ctypes
import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._context_owner import ContextOwner

from . import _native


class ProviderInitializationCloseUnavailable(RuntimeError):
    def __init__(
        self,
        status: int,
        detail_bytes: bytes | None = None,
        detail_query_error: BaseException | None = None,
    ) -> None:
        self.status = status
        self.detail_bytes = detail_bytes
        self.detail = (
            detail_bytes.decode("utf-8", errors="backslashreplace")
            if detail_bytes is not None
            else None
        )
        self.detail_query_error = detail_query_error
        diagnostic = (
            self.detail
            if self.detail is not None
            else "native close detail unavailable"
        )
        super().__init__(
            f"provider owner close unavailable: native status={status}; {diagnostic}; owner retained"
        )


class _Transaction:
    def __init__(
        self,
        root: _LifetimeRoot,
        library: Any,
        context_owner: ContextOwner | None,
        close: Any,
    ) -> None:
        self.root, self.library, self.context_owner = root, library, context_owner
        self.context = ctypes.c_void_p()
        self.system = ctypes.c_void_p()
        self.auxiliary_system = ctypes.c_void_p()
        self.calculation = ctypes.c_void_p()
        self.budget_lease = None
        self.native_close = close
        self.shutdown_unavailable = False
        self._executing = True
        self._execution_thread = threading.get_ident()

    def open_context(self, descriptor: _native.ContextDescriptor) -> ctypes.c_void_p:
        if self.context_owner is not None:
            self.context = self.context_owner.get(descriptor)
        else:
            _native.check(
                self.library,
                self.library.generativeqc_context_create(
                    ctypes.byref(descriptor), ctypes.byref(self.context)
                ),
            )
        return self.context

    def close(self, *, _finish_execution: bool = True) -> None:
        # Match the scalar transaction's lock order. Never acquire a context
        # owner lock while already holding the root lock.
        guard = (
            self.context_owner.lock if self.context_owner is not None else nullcontext()
        )
        with guard, self.root.lock:
            if self.root.active is not self:
                return
            if self._executing:
                if (
                    not _finish_execution
                    or threading.get_ident() != self._execution_thread
                ):
                    raise RuntimeError(
                        "provider transaction is still executing; owner retained"
                    )
                # Only the executing scalar call's finally path can end use.
                # Public close cannot retire a not-yet-created context or any
                # handle between prepare, execute and diagnostic reads.
                self._executing = False
            if self.context.value:
                status = int(self.native_close(self.context))
                if status:
                    detail_bytes = None
                    detail_query_error = None
                    try:
                        getter = self.library.generativeqc_context_get_last_detail
                        getter.argtypes = [ctypes.c_void_p]
                        getter.restype = ctypes.c_char_p
                        detail_bytes = getter(self.context)
                        if detail_bytes is not None and not isinstance(
                            detail_bytes, bytes
                        ):
                            raise TypeError("native close detail is not a C string")
                    except Exception as error:  # noqa: BLE001 - retain failed diagnostics with the failed close
                        detail_bytes = None
                        detail_query_error = error
                    raise ProviderInitializationCloseUnavailable(
                        status, detail_bytes, detail_query_error
                    )
            # Native close has proved complete owner closure before ANY delete.
            if self.calculation.value:
                self.library.generativeqc_calculation_destroy(self.calculation)
                self.calculation.value = None
            if self.system.value:
                self.library.generativeqc_system_destroy(self.system)
                self.system.value = None
            if self.auxiliary_system.value:
                self.library.generativeqc_system_destroy(self.auxiliary_system)
                self.auxiliary_system.value = None
            if self.context_owner is None:
                if self.context.value:
                    self.library.generativeqc_context_destroy(self.context)
                    self.context.value = None
            else:
                self.context_owner._provider_initialization_retained = False
            if self.budget_lease is not None:
                self.budget_lease.close()
                self.budget_lease = None
            self.root.active = None

    def close_preserving_primary(self, primary: BaseException | None) -> None:
        """Keep the original failure directly observable and attach failed close.

        A secondary close error never replaces the caller's primary object,
        status, detail or traceback. With no primary, close failure propagates.
        """
        try:
            self.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.provider_initialization_close_error = close_error
            # Python 3.10 has no exception notes; the attached secondary error
            # still preserves the close failure without replacing the primary.
            add_note = getattr(BaseException, "add_note", None)
            if add_note is not None:
                add_note(primary, f"Secondary provider close failure: {close_error}")


class _LifetimeRoot:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active: _Transaction | None = None

    def require_empty(self) -> None:
        with self.lock:
            if self.active is not None:
                raise RuntimeError(
                    "one optional provider transaction is still retained; close it first"
                )

    def begin(self, library: Any, context_owner: ContextOwner | None) -> _Transaction:
        with self.lock:
            self.require_empty()
            try:
                close = library.generativeqc_context_close_provider_initialization_v1
            except AttributeError as error:
                raise NotImplementedError(
                    "status-bearing provider close is unavailable"
                ) from error
            close.argtypes = [ctypes.c_void_p]
            close.restype = ctypes.c_int
            transaction = _Transaction(self, library, context_owner, close)
            # Occupied BEFORE native context creation; a different CDLL cannot
            # create another slot by loading the same SDK again.
            self.active = transaction
            if context_owner is not None:
                context_owner._provider_initialization_retained = True
            return transaction

    def keep_until_exit(self) -> None:
        with self.lock:
            if self.active is not None:
                self.active.shutdown_unavailable = True
                # No close, context switch, retry or void cleanup at shutdown.


_root = _LifetimeRoot()
# The callback strongly holds the root independently of Calculator/module GC.
# Exactly one registration and one slot; process exit is not cleanup evidence.
atexit.register(_root.keep_until_exit)


def require_no_pending_owner() -> None:
    _root.require_empty()


def begin(library: Any, context_owner: ContextOwner | None) -> _Transaction:
    return _root.begin(library, context_owner)


def has_pending_owner() -> bool:
    with _root.lock:
        return _root.active is not None


def close_retained_owner() -> None:
    """Close under the original ambient owner; failure retains every handle.

    No force-clear, caller context key, qualification bit or replacement owner.
    Native physical-close latching preserves an uncertain attempted raw outcome.
    """
    with _root.lock:
        active = _root.active
    if active is not None:
        active.close(_finish_execution=False)
