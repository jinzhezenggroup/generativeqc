"""Scoped numerical placement for consumers of the shared AD representations.

The context carries execution, not mathematical or generated-source identity.
It never imports a device array package, probes a device, or supplies a host
fallback. Context variables keep simultaneous CPU and CUDA consumers isolated.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy

_execution: ContextVar[Any] = ContextVar("generativeqc_array_execution", default=None)


def current_execution() -> Any:
    """Return the explicitly bound executor, or the ordinary NumPy contract."""
    return _execution.get()


@contextmanager
def array_execution(execution: Any) -> Iterator[Any]:
    """Bind one numerical owner for the lifetime of a derivative operation."""
    token = _execution.set(execution)
    try:
        yield execution
    finally:
        _execution.reset(token)


class _ArrayNamespace:
    """Resolve array operations without changing process-global module state."""

    def __getattr__(self, name: str) -> Any:
        execution = current_execution()
        return getattr(numpy if execution is None else execution.namespace, name)


arrays = _ArrayNamespace()


def branch_bytes(value: Any) -> bytes:
    """Hash branch metadata through the owner without staging numerical panels."""
    execution = current_execution()
    if execution is not None:
        return execution.branch_bytes(value)
    return value.tobytes()
