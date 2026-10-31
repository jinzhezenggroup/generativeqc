"""Bounded bindings for immutable native metadata protocols.

Keep scientific policy in native owners. This transport shares only typed
symbols and their library lifetime, never calculation/context/device resources.
"""

from __future__ import annotations

import ctypes as ct
from collections import OrderedDict
from threading import RLock
from typing import Any

_QUERIES: OrderedDict[tuple[int, str, Any], tuple[Any, Any]] = OrderedDict()
_QUERY_LOCK = RLock()


def bind(library: Any, name: str, query_type: Any) -> Any:
    """Bound shared symbol binding without a new CDLL/function reference cycle.

    ctypes attribute lookup normally installs a function on each fresh CDLL,
    while the function retains that CDLL. Audit Calculator construction repeats
    this for otherwise immutable metadata queries. Bind without installing the attribute
    and retain at most eight library/function pairs. Retaining each library also
    prevents handle reuse while its cached binding is live. Explicitly supplied
    bindings remain authoritative, including failure-injection adapters.
    """
    if not isinstance(library, ct.CDLL):
        return getattr(library, name, None)
    if name in library.__dict__:
        return library.__dict__[name]
    handle = (library._handle, name, query_type)
    with _QUERY_LOCK:
        if handle in _QUERIES:
            _QUERIES.move_to_end(handle)
            return _QUERIES[handle][1]
        try:
            query = query_type((name, library))
        except AttributeError:
            query = None
        _QUERIES[handle] = (library, query)
        if len(_QUERIES) > 8:
            _QUERIES.popitem(last=False)
        return query
