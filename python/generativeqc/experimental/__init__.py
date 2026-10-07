"""Experimental public GenerativeQC surfaces.

These APIs are intentionally versioned separately from stable user-facing
interfaces. Selecting one surface is lazy so unrelated compiler functionality
is not activated by importing the package.
"""

from __future__ import annotations

import importlib
import typing

API_VERSION = 1

_PUBLIC_MODULES = frozenset({"array_api"})


def __getattr__(name: str) -> typing.Any:
    """Load one experimental public surface only when requested."""
    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Advertise lazy experimental surfaces to interactive discovery."""
    return sorted(set(globals()) | _PUBLIC_MODULES)


__all__ = ["API_VERSION", "array_api"]
