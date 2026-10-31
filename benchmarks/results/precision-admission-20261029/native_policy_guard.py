"""Qualification-only guard for the explicitly migrated production boundary."""

import ctypes
import os
from typing import Any, NoReturn

import pytest


@pytest.fixture(autouse=True)
def require_native_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual GPU endpoints must never re-enter the missing-protocol adapter."""
    from generativeqc import precision_policy

    library = ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"])
    assert library.generativeqc_precision_admission_query_v1

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError(
            "current native precision policy used Python compatibility"
        )

    for name in ("budget_fock_work", "budget_fock_strict_warm", "standard_fallback"):
        monkeypatch.setattr(
            precision_policy, f"_compatibility_{name}_admitted", forbidden
        )
