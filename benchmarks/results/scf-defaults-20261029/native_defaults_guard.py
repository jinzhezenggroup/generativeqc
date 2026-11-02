"""Qualification-only guard for actual native default policy consumption."""

import ctypes
import os
from typing import Any, NoReturn

import pytest


@pytest.fixture(autouse=True)
def require_native_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject either retained Python policy owner during real CUDA endpoints."""
    from generativeqc import _scf_convergence

    library = ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"])
    assert library.generativeqc_scf_defaults_query_v1

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("native SCF defaults used Python compatibility")

    for name in ("_compatibility_default_scf_controls", "_standard_models"):
        monkeypatch.setattr(_scf_convergence, name, forbidden)
