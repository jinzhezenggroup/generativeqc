"""Qualification-only rejection of the retired topology reconstruction path."""

import ctypes
import os
from typing import Any, NoReturn

import pytest


@pytest.fixture(autouse=True)
def require_native_seed_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual audit/seed lifetimes must consume the native prepared metadata."""
    from generativeqc import _precision_batch

    library = ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"])
    assert library.generativeqc_batch_query_warm_seed_capacity_v1

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("prepared seed staging used Python topology")

    monkeypatch.setattr(_precision_batch, "_compatibility_seed_capacity", forbidden)
