"""Require actual native capacity and snapshot transport during qualification."""
import ctypes
import os
from typing import Any, NoReturn
import pytest

@pytest.fixture(autouse=True)
def require_native_seed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc import _precision_batch
    library = ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"])
    assert library.generativeqc_batch_capture_warm_seeds_v1
    assert library.generativeqc_batch_query_warm_seed_capacity_v1
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("native seed staging used Python descriptor compatibility")
    monkeypatch.setattr(_precision_batch, "_compatibility_seed_capacity", forbidden)
    monkeypatch.setattr(_precision_batch._SeedLease, "_capture_compatibility", forbidden)
    monkeypatch.setattr(_precision_batch, "_restore_states", forbidden)
