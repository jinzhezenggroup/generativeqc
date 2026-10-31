"""Qualify actual native budget selection without descriptor fallback."""
import ctypes
import os
from typing import Any, NoReturn
import pytest

@pytest.fixture(autouse=True)
def require_native_budget_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc import precision_policy
    assert ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"]).generativeqc_precision_controls_query_v1
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("selected budget policy used Python compatibility")
    monkeypatch.setattr(precision_policy, "_compatibility_budget_fock_controls", forbidden)
