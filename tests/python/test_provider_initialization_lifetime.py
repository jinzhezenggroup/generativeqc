"""SDK failure and handle-lifetime contracts, without loading a native library."""

from __future__ import annotations

import ctypes
import importlib
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def lifetime(monkeypatch: pytest.MonkeyPatch) -> Any:
    package = types.ModuleType("_provider_lifetime_contract")
    package.__path__ = [str(ROOT / "python/generativeqc")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    # These tests exercise the Python ownership transaction. The public C ABI
    # calls are supplied below, so importing the native library loader is unnecessary.
    monkeypatch.setitem(
        sys.modules,
        package.__name__ + "._native",
        types.ModuleType(package.__name__ + "._native"),
    )
    module = importlib.import_module(
        package.__name__ + "._provider_initialization_lifetime"
    )
    yield module
    for name in list(sys.modules):
        if name.startswith(package.__name__ + "."):
            sys.modules.pop(name, None)


class Function:
    def __init__(self, function: Any) -> None:
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: Any) -> Any:
        return self.function(*args)


class Library:
    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.calls: list[tuple[str, int | None]] = []
        self.generativeqc_context_close_provider_initialization_v1 = Function(
            lambda handle: self.record("close", handle) or self.status
        )
        self.generativeqc_context_get_last_detail = Function(
            lambda handle: b"operation=solver/release;state=returned;raw_status=7"
        )
        self.generativeqc_calculation_destroy = Function(
            lambda handle: self.record("calculation", handle)
        )
        self.generativeqc_system_destroy = Function(
            lambda handle: self.record("system", handle)
        )
        self.generativeqc_context_destroy = Function(
            lambda handle: self.record("context", handle)
        )

    def record(self, name: str, handle: Any) -> None:
        self.calls.append((name, handle.value))


def populated(lifetime: Any, library: Library, owner: Any = None) -> Any:
    transaction = lifetime.begin(library, owner)
    transaction.context = ctypes.c_void_p(11)
    transaction.system = ctypes.c_void_p(22)
    transaction.auxiliary_system = ctypes.c_void_p(33)
    transaction.calculation = ctypes.c_void_p(44)
    return transaction


def test_failed_close_retains_handles_and_native_detail(lifetime: Any) -> None:
    library = Library(3)
    transaction = populated(lifetime, library)
    lease = object()
    transaction.budget_lease = lease
    with pytest.raises(lifetime.ProviderInitializationCloseUnavailable) as raised:
        transaction.close()
    assert raised.value.status == 3
    assert (
        raised.value.detail_bytes
        == b"operation=solver/release;state=returned;raw_status=7"
    )
    assert lifetime.has_pending_owner()
    assert transaction.calculation.value == 44 and transaction.system.value == 22
    assert transaction.auxiliary_system.value == 33 and transaction.context.value == 11
    assert transaction.budget_lease is lease
    assert library.calls == [("close", 11)]


def test_primary_error_survives_secondary_close_failure(lifetime: Any) -> None:
    primary = ValueError("original scientific error")
    transaction = populated(lifetime, Library(3))
    transaction.close_preserving_primary(primary)
    assert str(primary) == "original scientific error"
    assert isinstance(
        primary.provider_initialization_close_error,
        lifetime.ProviderInitializationCloseUnavailable,
    )
    assert lifetime.has_pending_owner()
    if hasattr(BaseException, "add_note"):
        assert any(
            "Secondary provider close failure" in note for note in primary.__notes__
        )


def test_success_closes_dependents_before_releasing_transaction(lifetime: Any) -> None:
    library = Library()
    transaction = populated(lifetime, library)

    class Lease:
        def close(self) -> None:
            assert lifetime.has_pending_owner()
            assert transaction.context.value is None
            library.calls.append(("budget", None))

    transaction.budget_lease = Lease()
    transaction.close()
    assert library.calls == [
        ("close", 11),
        ("calculation", 44),
        ("system", 22),
        ("system", 33),
        ("context", 11),
        ("budget", None),
    ]
    assert not lifetime.has_pending_owner()


def test_retained_owner_blocks_a_second_library(lifetime: Any) -> None:
    first, second = Library(3), Library()
    transaction = populated(lifetime, first)
    with pytest.raises(RuntimeError, match="still retained"):
        lifetime.begin(second, None)
    assert second.calls == []
    with pytest.raises(lifetime.ProviderInitializationCloseUnavailable):
        transaction.close()
    with pytest.raises(RuntimeError, match="still retained"):
        lifetime.begin(second, None)


def test_failure_before_context_creation_releases_empty_transaction(
    lifetime: Any,
) -> None:
    library = Library()
    transaction = lifetime.begin(library, None)
    transaction.close_preserving_primary(ValueError("context was never created"))
    assert library.calls == []
    assert not lifetime.has_pending_owner()


def test_shared_context_survives_confirmed_provider_close(lifetime: Any) -> None:
    owner = types.SimpleNamespace(
        lock=threading.RLock(), _provider_initialization_retained=False
    )
    library = Library()
    transaction = populated(lifetime, library, owner)
    assert owner._provider_initialization_retained
    transaction.close()
    assert ("context", 11) not in library.calls
    assert transaction.context.value == 11
    assert not owner._provider_initialization_retained
    assert not lifetime.has_pending_owner()


@pytest.mark.parametrize("phase", ["before_create", "during_create", "after_create"])
def test_public_close_cannot_retire_an_executing_transaction(
    lifetime: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    admitted, resume = threading.Event(), threading.Event()
    library = Library()
    failures: list[BaseException] = []

    def pause() -> None:
        admitted.set()
        assert resume.wait(5), "public-close interleaving did not resume"

    def create(descriptor: Any, output: Any) -> int:
        if phase == "during_create":
            pause()
        output._obj.value = 11
        library.calls.append(("create", 11))
        return 0

    library.generativeqc_context_create = Function(create)
    monkeypatch.setattr(
        lifetime._native, "check", lambda library, status: None, raising=False
    )

    def execute() -> None:
        transaction = lifetime.begin(library, None)
        try:
            if phase == "before_create":
                pause()
            transaction.open_context(ctypes.c_int())
            if phase == "after_create":
                pause()
        except BaseException as error:  # noqa: BLE001 - report worker failures in the asserting thread
            failures.append(error)
        finally:
            transaction.close_preserving_primary(None)

    worker = threading.Thread(target=execute)
    worker.start()
    try:
        assert admitted.wait(5)
        with pytest.raises(RuntimeError, match="still executing"):
            lifetime.close_retained_owner()
        assert lifetime.has_pending_owner()
        with pytest.raises(RuntimeError, match="still retained"):
            lifetime.begin(Library(), None)
        assert library.calls == ([] if phase != "after_create" else [("create", 11)])
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert library.calls == [("create", 11), ("close", 11), ("context", 11)]
    assert not lifetime.has_pending_owner()
