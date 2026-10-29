"""Pure SDK intent/optional transport tests; never loads a native library."""

from __future__ import annotations

import ctypes
import dataclasses
import importlib
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sdk_intent_package(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    directory = ROOT / "python/generativeqc"
    package = types.ModuleType("_intent_sdk_contract_probe")
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    yield package.__name__
    for name in list(sys.modules):
        if name.startswith(package.__name__ + "."):
            sys.modules.pop(name, None)


def test_immutable_intent_available(sdk_intent_package: str) -> None:
    spec = importlib.util.find_spec(sdk_intent_package + ".provider_initialization")
    assert spec is not None, "immutable SDK intent is unavailable"


@pytest.fixture
def sdk_intent_modules(
    sdk_intent_package: str,
) -> Iterator[tuple[types.ModuleType, types.ModuleType]]:
    intent = importlib.import_module(sdk_intent_package + ".provider_initialization")
    adapter = importlib.import_module(
        sdk_intent_package + "._provider_initialization_native"
    )
    yield intent, adapter


class Function:
    def __init__(self, function: Callable[..., int]) -> None:
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: Any) -> int:
        return self.function(*args)


def test_immutable_sdk_intent(
    sdk_intent_modules: tuple[types.ModuleType, types.ModuleType],
) -> None:
    module, _ = sdk_intent_modules
    request = module.ProviderInitializationIntent(512 << 20)
    assert request.allowance_bytes == 512 << 20
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.allowance_bytes = 7
    for value in [0, -1, True, 1.5, "512", 1 << 63]:
        with pytest.raises((TypeError, ValueError)):
            module.ProviderInitializationIntent(value)
    with pytest.raises(TypeError):
        module.ProviderInitializationIntent(1, context_id=41)


def test_absent_intent_never_resolves_new_symbols(
    sdk_intent_modules: tuple[types.ModuleType, types.ModuleType],
) -> None:
    _, adapter = sdk_intent_modules
    calls = []

    class LegacyLibrary:
        generativeqc_calculation_prepare = Function(
            lambda *args: calls.append(args) or 0
        )

        def __getattr__(self, name: str) -> NoReturn:
            raise AssertionError("legacy request inspected a new symbol: " + name)

    assert adapter.prepare(LegacyLibrary(), 1, 2, 3, 4, None) == 0
    assert calls == [(1, 2, 3, 4)]


def test_explicit_intent_requires_optional_native_transport(
    sdk_intent_modules: tuple[types.ModuleType, types.ModuleType],
) -> None:
    module, adapter = sdk_intent_modules
    request = module.ProviderInitializationIntent(512 << 20)

    class OldLibrary:
        def __getattr__(self, name: str) -> NoReturn:
            raise AttributeError(name)

    with pytest.raises(NotImplementedError):
        adapter.prepare(OldLibrary(), 1, 2, 3, 4, request)


def test_python_transport_owns_a_finite_copy(
    sdk_intent_modules: tuple[types.ModuleType, types.ModuleType],
) -> None:
    module, adapter = sdk_intent_modules
    captured = []

    def capture(
        context: int, system: int, method: int, intent: Any, output: int
    ) -> int:
        value = ctypes.cast(
            intent, ctypes.POINTER(adapter.ProviderInitializationIntentDescriptor)
        ).contents
        captured.append(
            (
                context,
                system,
                method,
                output,
                value.struct_size,
                value.abi_version,
                value.mode,
                value.reserved,
                value.allowance_bytes,
            )
        )
        value.allowance_bytes = 7
        return 0

    class NativeLibrary:
        generativeqc_provider_initialization_intent_version = Function(lambda: 1)
        generativeqc_provider_initialization_capabilities_v1 = Function(lambda: 3)
        generativeqc_calculation_prepare_with_provider_initialization_v1 = Function(
            capture
        )

    request = module.ProviderInitializationIntent(512 << 20)
    assert adapter.prepare(NativeLibrary(), 1, 2, 3, 4, request) == 0
    assert captured == [(1, 2, 3, 4, 24, 1, 1, 0, 512 << 20)]
    assert request.allowance_bytes == 512 << 20
