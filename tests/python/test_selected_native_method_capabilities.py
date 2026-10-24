"""Selected-library method ownership cannot be replaced by global discovery."""

from __future__ import annotations

import ctypes
import typing

import pytest
from generativeqc import Calculator, _native, method_capabilities
from generativeqc import _generated_methods as manifest
from generativeqc._method_native import capabilities


@pytest.fixture
def library() -> typing.Any:
    return _native.load_library(device="cpu")


class Function:
    """Allow ordinary injected Python callbacks to carry ctypes ABI metadata."""

    def __init__(self, action: typing.Any) -> None:
        self.action = action

    def __call__(self, *arguments: typing.Any) -> typing.Any:
        return self.action(*arguments)


def test_selected_registry_is_not_global_public_cache(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_capabilities.cache_clear()
    try:
        with monkeypatch.context() as discovery:
            discovery.setattr(_native, "load_library", lambda *args, **kwargs: library)
            cached = method_capabilities("rhf")
        assert cached.supports_batch and "forces" in cached.supported_properties

        def selected_capabilities(method_id: int, output: typing.Any) -> int:
            status = library.generativeqc_method_get_capabilities(method_id, output)
            record = ctypes.cast(
                output, ctypes.POINTER(_native.MethodCapabilitiesDescriptor)
            ).contents
            record.supports_batch = 0
            record.supported_properties = _native.PROPERTY_ENERGY
            return status

        class SelectedLibrary:
            generativeqc_method_get_capabilities = Function(selected_capabilities)

            def __getattr__(self, name: str) -> typing.Any:
                return getattr(library, name)

        selected = SelectedLibrary()

        def selected_loader(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            if kwargs.get("device") != "cpu":
                raise AssertionError(
                    "selected Calculator rediscovered a default library"
                )
            return selected

        monkeypatch.setattr(_native, "load_library", selected_loader)
        calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
        assert calculator._library is selected
        assert not calculator.capabilities.supports_batch
        assert calculator.capabilities.supported_properties == frozenset({"energy"})
        assert method_capabilities("rhf") is cached
    finally:
        method_capabilities.cache_clear()


@pytest.mark.parametrize("alias,method_id", tuple(manifest.METHOD_NAME_TO_ID.items()))
def test_aliases_query_existing_native_carrier(
    library: typing.Any, alias: str, method_id: int
) -> None:
    expected = _native.MethodCapabilitiesDescriptor(
        ctypes.sizeof(_native.MethodCapabilitiesDescriptor), _native.ABI_VERSION
    )
    _native.check(
        library,
        library.generativeqc_method_get_capabilities(method_id, ctypes.byref(expected)),
    )
    actual = capabilities(library, alias, method_id)
    assert actual.method == alias
    assert actual.available == bool(expected.available)
    assert actual.supports_batch == bool(expected.supports_batch)
    assert ("energy" in actual.supported_properties) == bool(
        expected.supported_properties & _native.PROPERTY_ENERGY
    )
    assert ("forces" in actual.supported_properties) == bool(
        expected.supported_properties & _native.PROPERTY_FORCES
    )


def test_old_named_protocol_keeps_numeric_compatibility(library: typing.Any) -> None:
    class OldLibrary:
        generativeqc_method_from_name = None

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(library, name)

    assert capabilities(OldLibrary(), "rhf", _native.METHOD_RHF) == capabilities(
        library, "rhf", _native.METHOD_RHF
    )


@pytest.mark.parametrize("stage", ("lookup", "capabilities"))
@pytest.mark.parametrize("consumer", ("query", "calculator"))
def test_selected_native_failure_never_rediscovers(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, stage: str, consumer: str
) -> None:
    class FailedLibrary:
        def __getattr__(self, name: str) -> typing.Any:
            if (
                name
                == {
                    "lookup": "generativeqc_method_from_name",
                    "capabilities": "generativeqc_method_get_capabilities",
                }[stage]
            ):
                return Function(lambda *args: _native.STATUS_NUMERICAL_FAILURE)
            return getattr(library, name)

    selected = FailedLibrary()

    def selected_loader(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if consumer == "calculator" and kwargs.get("device") == "cpu":
            return selected
        raise AssertionError("selected native failure tried another library")

    monkeypatch.setattr(_native, "load_library", selected_loader)
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 5"):
        if consumer == "calculator":
            Calculator(method="rhf", basis="sto-3g", device="cpu")
        else:
            capabilities(selected, "rhf", _native.METHOD_RHF)


@pytest.mark.parametrize(
    "field", ("lookup-id", "method", "abi_version", "struct_size", "family")
)
def test_inconsistent_native_publication_is_not_metadata_fallback(
    library: typing.Any, field: str
) -> None:
    def lookup(name: bytes, output: typing.Any) -> int:
        status = library.generativeqc_method_from_name(name, output)
        if field == "lookup-id":
            ctypes.cast(
                output, ctypes.POINTER(ctypes.c_int32)
            ).contents.value = _native.METHOD_UHF
        return status

    def query(method_id: int, output: typing.Any) -> int:
        status = library.generativeqc_method_get_capabilities(method_id, output)
        record = ctypes.cast(
            output, ctypes.POINTER(_native.MethodCapabilitiesDescriptor)
        ).contents
        if field == "method":
            record.method = _native.METHOD_UHF
        elif field == "abi_version":
            record.abi_version += 1
        elif field == "struct_size":
            record.struct_size -= 1
        elif field == "family":
            record.family = -1
        return status

    class InconsistentLibrary:
        generativeqc_method_from_name = Function(lookup)
        generativeqc_method_get_capabilities = Function(query)

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(library, name)

    with pytest.raises(RuntimeError, match="disagrees|inconsistent"):
        capabilities(InconsistentLibrary(), "rhf", _native.METHOD_RHF)
