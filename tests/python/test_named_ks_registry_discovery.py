"""Installed named registry selection is independent of compiler graph resolution."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc import _native, ks, method_capabilities
from generativeqc._generated_methods import COMPOSITE_METHOD_ALIASES, METHOD_NAME_TO_ID
from generativeqc._named_ks import registry_carrier

from tools.generate_cli_method_catalog import rows, selector_aliases

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def uncached_registry() -> Iterator[None]:
    """Exercise registry discovery, rather than a cached presentation record."""
    method_capabilities.cache_clear()
    yield
    method_capabilities.cache_clear()


@pytest.fixture
def library() -> ctypes.CDLL:
    return _native.load_library(device="cpu")


def test_native_catalog_preserves_compiler_carrier_projection(
    library: ctypes.CDLL,
) -> None:
    catalog = rows()
    by_name = {item.name: item for item in catalog}
    for name, row in (
        *by_name.items(),
        *((alias, by_name[target]) for alias, target in selector_aliases(catalog)),
    ):
        if (
            (not row.cpu and not row.cuda)
            or (row.bound_basis and name.startswith("libxc"))
            or ks.parse_automatic_libxc_selector(name)
        ):
            assert registry_carrier(library, name) is None, name
            continue
        if name in METHOD_NAME_TO_ID:
            continue
        if name in COMPOSITE_METHOD_ALIASES:
            _, spin = COMPOSITE_METHOD_ALIASES[name]
            carrier = "r2scan-uks" if spin == "polarized" else "r2scan-rks"
        else:
            carrier = ks.native_dft_carrier(name)
        assert registry_carrier(library, name) == METHOD_NAME_TO_ID[carrier], name


@pytest.mark.parametrize(
    "selector, carrier",
    [
        ("PBEH-RKS", "pbe-rks"),
        ("B3LYP-UKS", "pbe-uks"),
        ("Libxc-RKS: PBE0 ", "pbe-rks"),
        ("b3lyp-d3(bj)-rks", "pbe-rks"),
        ("pbe-d4(bj-eeq-atm)-rks", "pbe-d4-rks"),
        ("r2scan-d4(bj-eeq-atm)-uks", "r2scan-uks"),
        ("r2scan-3c", "r2scan-rks"),
        ("r2scan-3c-uks", "r2scan-uks"),
    ],
)
def test_public_query_does_not_resolve_a_compiler_graph(
    monkeypatch: pytest.MonkeyPatch, selector: str, carrier: str
) -> None:
    expected = method_capabilities(carrier)

    def forbidden(*_args: Any) -> None:
        pytest.fail("named registry hit entered compiler graph resolution")

    monkeypatch.setattr(ks, "native_dft_carrier", forbidden)
    monkeypatch.setattr(ks, "resolve_method", forbidden)
    assert method_capabilities(selector) == replace(expected, method=selector.lower())


def test_catalog_miss_keeps_unsupported_composite_and_unknown_name_contract(
    library: ctypes.CDLL,
) -> None:
    name = "libxc:r2scan-3c"
    assert registry_carrier(library, name) is None
    with pytest.raises(NotImplementedError, match="electronic projection"):
        method_capabilities(name)
    with pytest.raises(ValueError, match="unknown method"):
        method_capabilities("no-such-functional-rks")


@pytest.mark.parametrize("failure", ["missing", "refusal", "non-dft-carrier"])
def test_native_error_is_not_a_catalog_miss(
    library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    class SelectedLibrary:
        def __getattr__(self, name: str) -> Any:
            if name != "generativeqc_ks_catalog_registry_carrier_v1":
                return getattr(library, name)
            if failure == "missing":
                raise AttributeError(name)

            def query(_name: bytes, _length: int, carrier: Any) -> int:
                if failure == "refusal":
                    return _native.STATUS_NUMERICAL_FAILURE
                ctypes.cast(carrier, ctypes.POINTER(ctypes.c_int32))[0] = (
                    _native.METHOD_RHF
                )
                return _native.STATUS_SUCCESS

            return query

    def forbidden(*_args: Any) -> None:
        pytest.fail("native registry error entered compiler graph resolution")

    selected = SelectedLibrary()
    monkeypatch.setattr(_native, "load_library", lambda: selected)
    monkeypatch.setattr(ks, "native_dft_carrier", forbidden)
    with pytest.raises(
        RuntimeError, match="error 5|lacks the named KS|invalid carrier"
    ):
        method_capabilities("pbeh-rks")


def test_native_counted_selector_rejects_embedded_nul_without_publication(
    library: ctypes.CDLL,
) -> None:
    assert registry_carrier(library, "pbeh-rks") is not None
    query = library.generativeqc_ks_catalog_registry_carrier_v1
    name = b"pbeh-rks\0trailing"
    carrier = ctypes.c_int32(-1)
    assert (
        query(name, len(name), ctypes.byref(carrier)) == _native.STATUS_INVALID_ARGUMENT
    )
    assert carrier.value == -1
