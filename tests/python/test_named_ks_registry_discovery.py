"""Installed named registry selection is independent of compiler graph resolution."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc import _native, ks, method_capabilities
from generativeqc._generated_methods import METHOD_NAME_TO_ID
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
        if (not row.cpu and not row.cuda) or row.dispersion[0]:
            assert registry_carrier(library, name) is None, name
            continue
        # Automatic Libxc resolution and the stable ABI registry are different
        # entry points; compare only names eligible for the new consumer.
        if name in METHOD_NAME_TO_ID or ks.parse_automatic_libxc_selector(name):
            continue
        assert (
            registry_carrier(library, name)
            == METHOD_NAME_TO_ID[ks.native_dft_carrier(name)]
        ), name


@pytest.mark.parametrize(
    "selector, carrier",
    [
        ("PBEH-RKS", "pbe-rks"),
        ("B3LYP-UKS", "pbe-uks"),
        ("Libxc-RKS: PBE0 ", "pbe-rks"),
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


def test_catalog_miss_keeps_the_correction_and_unknown_name_contract(
    library: ctypes.CDLL,
) -> None:
    name = "r2scan-d4(bj-eeq-atm)-uks"
    assert registry_carrier(library, name) is None
    assert method_capabilities(name) == replace(
        method_capabilities("r2scan-uks"), method=name
    )
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
