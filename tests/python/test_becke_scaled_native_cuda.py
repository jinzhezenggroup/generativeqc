"""Opt-in same-owner numerical and alias gates, always in finite Slurm jobs."""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace

import pytest
import test_stationary_becke_phased_cuda as retained
from test_stationary_becke_phased_cuda import native

__all__ = ["native"]


@pytest.mark.parametrize(
    "atoms,implicit,selection,external,primitive,serial",
    [
        (48, False, "full", False, False, False),
        (48, False, "subset", False, False, False),
        (48, False, "empty", True, False, False),
        (48, False, "full", False, False, True),
        (96, True, "full", False, False, False),
        (96, False, "subset", True, False, False),
        (128, False, "full", False, False, False),
        (48, False, "full", False, 2, False),
        (96, False, "subset", True, True, False),
    ],
)
@pytest.mark.parametrize("selection_mode", ["explicit", "automatic"])
def test_binary_native_sources_fallback_alias_and_invalid_publication(
    native: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    atoms: int,
    implicit: bool,
    selection: str,
    external: bool,
    primitive: bool | int,
    serial: bool,
    selection_mode: str,
) -> None:
    """Reuse complete source-slot guards and authenticate the actual native route.

    The retained test compares identical synthetic AO/XC panels on bounded and
    phased owners, including moved coordinates, zero rows, the point tail and
    rejection without publication. These remain routing/lifetime tests, not a
    complete molecular oracle or a timing result.
    """
    if selection_mode == "explicit":
        monkeypatch.setenv("GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS", "1")
    else:
        monkeypatch.delenv(
            "GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS", raising=False
        )
    metrics = getattr(native.library, "stationary_becke_binary_scaled_metrics_v1", None)
    if metrics is None:
        pytest.fail("binary-scaled qualification requires the new native metrics ABI")
    metrics.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t]
    observed: list[tuple[int, ...]] = []

    def call(name: str, *arguments: object) -> None:
        native.call(name, *arguments)
        if name == "stationary_configure_phased_becke_v1" and serial:
            native.call("stationary_configure_becke_normalize_v1", arguments[0], 0)
        if name == "stationary_finish":
            values = (ct.c_uint64 * 6)()
            assert metrics(arguments[0], values, 6) == 0
            observed.append(tuple(values))
            assert values[0] == 1
            selected = (
                values[4] > 0
                and atoms <= 128
                and not primitive
                and (selection_mode == "explicit" or atoms >= 96)
            )
            assert values[1] == selected
            if selected:
                assert values[2] > 0 and values[3] > 0
                assert values[5] == (not serial)
            else:
                assert values[2] == values[3] == 0

    wrapped = SimpleNamespace(cupy=native.cupy, library=native.library, call=call)
    retained.test_shared_owner_phases_preserve_sources_and_work(
        wrapped, atoms, implicit, selection, external, primitive, False
    )
    assert observed and any(values[4] > 0 for values in observed)
    assert any(values[4] == 0 for values in observed)
