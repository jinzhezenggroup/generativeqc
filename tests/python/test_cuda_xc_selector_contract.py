"""Exercise the actual selector/geometry boundary without creating a GPU owner."""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc import ks
from vibeqc._stationary_cuda import _CudaSources
from vibeqc_compiler.dft.grid import GridSpec
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc.geometry_cuda import _functional_code


class _View(ct.Structure):
    _fields_ = [("npoint", ct.c_size_t), ("stream", ct.c_void_p)]


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("LDA_XC_PW", 0),
        ("PBE", 1),
        ("R2SCAN", 2),
        ("PBE0", 1),
        ("B3LYP", 3),
        ("WB97M-V", 4),
    ],
)
def test_curated_selector_reaches_both_geometry_consumers(
    method: str, spin: str, expected: int
) -> None:
    code = ks._native_semilocal_family(resolve_method(method, spin=spin))
    assert type(code) is int
    assert code == expected
    assert _functional_code(code, None) == expected

    calls = []
    jets = []

    def density_jets(count: int) -> int:
        jets.append(count)
        return 123

    owner = SimpleNamespace(
        device=0,
        borrowed_streams=set(),
        handle=None,
        profile_device=False,
        _call=lambda *args: calls.append(args),
    )
    task = SimpleNamespace(
        view=_View(2, 17),
        _owner=SimpleNamespace(device_id=0),
        density_jets=density_jets,
    )
    _CudaSources.geometry(
        owner,
        task,
        np.array([0, 1], dtype=np.int64),
        np.ones(2),
        np.ones(2),
        functional=code,
    )
    assert jets == [1 if expected == 0 else 4]
    assert len(calls) == 1 and calls[0][0] == "stationary_geometry_enqueue"
    assert owner.borrowed_streams == {17}

    calls.clear()
    owner.profile_device = True
    _CudaSources.geometry(
        owner,
        task,
        np.array([0, 1], dtype=np.int64),
        np.ones(2),
        np.ones(2),
        functional=code,
    )
    assert len(calls) == 1 and calls[0][0] == "stationary_geometry"


def test_resident_nonlocal_seed_handoff_uses_device_pointer_and_stride() -> None:
    calls = []
    jets = []

    def density_jets(count: int) -> int:
        jets.append(count)
        return 456

    owner = SimpleNamespace(
        device=0,
        borrowed_streams=set(),
        handle=None,
        profile_device=False,
        _call=lambda *args: calls.append(args),
    )
    task = SimpleNamespace(
        view=_View(2, 19),
        _owner=SimpleNamespace(device_id=0),
        density_jets=density_jets,
    )
    _CudaSources.geometry_external_device(
        owner,
        task,
        np.array([0, 1], dtype=np.int64),
        np.ones(2),
        np.ones(2),
        0x1234,
        17,
        5,
    )
    assert jets == [4]
    assert len(calls) == 1
    assert calls[0][0] == "stationary_geometry_external_device_enqueue"
    assert calls[0][-3].value == 0x1234
    assert calls[0][-2:] == (17, 5)
    assert owner.borrowed_streams == {19}

    calls.clear()
    owner.profile_device = True
    _CudaSources.geometry_external_device(
        owner,
        task,
        np.array([0, 1], dtype=np.int64),
        np.ones(2),
        np.ones(2),
        ct.c_void_p(0x5678),
        17,
        7,
    )
    assert calls[0][0] == "stationary_geometry_external_device"

    with pytest.raises(ValueError, match="exceeds"):
        _CudaSources.geometry_external_device(
            owner,
            task,
            np.array([0, 1], dtype=np.int64),
            np.ones(2),
            np.ones(2),
            0x1234,
            8,
            7,
        )


def test_pbe_d4_selector_also_returns_builtin_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ks, "_is_pbe_d4_composition", lambda _: True)
    code = ks._native_semilocal_family(object())
    assert type(code) is int and code == 1
    assert _functional_code(code, None) == 1


@pytest.mark.parametrize("invalid", (True, 1.0, "1", -1))
def test_geometry_code_validation_is_not_weakened(invalid: object) -> None:
    with pytest.raises(ValueError, match="registered semilocal"):
        _functional_code(invalid, None)


def test_resolved_ks_payload_preserves_instance_tile_points() -> None:
    options = ks.resolve_ks_options(
        "pbe-rks", ks.KsOptions(grid=GridSpec(), tile_points=17)
    )
    assert options.to_payload()["tile_points"] == 17
