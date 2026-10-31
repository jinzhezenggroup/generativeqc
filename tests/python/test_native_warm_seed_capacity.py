"""Actual prepared topology bounds and authoritative native failure handling."""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc import _precision_batch as precision
from generativeqc import _warm_seed_native as native

SYSTEMS = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("He", (0.0, 0.0, 0.0))],
]


def forbidden(*args: Any, **kwargs: Any) -> None:
    """Fail if a current prepared owner re-enters Python shape reconstruction."""
    raise AssertionError("native capacity retried Python topology")


@pytest.mark.parametrize("method", ("rhf", "uhf", "pbe-rks", "pbe-uks"))
def test_prepared_capacity_covers_cold_and_retained_ragged_seeds(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = (
        {"ks_options": KsOptions(grid=GridSpec(8, 4, 8))} if "pbe" in method else {}
    )
    calculator = Calculator(method=method, initial_guess=None, **options)
    if not hasattr(
        calculator._library, "generativeqc_batch_query_warm_seed_capacity_v1"
    ):
        pytest.skip("requires native warm-seed capacity protocol")
    monkeypatch.setattr(precision, "_compatibility_seed_capacity", forbidden)
    spin_count = 2 if method in ("uhf", "pbe-uks") else 1
    density = 8 * spin_count * (2**2 + 1**2)
    expected = (density + 24 * 3, 8 * spin_count * 2**2)
    with calculator.prepare_batch(SYSTEMS) as owner:
        assert precision._seed_capacity(owner) == expected
        # Counts bound export before any retained density exists. A deficient
        # numeric staging budget must reject before materializing a seed.
        with pytest.raises(MemoryError, match="maximum_seed_bytes"):
            precision._SeedLease(owner, expected[0] - 1)
        result = owner.execute(properties=("energy",), strict=True)
        assert all(item.converged for item in result.items)
        assert precision._seed_capacity(owner) == expected
        lease = precision._SeedLease(owner, expected[0])
        assert lease.numeric_bytes == expected[0]
        assert lease.largest_density_bytes == expected[1]
        owner.clear_warm_starts()
        assert precision._seed_capacity(owner) == expected
        lease.restore(owner, updates=False)
        assert precision._SeedLease(owner, expected[0]).numeric_bytes == expected[0]
    calculator.clear_cache()


@pytest.mark.parametrize("representation", ("spherical", "cartesian"))
def test_prepared_representation_matches_independent_ao_count(
    representation: str,
) -> None:
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("Ne", (0.0, 0.0, 0.0))]
    reference = gto.M(
        atom=atoms, basis="def2-svp", cart=representation == "cartesian", verbose=0
    )
    calculator = Calculator(basis="def2-svp", basis_representation=representation)
    if not hasattr(
        calculator._library, "generativeqc_batch_query_warm_seed_capacity_v1"
    ):
        pytest.skip("requires native warm-seed capacity protocol")
    density_bytes = 8 * reference.nao_nr() ** 2
    with calculator.prepare_batch([atoms]) as owner:
        assert native.seed_capacity(owner) == (density_bytes + 24, density_bytes)
    calculator.clear_cache()


def test_missing_symbol_is_the_only_compatibility_boundary() -> None:
    assert native.seed_capacity(SimpleNamespace(_library=SimpleNamespace())) is None


def test_selected_native_refusal_never_retries_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any) -> int:
        return _native.STATUS_NOT_IMPLEMENTED

    batch = SimpleNamespace(
        _library=SimpleNamespace(
            generativeqc_batch_query_warm_seed_capacity_v1=refuse,
            generativeqc_status_message=lambda status: b"unsupported owner",
        ),
        _batch=None,
        _context=None,
    )
    monkeypatch.setattr(precision, "_compatibility_seed_capacity", forbidden)
    with pytest.raises(NotImplementedError):
        precision._seed_capacity(batch)


def test_malformed_native_receipt_fails_closed() -> None:
    def query(batch: Any, pointer: Any) -> int:
        output = ct.cast(pointer, ct.POINTER(native.Capacity)).contents
        output.total_bytes = 8
        return 0

    batch = SimpleNamespace(
        _library=SimpleNamespace(generativeqc_batch_query_warm_seed_capacity_v1=query),
        _batch=None,
        _context=None,
    )
    with pytest.raises(RuntimeError, match="invalid receipt"):
        native.seed_capacity(batch)
