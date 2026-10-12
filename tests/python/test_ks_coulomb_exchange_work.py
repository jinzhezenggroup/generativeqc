"""Narrow CPU value-provider observations, separate from precision evidence."""

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import Calculator, KsOptions, _native
from generativeqc.ks_diagnostics import _coulomb_exchange_work
from generativeqc_compiler.dft.grid import GridSpec

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("family", ("lda", "pbe", "pbe0"))
def test_singlepoint_observes_selected_providers(family: str, spins: int) -> None:
    """Even an empty beta spin counts if the hybrid's K contraction runs."""
    calculator = Calculator(
        method=f"{family}-{'rks' if spins == 1 else 'uks'}",
        device="cpu",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
        ),
    )
    result = calculator.singlepoint(
        ATOMS, charge=spins - 1, multiplicity=spins, properties=("energy",)
    )
    assert result.converged
    diagnostic = result.ks_diagnostic
    assert diagnostic is not None
    assert diagnostic.coulomb_exchange_work == {
        "complete": True,
        "coulomb_applications": diagnostic.fock_builds,
        "exchange_applications": spins * diagnostic.fock_builds
        if family == "pbe0"
        else 0,
    }
    assert not result.precision["complete"]
    assert not result.precision["operator_inventory_complete"]


def test_batch_query_validation_and_invalidation() -> None:
    """New queries share KS-record invalidation, and reject before copying."""
    calculator = Calculator(method="pbe-rks", device="cpu", initial_guess=None)
    with calculator.prepare_batch([ATOMS]) as batch:
        query = batch._library.generativeqc_batch_get_ks_coulomb_exchange_work_v1
        record = _native.KsCoulombExchangeWorkDescriptor(
            ctypes.sizeof(_native.KsCoulombExchangeWorkDescriptor), _native.ABI_VERSION
        )
        record.coulomb_applications = 19
        before = bytes(record)
        assert (
            query(batch._batch, 0, ctypes.byref(record))
            == _native.STATUS_NOT_IMPLEMENTED
        )
        assert bytes(record) == before
        assert query(batch._batch, 1, None) == _native.STATUS_INVALID_ARGUMENT
        result = batch.execute(properties=("energy",), strict=True).items[0]
        assert query(batch._batch, 0, None) == _native.STATUS_SUCCESS
        assert query(batch._batch, 0, ctypes.byref(record)) == _native.STATUS_SUCCESS
        assert (
            record.complete
            and record.coulomb_applications == result.ks_diagnostic.fock_builds
        )
        assert record.exchange_applications == 0
        record.struct_size = 0
        before = bytes(record)
        assert (
            query(batch._batch, 0, ctypes.byref(record)) == _native.STATUS_ABI_MISMATCH
        )
        assert bytes(record) == before
        outputs = (_native.BatchItemResultDescriptor * 1)()
        outputs[0].struct_size = ctypes.sizeof(_native.BatchItemResultDescriptor)
        outputs[0].abi_version = _native.ABI_VERSION
        assert (
            batch._library.generativeqc_batch_execute(batch._batch, None, 0, outputs, 0)
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert query(batch._batch, 0, None) == _native.STATUS_NOT_IMPLEMENTED


@pytest.mark.parametrize("complete", (False, True))
@pytest.mark.parametrize("index", (None, 0))
def test_decoder_never_certifies_missing_work(
    complete: bool, index: int | None
) -> None:
    """Partial/default native counters and older libraries are not zero-work proof."""

    def query(*arguments: Any) -> int:
        record = ctypes.cast(
            arguments[-1], ctypes.POINTER(_native.KsCoulombExchangeWorkDescriptor)
        ).contents
        record.complete = int(complete)
        record.coulomb_applications = 17
        record.exchange_applications = 3
        return _native.STATUS_SUCCESS

    name = (
        "generativeqc_calculation_get_ks_coulomb_exchange_work_v1"
        if index is None
        else "generativeqc_batch_get_ks_coulomb_exchange_work_v1"
    )
    assert _coulomb_exchange_work(SimpleNamespace(), None, index) is None
    assert _coulomb_exchange_work(SimpleNamespace(**{name: query}), None, index) == {
        "complete": complete,
        "coulomb_applications": 17 if complete else None,
        "exchange_applications": 3 if complete else None,
    }
