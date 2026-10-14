"""Host accounting and ctypes ABI behavior of the opt-in endpoint profiler."""

import ctypes

import pytest

from benchmarks.profile_dft_endpoint import WallLedger, _NativeCall


def test_wall_ledger_subtracts_immediate_children_without_double_counting() -> None:
    ticks = iter((0, 2, 3, 5, 7, 9, 10, 12))
    ledger = WallLedger(lambda: next(ticks))
    with ledger.scope("endpoint"):
        with ledger.scope("scf"), ledger.scope("xc"):
            pass
        with ledger.scope("force"):
            pass
    actual = ledger.snapshot()
    assert actual["exclusive_wall_seconds"] == {
        "endpoint": 6e-9,
        "scf": 3e-9,
        "xc": 2e-9,
        "force": 1e-9,
    }
    assert actual["root_seconds"] == 12e-9
    assert actual["reconciliation_error_seconds"] == 0


def test_wall_ledger_retains_exceptions_and_refuses_incomplete_intervals() -> None:
    ticks = iter((0, 6, 10, 11))
    ledger = WallLedger(lambda: next(ticks))
    with (
        pytest.raises(RuntimeError, match="scientific failure"),
        ledger.scope("failure"),
    ):
        raise RuntimeError("scientific failure")
    assert ledger.snapshot()["root_seconds"] == 6e-9
    with ledger.scope("pending"), pytest.raises(ValueError, match="incomplete"):
        ledger.snapshot()


def test_native_observer_forwards_late_abi_signature_configuration() -> None:
    callback = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(lambda value: value + 3)
    observations = []

    def observe(function: object, *args: object) -> object:
        observations.append(args)
        return function(*args)

    proxy = _NativeCall(callback, observe)
    proxy.argtypes = [ctypes.c_int64]
    proxy.restype = ctypes.c_int64
    assert callback.argtypes == [ctypes.c_int64]
    assert callback.restype is ctypes.c_int64
    assert proxy(2**40) == 2**40 + 3
    assert observations == [(2**40,)]
