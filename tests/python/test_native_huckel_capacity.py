"""Shared Huckel topology/admission capacity and explicit compatibility behavior."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import (
    Atom,
    Calculator,
    InitialGuessSpec,
    Primitive,
    ResourceBudget,
    Shell,
    _native,
)
from generativeqc import initial_guess as guesses

H2 = (Atom(1, (0.0, 0.0, -0.7)), Atom(1, (0.0, 0.0, 0.7)))
SHELLS = (Shell(0, 0, (Primitive(1.0, 1.0),)), Shell(1, 0, (Primitive(1.0, 1.0),)))


@pytest.fixture(scope="module")
def library() -> Any:
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_huckel_numeric_capacity_v1")
    # Configure the actual ctypes signature through its production adapter.
    guesses._huckel_preparation_capacity(2, H2, SHELLS, "cartesian", 8, loaded)
    return loaded


@pytest.mark.parametrize("number", range(1, 19))
@pytest.mark.parametrize(
    "representation,dimension", (("cartesian", 35), ("spherical", 25))
)
@pytest.mark.parametrize("history", (1, 8, 16))
def test_native_topology_matches_independent_occupied_channel_counts(
    library: Any, number: int, representation: str, dimension: int, history: int
) -> None:
    """Unoccupied p and all polarization shells are absent from atomic solves."""
    atoms = (Atom(number, (0.0, 0.0, 0.0)),)
    shells = tuple(
        Shell(0, angular, (Primitive(1.0, 0.5), Primitive(0.5, 0.5)))
        for angular in range(5)
    )
    atomic = 1 if number <= 4 else 4
    expected = guesses._huckel_numeric_capacity(dimension, atomic, 1, 10, history)
    assert (
        guesses._huckel_preparation_capacity(
            dimension, atoms, shells, representation, history, library
        )
        == expected
    )


@pytest.mark.parametrize("loaded", (None, SimpleNamespace()))
def test_missing_query_retains_portable_planning_without_library_discovery(
    loaded: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("compatibility sizing loaded a native library")

    monkeypatch.setattr(_native, "load_library", forbidden)
    assert guesses._huckel_preparation_capacity(
        2, H2, SHELLS, "cartesian", 8, loaded
    ) == (guesses._huckel_numeric_capacity(2, 1, 2, 2, 8))


@pytest.mark.parametrize("status", (1, -1, 2))
def test_selected_native_rejection_never_retries_python(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(*args: Any) -> int:
        return status

    def forbidden(*args: Any) -> Any:
        pytest.fail("selected native sizing retried the portable polynomial")

    monkeypatch.setattr(guesses, "_huckel_numeric_capacity", forbidden)
    library = SimpleNamespace(generativeqc_resource_huckel_numeric_capacity_v1=reject)
    with pytest.raises(
        ValueError, match="native Huckel numeric capacity query rejected"
    ):
        guesses._huckel_preparation_capacity(2, H2, SHELLS, "cartesian", 8, library)


def test_raw_abi_rejects_invalid_shapes_transactionally(library: Any) -> None:
    """Exercise the exported boundary, including overflow before publication."""
    query = library.generativeqc_resource_huckel_numeric_capacity_v1
    numbers = (ctypes.c_int32 * 1)(2)
    owners = (ctypes.c_uint32 * 1)(0)
    angular = (ctypes.c_uint32 * 1)(1)
    primitives = (ctypes.c_uint64 * 1)(1)
    output = ctypes.c_uint64(12345)
    arguments = [
        numbers,
        1,
        owners,
        angular,
        primitives,
        1,
        _native.BASIS_CARTESIAN,
        8,
        ctypes.byref(output),
    ]
    assert query(*arguments) == 0
    # A He p-only target is a valid empty occupied atomic subspace.
    assert output.value == guesses._huckel_numeric_capacity(3, 0, 1, 1, 8)
    for index, bad in (
        (0, None),
        (1, 0),
        (1, 2**32),
        (2, None),
        (3, None),
        (4, None),
        (5, 0),
        (6, 12345),
        (7, 0),
        (7, 17),
        (8, None),
        (0, (ctypes.c_int32 * 1)(19)),
        (2, (ctypes.c_uint32 * 1)(1)),
        (3, (ctypes.c_uint32 * 1)(2**32 - 1)),
        (4, (ctypes.c_uint64 * 1)(0)),
        (4, (ctypes.c_uint64 * 1)(2**63 - 1)),
        (4, (ctypes.c_uint64 * 1)(2**64 - 1)),
    ):
        output.value = 12345
        invalid = arguments.copy()
        invalid[index] = bad
        assert query(*invalid) != 0
        assert output.value == 12345


@pytest.mark.parametrize("guess", ("auto", InitialGuessSpec("huckel")))
def test_runtime_plans_use_native_capacity_and_exact_host_budget(
    library: Any, monkeypatch: pytest.MonkeyPatch, guess: Any
) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("current runtime planner executed portable Huckel sizing")

    baseline = Calculator("rhf", "sto-3g", device="cpu", initial_guess=guess)
    systems = [H2, H2 + (Atom(2, (0.0, 0.0, 3.0)),)]
    before = baseline.estimate_resources(systems)
    monkeypatch.setattr(guesses, "_huckel_numeric_capacity", forbidden)
    after = baseline.estimate_resources(systems)
    assert before.to_dict() == after.to_dict()
    peak = after.peak_bytes["host"]
    bounded = Calculator(
        "rhf",
        "sto-3g",
        device="cpu",
        initial_guess=guess,
        resource_budget=ResourceBudget(host_bytes=peak),
    )
    with bounded.prepare_batch(systems) as batch:
        result = batch.execute(strict=True)
        assert all(item.converged for item in result.items)
        assert all(item.initial_guess["outcome"] == "used" for item in result.items)
        warm = batch.execute(strict=True)
        assert all(item.converged for item in warm.items)
    rejected = Calculator(
        "rhf",
        "sto-3g",
        device="cpu",
        initial_guess=guess,
        resource_budget=ResourceBudget(host_bytes=peak - 1),
    )
    assert rejected.estimate_resources(systems).status == "infeasible"
    with pytest.raises((ValueError, MemoryError)):
        rejected.prepare_batch(systems)


def test_actual_preparation_obeys_exact_native_numeric_cap(library: Any) -> None:
    """The shared query is the admission threshold, not just a planner estimate."""
    capacity = guesses._huckel_preparation_capacity(
        2, H2, SHELLS, "cartesian", 8, library
    )
    results = []
    for cap in (capacity, capacity - 1):
        calculator = Calculator(
            "rhf",
            SHELLS,
            device="cpu",
            initial_guess=InitialGuessSpec("huckel", maximum_numeric_bytes=cap),
        )
        result = calculator.singlepoint(H2)
        assert result.converged
        assert result.initial_guess["preparation_numeric_capacity"] == capacity
        results.append(result)
    assert results[0].initial_guess["outcome"] == "used"
    assert results[1].initial_guess["outcome"] == "budget_skipped"
    assert results[1].initial_guess["preliminary_fock_builds"] == 0
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-10)
    assert results[0].forces == pytest.approx(results[1].forces, abs=1e-9)
