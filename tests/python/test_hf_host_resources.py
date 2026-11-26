"""CPU HF sizing has one live native owner and an offline comparison."""

import ctypes
import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import (
    Calculator,
    ResourceBudget,
    _native,
    estimate_hf_resources,
    resources_hf,
)


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Require the source-matched bridge rather than silently skipping it."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_hf_cpu_host_v1")
    assert hasattr(loaded, "generativeqc_resource_hf_cpu_fleet_v1")
    return loaded


@pytest.fixture(scope="module")
def phase_query(library: typing.Any) -> typing.Any:
    """Exercise the standalone leaf ABI directly, not the fleet fallback."""
    operation = library.generativeqc_resource_hf_cpu_host_v1
    pointer = ctypes.POINTER(ctypes.c_uint64)
    operation.argtypes = [pointer, ctypes.c_size_t, pointer, ctypes.c_size_t]
    operation.restype = ctypes.c_int
    return operation


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("history", (0, 1, 8, 64))
@pytest.mark.parametrize(
    "orbital_aos,cartesian_aos,angular,auxiliary",
    (
        (2, 2, 0, None),
        (7, 8, 1, None),
        (5, 6, 2, None),
        (9, 15, 4, None),
        (7, 8, 1, (12, 13, 2)),
        (5, 6, 2, (9, 15, 4)),
        (500, 500, 0, (1000, 1000, 1)),
    ),
)
def test_native_matches_independent_portable_phase_bound(
    phase_query: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    history: int,
    orbital_aos: int,
    cartesian_aos: int,
    angular: int,
    auxiliary: tuple[int, int, int] | None,
) -> None:
    """Cover direct/DF, public/Cartesian coexistence, through-g and ECP storage."""
    item = {
        "atoms": 3,
        "spins": spins,
        "orbital": {
            "nbf": orbital_aos,
            "cartesian_nbf": cartesian_aos,
            "maximum_angular": angular,
        },
        "auxiliary": (
            dict(
                zip(("nbf", "cartesian_nbf", "maximum_angular"), auxiliary, strict=True)
            )
            if auxiliary
            else None
        ),
    }
    monkeypatch.setattr(resources_hf, "_ecp_workspace", lambda item, **kwargs: 768)
    expected = resources_hf._cpu_item_inventory(item, history)

    shape = (ctypes.c_uint64 * 10)(
        3,
        orbital_aos,
        cartesian_aos,
        spins,
        angular,
        history,
        *(auxiliary or (0, 0, 0)),
        768,
    )
    result = (ctypes.c_uint64 * 4)()
    assert phase_query(shape, len(shape), result, len(result)) == 0
    assert tuple(result) == tuple(expected.values())


def test_invalid_standalone_phase_query_does_not_publish(
    phase_query: typing.Any,
) -> None:
    shape = (ctypes.c_uint64 * 10)(2, 2, 2, 3, 0, 8, 0, 0, 0, 0)
    result = (ctypes.c_uint64 * 4)(5, 7, 11, 13)
    before = bytes(result)
    assert phase_query(shape, len(shape), result, len(result)) != 0
    assert bytes(result) == before


@pytest.mark.parametrize("method,charge,multiplicity", (("rhf", 0, 1), ("uhf", 1, 2)))
@pytest.mark.parametrize("fitted", (False, True))
def test_offline_and_native_plans_share_bounds_but_not_owner_identity(
    library: typing.Any, method: str, charge: int, multiplicity: int, fitted: bool
) -> None:
    """The independent library-free dry run retains its capacity contract."""

    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    controls = {
        "method": method,
        "charges": [charge] * 3,
        "multiplicities": [multiplicity] * 3,
        "density_fitting": "cpu" if fitted else "none",
    }
    native = estimate_hf_resources([atoms] * 3, library=library, **controls)
    legacy = estimate_hf_resources([atoms] * 3, **controls)
    assert native.status == legacy.status == "feasible"
    assert native.peak_bytes == legacy.peak_bytes
    assert native.resident_bytes == legacy.resident_bytes
    assert native.identity != legacy.identity
    assert (
        json.loads(native.requests[0].identity.schedule)["host_inventory"]
        == "native-cpu-v1"
    )
    assert (
        json.loads(legacy.requests[0].identity.schedule)["host_inventory"]
        == "portable-metadata-v1"
    )
    assert (
        json.loads(native.requests[0].identity.schedule)["fleet_inventory"]
        == "native-cpu-v1"
    )
    assert (
        json.loads(legacy.requests[0].identity.schedule)["fleet_inventory"]
        == "portable-metadata-v1"
    )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("history", (0, 8))
@pytest.mark.parametrize("count", (1, 3))
def test_native_ragged_fleet_matches_portable_lifetimes_without_phase_requeries(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    history: int,
    count: int,
) -> None:
    """One bridge owns phase diagnostics, retained sums and serialized maxima."""
    items = []
    for atoms, orbital, auxiliary, ecp in (
        (2, (2, 2, 0, 2, 6), None, 0),
        (3, (5, 6, 2, 1, 3), (9, 15, 4, 1, 3), 0),
        (4, (7, 8, 1, 3, 9), None, 768),
    )[:count]:
        keys = ("nbf", "cartesian_nbf", "maximum_angular", "shells", "primitives")
        items.append(
            {
                "atoms": atoms,
                "spins": spins,
                "orbital": dict(zip(keys, orbital, strict=True)),
                "auxiliary": dict(zip(keys, auxiliary, strict=True))
                if auxiliary
                else None,
                "ecp_workspace": ecp,
            }
        )
    monkeypatch.setattr(
        resources_hf, "_ecp_workspace", lambda item, **kwargs: item["ecp_workspace"]
    )
    expected_phases = tuple(
        resources_hf._cpu_item_inventory(item, history) for item in items
    )
    expected_retained = sum(
        8 * (4 * item["spins"] * item["orbital"]["nbf"] ** 2 + 24 * item["atoms"])
        + 128
        * (item["orbital"]["primitives"] + item["orbital"]["shells"] + item["atoms"])
        + (
            128 * (item["auxiliary"]["primitives"] + item["auxiliary"]["shells"])
            if item["auxiliary"]
            else 0
        )
        for item in items
    )
    expected_workspace = max(
        max(row["preparation"], row["iteration"], row["finalization"])
        for row in expected_phases
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail(
            "native fleet sizing retried individual phases or portable arithmetic"
        )

    monkeypatch.setattr(resources_hf, "_cpu_item_inventory", forbidden)
    monkeypatch.setattr(resources_hf, "byte_product", forbidden)
    assert resources_hf._native_cpu_fleet_inventory(library, items, history) == (
        expected_phases,
        expected_retained,
        expected_workspace,
    )


def test_discovered_fleet_rejection_never_retries_phase_or_portable_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: typing.Any) -> int:
        return -1

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("rejected native fleet query silently fell back")

    monkeypatch.setattr(resources_hf, "_cpu_item_inventory", forbidden)
    monkeypatch.setattr(resources_hf, "byte_product", forbidden)
    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    with pytest.raises(ValueError, match="native CPU HF fleet inventory rejected"):
        estimate_hf_resources(
            [atoms],
            library=SimpleNamespace(generativeqc_resource_hf_cpu_fleet_v1=reject),
        )


def test_phase_only_library_refuses_instead_of_composing_in_python(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("phase-only library cannot authorize a live fleet")

    class PhaseOnlyLibrary:
        """Model the first additive phase-query ABI without the fleet query."""

        def __getattr__(self, name: str) -> typing.Any:
            if name == "generativeqc_resource_hf_cpu_fleet_v1":
                raise AttributeError(name)
            if name.startswith(
                (
                    "generativeqc_batch_prepare",
                    "generativeqc_system_create",
                    "generativeqc_calculation_create",
                )
            ):
                return forbidden
            return getattr(library, name)

    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]

    monkeypatch.setattr(resources_hf, "_cpu_item_inventory", forbidden)
    phase_only = estimate_hf_resources([atoms] * 3, library=PhaseOnlyLibrary())
    assert phase_only.status == "unsupported" and not phase_only.requests[0].candidates
    with pytest.raises(NotImplementedError, match="CPU HF fleet inventory"):
        phase_only.require_feasible()

    calculator = Calculator(resource_budget=ResourceBudget(host_bytes=2**30))
    monkeypatch.setattr(calculator, "_library", PhaseOnlyLibrary())
    with pytest.raises(NotImplementedError, match="CPU HF fleet inventory"):
        calculator.prepare_batch([atoms])


def test_public_fleet_planning_uses_one_native_owner(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    expected = estimate_hf_resources([atoms] * 3)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("public native fleet planning delegated back to Python phases")

    monkeypatch.setattr(resources_hf, "_cpu_item_inventory", forbidden)
    monkeypatch.setattr(resources_hf, "byte_product", forbidden)
    actual = estimate_hf_resources([atoms] * 3, library=library)
    assert actual.peak_bytes == expected.peak_bytes
    assert actual.resident_bytes == expected.resident_bytes
