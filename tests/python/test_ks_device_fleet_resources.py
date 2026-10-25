"""Execute native CUDA fleet composition without claiming GPU leaf qualification."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import ResourceBudget, _native, resources_ks
from generativeqc.resources_planning import plan_resources


@pytest.fixture(scope="module")
def library() -> Any:
    """Require the actual compiled bridge even in a CPU-only library."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ks_cuda_fleet_v1")
    return loaded


def topology(cartesian_aos: int, atoms: int = 2) -> dict[str, Any]:
    return {
        "atoms": atoms,
        "orbital": {
            "cartesian_nbf": cartesian_aos,
            "shells": atoms,
            "primitives": 3 * atoms,
        },
    }


@pytest.mark.parametrize("cores", ((0, 0), (10, 0), (28, 46)))
def test_stationary_shape_reduces_per_atom_ecp_core_counts(
    cores: tuple[int, ...],
) -> None:
    """The seven-scalar native shape receives the total, never an atom tuple."""
    observed = []

    class Query:
        def __call__(
            self, descriptor: Any, shape: Any, count: int, output: Any, size: int
        ) -> int:
            observed.append(tuple(shape))
            assert count == 7 and size == 9
            return _native.STATUS_NOT_IMPLEMENTED

    item = {
        "atoms": 2,
        "grid_points": 768,
        "orbital": {
            "nbf": 7,
            "primitives": 21,
            "maximum_angular": 1,
            "ecp_terms": 3,
            "ecp_core_electrons": cores,
        },
    }
    library = SimpleNamespace(generativeqc_resource_ks_stationary_cuda_v1=Query())
    descriptor = _native.MethodDescriptor()
    assert (
        resources_ks._native_cuda_stationary_inventory(library, item, descriptor)
        is None
    )
    assert observed == [(2, 7, 21, 768, 1, 3, sum(cores))]


def leaves(item: dict[str, Any], *, ecp: bool = False) -> dict[str, int]:
    atoms = item["atoms"]
    return {
        "state": atoms * 100,
        "xc": atoms * 200,
        "coulomb": atoms * 300,
        "quadrature_setup": atoms * 400,
        "matrix_provider": atoms * 500,
        "ecp_workspace": (1 << 20) if ecp else 0,
    }


@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("ecp", (False, True))
@pytest.mark.parametrize("stationary", (False, True))
@pytest.mark.parametrize("cartesian_aos", (2, 6, 18, 1024))
def test_ragged_fleet_matches_independent_portable_lifetimes(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
    ecp: bool,
    stationary: bool,
    cartesian_aos: int,
) -> None:
    """Vary Cartesian expansion, ECP, XC schedule and stationary candidates."""
    items = [topology(cartesian_aos), topology(2, atoms=1)]
    monkeypatch.setattr(
        resources_ks,
        "_cuda_leaf_inventory",
        lambda _, item, **__: leaves(item, ecp=ecp),
    )
    consumers = [{"minimum": 700, "cached": 900}, {"minimum": 10, "cached": 20}]
    if not stationary:
        consumers = [None, consumers[1]]
    expected = [resources_ks._cuda_item_inventory(library, item) for item in items]
    if not fused:
        for inventory in expected:
            inventory["xc"] = 0
    if ecp and stationary:
        # The force's ECP quadrature and bridge temporaries coexist with SCF;
        # the old energy setup retires those arenas before allocating scratch.
        for item, inventory in zip(items, expected, strict=True):
            shape = item["orbital"]
            n, a, s, p = (
                shape["cartesian_nbf"],
                item["atoms"],
                shape["shells"],
                shape["primitives"],
            )
            bridge = 64 * (1 + a + s + p + n + s * (s + 1) // 2 + n * n) + 48 * a
            retained = sum(inventory[key] for key in ("state", "xc", "coulomb"))
            inventory["setup"] = max(inventory["setup"], retained + (1 << 20) + bridge)
    actual, aggregate = resources_ks._native_cuda_fleet_inventory(
        library, items, consumers, device_fused_xc=fused
    )
    assert actual == expected
    for key in ("state", "xc", "coulomb", "matrix_provider"):
        assert aggregate[key] == sum(inventory[key] for inventory in expected)
    assert aggregate["setup_extra"] == max(
        max(
            0,
            inventory["setup"]
            - sum(inventory[key] for key in ("state", "xc", "coulomb")),
        )
        for inventory in expected
    )
    common = sum(
        aggregate[key]
        for key in ("state", "xc", "coulomb", "matrix_provider", "setup_extra")
    )
    for mode in ("minimum", "cached"):
        assert aggregate[mode] == (
            sum(consumer[mode] for consumer in consumers) if stationary else 0
        )
        assert aggregate[f"{mode}_peak"] == common + aggregate[mode]


@pytest.mark.parametrize("minimum,cached", ((700, 900), (900, 700), (0, 0)))
def test_cache_candidates_remain_independent(
    library: Any, monkeypatch: pytest.MonkeyPatch, minimum: int, cached: int
) -> None:
    """Composition must not invent a cache ordering or scientific policy."""
    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )
    _, aggregate = resources_ks._native_cuda_fleet_inventory(
        library,
        [topology(2)],
        [{"minimum": minimum, "cached": cached}],
        device_fused_xc=True,
    )
    assert aggregate["minimum"] == minimum
    assert aggregate["cached"] == cached
    assert aggregate["minimum_peak"] - minimum == aggregate["cached_peak"] - cached


def test_missing_symbol_does_not_query_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional composition is independent from execution support."""

    def forbidden(*args: Any, **options: Any) -> Any:
        raise AssertionError("missing bridge unexpectedly requested CUDA leaves")

    monkeypatch.setattr(resources_ks, "_cuda_leaf_inventory", forbidden)
    assert (
        resources_ks._native_cuda_fleet_inventory(
            SimpleNamespace(), [topology(2)], [None], device_fused_xc=True
        )
        is None
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "refusal",
        "overflow",
        "leaf",
        "peak",
        "setup",
        "setup_aggregate",
        "minimum",
        "cached",
        "swapped_candidates",
    ),
)
def test_native_refusal_or_corruption_never_retries_portable_composition(
    library: Any, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """Exercise the real ABI first, then corrupt its publication at the boundary."""
    query = library.generativeqc_resource_ks_cuda_fleet_v1
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int

    def corrupted(shape: Any, count: int, output: Any, size: int) -> int:
        assert query(shape, count, output, size) == 0
        if corruption == "refusal":
            return -1
        if corruption == "swapped_candidates":
            output[5], output[6] = output[6], output[5]
        elif corruption in ("setup_aggregate", "minimum", "cached"):
            slot = {"setup_aggregate": 4, "minimum": 5, "cached": 6}[corruption]
            output[slot] = 0
        else:
            slot = {"overflow": 0, "leaf": 9, "peak": 7, "setup": 12}[corruption]
            output[slot] = 2**64 - 1 if corruption == "overflow" else 0
        if corruption in ("setup_aggregate", "minimum", "cached", "swapped_candidates"):
            # Keep both internal peak equations coherent while contradicting
            # returned setup diagnostics or the supplied stationary leaves.
            output[7] = sum(output[:5]) + output[5]
            output[8] = sum(output[:5]) + output[6]
        return 0

    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )

    def forbidden(*args: Any, **options: Any) -> Any:
        raise AssertionError("selected native failure retried portable composition")

    monkeypatch.setattr(resources_ks, "_cuda_item_inventory", forbidden)
    with pytest.raises(ValueError):
        resources_ks._native_cuda_fleet_inventory(
            SimpleNamespace(generativeqc_resource_ks_cuda_fleet_v1=corrupted),
            [topology(2)],
            [{"minimum": 700, "cached": 900}],
            device_fused_xc=True,
        )


@pytest.mark.parametrize("mode", ("minimum", "cached"))
def test_stationary_candidates_admit_exact_signed_int64_peak(
    library: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Validate the wire relations without imposing ordering or losing one byte."""
    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )
    _, empty = resources_ks._native_cuda_fleet_inventory(
        library,
        [topology(2)],
        [{"minimum": 0, "cached": 0}],
        device_fused_xc=True,
    )
    maximum = 2**63 - 1
    stationary = {"minimum": 0, "cached": 0}
    stationary[mode] = maximum - empty[f"{mode}_peak"]
    _, exact = resources_ks._native_cuda_fleet_inventory(
        library, [topology(2)], [stationary], device_fused_xc=True
    )
    assert exact[f"{mode}_peak"] == maximum
    stationary[mode] += 1
    with pytest.raises(ValueError):
        resources_ks._native_cuda_fleet_inventory(
            library, [topology(2)], [stationary], device_fused_xc=True
        )


def test_public_candidate_and_exact_budget_preserve_legacy_bounds(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only leaf metadata is doubled; native fleet and public admission execute."""
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    monkeypatch.setattr(
        resources_ks, "_cuda_leaf_inventory", lambda _, item, **__: leaves(item)
    )
    monkeypatch.setattr(
        resources_ks,
        "_native_cuda_stationary_inventory",
        lambda _, item, descriptor: {
            "minimum": 700 * item["atoms"],
            "cached": 900 * item["atoms"],
        },
    )
    common = {
        "generativeqc_ks_options_version": lambda: 1,
        "generativeqc_ks_resource_inventory_version_v1": lambda: 1,
    }
    native_library = SimpleNamespace(
        **common,
        generativeqc_resource_ks_cuda_fleet_v1=library.generativeqc_resource_ks_cuda_fleet_v1,
    )
    portable_library = SimpleNamespace(**common)
    systems = [[(1, (0, 0, -0.7)), (1, (0, 0, 0.7))], [(2, (0, 0, 0))]]
    native_request = resources_ks.ks_resource_request(
        systems, backend="cuda", library=native_library
    )
    portable_request = resources_ks.ks_resource_request(
        systems, backend="cuda", library=portable_library
    )
    assert native_request.unsupported_reason is None
    assert portable_request.unsupported_reason is None
    assert len(native_request.candidates) == len(portable_request.candidates) == 2
    for native, portable in zip(
        native_request.candidates, portable_request.candidates, strict=True
    ):
        assert native.estimates == portable.estimates
        native_decisions = dict(native.decisions)
        portable_decisions = dict(portable.decisions)
        assert (
            native_decisions.pop("cuda_fleet_composition") == "native-declared-capacity"
        )
        assert (
            portable_decisions.pop("cuda_fleet_composition")
            == "legacy-leaf-compatibility"
        )
        assert native_decisions == portable_decisions
    request = replace(native_request, candidates=(native_request.candidates[0],))
    portable = replace(portable_request, candidates=(portable_request.candidates[0],))
    plan = plan_resources(
        (request,), ResourceBudget(), library=native_library
    ).require_feasible()
    reference = plan_resources(
        (portable,), ResourceBudget(), library=portable_library
    ).require_feasible()
    assert plan.peak_bytes == reference.peak_bytes
    exact = ResourceBudget(
        host_bytes=plan.peak_bytes["host"], device_bytes=plan.peak_bytes["device"]
    )
    plan_resources((request,), exact, library=native_library).require_feasible()
    short = replace(exact, device_bytes=exact.device_bytes - 1)
    with pytest.raises(MemoryError):
        plan_resources((request,), short, library=native_library).require_feasible()
