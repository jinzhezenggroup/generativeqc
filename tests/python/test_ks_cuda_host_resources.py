"""Compiled CUDA host inventories, retained consumers and authoritative failure."""

import ctypes
import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import (
    KsOptions,
    ResourceBudget,
    _native,
    estimate_ks_resources,
    resources_ks,
)


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Require the compiled shape bridge even when the product has no CUDA."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ks_cuda_host_fleet_v1")
    return loaded


def topology(
    nao: int, cartesian_nao: int, angular: int, spins: int, points: int
) -> dict:
    """Describe storage without constructing a molecule or scientific arrays."""
    return {
        "atoms": 3,
        "spins": spins,
        "grid_points": points,
        "orbital": {
            "nbf": nao,
            "cartesian_nbf": cartesian_nao,
            "shells": 1,
            "primitives": 3,
            "maximum_angular": angular,
            "ecp_terms": 2,
        },
    }


def options(schedule: str = "device_fused", nonlocal_correlation: bool = False) -> dict:
    """Keep XC schedule and its independent nonlocal sub-budget explicit."""
    return {
        "diis_history": 8,
        "max_iterations": 204,
        "requires_first_ao_derivatives": True,
        "model": SimpleNamespace(
            grid=SimpleNamespace(radial_points=24, angular_polar=8),
            tile_points=256,
            xc_schedule=schedule,
            has_nonlocal_correlation=nonlocal_correlation,
            nonlocal_memory_budget_bytes=1 << 30,
        ),
    }


class LibraryView:
    """Keep the real compiled authority while selectively hiding an additive ABI."""

    def __init__(self, loaded: typing.Any, hidden: str = "") -> None:
        self.loaded, self.hidden = loaded, hidden

    def __getattr__(self, name: str) -> typing.Any:
        if name == self.hidden:
            raise AttributeError(name)
        return getattr(self.loaded, name)


@pytest.mark.parametrize("schedule", ("device_fused", "host_unfused"))
@pytest.mark.parametrize("nonlocal_correlation", (False, True))
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("allowance,floor", ((0, 0), (384 << 20, 256 << 20)))
@pytest.mark.parametrize(
    "nao,cartesian_nao,angular,points",
    ((2, 2, 0, 6144), (17, 21, 5, 1234), (100000, 100000, 0, 2)),
)
def test_native_ragged_host_fleet_matches_portable_lifetimes(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    schedule: str,
    nonlocal_correlation: bool,
    spins: int,
    allowance: int,
    floor: int,
    nao: int,
    cartesian_nao: int,
    angular: int,
    points: int,
) -> None:
    """Cover ECP, solver threshold, high angular momentum and non-CPU powers."""
    items = [
        topology(nao, cartesian_nao, angular, spins, points),
        topology(1, 1, 0, 1, 12),
    ]
    settings = options(schedule, nonlocal_correlation)
    expected = [
        resources_ks._item_host_inventory(item, backend="cuda", **settings)
        for item in items
    ]
    totals = {
        key: sum(row[key] for row in expected)
        for key in resources_ks._CPU_HOST_PERSISTENT_KEYS
    }
    workspace = max(
        floor, *(max(row["setup_workspace"], row["scf_workspace"]) for row in expected)
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail(
            "compiled host authority retried portable sizing or loaded a provider"
        )

    monkeypatch.setattr(resources_ks, "_item_host_inventory", forbidden)
    monkeypatch.setattr(resources_ks, "byte_product", forbidden)
    monkeypatch.setattr(_native, "load_library", forbidden)
    assert resources_ks._native_host_fleet_inventory(
        library,
        items,
        backend="cuda",
        retained_allowance=allowance,
        workspace_floor=floor,
        **settings,
    ) == (expected, totals, workspace)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
@pytest.mark.parametrize("schedule", ("device_fused", "host_unfused"))
def test_public_bounds_legacy_compatibility_and_exact_budgets(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, method: str, schedule: str
) -> None:
    """Device leaves are stubs; host composition uses the actual CPU-loaded ABI."""
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    monkeypatch.setattr(
        resources_ks,
        "_cuda_item_inventory",
        lambda *args, **kwargs: {
            "state": 8,
            "xc": 8,
            "coulomb": 8,
            "setup": 24,
            "matrix_provider": 8,
        },
    )
    systems = [
        [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))],
        [(2, (0, 0, 0))],
    ]
    settings = {
        "backend": "cuda",
        "method": method,
        "ks_options": KsOptions(xc_schedule=schedule),
    }
    native = estimate_ks_resources(
        systems, library=library, **settings
    ).require_feasible()
    legacy = estimate_ks_resources(
        systems,
        library=LibraryView(library, "generativeqc_resource_ks_cuda_host_fleet_v1"),
        **settings,
    ).require_feasible()
    assert native.peak_bytes == legacy.peak_bytes
    assert native.resident_bytes == legacy.resident_bytes
    assert native.requests[0].candidates == legacy.requests[0].candidates
    assert native.identity != legacy.identity
    controls = json.loads(native.requests[0].identity.schedule)
    assert controls["cuda_host_fleet_composition"] == "native-declared-capacity"
    rows = json.loads(
        dict(native.requests[0].candidates[0].decisions)["item_host_inventory"]
    )
    retained = sum(row["retained"] for row in rows)
    assert native.resident_bytes[
        "host"
    ] == retained + resources_ks.SNAPSHOT_GRID_CACHE_BYTES + (256 << 20)
    budget = ResourceBudget(
        host_bytes=native.peak_bytes["host"], device_bytes=native.peak_bytes["device"]
    )
    estimate_ks_resources(
        systems, library=library, budget=budget, **settings
    ).require_feasible()
    short = ResourceBudget(
        host_bytes=budget.host_bytes - 1, device_bytes=budget.device_bytes
    )
    assert (
        estimate_ks_resources(systems, library=library, budget=short, **settings).status
        == "infeasible"
    )


@pytest.mark.parametrize(
    "corruption",
    ("refusal", "peak", "components", "item", "workspace", "phase", "overflow"),
)
def test_native_refusal_or_coherently_corrupt_publication_never_falls_back(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """Validate wire invariants, including corruption that preserves peak sums."""
    actual = library.generativeqc_resource_ks_cuda_host_fleet_v1
    actual.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    actual.restype = ctypes.c_int

    def corrupt(
        shapes: typing.Any,
        count: int,
        allowance: int,
        floor: int,
        values: typing.Any,
        output_count: int,
    ) -> int:
        if corruption == "refusal":
            return -1
        assert actual(shapes, count, allowance, floor, values, output_count) == 0
        if corruption == "peak":
            values[12] += 1
        elif corruption == "components":
            values[0] += 1
            values[12] += 1
            values[14] += 1
        elif corruption == "item":
            values[28] += 1
        elif corruption == "workspace":
            values[13] += 1
            values[14] += 1
        elif corruption == "phase":
            values[29] += values[13]
        else:
            values[0] = 1 << 63
        return 0

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("authoritative native rejection retried portable host composition")

    view = LibraryView(library)
    monkeypatch.setattr(view, "generativeqc_resource_ks_cuda_host_fleet_v1", corrupt)
    monkeypatch.setattr(resources_ks, "_item_host_inventory", forbidden)
    with pytest.raises(ValueError, match="native|signed|capacity"):
        resources_ks._native_host_fleet_inventory(
            view,
            [topology(2, 2, 0, 1, 64)],
            backend="cuda",
            retained_allowance=384 << 20,
            **options(),
        )


def test_nonlocal_subbudget_and_missing_symbol_remain_explicit(
    library: typing.Any,
) -> None:
    """A discovered inventory does not waive a caller's independent provider cap."""
    settings = options(nonlocal_correlation=True)
    settings["model"].nonlocal_memory_budget_bytes = 1
    items = [topology(2, 2, 0, 1, 64)]
    with pytest.raises(ValueError, match="nonlocal_memory_budget_bytes"):
        resources_ks._native_host_fleet_inventory(
            library, items, backend="cuda", **settings
        )
    assert (
        resources_ks._native_host_fleet_inventory(
            LibraryView(library, "generativeqc_resource_ks_cuda_host_fleet_v1"),
            items,
            backend="cuda",
            **settings,
        )
        is None
    )
