"""Native CPU host sizing shares exact portable bounds without scientific work."""

import ctypes
import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import _native, estimate_ks_resources, resources_ks


@pytest.fixture
def library() -> typing.Any:
    """Require the new bridge in source qualification, not an optional skip."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ks_cpu_host_v1")
    return loaded


@pytest.fixture
def phase_query(library: typing.Any) -> typing.Any:
    """Bind the standalone C ABI independently of public fleet requests."""
    operation = library.generativeqc_resource_ks_cpu_host_v1
    pointer = ctypes.POINTER(ctypes.c_uint64)
    operation.argtypes = [pointer, ctypes.c_size_t, pointer, ctypes.c_size_t]
    operation.restype = ctypes.c_int
    return operation


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("nonlocal_correlation", (False, True))
@pytest.mark.parametrize(
    "orbital_aos,cartesian_aos,angular,points,tile",
    (
        (7, 7, 1, 9216, 256),
        (5, 6, 2, 100, 32),
        (9, 15, 4, 1234, 7),
        (7, 7, 1, (64 << 20) // 32 // 7, 256),
        (7, 7, 1, (64 << 20) // 32 // 7 + 1, 256),
        (500, 500, 0, 9216, 512),
    ),
)
def test_native_matches_independent_portable_lifetime_bound(
    library: typing.Any,
    phase_query: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    nonlocal_correlation: bool,
    orbital_aos: int,
    cartesian_aos: int,
    angular: int,
    points: int,
    tile: int,
) -> None:
    """Cover representation, exact cache admission, tiles and retained sources."""
    model = SimpleNamespace(
        grid=SimpleNamespace(radial_points=24, angular_polar=8),
        tile_points=tile,
        has_nonlocal_correlation=nonlocal_correlation,
        nonlocal_memory_budget_bytes=1 << 30,
    )
    item = {
        "atoms": 3,
        "spins": spins,
        "grid_points": points,
        "orbital": {
            "nbf": orbital_aos,
            "cartesian_nbf": cartesian_aos,
            "shells": 1,
            "primitives": 3,
            "maximum_angular": angular,
            "ecp_terms": 2,
        },
    }
    monkeypatch.setattr(resources_ks, "_ecp_workspace", lambda item, **kwargs: 768)
    options = {
        "diis_history": 8,
        "max_iterations": 100,
        "requires_first_ao_derivatives": True,
        "backend": "cpu",
        "model": model,
    }
    expected = resources_ks._item_host_inventory(item, **options)

    options.pop("backend")
    dimensions = resources_ks._cpu_host_shape(item, library=library, **options)
    shape = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    result = (ctypes.c_uint64 * len(resources_ks._CPU_HOST_INVENTORY_KEYS))()
    assert phase_query(shape, len(shape), result, len(result)) == 0
    actual = resources_ks._cpu_host_result(result, model)
    assert actual == expected


def test_invalid_phase_query_does_not_publish(
    phase_query: typing.Any,
) -> None:
    shape = (ctypes.c_uint64 * 17)()
    result = (ctypes.c_uint64 * 16)(9, 11, 13)
    before = bytes(result)
    assert phase_query(shape, len(shape), result, len(result)) != 0
    assert bytes(result) == before


def test_production_query_and_legacy_metadata_have_equal_bounds_distinct_identity(
    library: typing.Any,
) -> None:
    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    native = estimate_ks_resources([atoms], library=library)
    legacy = estimate_ks_resources([atoms])
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


def test_native_nonlocal_bound_still_enforces_explicit_provider_budget(
    library: typing.Any,
) -> None:
    """Shape discovery does not waive the separately configured VV10 budget."""
    item = {
        "atoms": 1,
        "spins": 1,
        "grid_points": 10,
        "orbital": {
            "nbf": 1,
            "cartesian_nbf": 1,
            "shells": 1,
            "primitives": 1,
            "maximum_angular": 0,
        },
    }
    model = SimpleNamespace(
        grid=SimpleNamespace(radial_points=1, angular_polar=1),
        tile_points=1,
        has_nonlocal_correlation=True,
        nonlocal_memory_budget_bytes=479,
    )
    options = {
        "diis_history": 8,
        "max_iterations": 100,
        "requires_first_ao_derivatives": True,
        "model": model,
    }
    with pytest.raises(ValueError, match="nonlocal provider workspace exceeds"):
        resources_ks._native_cpu_fleet_inventory(
            library, [item], workspace_floor=0, **options
        )
    model.nonlocal_memory_budget_bytes = 480
    inventories, _, _ = resources_ks._native_cpu_fleet_inventory(
        library, [item], workspace_floor=0, **options
    )
    assert inventories[0]["nonlocal_provider"] == 480
