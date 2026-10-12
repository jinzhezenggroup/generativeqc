"""Native CPU host sizing shares exact portable bounds without scientific work."""

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
    monkeypatch.setattr(resources_ks, "_ecp_workspace", lambda item: 768)
    options = {
        "diis_history": 8,
        "max_iterations": 100,
        "requires_first_ao_derivatives": True,
        "backend": "cpu",
        "model": model,
    }
    expected = resources_ks._item_host_inventory(item, **options)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("discovered native sizing retried the portable arithmetic")

    monkeypatch.setattr(resources_ks, "byte_product", forbidden)
    actual = resources_ks._item_host_inventory(item, library=library, **options)
    assert actual == expected


def test_discovered_rejection_never_retries_portable_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: typing.Any) -> int:
        return -1

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("rejected native inventory silently used a portable estimate")

    monkeypatch.setattr(resources_ks, "byte_product", forbidden)
    item = {
        "atoms": 1,
        "spins": 1,
        "grid_points": 1,
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
        has_nonlocal_correlation=False,
    )
    with pytest.raises(ValueError, match="native CPU KS host inventory rejected"):
        resources_ks._item_host_inventory(
            item,
            library=SimpleNamespace(generativeqc_resource_ks_cpu_host_v1=reject),
            diis_history=8,
            max_iterations=100,
            requires_first_ao_derivatives=False,
            backend="cpu",
            model=model,
        )


def test_production_query_and_legacy_metadata_have_equal_bounds_distinct_identity(
    library: typing.Any,
) -> None:
    class LegacyLibrary:
        """Hide only the new query while retaining the installed ABI contract."""

        def __getattr__(self, name: str) -> typing.Any:
            if name == "generativeqc_resource_ks_cpu_host_v1":
                raise AttributeError(name)
            return getattr(library, name)

    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    native = estimate_ks_resources([atoms], library=library)
    legacy = estimate_ks_resources([atoms], library=LegacyLibrary())
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
        "library": library,
        "diis_history": 8,
        "max_iterations": 100,
        "requires_first_ao_derivatives": True,
        "backend": "cpu",
        "model": model,
    }
    with pytest.raises(ValueError, match="nonlocal provider workspace exceeds"):
        resources_ks._item_host_inventory(item, **options)
    model.nonlocal_memory_budget_bytes = 480
    assert (
        resources_ks._item_host_inventory(item, **options)["nonlocal_provider"] == 480
    )
