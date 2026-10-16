"""CPU KS fleet capacities, mixed-age ABIs and authoritative failure handling."""

import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import ResourceBudget, _native, estimate_ks_resources, resources_ks


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Source qualification requires the new bridge rather than skipping it."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ks_cpu_fleet_v1")
    return loaded


def topology(
    nao: int, cartesian_nao: int, angular: int, spins: int, points: int
) -> dict[str, typing.Any]:
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


def model_options(*, nonlocal_correlation: bool = False) -> dict[str, typing.Any]:
    return {
        "diis_history": 8,
        "max_iterations": 204,
        "requires_first_ao_derivatives": True,
        "model": SimpleNamespace(
            grid=SimpleNamespace(radial_points=24, angular_polar=8),
            tile_points=256,
            has_nonlocal_correlation=nonlocal_correlation,
            nonlocal_memory_budget_bytes=1 << 30,
        ),
    }


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("nonlocal_correlation", (False, True))
@pytest.mark.parametrize("workspace_floor", (0, 256 << 20))
@pytest.mark.parametrize(
    "nao,cartesian_nao,angular,points",
    (
        (2, 2, 0, 6144),
        (5, 6, 2, 100),
        (9, 15, 4, 1234),
        (7, 7, 1, (64 << 20) // 32 // 7),
        (7, 7, 1, (64 << 20) // 32 // 7 + 1),
    ),
)
def test_ragged_native_fleet_matches_independent_portable_bounds(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    nonlocal_correlation: bool,
    workspace_floor: int,
    nao: int,
    cartesian_nao: int,
    angular: int,
    points: int,
) -> None:
    """Cover spin, spherical storage, cache gates, ECP and nonlocal owners."""
    items = [
        topology(nao, cartesian_nao, angular, spins, points),
        topology(1, 1, 0, 1, 12),
    ]
    options = model_options(nonlocal_correlation=nonlocal_correlation)
    expected = [
        resources_ks._item_host_inventory(item, backend="cpu", **options)
        for item in items
    ]
    totals = {
        key: sum(inventory[key] for inventory in expected)
        for key in resources_ks._CPU_HOST_PERSISTENT_KEYS
    }
    workspace = max(
        workspace_floor,
        *(max(row["setup_workspace"], row["scf_workspace"]) for row in expected),
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("native fleet retried per-item sizing or loaded a provider")

    monkeypatch.setattr(resources_ks, "_item_host_inventory", forbidden)
    monkeypatch.setattr(resources_ks, "_native_cpu_host_inventory", forbidden)
    monkeypatch.setattr(resources_ks, "byte_product", forbidden)
    monkeypatch.setattr(_native, "load_library", forbidden)
    assert resources_ks._native_cpu_fleet_inventory(
        library, items, workspace_floor=workspace_floor, **options
    ) == (expected, totals, workspace)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
@pytest.mark.parametrize("history,iterations", ((0, 1), (64, 250)))
def test_public_native_phase_only_and_legacy_fleets_keep_bounds_and_provenance(
    library: typing.Any, method: str, history: int, iterations: int
) -> None:
    """Source ownership is distinct from unchanged public capacity contracts."""

    class LibraryView:
        def __init__(self, *hidden: str) -> None:
            self.hidden = hidden

        def __getattr__(self, name: str) -> typing.Any:
            if name in self.hidden:
                raise AttributeError(name)
            return getattr(library, name)

    unrestricted = method.endswith("uks")
    systems = [
        [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))],
        [(1 if unrestricted else 2, (0, 0, 0))],
    ]
    options = {
        "method": method,
        "diis_history": history,
        "max_iterations": iterations,
        "charges": [int(unrestricted), 0],
        "multiplicities": [2, 2] if unrestricted else [1, 1],
    }
    native = estimate_ks_resources(
        systems, library=library, **options
    ).require_feasible()
    phase = estimate_ks_resources(
        systems, library=LibraryView("generativeqc_resource_ks_cpu_fleet_v1"), **options
    ).require_feasible()
    legacy = estimate_ks_resources(
        systems,
        library=LibraryView(
            "generativeqc_resource_ks_cpu_fleet_v1",
            "generativeqc_resource_ks_cpu_host_v1",
        ),
        **options,
    ).require_feasible()
    fleet_only = estimate_ks_resources(
        systems, library=LibraryView("generativeqc_resource_ks_cpu_host_v1"), **options
    ).require_feasible()
    assert native.identity == fleet_only.identity
    assert len({native.identity, phase.identity, legacy.identity}) == 3
    for plan in (phase, legacy, fleet_only):
        assert native.peak_bytes == plan.peak_bytes
        assert native.resident_bytes == plan.resident_bytes
        assert (
            native.requests[0].candidates[0].estimates
            == plan.requests[0].candidates[0].estimates
        )
    for plan, host_owner, fleet_owner in (
        (native, "native-cpu-v1", "native-cpu-v1"),
        (phase, "native-cpu-v1", "portable-metadata-v1"),
        (legacy, "portable-metadata-v1", "portable-metadata-v1"),
    ):
        controls = json.loads(plan.requests[0].identity.schedule)
        assert controls["host_inventory"] == host_owner
        assert controls["fleet_inventory"] == fleet_owner
    peak = native.peak_bytes["host"]
    assert (
        estimate_ks_resources(
            systems, library=library, budget=ResourceBudget(host_bytes=peak), **options
        ).status
        == "feasible"
    )
    assert (
        estimate_ks_resources(
            systems,
            library=library,
            budget=ResourceBudget(host_bytes=peak - 1),
            **options,
        ).status
        == "infeasible"
    )


def test_discovered_fleet_rejection_never_retries_older_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: typing.Any) -> int:
        return -1

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("rejected fleet query retried per-item or portable sizing")

    monkeypatch.setattr(resources_ks, "_item_host_inventory", forbidden)
    monkeypatch.setattr(resources_ks, "_native_cpu_host_inventory", forbidden)
    with pytest.raises(ValueError, match="native CPU KS fleet inventory rejected"):
        estimate_ks_resources(
            [[(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]],
            library=SimpleNamespace(generativeqc_resource_ks_cpu_fleet_v1=reject),
        )


def test_native_fleet_does_not_waive_nonlocal_provider_sub_budget(
    library: typing.Any,
) -> None:
    options = model_options(nonlocal_correlation=True)
    options["model"].nonlocal_memory_budget_bytes = 1
    with pytest.raises(ValueError, match="nonlocal provider workspace"):
        resources_ks._native_cpu_fleet_inventory(
            library, [topology(2, 2, 0, 1, 10)], workspace_floor=0, **options
        )


@pytest.mark.parametrize("oversized", (False, True))
def test_invalid_native_aggregate_cannot_become_a_budget(oversized: bool) -> None:
    def malformed(*args: typing.Any) -> int:
        args[3][14] = (1 << 63) if oversized else 1
        return 0

    loaded = SimpleNamespace(generativeqc_resource_ks_cpu_fleet_v1=malformed)
    with pytest.raises(ValueError, match="CPU KS fleet"):
        resources_ks._native_cpu_fleet_inventory(
            loaded, [topology(2, 2, 0, 1, 10)], workspace_floor=0, **model_options()
        )
