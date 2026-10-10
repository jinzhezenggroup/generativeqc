"""Fail-closed tests for cross-functional force active-AO policy."""

from dataclasses import fields, replace

import pytest
from generativeqc._force_active_ao import (
    QUALIFIED_FORCE_ACTIVE_AO_PROFILES,
    ForceActiveAoWorkload,
    QualifiedForceActiveAoProfile,
    force_active_ao_policy_record,
    resolve_force_active_ao_policy,
)


def _workload(**updates: object) -> ForceActiveAoWorkload:
    value = ForceActiveAoWorkload(
        architecture="sm_120",
        derivative_order=2,
        spin_blocks=1,
        composition="ordinary",
        hamiltonian="all-electron",
        density_fitted=False,
        atoms=48,
        aos=384,
        grid_points=1_000_000,
        tile_policy="fixed",
        tile_points=256,
        max_device_bytes=512 << 20,
        max_host_bytes=256 << 20,
        resident_grid=True,
    )
    return replace(value, **updates)


def _profile(**updates: object) -> QualifiedForceActiveAoProfile:
    values = {
        "profile_id": "test-qualified-domain",
        "evidence": ("test:complete-cold-warm-moved",),
        "compositions": ("ordinary", "composite"),
        "derivative_orders": (1, 2),
        "spin_blocks": (1, 2),
        "density_fitted": False,
        "min_dense_point_ao_square_work": 100_000_000_000,
        "tile_policy": "fixed",
        "tile_points": 256,
        "min_device_bytes": 512 << 20,
        "min_host_bytes": 256 << 20,
        "cutoff": 1e-16,
        "cache_bytes": 16 << 20,
    }
    values.update(updates)
    return QualifiedForceActiveAoProfile(**values)


def test_production_auto_policy_uses_dense_work_crossover_not_size_window() -> None:
    assert tuple(
        profile.profile_id for profile in QUALIFIED_FORCE_ACTIVE_AO_PROFILES
    ) == (
        "ordinary-direct-active-ao-cost-v3",
        "cuda-resident-preao-native-csr-v1",
        "fitted-exact-jet-bitmask-v1",
        "ordinary-direct-active-ao-cost-v3-budget-auto",
        "cuda-resident-preao-native-csr-v1-budget-auto",
        "fitted-exact-jet-bitmask-v1-budget-auto",
    )

    below_crossover = resolve_force_active_ao_policy(_workload())
    assert below_crossover.selected
    assert below_crossover.producer == "pre-ao-envelope-native-csr"

    # Admission is continuous in predicted dense point×AO² work. Atom count,
    # AO count and grid count are not benchmark fingerprints.
    for workload in (
        _workload(grid_points=1_179_648),
        _workload(atoms=40, aos=512, grid_points=700_000),
        _workload(atoms=120, aos=960, grid_points=3_000_000),
        _workload(atoms=200, aos=384, grid_points=1_179_648),
        _workload(derivative_order=1, grid_points=1_179_648),
        _workload(spin_blocks=2, grid_points=1_179_648),
        _workload(architecture="sm_90", grid_points=1_179_648),
        _workload(architecture="sm_89", aos=512, grid_points=700_000),
    ):
        decision = resolve_force_active_ao_policy(workload)
        assert decision.selected
        assert decision.profile_id == "ordinary-direct-active-ao-cost-v3"
        assert decision.cutoff == 1e-16
        assert decision.cache_bytes == 16 << 20


@pytest.mark.parametrize(
    "updates",
    [
        {"composition": "composite", "grid_points": 1_179_648},
        {"density_fitted": True, "grid_points": 1_000_000},
        {"tile_points": 128, "grid_points": 1_179_648},
        {"max_device_bytes": (512 << 20) - 1, "grid_points": 1_179_648},
        {"max_host_bytes": (256 << 20) - 1, "grid_points": 1_179_648},
    ],
)
def test_production_cost_profile_keeps_capability_or_profitability_misses_dense(
    updates: dict[str, object],
) -> None:
    decision = resolve_force_active_ao_policy(_workload(**updates))
    assert not decision.selected
    assert decision.reason == "no-qualified-profile"


def test_qualified_default_selects_native_csr() -> None:
    workload = _workload(
        device_name="NVIDIA GeForce RTX 5090", grid_points=589_824, aos=192
    )
    decision = resolve_force_active_ao_policy(workload)
    assert decision.selected
    assert decision.producer == "pre-ao-envelope-native-csr"
    assert decision.cache_bytes == 64 << 20
    assert decision.max_active_fraction == 0.8


@pytest.mark.parametrize("derivative_order", [1, 2])
@pytest.mark.parametrize("spin_blocks", [1, 2])
@pytest.mark.parametrize("tile_policy", ["fixed", "budget-auto"])
def test_fitted_force_qualifies_exact_jets_without_relaxing_cutoff(
    derivative_order: int, spin_blocks: int, tile_policy: str
) -> None:
    """The new domain is integral-provider based, not a molecule whitelist."""
    workload = _workload(
        density_fitted=True,
        derivative_order=derivative_order,
        spin_blocks=spin_blocks,
        tile_policy=tile_policy,
        tile_points=256 if tile_policy == "fixed" else None,
        grid_points=1_179_648,
    )
    decision = resolve_force_active_ao_policy(workload)
    assert decision.selected
    assert decision.producer == "exact-jets-native-bitmask"
    assert decision.cutoff == 1e-16
    assert decision.cache_bytes == 64 << 20
    assert decision.max_active_fraction == 0.8
    assert decision.profile_id == "fitted-exact-jet-bitmask-v1" + (
        "-budget-auto" if tile_policy == "budget-auto" else ""
    )
    for miss in (
        replace(workload, composition="composite"),
        replace(workload, resident_grid=False),
        replace(workload, grid_points=1_000_000),
        replace(workload, max_device_bytes=(512 << 20) - 1),
        replace(workload, max_host_bytes=(256 << 20) - 1),
    ):
        assert not resolve_force_active_ao_policy(miss).selected


@pytest.mark.parametrize("architecture", ["sm_80", "sm_89", "sm_90", "sm_120"])
@pytest.mark.parametrize("device_name", [None, "arbitrary CUDA device"])
@pytest.mark.parametrize("grid_points", [589_824, 1_179_648])
def test_automatic_tiles_preserve_producer_and_guards_without_product_whitelists(
    architecture: str, device_name: str | None, grid_points: int
) -> None:
    fixed = _workload(
        architecture=architecture, device_name=device_name, grid_points=grid_points
    )
    automatic = replace(fixed, tile_policy="budget-auto", tile_points=None)
    incumbent = resolve_force_active_ao_policy(fixed)
    selected = resolve_force_active_ao_policy(automatic)
    assert selected.selected
    assert selected.profile_id == f"{incumbent.profile_id}-budget-auto"
    assert (selected.producer, selected.cutoff, selected.cache_bytes) == (
        incumbent.producer,
        incumbent.cutoff,
        incumbent.cache_bytes,
    )
    assert selected.max_active_fraction == incumbent.max_active_fraction
    for miss in (
        replace(automatic, density_fitted=True, grid_points=1_000_000),
        replace(automatic, resident_grid=False),
        replace(automatic, max_device_bytes=(512 << 20) - 1),
        replace(automatic, max_host_bytes=(256 << 20) - 1),
    ):
        assert not resolve_force_active_ao_policy(miss).selected


@pytest.mark.parametrize(
    "updates",
    [
        {"composition": "composite"},
        {"architecture": "cpu"},
        {"density_fitted": True, "grid_points": 1_000_000},
        {"resident_grid": False},
        {"tile_points": 128},
        {"max_device_bytes": (512 << 20) - 1},
        {"max_host_bytes": (256 << 20) - 1},
    ],
)
def test_default_native_profile_misses_remain_dense(updates: dict) -> None:
    workload = _workload(device_name="NVIDIA GeForce RTX 5090", grid_points=1_179_648)
    decision = resolve_force_active_ao_policy(replace(workload, **updates))
    assert not decision.selected


@pytest.mark.parametrize(
    "shape",
    [
        (3, 7, 9216),
        (24, 192, 589_824),
        (48, 384, 1_179_648),
        (96, 768, 2_359_296),
        (128, 1024, 3_145_728),
    ],
)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("spins", [1, 2])
def test_default_profile_does_not_whitelist_benchmark_shapes(
    shape: tuple[int, int, int],
    order: int,
    spins: int,
) -> None:
    atoms, aos, points = shape
    decision = resolve_force_active_ao_policy(
        _workload(
            device_name="NVIDIA GeForce RTX 5090",
            atoms=atoms,
            aos=aos,
            grid_points=points,
            derivative_order=order,
            spin_blocks=spins,
        )
    )
    assert decision.selected
    work = points * aos * aos
    expected = (
        "sampled-jets"
        if work >= QUALIFIED_FORCE_ACTIVE_AO_PROFILES[0].min_dense_point_ao_square_work
        else "pre-ao-envelope-native-csr"
    )
    assert decision.producer == expected


@pytest.mark.parametrize(
    "architecture,name",
    [
        ("sm_80", "NVIDIA A100"),
        ("sm_86", "NVIDIA GeForce RTX 3090"),
        ("sm_90", "NVIDIA H100"),
        ("sm_120", "NVIDIA RTX PRO 6000 Blackwell"),
        ("sm_120", "NVIDIA GeForce RTX 5090"),
        ("sm_120", None),
    ],
)
def test_default_selection_has_no_gpu_product_whitelist(
    architecture: str,
    name: str | None,
) -> None:
    decision = resolve_force_active_ao_policy(
        _workload(
            architecture=architecture,
            device_name=name,
        )
    )
    assert decision.selected and decision.producer == "pre-ao-envelope-native-csr"


@pytest.mark.parametrize("spin_blocks", [1, 2])
@pytest.mark.parametrize("derivative_order", [1, 2])
def test_profile_admits_rks_uks_and_supported_jet_orders(
    spin_blocks: int, derivative_order: int
) -> None:
    decision = resolve_force_active_ao_policy(
        _workload(
            spin_blocks=spin_blocks,
            derivative_order=derivative_order,
        ),
        profiles=(_profile(),),
    )
    assert decision.selected
    assert decision.profile_id == "test-qualified-domain"


def test_same_resolver_can_cover_composite_without_method_identity() -> None:
    decision = resolve_force_active_ao_policy(
        _workload(
            composition="composite",
            tile_policy="budget-auto",
            tile_points=None,
            max_device_bytes=1 << 30,
            max_host_bytes=2 << 30,
        ),
        profiles=(
            _profile(
                tile_policy="budget-auto",
                tile_points=None,
                min_device_bytes=1 << 30,
                min_host_bytes=2 << 30,
                cache_bytes=64 << 20,
            ),
        ),
    )
    assert decision.selected
    names = {field.name.lower() for field in fields(QualifiedForceActiveAoProfile)}
    for forbidden in ("method", "functional", "selector", "architecture"):
        assert not any(forbidden in name for name in names)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"hamiltonian": "scalar-semilocal-ecp"}, "unsupported-hamiltonian"),
        ({"derivative_order": 3}, "unsupported-derivative-order"),
        ({"resident_grid": False}, "resident-grid-unavailable"),
    ],
)
def test_capability_miss_is_dense(updates: dict[str, object], reason: str) -> None:
    decision = resolve_force_active_ao_policy(
        _workload(**updates),
        profiles=(_profile(),),
    )
    assert not decision.selected
    assert decision.reason == reason


@pytest.mark.parametrize(
    "updates",
    [
        {"density_fitted": True},
        {"aos": 64},
        {"grid_points": 50_000},
        {"tile_points": 128},
        {"max_device_bytes": (512 << 20) - 1},
        {"max_host_bytes": (256 << 20) - 1},
    ],
)
def test_adjacent_unqualified_workloads_fall_back_dense(
    updates: dict[str, object],
) -> None:
    decision = resolve_force_active_ao_policy(
        _workload(**updates),
        profiles=(_profile(),),
    )
    assert not decision.selected
    assert decision.reason == "no-qualified-profile"


@pytest.mark.parametrize("architecture", ("sm_89", "sm_90", "sm_120"))
def test_custom_profile_admission_is_architecture_independent(
    architecture: str,
) -> None:
    decision = resolve_force_active_ao_policy(
        _workload(architecture=architecture), profiles=(_profile(),)
    )
    assert decision.selected
    assert decision.profile_id == "test-qualified-domain"


def test_overlapping_profiles_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="overlapping"):
        resolve_force_active_ao_policy(
            _workload(),
            profiles=(
                _profile(),
                _profile(profile_id="duplicate"),
            ),
        )


def test_policy_record_preserves_selected_and_dense_ao_work() -> None:
    decision = resolve_force_active_ao_policy(
        _workload(),
        profiles=(_profile(),),
    )
    record = force_active_ao_policy_record(
        decision,
        {
            "point_ao_square_sum": 25,
            "dense_point_ao_square_sum": 100,
            "empty_tiles": 1,
        },
    )
    assert record["actual_mode"] == "selected"
    assert record["selected_point_ao_square_sum"] == 25
    assert record["dense_point_ao_square_sum"] == 100


def test_missing_map_work_never_invents_sparse_execution() -> None:
    dense = resolve_force_active_ao_policy(_workload(), profiles=())
    assert force_active_ao_policy_record(dense, None)["actual_mode"] == "dense"
    selected = resolve_force_active_ao_policy(
        _workload(),
        profiles=(_profile(),),
    )
    assert (
        force_active_ao_policy_record(selected, None)["actual_mode"] == "dense-fallback"
    )
