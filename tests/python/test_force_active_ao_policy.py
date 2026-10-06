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
        "architectures": ("sm_120",),
        "compositions": ("ordinary", "composite"),
        "derivative_orders": (1, 2),
        "spin_blocks": (1, 2),
        "density_fitted": False,
        "min_atoms": 24,
        "max_atoms": 96,
        "min_aos": 128,
        "max_aos": 1024,
        "min_grid_points": 100_000,
        "max_grid_points": 4_000_000,
        "tile_policy": "fixed",
        "tile_points": 256,
        "min_device_bytes": 512 << 20,
        "min_host_bytes": 256 << 20,
        "cutoff": 1e-16,
        "cache_bytes": 16 << 20,
    }
    values.update(updates)
    return QualifiedForceActiveAoProfile(**values)


def test_production_auto_policy_promotes_only_the_measured_large_rks_envelope() -> None:
    assert tuple(
        profile.profile_id for profile in QUALIFIED_FORCE_ACTIVE_AO_PROFILES
    ) == ("sm120-ordinary-rks-second-jet-v1",)

    below_envelope = resolve_force_active_ao_policy(_workload())
    assert not below_envelope.selected
    assert below_envelope.reason == "no-qualified-profile"

    for workload in (
        _workload(grid_points=1_179_648),
        _workload(atoms=96, aos=768, grid_points=2_359_296),
        _workload(atoms=72, aos=576, grid_points=1_769_472),
    ):
        decision = resolve_force_active_ao_policy(workload)
        assert decision.selected
        assert decision.profile_id == "sm120-ordinary-rks-second-jet-v1"
        assert decision.cutoff == 1e-16
        assert decision.cache_bytes == 16 << 20


@pytest.mark.parametrize(
    "updates",
    [
        {"architecture": "sm_90", "grid_points": 1_179_648},
        {"derivative_order": 1, "grid_points": 1_179_648},
        {"spin_blocks": 2, "grid_points": 1_179_648},
        {"composition": "composite", "grid_points": 1_179_648},
        {"density_fitted": True, "grid_points": 1_179_648},
        {"atoms": 47, "grid_points": 1_179_648},
        {"aos": 383, "grid_points": 1_179_648},
        {"atoms": 97, "aos": 768, "grid_points": 2_359_296},
        {"aos": 769, "grid_points": 2_359_296},
        {"grid_points": 2_359_297},
        {"tile_points": 128, "grid_points": 1_179_648},
        {"max_device_bytes": (512 << 20) - 1, "grid_points": 1_179_648},
        {"max_host_bytes": (256 << 20) - 1, "grid_points": 1_179_648},
    ],
)
def test_production_profile_keeps_adjacent_unqualified_domains_dense(
    updates: dict[str, object],
) -> None:
    decision = resolve_force_active_ao_policy(_workload(**updates))
    assert not decision.selected
    assert decision.reason == "no-qualified-profile"


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
    for forbidden in ("method", "functional", "selector"):
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
        {"architecture": "sm_90"},
        {"density_fitted": True},
        {"atoms": 12},
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
    dense = resolve_force_active_ao_policy(_workload())
    assert force_active_ao_policy_record(dense, None)["actual_mode"] == "dense"
    selected = resolve_force_active_ao_policy(
        _workload(),
        profiles=(_profile(),),
    )
    assert (
        force_active_ao_policy_record(selected, None)["actual_mode"] == "dense-fallback"
    )
