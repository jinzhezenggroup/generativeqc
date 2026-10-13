"""The static report charges the same default force owners as dry admission."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _stationary_cuda as runtime

from tools.dft_mp_v1 import qualify_capacity

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=("water", "water32"))
def frozen_basis(request: pytest.FixtureRequest) -> Any:
    root = ROOT / "tools/dft_mp_v1"
    manifest = json.loads((root / "manifest.json").read_text())
    value = json.loads((root / manifest["cases"][request.param]["input"]).read_text())
    atoms = tuple(qualify_capacity.Atom.from_value(atom) for atom in value["atoms"])
    _, (basis, _) = qualify_capacity._basis_shape(
        atoms, value["charge"], value["multiplicity"]
    )
    return basis


@pytest.mark.parametrize("functional", (0, 1), ids=("lda", "pbe"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"), ids=("rks", "uks"))
def test_report_matches_default_dense_feature_force_owners(
    frozen_basis: Any,
    functional: int,
    spin: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unused indexed packing and writable potential staging are not charged."""
    defaults = inspect.signature(runtime.complete_rks_cuda_gradient_diagnostic)
    limits = {
        "tile_points": defaults.parameters["tile_points"].default,
        "primitive_tile": defaults.parameters["primitive_tile"].default,
        "integral_terms": defaults.parameters["integral_terms"].default,
        "additional_device_bytes": defaults.parameters["max_device_bytes"].default,
        "additional_host_bytes": defaults.parameters["max_host_bytes"].default,
    }
    observed_plans = []
    plan_tiles = qualify_capacity.plan_tiles

    def observe_grid_plan(*args: Any, **kwargs: Any) -> Any:
        result = plan_tiles(*args, **kwargs)
        observed_plans.append(result)
        return result

    monkeypatch.setattr(qualify_capacity, "plan_tiles", observe_grid_plan)
    memory, plan = qualify_capacity._method_resources(
        frozen_basis,
        atom_count=frozen_basis.natom,
        functional=functional,
        spin=spin,
        limits=limits,
    )
    layout = runtime._plan_stationary_cuda_tile(
        SimpleNamespace(
            _source=SimpleNamespace(cuda_integral_derivatives=lambda *_: None)
        ),
        frozen_basis,
        plan=plan,
        target=qualify_capacity.cuda_target_info("sm_120"),
        needs_first=functional != 0,
        tile_points=limits["tile_points"],
        primitive_tile=limits["primitive_tile"],
        integral_terms=limits["integral_terms"],
        source_names=qualify_capacity.stationary_runtime_sources(plan),
        ecp=False,
        max_device_bytes=limits["additional_device_bytes"],
        max_host_bytes=limits["additional_host_bytes"],
        max_ecp_pair_samples=defaults.parameters["max_ecp_pair_samples"].default,
        reserve_active_ao=False,
        features_only=True,
    )

    # The default force has no AO maps and cannot consume potential matrices.
    # Observe the planned capability as well as its bytes: the features-only
    # flag need not change numeric storage on a dense, non-indexed owner.
    assert defaults.parameters["resident_ao_cutoff"].default is None
    assert observed_plans
    assert all(owner.active_ao_capacity is None for owner in observed_plans)
    assert all(owner.features_only for owner in observed_plans)
    assert memory["grid_active_ao_capacity"] is None
    assert memory["grid_features_only"] is True
    assert memory["grid_tile_peak_bytes"] == layout.grid_plan.peak_bytes
    assert memory["stationary_source_bytes"] == layout.source_resources.allocation_bytes
    assert memory["additional_host_numeric_bound"] == layout.host_bound
    assert memory["stationary_native_integral_host_reserve_bytes"] == (
        layout.native_integral_host_reserve
    )
    assert memory["stationary_native_pair_reserve_bytes"] == (
        layout.native_geometry_reserve
    )
    force_peak = layout.grid_plan.peak_bytes + layout.source_resources.allocation_bytes
    assert memory["stationary_grid_device_peak_bound"] == force_peak
    assert memory["additional_device_peak_bound"] == (
        force_peak + layout.native_geometry_reserve
    )
    assert memory["native_integral_device_budget"] == (
        limits["additional_device_bytes"] - force_peak
    )
