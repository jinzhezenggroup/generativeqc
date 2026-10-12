"""Slurm-only acceptance of complete, context-admitted native CUDA forces."""

from __future__ import annotations

import ctypes
import json
import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions, ResourceBudget, _native
from generativeqc_compiler.common.resources import CpuResourceObservation
from test_dft_complete_cpu import (
    ATOMS,
    GRID,
    NativeAO,
    StationaryKsState,
    independent_global_hybrid_gradient,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated Slurm GPU job",
)
H2 = [(1, (0.12, -0.09, -0.71)), (1, (-0.08, 0.13, 0.79))]


def calculator(method: str, **options: Any) -> Calculator:
    """Use production schedules and strict SCF controls, not a kernel override."""
    return Calculator(
        method=method,
        device="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
        **options,
    )


def forbid_python_force(batch: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A public native request must not discover compilers or scientific frames."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail(
            "admitted CUDA force invoked Python scientific/compiler orchestration"
        )

    for name in (
        "_public_dft_cuda_force",
        "_public_dft_cpu_force",
        "_stationary_cuda_compiler",
    ):
        monkeypatch.setattr(batch, name, forbidden)
    monkeypatch.setattr(StationaryKsState, "from_native", forbidden)


@pytest.mark.parametrize(
    "method,default_grid",
    [(method, False) for method in ("pbe-rks", "pbe0-rks", "pbe-uks", "pbe0-uks")]
    + [("pbe-rks", True), ("pbe-uks", True)],
)
def test_public_native_force_matches_independent_grid_and_final_state(
    method: str, default_grid: bool
) -> None:
    """Accept both the explicit fixture grid and actual public v2 defaults."""
    pytest.importorskip("pyscf", reason="independent native CUDA acceptance")
    unrestricted = method.endswith("uks")
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    options = {} if default_grid else {"ks_options": KsOptions(grid=GRID)}
    calc = calculator(method, **options)
    # The default restricted MINAO proposal legitimately transfers setup
    # matrices. Compare identical cold trajectories instead of mistaking
    # pre-existing initial-guess work for a final D/W export by the force.
    with calc.prepare_batch(
        [ATOMS], charges=[charge], multiplicities=[multiplicity]
    ) as energy_batch:
        baseline = energy_batch.execute(strict=True, properties=("energy",)).items[0]
        baseline_movement = energy_batch.ks_transport_diagnostics[0]
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        assert batch._native_dft_force_eligible()
        with pytest.MonkeyPatch.context() as patch:
            forbid_python_force(batch, patch)
            result = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
        movement = batch.ks_transport_diagnostics[0]
        assert movement.matrix_d2h_bytes == baseline_movement.matrix_d2h_bytes
        assert result.energy == pytest.approx(baseline.energy, abs=2e-9)
        state = StationaryKsState.from_native(batch, basis)
        energy, gradient, reference = independent_global_hybrid_gradient(
            basis,
            state,
            method,
            xc="PBE0" if "pbe0" in method else "PBE",
            return_mean_field=True,
        )
        assert result.energy == pytest.approx(energy, abs=2e-9)
        np.testing.assert_allclose(result.forces, -gradient, atol=2e-7, rtol=0)
        np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-10, rtol=0)
        points = np.random.default_rng(2227).normal(size=(3 * basis.nao + 20, 3))
        native_ao = basis.evaluate(points, order=0)[0]
        reference_ao = reference.mol.eval_gto("GTOval_cart", points)
        transform, _, rank, _ = np.linalg.lstsq(native_ao, reference_ao, rcond=1e-13)
        assert rank == basis.nao
        np.testing.assert_allclose(
            native_ao @ transform, reference_ao, atol=2e-11, rtol=2e-11
        )
        densities = reference.make_rdm1()
        coefficients, energies, occupations = (
            reference.mo_coeff,
            reference.mo_energy,
            reference.mo_occ,
        )
        if not unrestricted:
            densities = [densities]
            coefficients, energies, occupations = (
                [coefficients],
                [energies],
                [occupations],
            )
        for spin, (density, vectors, values, occupied) in enumerate(
            zip(densities, coefficients, energies, occupations, strict=True)
        ):
            weighted = (vectors * (values * occupied)) @ vectors.T
            for actual, expected in (
                (state.density[spin], density),
                (state.weighted_density[spin], weighted),
            ):
                np.testing.assert_allclose(
                    actual, transform @ expected @ transform.T, atol=3e-7, rtol=0
                )


@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe-uks", "pbe0-uks"))
def test_native_minimum_fleet_budget_survives_cold_warm_move_and_rollback(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every retained item fits; no early optional cache can starve a neighbor."""
    unrestricted = method.endswith("uks")
    charges = [1, 1] if unrestricted else [0, 0]
    multiplicities = [2, 2] if unrestricted else [1, 1]
    options = {"ks_options": KsOptions(grid=GRID)}
    probe = calculator(method, **options)
    request = probe._resource_request(
        [H2, ATOMS], charges=charges, multiplicities=multiplicities
    )
    minimum = next(
        item for item in request.candidates if item.name.endswith("-minimum")
    )
    from dataclasses import replace

    from generativeqc_compiler.common.resources import plan_resources

    minimum_plan = plan_resources(
        [replace(request, candidates=(minimum,))], ResourceBudget()
    )
    budget = ResourceBudget(
        host_bytes=minimum_plan.peak_bytes["host"],
        device_bytes=minimum_plan.peak_bytes["device"],
    )
    calc = calculator(method, resource_budget=budget, **options)
    with calc.prepare_batch(
        [H2, ATOMS], charges=charges, multiplicities=multiplicities
    ) as batch:
        ledger = batch._resource_ledger
        assert ledger is not None and batch._native_dft_force_eligible()
        chosen = next(
            item for item in batch.resource_plan.requests if item.name == "ks"
        )
        selected = next(
            item
            for item in chosen.candidates
            if item.name == dict(batch.resource_plan.selections)["ks"]
        )
        assert dict(selected.decisions)["cuda_stationary_cache"] == "minimum"
        inventory = json.loads(dict(selected.decisions)["item_device_inventory"])
        expected_reserved = sum(
            item["stationary"]["provider_allowance"] for item in inventory
        )
        forbid_python_force(batch, monkeypatch)
        cold = batch.execute(strict=True, properties=("energy", "forces"))
        warm = batch.execute(strict=True, properties=("energy", "forces"))
        for actual, expected in zip(warm.items, cold.items, strict=True):
            assert actual.warm_start_used
            np.testing.assert_allclose(
                actual.forces, expected.forces, atol=2e-7, rtol=0
            )
        moved = [np.array([position for _, position in atoms]) for atoms in (H2, ATOMS)]
        moved[0][1] += [0.02, -0.01, 0.03]
        moved[1][2] += [-0.01, 0.02, 0.01]
        changed = batch.execute(moved, strict=True, properties=("energy", "forces"))
        assert abs(changed.items[0].energy - cold.items[0].energy) > 1e-6
        failed = batch.execute(
            [np.zeros((1, 3)), moved[1]], properties=("energy", "forces")
        )
        assert failed.failure_indices == (0,) and failed.items[0].forces is None
        assert failed.items[1].converged and failed.items[1].forces is not None
        recovered = batch.execute(moved, strict=True, properties=("energy", "forces"))
        for actual, expected in zip(recovered.items, changed.items, strict=True):
            np.testing.assert_allclose(
                actual.forces, expected.forces, atol=2e-7, rtol=0
            )
        receipt = ledger.to_dict()
        assert not receipt["rejected_allocations"]
        assert receipt["admission"]["provider_reserved_bytes"] == expected_reserved
        assert receipt["admission"]["combined_peak_bytes"] <= ledger.limit
        batch._resource_ledger = None
    assert ledger.to_dict()["live_bytes"] == 0
    assert ledger.to_dict()["admission"]["provider_reserved_bytes"] == 0
    ledger.close()


@pytest.mark.parametrize(
    "hook_name,expected_status",
    (
        ("grid_cuda_fail_next_allocation_for_test_v1", _native.STATUS_OUT_OF_MEMORY),
        (
            "grid_cuda_fail_next_host_allocation_for_test_v1",
            _native.STATUS_OUT_OF_MEMORY,
        ),
        ("grid_cuda_fail_next_runtime_for_test_v1", _native.STATUS_CUDA_ERROR),
    ),
)
def test_public_cuda_failure_preserves_caller_output_and_warm_state(
    hook_name: str, expected_status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recoverable injected failures are typed, atomic and preserve warm state."""
    calc = calculator(
        "pbe-rks", ks_options=KsOptions(grid=GRID), resource_budget=ResourceBudget()
    )
    with calc.prepare_batch([H2]) as batch:
        assert batch._native_dft_force_eligible()
        baseline = batch.execute(strict=True, properties=("energy",)).items[0]
        ledger = batch._resource_ledger
        before = ledger.to_dict()
        forces = (ctypes.c_double * 6)(*[991.0] * 6)
        output = (_native.BatchItemResultDescriptor * 1)(
            _native.BatchItemResultDescriptor(
                struct_size=ctypes.sizeof(_native.BatchItemResultDescriptor),
                abi_version=_native.ABI_VERSION,
                energy=992.0,
                forces=forces,
                force_count=len(forces),
            )
        )
        hook = getattr(calc._library, hook_name)
        hook.argtypes, hook.restype = [], None
        hook()
        forbid_python_force(batch, monkeypatch)
        with CpuResourceObservation(calc._library, ledger=ledger):
            status = calc._library.generativeqc_batch_execute(
                batch._batch, None, 0, output, 1
            )
        assert status == _native.STATUS_SUCCESS
        assert output[0].status == expected_status
        assert output[0].energy == 992.0 and list(forces) == [991.0] * 6
        after = ledger.to_dict()
        assert after["live_bytes"] == before["live_bytes"]
        assert after["admission"]["provider_reserved_bytes"] == 0
        recovered = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        assert recovered.warm_start_used and recovered.energy == pytest.approx(
            baseline.energy, abs=2e-9
        )
        assert recovered.forces is not None and np.all(np.isfinite(recovered.forces))
