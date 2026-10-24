"""Native spin-root retained numeric ownership and public budget admission."""

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._spin_root_native import _NativeSpinModel
from generativeqc.spin_roots import SpinRootPolicy, SpinRootUnqualified
from generativeqc_compiler.dft import NativeAO

ATOMS = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]


def calculator() -> Calculator:
    return Calculator(
        "lda-uks",
        basis="sto-3g",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=16, angular_azimuth=32)
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-11,
        initial_guess=None,
    )


def retained_grid_storage(model: _NativeSpinModel) -> int:
    """Count independent retained storage, excluding aliases and native arenas."""
    arrays = (
        model._snapshot.values,
        model.state.grid.points,
        model.state.grid.weights,
    )
    assert all(
        not np.shares_memory(left, right)
        for i, left in enumerate(arrays)
        for right in arrays[i + 1 :]
    )
    return sum(array.nbytes for array in arrays)


def model_for(calc: Calculator, basis: NativeAO) -> _NativeSpinModel:
    return _NativeSpinModel(
        calc,
        basis,
        "native-retained-storage-test",
        charge=0,
        multiplicity=1,
        properties=("energy",),
        policy=SpinRootPolicy(recover=False),
    )


def test_native_uks_retained_grid_storage_is_accounted_after_close() -> None:
    with NativeAO(ATOMS, basis="sto-3g") as basis:
        model = model_for(calculator(), basis)
        try:
            retained = retained_grid_storage(model)
            assert model.workspace_bytes >= retained
        finally:
            model.close()
        # Closing native owners leaves Python buffers live for the workflow's
        # continuation/discarded-model lifetime; their reservation must remain.
        assert retained_grid_storage(model) == retained
        assert model.workspace_bytes >= retained


def test_native_uks_rejects_budget_smaller_than_retained_grid_storage() -> None:
    calc = calculator()
    with NativeAO(ATOMS, basis="sto-3g") as basis:
        model = model_for(calc, basis)
        try:
            retained = retained_grid_storage(model)
        finally:
            model.close()
    with pytest.raises(SpinRootUnqualified, match="numeric budget"):
        calc.stable_singlepoint(
            ATOMS,
            properties=("energy",),
            policy=SpinRootPolicy(recover=False, memory_budget_bytes=retained - 1),
        )
    root = calc.stable_singlepoint(
        ATOMS,
        properties=("energy",),
        policy=SpinRootPolicy(recover=False, memory_budget_bytes=4 * retained),
    )
    assert root.diagnostics["status"] == "STABLE"
