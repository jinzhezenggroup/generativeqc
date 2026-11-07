"""Low-level scalar-def2-ECP RKS curvature, without public domain promotion."""

from time import perf_counter

import numpy as np
import pytest
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.composite_rks_hessian import CompositeRKSResponse, composite_rks_hvp
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from test_global_hybrid_ecp_basis import canonical_records as _canonical_records
from test_global_hybrid_ecp_forces import (
    independent_solver,
    make_calculator,
    molecule_atoms,
)
from test_integral_autodiff_second import require_native_library  # noqa: F401

ecp_records = _canonical_records
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


def independent_moved_gradient(
    raw: dict, atoms: list, atomic_rules: dict
) -> np.ndarray:
    """Reconverge the oracle on its own analytic-gradient partition arithmetic.

    Far-field rational-grid volumes amplify otherwise harmless Becke rounding.
    Bound differences in the AO/jet space instead of comparing bare
    volumes, then reconverge on the exact reference weights used by its gradient.
    Physical force/HVP acceptance thresholds remain unchanged.
    """
    from pyscf.dft import gen_grid

    solver = independent_solver(raw, atoms, 0, 1)
    solver.grids.gen_atomic_grids = lambda *args, **kwargs: atomic_rules
    points, weights = gen_grid.get_partition(
        solver.mol, atomic_rules, radii_adjust=None
    )
    np.testing.assert_allclose(points, solver.grids.coords, atol=1e-12, rtol=0)
    jets = solver._numint.eval_ao(solver.mol, points, deriv=1)
    envelope = np.max(abs(jets), axis=0)
    difference = abs(weights - solver.grids.weights)
    bound = envelope.T @ (difference[:, None] * envelope)
    scale = envelope.T @ (abs(weights)[:, None] * envelope)
    assert np.max(bound) <= 1e-12 * max(1.0, float(np.max(scale)))
    original_density = solver.make_rdm1()
    original_energy = solver.e_tot
    solver.grids.coords = np.array(points)
    solver.grids.weights = np.array(weights)
    solver.kernel(dm0=original_density)
    assert solver.converged and abs(solver.e_tot - original_energy) <= 1e-10
    np.testing.assert_allclose(solver.make_rdm1(), original_density, atol=2e-8, rtol=0)
    gradient = solver.nuc_grad_method()
    gradient.grid_response = True
    return gradient.kernel()


# Two complete Cs-ECP moving-grid HVP oracles take >14 minutes each on
# hosted CPU runners. Preserve both strict numerical gates for nightly/manual
# qualification; routine PR/merge-group CI keeps the shorter response tests.
@pytest.mark.ci_nightly
@pytest.mark.parametrize(
    "direction",
    (
        np.array(((0.02, -0.03, 0.1), (0.0, 0.0, 0.0))),
        np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    ),
    ids=("original-direction", "unit-cs-x"),
)
def test_low_level_def2_cs_rks_hvp_matches_independent_moving_grid_gradients(
    ecp_records: tuple,
    tmp_path: object,
    record_property: object,
    direction: np.ndarray,
) -> None:
    """Retain the 1e-5 oracle gate and the original unscaled unit-column failure."""
    pytest.importorskip("pyscf")
    _, raw, record = ecp_records
    atoms = molecule_atoms("Cs")
    calculator = make_calculator(record)
    assert not calculator.capabilities.supported_second_order
    started = perf_counter()
    with (
        calculator.prepare_batch([atoms], warm_start=False) as batch,
        NativeAO(atoms, basis=record, representation="spherical") as basis,
    ):
        native = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        state = StationaryKsState.from_native(batch, basis)
        try:
            with CompositeRKSResponse.from_native(
                batch, basis, grid_budget_bytes=512 << 20, max_nonlocal_pairs=1
            ) as operator:
                result = composite_rks_hvp(
                    operator, direction, cache=tmp_path, solver_options=SOLVER
                )
                record_property(
                    "whole_low_level_native_hvp_seconds", perf_counter() - started
                )
                record_property("candidate_hvp_value", result.value.tolist())
                record_property("hvp_diagnostics", str(dict(result.diagnostics)))
                record_property("ecp_work_observation", dict(operator.ecp.statistics))
                assert not result.diagnostics["public_calculator_endpoint"]
                assert result.diagnostics["ecp_ad_backend"] == "cpu"
                assert result.diagnostics["displaced_scf_evaluations"] == 0
                assert result.diagnostics["response_residual_norm"] <= 1e-9
                assert set(result.components) == {
                    "one_electron",
                    "coulomb",
                    "xc_ao",
                    "xc_grid",
                    "xc_weight",
                    "exact_exchange",
                    "overlap_pulay",
                    "nuclear",
                    "ecp_local",
                    "ecp_nonlocal",
                }
                assert (
                    max(
                        np.max(abs(result.components[name]))
                        for name in ("ecp_local", "ecp_nonlocal")
                    )
                    > 1e-5
                )
                np.testing.assert_allclose(
                    result.value.sum(axis=0), 0, atol=2e-7, rtol=0
                )
                reference = independent_solver(raw, atoms, 0, 1, state)
                assert abs(native.energy - reference.e_tot) <= 2e-8
                np.testing.assert_allclose(
                    state.density[0], reference.make_rdm1(), atol=2e-7, rtol=0
                )
                gradient = reference.nuc_grad_method()
                gradient.grid_response = True
                np.testing.assert_allclose(
                    native.forces, -gradient.kernel(), atol=1e-5, rtol=0
                )
                owners = np.asarray(state.grid.owners)
                atomic_rules = {
                    symbol: (
                        state.grid.points[owners == index] - np.asarray(center),
                        np.asarray(state._source.atomic_weights)[owners == index],
                    )
                    for index, (symbol, center) in enumerate(atoms)
                }
                errors = []
                for step in (3e-3, 1e-3, 3e-4):
                    gradients = []
                    for sign in (1, -1):
                        moved = [
                            (symbol, np.asarray(center) + sign * step * motion)
                            for (symbol, center), motion in zip(
                                atoms, direction, strict=True
                            )
                        ]
                        gradients.append(
                            independent_moved_gradient(raw, moved, atomic_rules)
                        )
                    finite = (gradients[0] - gradients[1]) / (2 * step)
                    errors.append(
                        float(
                            np.max(abs(result.value - finite))
                            / np.linalg.norm(direction)
                        )
                    )
                record_property("independent_normalized_hvp_errors", errors)
                assert max(errors) <= 1e-5
        finally:
            state._source.close()
