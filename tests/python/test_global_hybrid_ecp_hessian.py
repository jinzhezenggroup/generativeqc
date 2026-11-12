"""Low-level scalar-def2-ECP RKS curvature, without public domain promotion."""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.composite_rks_hessian import CompositeRKSResponse, composite_rks_hvp
from generativeqc.d3_hessian import compose_second_order
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import DispersionCorrectionPrimitive
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec
from test_global_hybrid_ecp_basis import canonical_records as _canonical_records
from test_global_hybrid_ecp_forces import (
    independent_solver,
    make_calculator,
    molecule_atoms,
)
from test_integral_autodiff_second import require_native_library  # noqa: F401

ecp_records = _canonical_records
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)
D3_MANIFEST = json.loads(
    (Path(__file__).parents[1] / "data/d3_second_order/ecp_manifest.json").read_text()
)


def assert_independent_d3_total(
    electronic: Any,
    method: Any,
    atoms: list,
    direction: np.ndarray,
    independent_actions: list,
    cache: Any,
    record_property: Any,
) -> None:
    """Compose only after electronic closure, keeping elemental Cs distinct from ECP9.

    The already-converged independent electronic actions are reused; no production
    displacement or additional electronic response supplies the correction.
    """
    interface = pytest.importorskip("dftd3.interface")
    specification = D3Spec(**d3_parameters("B3LYP-D3(BJ)"))
    total_method = replace(
        method,
        primitives=(*method.primitives, DispersionCorrectionPrimitive(specification)),
    )
    native_atoms = tuple(Atom.from_value(atom) for atom in atoms)
    positions = np.array([atom.position for atom in native_atoms])
    numbers = np.array([atom.atomic_number for atom in native_atoms])
    np.testing.assert_array_equal(numbers, D3_MANIFEST["elemental_atomic_numbers"])
    total = compose_second_order(
        electronic,
        total_method,
        native_atoms,
        cache=cache,
        max_workspace_bytes=64 << 20,
    )
    damping = interface.RationalDampingParam(
        s6=specification.s6,
        s8=specification.s8,
        a1=specification.a1,
        a2=specification.a2,
        s9=specification.s9,
    )
    gates = D3_MANIFEST["gates"]
    correction = total.value - electronic.value
    assert np.max(abs(correction)) > 20 * gates["additive_source_abs"]
    errors = []
    for step, electronic_action in independent_actions:
        gradients = [
            interface.DispersionModel(
                numbers, positions + sign * step * direction
            ).get_dispersion(damping, grad=True)["gradient"]
            for sign in (1, -1)
        ]
        finite = (gradients[0] - gradients[1]) / (2 * step)
        np.testing.assert_allclose(
            correction,
            finite,
            atol=gates["d3_hvp_abs_hartree_per_bohr2"],
            rtol=0,
        )
        errors.append(
            float(
                np.max(abs(total.value - electronic_action - finite))
                / np.linalg.norm(direction)
            )
        )
    assert max(errors) < gates["normalized_total_hvp_abs_hartree_per_bohr2"]
    assert set(total.components) == {*electronic.components, "d3_bj_two_body"}
    np.testing.assert_allclose(
        sum(total.components.values()), total.value, atol=2e-12, rtol=0
    )
    assert (
        total.diagnostics["nuclear_response_solves"]
        == electronic.diagnostics["nuclear_response_solves"]
    )
    assert total.diagnostics["dispersion"]["electronic_response_solves"] == 0
    assert total.diagnostics["displaced_scf_evaluations"] == 0
    assert not total.diagnostics["public_calculator_endpoint"]
    record_property("independent_total_normalized_hvp_errors", errors)
    record_property("composed_d3_hvp_diagnostics", json.dumps(total.diagnostics))


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
                electronic_method = operator.state._source.method_ir
                result = replace(
                    result,
                    _diagnostics=MappingProxyType(
                        {
                            **result.diagnostics,
                            "electronic_method_identity": electronic_method.identity,
                            "geometry_identity": operator.state.identity.geometry_identity,
                        }
                    ),
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
                independent_actions = []
                for step in D3_MANIFEST["difference_steps_bohr"]:
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
                    independent_actions.append((step, finite))
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
    assert_independent_d3_total(
        result,
        electronic_method,
        atoms,
        direction,
        independent_actions,
        tmp_path,
        record_property,
    )
    record_property(
        "whole_low_level_total_validation_seconds", perf_counter() - started
    )
