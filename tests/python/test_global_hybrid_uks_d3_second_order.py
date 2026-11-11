"""Independent total UKS/D3 curvature and matching public composition."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions
from generativeqc.composite_rks_hessian import (
    CompositeRKSResponse,
    CompositeUKSResponse,
    composite_rks_hessian,
    composite_rks_hvp,
)
from generativeqc.d3_hessian import NativeD3SecondOrder, compose_second_order
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import DispersionCorrectionPrimitive, resolve_method
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec
from test_dft_complete_cpu import independent_global_hybrid_gradient
from test_integral_autodiff_second import require_native_library  # noqa: F401

MANIFEST = json.loads(
    (Path(__file__).parents[1] / "data/d3_second_order/uks_manifest.json").read_text()
)
ATOMS = tuple(Atom.from_value(atom) for atom in MANIFEST["atoms_bohr"])
GRID = GridSpec(**MANIFEST["grid"])
DIRECTION = np.array(MANIFEST["direction"])
GATES = MANIFEST["gates"]
SPEC = D3Spec(**d3_parameters("B3LYP-D3(BJ)"))
ELECTRONIC = resolve_method("B3LYP", spin="polarized")
METHOD = replace(
    ELECTRONIC,
    identifier="low-level-b3lyp-uks-d3-bj-qualification",
    primitives=(*ELECTRONIC.primitives, DispersionCorrectionPrimitive(SPEC)),
)
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


def independent_total(atoms: tuple) -> tuple:
    """Reconverge only the independent UKS on the same moving atomic grid."""
    interface = pytest.importorskip("dftd3.interface")
    pytest.importorskip("pyscf")
    grid = MolecularGrid(atoms, GRID, 0, 2).explicit()
    atomic_weights = np.concatenate(
        [MolecularGrid((atom,), GRID, 0, 1).explicit().weights for atom in atoms]
    )
    state = SimpleNamespace(
        grid=grid,
        _source=SimpleNamespace(grid_spec=GRID, atomic_weights=atomic_weights),
    )
    functional = ELECTRONIC.primitives[0].functional
    code = "+".join(f"{float(weight)}*{name}" for name, weight in functional.components)
    code += f"+{float(ELECTRONIC.primitives[1].coefficient)}*HF"
    with NativeAO(atoms, representation="spherical", multiplicity=2) as basis:
        energy, gradient, reference = independent_global_hybrid_gradient(
            basis, state, "b3lyp-uks", xc=code, return_mean_field=True
        )
    dispersion = interface.DispersionModel(
        np.array([atom.atomic_number for atom in atoms]),
        np.array([atom.position for atom in atoms]),
    ).get_dispersion(
        interface.RationalDampingParam(
            s6=SPEC.s6, s8=SPEC.s8, a1=SPEC.a1, a2=SPEC.a2, s9=SPEC.s9
        ),
        grad=True,
    )
    return (
        energy + float(dispersion["energy"]),
        gradient + dispersion["gradient"],
        reference,
    )


def independent_action(direction: np.ndarray, step: float) -> np.ndarray:
    """Use independent gradients, never displaced production electronic states."""
    gradients = {}
    for offset in (-2, -1, 1, 2):
        atoms = tuple(
            Atom(atom.atomic_number, np.asarray(atom.position) + offset * step * motion)
            for atom, motion in zip(ATOMS, direction, strict=True)
        )
        gradients[offset] = independent_total(atoms)[1]
    return (8 * (gradients[1] - gradients[-1]) - (gradients[2] - gradients[-2])) / (
        12 * step
    )


@pytest.fixture(scope="module")
def total_results(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Close electronic owners before additive publication and oracle replay."""
    cache = tmp_path_factory.mktemp("uks-d3-curvature")
    calculator = Calculator(
        method=METHOD,
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    assert calculator.capabilities.supported_second_order == {"hvp", "hessian"}
    try:
        with (
            calculator.prepare_batch(
                [ATOMS], multiplicities=[2], warm_start=False
            ) as batch,
            NativeAO(ATOMS, representation="spherical", multiplicity=2) as basis,
        ):
            primal = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            with CompositeRKSResponse.from_native(batch, basis) as operator:
                assert isinstance(operator, CompositeUKSResponse)
                assert operator.layout.occupied == tuple(MANIFEST["occupations"])
                density = np.array(operator.state.density)
                overlap = np.array(operator.state.overlap)
                matrix = composite_rks_hessian(
                    operator, cache=cache, solver_options=SOLVER
                )
                action = composite_rks_hvp(
                    operator, DIRECTION, cache=cache, solver_options=SOLVER
                )
                identities = {
                    "geometry_identity": operator.state.identity.geometry_identity,
                    "electronic_method_identity": operator.state._source.method_ir.identity,
                    "scf_density_sha256": hashlib.sha256(
                        np.asarray(density, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                    "scf_basis_identity": operator.state.identity.basis_identity,
                    "scf_grid_identity": operator.state.identity.grid_identity,
                    "scf_grid_sha256": hashlib.sha256(
                        b"generativeqc.scf-grid-v1\0"
                        + str(len(operator.state.grid.points)).encode("ascii")
                        + b"\0"
                        + np.asarray(operator.state.grid.points, dtype="<f8").tobytes()
                        + np.asarray(operator.state.grid.weights, dtype="<f8").tobytes()
                        + np.asarray(operator.state.grid.owners, dtype="<i4").tobytes()
                    ).hexdigest(),
                }
                matrix, action = (
                    replace(
                        result,
                        _diagnostics=MappingProxyType(
                            {**result.diagnostics, **identities}
                        ),
                    )
                    for result in (matrix, action)
                )
        total_matrix = compose_second_order(
            matrix,
            METHOD,
            ATOMS,
            cache=cache,
            max_workspace_bytes=64 << 20,
            output_budget_bytes=64 << 20,
        )
        total_action = compose_second_order(
            action,
            METHOD,
            ATOMS,
            cache=cache,
            max_workspace_bytes=64 << 20,
        )
        return (
            primal,
            density,
            overlap,
            matrix,
            action,
            total_matrix,
            total_action,
            cache,
        )
    finally:
        calculator.clear_cache()


def test_public_uks_d3_matches_independently_qualified_total(
    total_results: tuple,
) -> None:
    """Dispatch the same coupled state and add the correction exactly once."""
    total_matrix, total_action, cache = total_results[5:]
    calculator = Calculator(
        method=METHOD,
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    try:
        result = calculator.hessian(
            ATOMS, multiplicity=2, cache=cache, solver_options=SOLVER
        )
        np.testing.assert_allclose(
            result.matrix, total_matrix.matrix, atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(
            result.matrix @ DIRECTION.ravel(),
            total_action.value.ravel(),
            atol=2e-8,
            rtol=0,
        )
        assert result.diagnostics["public_calculator_endpoint"]
        assert result.diagnostics["spin"] == "polarized"
        assert result.diagnostics["scf_fock_builds"] > 0
        for key in ("scf_density_sha256", "scf_basis_identity", "scf_grid_sha256"):
            assert result.diagnostics[key] == total_matrix.diagnostics[key], key
        assert result.diagnostics["dispersion"]["electronic_response_solves"] == 0
    finally:
        calculator.clear_cache()


def test_uks_d3_energy_forces_and_spin_state_match_independent_total(
    total_results: tuple,
) -> None:
    """Require the same coupled doublet, not merely a converged total energy."""
    primal, density, overlap, *_ = total_results
    energy, gradient, reference = independent_total(ATOMS)
    assert primal.converged and primal.dispersion is not None
    assert abs(primal.energy - energy) < GATES["energy_abs_hartree"]
    np.testing.assert_allclose(
        primal.forces, -gradient, atol=GATES["gradient_abs_hartree_per_bohr"], rtol=0
    )
    np.testing.assert_allclose(
        density, reference.make_rdm1(), atol=GATES["density_max"], rtol=0
    )
    np.testing.assert_allclose(
        np.einsum("sij,ji->s", density, overlap),
        MANIFEST["occupations"],
        atol=2e-12,
        rtol=0,
    )
    assert (
        np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
        < GATES["orbital_gradient_norm"]
    )


def test_uks_d3_composition_is_once_complete_and_raw(total_results: tuple) -> None:
    """Keep the coupled electronic solve and add exactly one geometry-only source."""
    _, _, _, electronic_matrix, electronic_action, matrix, action, cache = total_results
    correction = NativeD3SecondOrder(
        SPEC,
        [atom.atomic_number for atom in ATOMS],
        [atom.position for atom in ATOMS],
        cache=cache,
    ).hessian()
    np.testing.assert_allclose(
        matrix.matrix - electronic_matrix.matrix,
        correction.matrix,
        atol=GATES["additive_source_abs"],
        rtol=0,
    )
    assert np.max(abs(correction.matrix)) > 20 * GATES["additive_source_abs"]
    for incorrect in (
        electronic_matrix.matrix,
        electronic_matrix.matrix - correction.matrix,
        electronic_matrix.matrix + 2 * correction.matrix,
    ):
        assert (
            np.max(abs(matrix.matrix - incorrect)) > 20 * GATES["additive_source_abs"]
        )
    np.testing.assert_allclose(
        matrix.matrix, matrix.matrix.T, atol=GATES["raw_invariants_abs"], rtol=0
    )
    np.testing.assert_allclose(
        matrix.matrix @ np.tile(np.eye(3), (len(ATOMS), 1)),
        0,
        atol=GATES["raw_invariants_abs"],
        rtol=0,
    )
    np.testing.assert_allclose(
        matrix.matrix @ DIRECTION.ravel(),
        action.value.ravel(),
        atol=GATES["raw_invariants_abs"],
        rtol=0,
    )
    assert set(action.components) == {*electronic_action.components, "d3_bj_two_body"}
    np.testing.assert_allclose(
        sum(action.components.values()), action.value, atol=2e-12, rtol=0
    )
    for total, electronic in ((matrix, electronic_matrix), (action, electronic_action)):
        assert (
            total.diagnostics["nuclear_response_solves"]
            == electronic.diagnostics["nuclear_response_solves"]
        )
        assert total.diagnostics["displaced_scf_evaluations"] == 0
        assert total.diagnostics["dispersion"]["electronic_response_solves"] == 0
        assert total.diagnostics["total_method_identity"] == METHOD.identity
        assert not total.diagnostics["public_calculator_endpoint"]
    assert (
        action.diagnostics["response_residual_norm"] < GATES["response_residual_norm"]
    )
    with pytest.raises(ValueError, match="already present"):
        compose_second_order(
            action,
            METHOD,
            ATOMS,
            cache=cache,
            max_workspace_bytes=64 << 20,
        )


def test_uks_d3_total_curvature_matches_independent_multistep_gradients(
    total_results: tuple,
    record_property: Any,
) -> None:
    """Retain frozen total-method gates for all raw columns and an independent HVP."""
    *_, matrix, action, _ = total_results
    directions = np.eye(DIRECTION.size).reshape(DIRECTION.size, *DIRECTION.shape)
    for step in MANIFEST["difference_steps_bohr"]:
        reference = np.column_stack(
            [independent_action(direction, step).ravel() for direction in directions]
        )
        error = float(np.max(abs(matrix.matrix - reference)))
        record_property(f"independent_total_raw_error_{step}", error)
        assert error < GATES["total_hessian_abs_hartree_per_bohr2"]
        np.testing.assert_allclose(
            action.value,
            independent_action(DIRECTION, step),
            atol=GATES["total_hvp_abs_hartree_per_bohr2"],
            rtol=0,
        )
    record_property("total_hessian_diagnostics", json.dumps(matrix.diagnostics))
    record_property("total_hvp_diagnostics", json.dumps(action.diagnostics))
