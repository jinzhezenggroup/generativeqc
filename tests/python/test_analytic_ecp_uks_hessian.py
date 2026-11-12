"""Synthetic coupled-spin ECP candidate gates, not public def2 qualification."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from generativeqc import BasisProvenance, GridSpec
from generativeqc.composite_rks_hessian import (
    CompositeRKSResponse,
    CompositeUKSResponse,
    composite_rks_hessian,
    composite_rks_hvp,
)
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from test_ecp import _record_from_resolved_basis
from test_global_hybrid_ecp_forces import make_calculator
from test_integral_autodiff_second import require_native_library  # noqa: F401

ATOMS = [
    ("Na", (3.7, 0.21, 0.17)),
    ("H", (0.13, 0.19, -0.7)),
    ("H", (-0.11, 0.07, 0.7)),
]
BASIS = {"Na": [[0, [0.65, 1.0]], [1, [0.45, 1.0]]], "H": [[0, [0.8, 1.0]]]}
TERMS = [
    [channel, [[], [], [[0.67, -0.12], [1.23, 0.081]]]] for channel in (-1, 0, 1, 2)
]
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


def independent_state(atoms: list, atomic_rules: dict) -> object:
    """Reconverge an independent UKS Hamiltonian on its own moving partition.

    Shared atomic nodes define the requested quadrature, not a shared electronic
    state. Each solve starts from PySCF's own guess, never a native density.
    """
    from pyscf import dft, gto, lib
    from pyscf.dft import gen_grid

    lib.num_threads(1)
    molecule = gto.M(
        atom=atoms,
        basis=BASIS,
        ecp={"Na": [10, TERMS]},
        unit="Bohr",
        spin=1,
        verbose=0,
    )
    reference = dft.UKS(molecule)
    reference.xc = "0.81*GGA_C_LYP+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.08*LDA_X+0.2*HF"
    reference.grids.radii_adjust = None
    reference.grids.coords, reference.grids.weights = gen_grid.get_partition(
        molecule, atomic_rules, radii_adjust=None
    )
    reference.grids.gen_atomic_grids = lambda *args, **kwargs: atomic_rules
    reference.small_rho_cutoff = 0
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-13, 1e-10, 240
    reference.kernel()
    assert reference.converged
    assert (
        np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ)) < 2e-9
    )
    return reference


def independent_gradient(atoms: list, atomic_rules: dict) -> np.ndarray:
    """Differentiate the independently reconverged moving-grid state."""
    reference = independent_state(atoms, atomic_rules)
    gradient = reference.nuc_grad_method()
    gradient.grid_response = True
    return gradient.kernel()


def independent_fixed_field_response(reference: object, fields: np.ndarray) -> object:
    """Differentiate exactly the supplied primal fields, without SCF seeding.

    Arbitrary AO directions at minority-spin nodes amplify tiny state differences
    and Libxc's spin-fraction cancellation. XCFun supplies an independent kernel
    at the native Cartesian fields; PySCF evaluates its own AOs and J/K integrals.
    The independently converged Libxc state remains the molecular/physical-OV
    oracle. This fixed-state check does not replace that state's density or SCF.
    """
    from pyscf.dft import numint

    integration = numint.NumInt()
    integration.libxc = pytest.importorskip("pyscf.dft.xcfun")
    code = "0.81*LYPC+0.72*BECKEX+0.19*VWN3C+0.08*SLATERX"
    rho = np.stack((fields[:4], fields[5:9]))
    _, potential, kernel, _ = integration.eval_xc_eff(code, rho, deriv=2, xctype="GGA")

    def response(moving: np.ndarray) -> np.ndarray:
        """Retain both output spins for every unscaled input-spin AO probe."""
        semilocal = integration.nr_uks_fxc(
            reference.mol,
            reference.grids,
            code,
            None,
            moving,
            hermi=1,
            rho0=rho,
            vxc=potential,
            fxc=kernel,
        )
        coulomb, exchange = reference.get_jk(reference.mol, moving, hermi=1)
        return semilocal + coulomb.sum(axis=0)[None] - 0.2 * exchange

    return response


def test_analytic_ecp_uks_raw_matrix_and_fresh_hvps_match_independent_gradients(
    tmp_path: Path, record_property: object
) -> None:
    """Retain full raw 1e-5 molecular and 2e-9 fixed-state/physical-OV gates.

    Na plus bonded H2 avoids the separate failed, different-root separated-H
    screening fixture. It does not resolve or relabel that failed qualification.
    The separate failed arbitrary-AO comparison between two slightly different
    SCF states remains recorded, not promoted to a fixed-state derivative oracle.
    """
    pytest.importorskip("pyscf")
    record = _record_from_resolved_basis(BASIS, {"Na": [10, TERMS]}, "spherical")
    record = replace(
        record,
        name="Synthetic analytic-ECP UKS Hessian oracle",
        provenance=BasisProvenance(
            "Explicit test-only coefficients in test_analytic_ecp_uks_hessian.py",
            "1",
            "Repository fixture, not canonical def2-ECP",
            record.provenance.checksum,
        ),
    )
    calculator = make_calculator(
        record, 2, grid=GridSpec(radial_points=16, angular_polar=8, angular_azimuth=16)
    )
    assert not calculator.capabilities.supported_second_order
    started = perf_counter()
    with (
        calculator.prepare_batch(
            [ATOMS], multiplicities=[2], warm_start=False
        ) as batch,
        NativeAO(
            ATOMS, basis=record, representation="spherical", multiplicity=2
        ) as basis,
    ):
        point = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        with CompositeRKSResponse.from_native(
            batch, basis, grid_budget_bytes=128 << 20, max_nonlocal_pairs=1
        ) as operator:
            assert isinstance(operator, CompositeUKSResponse)
            assert operator.layout.occupied == (2, 1)
            assert operator.ecp.statistics["analytic_gaussian"]
            assert operator.ecp.statistics["radial_layers"] == 0
            state = operator.state
            owners = np.asarray(state.grid.owners)
            atomic_rules = {
                symbol: (
                    state.grid.points[owners == index] - center,
                    np.asarray(state._source.atomic_weights)[owners == index],
                )
                for index, (symbol, center) in enumerate(ATOMS)
            }
            reference = independent_state(ATOMS, atomic_rules)
            assert abs(point.energy - reference.e_tot) < 2e-8
            np.testing.assert_allclose(
                state.density, reference.make_rdm1(), atol=2e-7, rtol=0
            )
            gradient = reference.nuc_grad_method()
            gradient.grid_response = True
            np.testing.assert_allclose(
                point.forces, -gradient.kernel(), atol=1e-5, rtol=0
            )
            np.testing.assert_allclose(
                np.einsum("sij,ji->s", state.density, state.overlap),
                [2, 1],
                atol=2e-10,
                rtol=0,
            )
            response = reference.gen_response(hermi=1)
            fixed_response = independent_fixed_field_response(
                reference, operator.xc_kernel.fields
            )
            random = np.random.default_rng(2847)
            unconditioned_errors, fixed_errors = [], []
            for spin in range(2):
                moving = np.zeros_like(state.density)
                trial = random.normal(size=(basis.nao, basis.nao)) * 0.03
                moving[spin] = trial + trial.T
                actual = operator.induced_fock(moving)
                unconditioned_errors.append(
                    float(np.max(abs(actual - response(moving))))
                )
                expected = fixed_response(moving)
                fixed_errors.append(float(np.max(abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
            record_property(
                "unconstrained_induced_fock_max_errors",
                unconditioned_errors,
            )
            record_property("fixed_field_random_ao_max_errors", fixed_errors)
            complete_errors, physical_errors = [], []
            for spin in range(2):
                for row in range(basis.nao):
                    for column in range(row, basis.nao):
                        moving = np.zeros_like(state.density)
                        moving[spin, row, column] = moving[spin, column, row] = 1
                        actual, expected = (
                            operator.induced_fock(moving),
                            fixed_response(moving),
                        )
                        complete_errors.append(float(np.max(abs(actual - expected))))
                        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
                occupied = operator.layout.occupied[spin]
                for virtual in range(occupied, basis.nao):
                    for orbital in range(occupied):
                        moving = np.zeros_like(state.density)
                        change = np.outer(
                            state.coefficients[spin, :, virtual],
                            state.coefficients[spin, :, orbital],
                        )
                        moving[spin] = change + change.T
                        difference = operator.induced_fock(moving) - response(moving)
                        for output_spin, count in enumerate(operator.layout.occupied):
                            coefficients = state.coefficients[output_spin]
                            block = (
                                coefficients[:, count:].T
                                @ difference[output_spin]
                                @ coefficients[:, :count]
                            )
                            physical_errors.append(float(np.max(abs(block))))
                            np.testing.assert_allclose(block, 0, atol=2e-9, rtol=0)
            record_property("fixed_field_complete_ao_max_error", max(complete_errors))
            record_property("complete_physical_ov_max_error", max(physical_errors))
            record_property("fixed_field_complete_ao_columns", len(complete_errors))
            record_property("physical_ov_output_blocks", len(physical_errors))
            directions = (
                np.eye(9)[0].reshape(3, 3),
                np.array([[0.2, -0.3, 0.4], [-0.1, 0.5, 0.2], [0.3, 0.1, -0.2]]),
            )
            actions = [
                composite_rks_hvp(
                    operator, direction, cache=tmp_path, solver_options=SOLVER
                )
                for direction in directions
            ]
            hessian = composite_rks_hessian(
                operator, cache=tmp_path, solver_options=SOLVER
            )
            assert not hessian.diagnostics["public_calculator_endpoint"]
            assert not hessian.diagnostics["posthoc_symmetrization"]
            assert hessian.diagnostics["displaced_scf_evaluations"] == 0
            assert max(hessian.diagnostics["response_residual_norms"]) < 2e-10
            np.testing.assert_allclose(
                hessian.matrix, hessian.matrix.T, atol=2e-8, rtol=0
            )
            np.testing.assert_allclose(
                hessian.matrix @ np.tile(np.eye(3), (3, 1)), 0, atol=2e-8, rtol=0
            )
            for direction, action in zip(directions, actions, strict=True):
                assert action.diagnostics["ecp_ad_backend"] == "cpu"
                assert action.diagnostics["displaced_scf_evaluations"] == 0
                assert {"ecp_local", "ecp_nonlocal"} <= set(action.components)
                np.testing.assert_allclose(
                    sum(action.components.values()), action.value, atol=2e-12, rtol=0
                )
                np.testing.assert_allclose(
                    hessian.matrix @ direction.reshape(-1),
                    action.value.reshape(-1),
                    atol=2e-8,
                    rtol=0,
                )
            record_property(
                "native_complete_matrix_and_two_hvps_seconds", perf_counter() - started
            )
            record_property("hessian_diagnostics", str(hessian.diagnostics))
            record_property("raw_native_matrix", hessian.matrix.tolist())
            record_property(
                "raw_native_hvps", [action.value.tolist() for action in actions]
            )

    positions = np.asarray([center for _, center in ATOMS])
    for step in (0.003, 0.001, 0.0003):
        independent = np.empty((9, 9))
        for column, direction in enumerate(np.eye(9).reshape(9, 3, 3)):
            gradients = [
                independent_gradient(
                    [
                        (symbol, center)
                        for (symbol, _), center in zip(
                            ATOMS, positions + sign * step * direction, strict=True
                        )
                    ],
                    atomic_rules,
                )
                for sign in (-1, 1)
            ]
            independent[:, column] = (
                (gradients[1] - gradients[0]) / (2 * step)
            ).reshape(-1)
        error = float(np.max(abs(hessian.matrix - independent)))
        record_property(f"raw_independent_matrix_{step}", independent.tolist())
        record_property(f"raw_independent_matrix_error_{step}", error)
        assert error < 1e-5
        for direction, action in zip(directions, actions, strict=True):
            np.testing.assert_allclose(
                action.value.reshape(-1),
                independent @ direction.reshape(-1),
                atol=1e-5,
                rtol=0,
            )
