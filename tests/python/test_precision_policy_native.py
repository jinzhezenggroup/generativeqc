"""Native precision endpoints must retain scientific identity and oracle gates."""

from dataclasses import replace

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    GridSpec,
    KsOptions,
    ObservableTarget,
    PrecisionCalculator,
    PrecisionPolicy,
    TargetAccuracy,
)
from generativeqc.precision_policy import _model_identity
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import resolve_method

from tools.generativeqc_validation.fixtures import calculator_inputs, load_fixtures

H2 = tuple(
    Atom.from_value(atom) for atom in (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
)


@pytest.mark.parametrize("name", ("h2", "h2o", "hf-plus-uhf"))
def test_fp64_single_fp64_endpoint_against_independent_reference(name: str) -> None:
    """FP64 preserves native forces without using the oracle in production."""
    fixture = next(
        record for record in load_fixtures() if record["inputs"]["name"] == name
    )
    inputs = fixture["inputs"]
    calculator = PrecisionCalculator(policy="fp64", **calculator_inputs(inputs))
    output = calculator.singlepoint(
        zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True),
        charge=inputs["charge"],
        multiplicity=inputs["multiplicity"],
    )
    assert output.precision_validation.status == "strict_fp64"
    assert output.precision_validation.solve_count == 1
    assert output.precision_validation.target is None
    assert abs(output.energy - fixture["data"]["energy"]) < 1e-9
    np.testing.assert_allclose(
        output.forces, fixture["data"]["forces"], atol=1e-8, rtol=0
    )


@pytest.mark.parametrize("profile", ("audit_auto", "balanced", "strict"))
@pytest.mark.parametrize("name", ("h2", "h2o", "hf-plus-uhf"))
def test_hf_endpoints_against_independent_pinned_pyscf(name: str, profile: str) -> None:
    fixture = next(
        record for record in load_fixtures() if record["inputs"]["name"] == name
    )
    inputs = fixture["inputs"]
    atoms = zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True)
    calculator = PrecisionCalculator(policy=profile, **calculator_inputs(inputs))
    observed = calculator.singlepoint(
        atoms, charge=inputs["charge"], multiplicity=inputs["multiplicity"]
    )
    assert observed.converged and observed.precision_validation.status == "observed_met"
    assert observed.precision_validation.solve_count == 2
    assert abs(observed.energy - fixture["data"]["energy"]) < 1e-9
    np.testing.assert_allclose(
        observed.forces, fixture["data"]["forces"], atol=1e-8, rtol=0
    )
    assert all(
        stage.executed_backend == "cpu_reference"
        for stage in observed.precision_validation.solves
    )


@pytest.mark.parametrize("method", ("rhf", "pbe-rks"))
def test_identity_excludes_solver_controls_but_binds_geometry_and_basis(
    method: str,
) -> None:
    baseline = Calculator(method=method)
    tight = Calculator(
        method=method,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        screening_tolerance=1e-15,
    )
    original = _model_identity(baseline, H2, 0, 1)
    assert original == _model_identity(tight, H2, 0, 1)
    assert original != _model_identity(
        Calculator(method=method, basis="def2-svp"), H2, 0, 1
    )
    moved = (replace(H2[0], position=(0.0, 0.0, -0.8)), H2[1])
    assert original != _model_identity(baseline, moved, 0, 1)


def test_ks_identity_binds_grid_functional_and_fitted_model() -> None:
    baseline = Calculator(method="pbe-rks")
    options = baseline.ks_options
    original = _model_identity(baseline, H2, 0, 1)
    scheduled = Calculator(
        method="pbe-rks", ks_options=replace(options, tile_points=31)
    )
    assert original == _model_identity(scheduled, H2, 0, 1)
    assert original != _model_identity(Calculator(method="lda-rks"), H2, 0, 1)
    grid = replace(options.grid, element_radii=((1, 1.7),))
    changed = Calculator(method="pbe-rks", ks_options=KsOptions(grid=grid))
    assert original != _model_identity(changed, H2, 0, 1)
    fitted = Calculator(method="pbe-rks", density_fitting="cpu")
    other_metric = Calculator(
        method="pbe-rks", density_fitting="cpu", density_fitting_relative_threshold=1e-8
    )
    assert (
        len(
            {
                original,
                _model_identity(fitted, H2, 0, 1),
                _model_identity(other_metric, H2, 0, 1),
            }
        )
        == 3
    )


def test_basis_snapshot_and_energy_only_policy_are_supported() -> None:
    budget = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", absolute=1e-8),)
    )
    calculator = PrecisionCalculator(
        basis=Calculator(basis="sto-3g")._basis,
        policy=PrecisionPolicy(target=budget),
    )
    output = calculator.singlepoint(iter(H2), properties=iter(("energy",)))
    assert (
        output.forces is None and output.precision_validation.status == "observed_met"
    )


@pytest.mark.parametrize("method", ("pbe0-rks", "b3lyp-rks", resolve_method("PBE0")))
def test_hybrid_audit_replays_constructor_inputs_not_resolved_outputs(
    method: object,
) -> None:
    """Discovered/MethodIR selectors already own their KS composition."""
    options = KsOptions(grid=GridSpec(16, 8, 16))
    calculator = PrecisionCalculator(method=method, ks_options=options)
    reference = calculator._reference()
    assert calculator._options["ks_options"] is options
    assert _model_identity(calculator._candidate, H2, 0, 1) == _model_identity(
        reference, H2, 0, 1
    )


def _pyscf_pbe_energy(calculator: Calculator, atoms: tuple[Atom, ...]) -> float:
    """Independent SCF/integrals/XC at the same discrete moving-grid target."""
    from pyscf import dft, gto

    labels = [f"H{index}" for index in range(len(atoms))]
    basis = {label: [] for label in labels}
    for shell in calculator._shells_for_atoms(atoms):
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ],
            ]
        )
    molecule = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=basis,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    grid = MolecularGrid(atoms, spec=calculator.ks_options.grid).explicit()
    reference = dft.RKS(molecule)
    reference.xc = "PBE"
    reference.grids.coords, reference.grids.weights = (
        np.array(grid.points),
        np.array(grid.weights),
    )
    reference.small_rho_cutoff = 0
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
    reference.kernel()
    assert reference.converged
    return float(reference.e_tot)


def test_pbe_energy_and_analytic_force_against_independent_scf_difference() -> None:
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(1)
    calculator = PrecisionCalculator(method="pbe-rks")
    observed = calculator.singlepoint(H2)
    oracle = Calculator(method="pbe-rks")
    energy = _pyscf_pbe_energy(oracle, H2)
    step = 2e-4
    energies = []
    for displacement in (-step, step):
        moved = (replace(H2[0], position=(0.0, 0.0, -0.7 + displacement)), H2[1])
        energies.append(_pyscf_pbe_energy(oracle, moved))
    force = -(energies[1] - energies[0]) / (2 * step)
    assert observed.precision_validation.status == "observed_met"
    assert abs(observed.energy - energy) < 1e-8
    assert abs(observed.forces[0, 2] - force) < 2e-6
