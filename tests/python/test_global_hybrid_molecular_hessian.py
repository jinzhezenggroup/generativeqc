"""Low-level full-range RKS qualification; no public Ni capability is implied."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.composite_rks_hessian import (
    CompositeRKSResponse,
    composite_rks_hessian,
    composite_rks_hvp,
)
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from test_integral_autodiff_second import require_native_library  # noqa: F401

ATOMS = [("H", (0.03, -0.02, -0.72)), ("H", (0.11, 0.13, 0.72))]
DIRECTION = np.array([[0.02, -0.03, -0.1], [-0.02, 0.03, 0.1]])
GRID = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("functional", ("B3LYP", "PBE0"))
def test_full_range_shared_owner_matches_reconverged_native_gradients(
    representation: str, functional: str, tmp_path: Path
) -> None:
    """Freeze source/HVP and raw-matrix gates separately from endpoint promotion."""
    method = resolve_method(functional)
    method = replace(
        method,
        identifier="full-range-stationary-qualification",
        primitives=(
            method.primitives[0],
            method.primitives[1],
        ),
    )
    calculator = Calculator(
        method=method,
        device="cpu",
        basis_representation=representation,
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    assert not calculator.capabilities.supported_second_order
    with calculator.prepare_batch([ATOMS], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        with (
            NativeAO(ATOMS, basis="sto-3g", representation=representation) as basis,
            CompositeRKSResponse.from_native(batch, basis) as operator,
        ):
            result = composite_rks_hvp(
                operator, DIRECTION, cache=tmp_path, solver_options=SOLVER
            )
            hessian = composite_rks_hessian(
                operator, cache=tmp_path, solver_options=SOLVER
            )
            assert operator.plan.method.semantic_payload() == method.semantic_payload()
            assert (
                operator.xc_kernel.spec.components
                == method.primitives[0].functional.components
            )
            assert "exact_exchange" in result.components
            assert np.linalg.norm(result.components["exact_exchange"]) > 1e-10
            assert not any(name.startswith("nonlocal_") for name in result.components)
            assert not result.diagnostics["public_calculator_endpoint"]
            assert result.diagnostics["pair_evaluations"] == 0
            assert result.diagnostics["response_residual_norm"] < 2e-10
            assert result.diagnostics["nuclear_response_solves"] == 1
            assert result.diagnostics["displaced_scf_evaluations"] == 0
            np.testing.assert_allclose(
                hessian.matrix @ DIRECTION.reshape(-1),
                result.value.reshape(-1),
                atol=2e-8,
                rtol=0,
            )
            np.testing.assert_allclose(
                hessian.matrix, hessian.matrix.T, atol=2e-8, rtol=0
            )
            translations = np.tile(np.eye(3), (len(ATOMS), 1))
            np.testing.assert_allclose(
                hessian.matrix @ translations, 0, atol=2e-8, rtol=0
            )
            with pytest.raises(MemoryError):
                composite_rks_hvp(
                    operator, DIRECTION, cache=tmp_path, integral_budget_bytes=1
                )
        with pytest.raises(ValueError):
            operator.validate_current()
    for step in (3e-3, 1e-3, 3e-4):
        forces = []
        for sign in (-1, 1):
            atoms = [
                (element, np.asarray(position) + sign * step * direction)
                for (element, position), direction in zip(ATOMS, DIRECTION, strict=True)
            ]
            point = calculator.singlepoint(atoms, properties=("energy", "forces"))
            assert point.converged
            forces.append(point.forces)
        np.testing.assert_allclose(
            result.value, -(forces[1] - forces[0]) / (2 * step), atol=3e-6, rtol=3e-5
        )


def test_custom_exchange_does_not_inherit_public_molecular_qualification() -> None:
    """A representable coefficient is not evidence of a qualified native SCF/Hessian."""
    method = resolve_method("B3LYP")
    method = replace(
        method,
        identifier="custom-full-range-qualification-pending",
        primitives=(
            method.primitives[0],
            replace(method.primitives[1], coefficient=Fraction(37, 100)),
        ),
    )
    calculator = Calculator(
        method=method, device="cpu", ks_options=KsOptions(grid=GRID)
    )
    assert not calculator.capabilities.supported_second_order
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(ATOMS)


@pytest.mark.parametrize(
    ("functional", "exchange"),
    (
        ("B3LYP", Fraction(1, 5)),
        ("B3LYP", Fraction(37, 100)),
        ("PBE0", Fraction(1, 4)),
        ("PBE0", Fraction(37, 100)),
    ),
)
@pytest.mark.parametrize("case", ("h2", "heh+"))
def test_complete_hvp_has_matched_state_independent_energy_curvature(
    case: str,
    functional: str,
    exchange: Fraction,
    tmp_path: Path,
    record_property: object,
) -> None:
    """Independent displaced energies are an oracle, never a derivative provider.

    Match raw contractions, the explicit moving quadrature and the VWN-RPA
    definition. Freeze a 1e-6 Hartree/Bohr² directional-curvature gate.
    Full raw-matrix gates use three independent energy-difference steps and
    1e-5 Hartree/Bohr² maximum error. HeH+ exercises a nontrivial asymmetric
    response; these s-shell sentinels do not qualify Ni or default-grid work.
    """
    gto = pytest.importorskip("pyscf.gto")
    dft = pytest.importorskip("pyscf.dft")
    method = resolve_method(functional)
    method = replace(
        method,
        identifier="independent-composed-full-range-reference",
        primitives=(
            method.primitives[0],
            replace(method.primitives[1], coefficient=exchange),
        ),
    )
    fixture_atoms = ATOMS if case == "h2" else [("He", ATOMS[0][1]), ATOMS[1]]
    charge = int(case == "heh+")
    calculator = Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    with calculator.prepare_batch(
        [fixture_atoms], charges=[charge], warm_start=False
    ) as batch:
        batch.execute(strict=True, properties=("energy",))
        with (
            NativeAO(fixture_atoms, basis="sto-3g", charge=charge) as basis,
            CompositeRKSResponse.from_native(batch, basis) as operator,
        ):
            native = composite_rks_hvp(
                operator, DIRECTION, cache=tmp_path, solver_options=SOLVER
            )
            matrix = composite_rks_hessian(
                operator, cache=tmp_path, solver_options=SOLVER
            ).matrix
            np.testing.assert_allclose(matrix, matrix.T, atol=2e-8, rtol=0)
            np.testing.assert_allclose(
                matrix @ DIRECTION.reshape(-1),
                native.value.reshape(-1),
                atol=2e-8,
                rtol=0,
            )
            native_energy = operator.state._source.energy()
            native_density = operator.state.density[0]
            native_overlap = operator.state.overlap
            shells = basis.shells
            center_points = operator.state.grid.points.copy()
            center_weights = operator.state.grid.weights.copy()
    labels = tuple(
        f"{element}{index}" for index, (element, _) in enumerate(fixture_atoms)
    )
    records = {label: [] for label in labels}
    for shell in shells:
        records[labels[shell.atom_index]].append(
            [shell.angular_momentum]
            + [
                [primitive.exponent, primitive.coefficient]
                for primitive in shell.primitives
            ]
        )
    components = method.primitives[0].functional.components
    code = "+".join(f"{float(weight)}*{name}" for name, weight in components)
    code += f"+{float(method.primitives[1].coefficient)}*HF"

    def reference(offset: np.ndarray) -> tuple:
        atoms = [
            (element, np.asarray(position) + motion)
            for (element, position), motion in zip(fixture_atoms, offset, strict=True)
        ]
        molecule = gto.M(
            atom=list(zip(labels, [position for _, position in atoms], strict=True)),
            basis=records,
            unit="Bohr",
            cart=True,
            charge=charge,
            verbose=0,
        )
        points, weights = center_points, center_weights
        if np.any(offset):
            # Match the exact numerical quadrature as an input, not an oracle
            # density or derivative. Independent SCFs never consume its state.
            with (
                calculator.prepare_batch(
                    [atoms], charges=[charge], warm_start=False
                ) as displaced,
                NativeAO(atoms, basis=shells, charge=charge) as displaced_basis,
            ):
                displaced.execute(strict=True, properties=("energy",))
                state = StationaryKsState.from_native(displaced, displaced_basis)
                try:
                    points, weights = (
                        state.grid.points.copy(),
                        state.grid.weights.copy(),
                    )
                finally:
                    state._source.close()
        solver = dft.RKS(molecule)
        solver.xc = code
        solver.grids.coords, solver.grids.weights = points, weights
        solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-10, 240
        energy = solver.kernel()
        assert solver.converged
        return energy, solver.make_rdm1(), molecule.intor("int1e_ovlp")

    center, density, overlap = reference(np.zeros_like(DIRECTION))
    assert native_energy == pytest.approx(center, abs=2e-10)
    np.testing.assert_allclose(native_overlap, overlap, atol=2e-12, rtol=0)
    np.testing.assert_allclose(native_density, density, atol=2e-9, rtol=0)
    curvature = float(np.sum(DIRECTION * native.value))
    energies = {tuple(np.zeros(DIRECTION.size)): center}

    def energy(offset: np.ndarray) -> float:
        key = tuple(offset.reshape(-1))
        if key not in energies:
            energies[key] = reference(offset)[0]
        return energies[key]

    for step in (3e-3, 1e-3, 3e-4):
        finite = (
            energy(step * DIRECTION) - 2 * center + energy(-step * DIRECTION)
        ) / step**2
        assert curvature == pytest.approx(finite, abs=1e-6)
    directions = np.eye(DIRECTION.size).reshape(DIRECTION.size, *DIRECTION.shape)
    for step in (1e-3, 5e-4, 3e-4):
        independent = np.empty_like(matrix)
        for row, left in enumerate(directions):
            independent[row, row] = (
                energy(step * left) - 2 * center + energy(-step * left)
            ) / step**2
            for column in range(row):
                right = directions[column]
                mixed = (
                    energy(step * (left + right))
                    - energy(step * (left - right))
                    - energy(step * (-left + right))
                    + energy(-step * (left + right))
                ) / (4 * step**2)
                independent[row, column] = independent[column, row] = mixed
        error = float(np.max(abs(matrix - independent)))
        record_property(f"raw_hessian_max_error_step_{step:g}", error)
        np.testing.assert_allclose(matrix, independent, atol=1e-5, rtol=0)
