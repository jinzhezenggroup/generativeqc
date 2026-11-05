"""Independent molecular full-range qualification through shared and public owners."""

import json
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
from generativeqc.rks_hessian_integrals import (
    generated_directional_semilocal_rks_integral_first_order,
)
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
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
    assert calculator.second_order_capabilities == frozenset(("hvp", "hessian"))
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
            assert not hessian.diagnostics["public_calculator_endpoint"]
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


def test_changed_semilocal_weights_do_not_inherit_public_qualification() -> None:
    """Exchange admission must not promote unqualified semilocal compositions."""
    method = resolve_method("PBE0")
    method = replace(
        method,
        identifier="custom-full-range-qualification-pending",
        primitives=(
            replace(
                method.primitives[0],
                functional=replace(
                    method.primitives[0].functional,
                    components=tuple(
                        (name, weight * Fraction(3, 4))
                        for name, weight in method.primitives[0].functional.components
                    ),
                ),
            ),
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
    reference_gradient = pytest.importorskip("pyscf.grad.rks")
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
            generator = np.random.default_rng(2224)
            density_directions = generator.normal(size=(3, basis.nao, basis.nao)) * 0.03
            density_directions += density_directions.transpose(0, 2, 1)
            native_actions = [
                operator.induced_fock(delta) for delta in density_directions
            ]
            pure, _ = generated_directional_semilocal_rks_integral_first_order(
                operator.topology, native_density, DIRECTION, cache=tmp_path
            )
            mixed, _ = generated_directional_semilocal_rks_integral_first_order(
                operator.topology,
                native_density,
                DIRECTION,
                cache=tmp_path,
                exchange_operators=(
                    (four_center_eri_operator(CoulombKernel("full_range")), exchange),
                ),
            )
            native_frozen_exchange = mixed - pure
            shells = basis.shells
            center_points = operator.state.grid.points.copy()
            center_weights = operator.state.grid.weights.copy()
            center_owners = np.asarray(operator.state.grid.owners)
            atomic_weights = operator.state._source.atomic_weights.copy()
    public_action = calculator.hessian_vector_product(
        fixture_atoms, DIRECTION, charge=charge, cache=tmp_path, solver_options=SOLVER
    )
    public_matrix = calculator.hessian(
        fixture_atoms, charge=charge, cache=tmp_path, solver_options=SOLVER
    )
    for result in (public_action, public_matrix):
        assert result.diagnostics["public_calculator_endpoint"]
        assert result.diagnostics["scf_energy"] == pytest.approx(
            native_energy, abs=2e-11
        )
        assert result.diagnostics["scf_iterations"] > 0
        assert result.diagnostics["scf_fock_builds"] > 0
        assert (
            result.diagnostics["endpoint_seconds"]
            >= result.diagnostics["assembly_seconds"]
        )
        assert result.diagnostics["displaced_scf_evaluations"] == 0
    np.testing.assert_allclose(public_action.value, native.value, atol=2e-8, rtol=0)
    np.testing.assert_allclose(public_matrix.matrix, matrix, atol=2e-8, rtol=0)
    for name, result in (("hvp", public_action), ("hessian", public_matrix)):
        record_property(
            f"public_{name}_diagnostics",
            json.dumps(dict(result.diagnostics), default=str),
        )
    labels = tuple(
        f"{element}{index}" for index, (element, _) in enumerate(fixture_atoms)
    )
    records = {label: [] for label in labels}
    atomic_grids = {
        label: (
            center_points[center_owners == index] - fixture_atoms[index][1],
            atomic_weights[center_owners == index],
        )
        for index, label in enumerate(labels)
    }
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
        solver.grids.gen_atomic_grids = lambda *unused, **kwargs: atomic_grids
        solver.grids.radii_adjust = None
        solver.grids.becke_scheme = dft.gen_grid.original_becke
        solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-10, 240
        energy = solver.kernel()
        assert solver.converged
        return energy, solver.make_rdm1(), molecule.intor("int1e_ovlp"), solver

    center, density, overlap, solver = reference(np.zeros_like(DIRECTION))
    assert native_energy == pytest.approx(center, abs=2e-10)
    np.testing.assert_allclose(native_overlap, overlap, atol=2e-12, rtol=0)
    np.testing.assert_allclose(native_density, density, atol=2e-9, rtol=0)
    record_property("independent_energy_error", abs(native_energy - center))
    record_property(
        "independent_density_error", float(np.max(abs(native_density - density)))
    )
    response = solver.gen_response(hermi=1)
    for delta, actual in zip(density_directions, native_actions, strict=True):
        np.testing.assert_allclose(actual, response(delta), atol=2e-9, rtol=1e-10)
    molecule = solver.mol
    ao_owners = np.repeat(
        np.arange(molecule.natm),
        np.diff(molecule.aoslice_by_atom()[:, 2:4], axis=1).reshape(-1),
    )
    motion = DIRECTION[ao_owners].T
    # Libcint differentiates electron coordinates on its first AO leg. Negate
    # and permute all four legs to obtain the physical nuclear K' source.
    derivative = molecule.intor("int2e_ip1", comp=3)
    eri_direction = -np.einsum("xp,xpqrs->pqrs", motion, derivative)
    eri_direction -= np.einsum("xq,xqprs->pqrs", motion, derivative)
    eri_direction -= np.einsum("xr,xrspq->pqrs", motion, derivative)
    eri_direction -= np.einsum("xs,xsrpq->pqrs", motion, derivative)
    reference_exchange = (
        -float(exchange) / 2 * np.einsum("ikjl,kl->ij", eri_direction, native_density)
    )
    np.testing.assert_allclose(
        native_frozen_exchange, reference_exchange, atol=2e-10, rtol=0
    )
    record_property(
        "frozen_nuclear_k_error",
        float(np.max(abs(native_frozen_exchange - reference_exchange))),
    )
    component_names = (
        "one_electron",
        "coulomb",
        "exact_exchange",
        "overlap_pulay",
        "xc_total",
        "nuclear",
    )

    def gradient_components(independent: object) -> np.ndarray:
        """Partition the independent analytic force equation, not native AD algebra."""
        molecule = independent.mol
        gradient = independent.nuc_grad_method()
        gradient.grid_response = True
        total = gradient.kernel()
        density = independent.make_rdm1()
        weighted = gradient.make_rdm1e(
            independent.mo_energy, independent.mo_coeff, independent.mo_occ
        )
        hcore = gradient.hcore_generator(molecule)
        overlap = gradient.get_ovlp(molecule)
        coulomb, exchange_matrix = gradient.get_jk(molecule, density)
        grid_gradient, potential = reference_gradient.get_vxc_full_response(
            independent._numint, molecule, independent.grids, independent.xc, density
        )
        values = {name: np.zeros_like(DIRECTION) for name in component_names}
        values["nuclear"] = gradient.grad_nuc(molecule)
        values["xc_total"] = grid_gradient.copy()
        for index, (_, _, begin, end) in enumerate(molecule.aoslice_by_atom()):
            values["one_electron"][index] = np.einsum(
                "xij,ij->x", hcore(index), density
            )
            values["coulomb"][index] = 2 * np.einsum(
                "xij,ij->x", coulomb[:, begin:end], density[begin:end]
            )
            values["exact_exchange"][index] = -float(exchange) * np.einsum(
                "xij,ij->x", exchange_matrix[:, begin:end], density[begin:end]
            )
            values["overlap_pulay"][index] = -2 * np.einsum(
                "xij,ij->x", overlap[:, begin:end], weighted[begin:end]
            )
            values["xc_total"][index] += 2 * np.einsum(
                "xij,ij->x", potential[:, begin:end], density[begin:end]
            )
        grouped = np.asarray([values[name] for name in component_names])
        np.testing.assert_allclose(grouped.sum(axis=0), total, atol=2e-10, rtol=0)
        return grouped

    native_groups = dict(native.components)
    native_groups["xc_total"] = sum(
        native_groups[name] for name in ("xc_ao", "xc_grid", "xc_weight")
    )
    native_grouped = np.asarray([native_groups[name] for name in component_names])
    for step in (3e-3, 1e-3, 3e-4):
        plus = gradient_components(reference(step * DIRECTION)[3])
        minus = gradient_components(reference(-step * DIRECTION)[3])
        finite = (plus - minus) / (2 * step)
        errors = np.max(abs(finite - native_grouped), axis=(1, 2)) / np.linalg.norm(
            DIRECTION
        )
        record_property(
            f"analytic_gradient_component_errors_step_{step:g}", errors.tolist()
        )
        np.testing.assert_allclose(errors, 0, atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            (finite.sum(axis=0) - native.value) / np.linalg.norm(DIRECTION),
            0,
            atol=1e-5,
            rtol=0,
        )
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
