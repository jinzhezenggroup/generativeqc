"""Small B3LYP scalar-def2-ECP force gates, not mixed-domain or scale promotion."""

from time import perf_counter

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget
from generativeqc._dft_gradient import StationaryKsState
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from test_global_hybrid_ecp_basis import (
    CORES,
    _independent_ecp,
)
from test_global_hybrid_ecp_basis import (
    canonical_records as _canonical_records,
)
from test_integral_autodiff_second import require_native_library  # noqa: F401

GRID = GridSpec(radial_points=16, angular_polar=6, angular_azimuth=12)
FORCE_GATE = 1e-5
ecp_records = _canonical_records


def make_calculator(
    record: object,
    multiplicity: int = 1,
    *,
    resource_budget: object = None,
    grid: GridSpec | None = GRID,
    max_iterations: int = 240,
) -> Calculator:
    """Keep physical Gaussian/VWN-RPA B3LYP and the caller's spin unchanged."""
    return Calculator(
        method=resolve_method(
            "B3LYP", spin="polarized" if multiplicity == 2 else "unpolarized"
        ),
        basis=record,
        basis_representation="spherical",
        device="cpu",
        ks_options=KsOptions() if grid is None else KsOptions(grid=grid),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=max_iterations,
        resource_budget=resource_budget,
    )


def molecule_atoms(symbol: str) -> list:
    """Return declared Bohr coordinates; only Cs uses the longer bond."""
    return [
        (symbol, (0.03, -0.02, 0.01)),
        ("H", (0.17, 0.11, 3.1 if symbol == "I" else 3.4)),
    ]


def independent_solver(
    raw: dict, atoms: list, charge: int, multiplicity: int, state: object = None
) -> object:
    """Use independent ECP integrals/SCF with the identical moving quadrature.

    The explicit Libxc expression avoids PySCF's configurable B3LYP alias.
    Both bare elemental identities and ECP-adjusted ionic charges are supplied
    by the retained canonical records, never reconstructed from the native state.
    Without a native state this is an independent displaced-energy oracle, not
    an analytic-gradient owner or a production continuation/fallback.
    """
    from generativeqc_compiler.dft.grid import MolecularGrid
    from pyscf import dft, gto, lib

    from tools.generate_external_basis_references import pyscf_basis

    lib.num_threads(1)
    molecule = gto.M(
        atom=atoms,
        basis=pyscf_basis(raw),
        ecp=_independent_ecp(raw),
        unit="Bohr",
        cart=False,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    solver = dft.UKS(molecule) if multiplicity == 2 else dft.RKS(molecule)
    solver.xc = "0.81*GGA_C_LYP+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.08*LDA_X+0.2*HF"
    solver.grids.radii_adjust = None
    if state is not None and state._source.grid_spec.version == 4:
        from pyscf.dft import radi

        radii = np.array(radi.BRAGG_RADII, copy=True)
        for number, radius in state._source.grid_spec.element_radii:
            radii[number] = radius
        solver.grids.atomic_radii = radii
        solver.grids.radii_adjust = radi.treutler_atomic_radii_adjust
        elemental_charges = np.array(
            [
                molecule.atom_charge(index) + molecule.atom_nelec_core(index)
                for index in range(molecule.natm)
            ]
        )
        # PySCF 2.14's primal radius adjustment uses elemental symbols, but its
        # grid gradient indexes atom_charges(), which contains ECP ionic charges.
        # Correct only the grid view; the independent Hamiltonian stays untouched.
        grid_molecule = molecule.copy()
        grid_molecule.atom_charges = lambda: elemental_charges.copy()
        solver.grids.mol = grid_molecule
    grid = (
        state.grid
        if state is not None
        else MolecularGrid(tuple(atoms), GRID, charge, multiplicity).explicit()
    )
    solver.grids.coords = np.array(grid.points)
    solver.grids.weights = np.array(grid.weights)
    if state is not None:
        owners = np.asarray(grid.owners)
        atomic = {
            symbol: (
                grid.points[owners == index] - np.asarray(center),
                np.asarray(state._source.atomic_weights)[owners == index],
            )
            for index, (symbol, center) in enumerate(atoms)
        }
        solver.grids.gen_atomic_grids = lambda *args, **kwargs: atomic
    solver.small_rho_cutoff = 0
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-10, 240
    solver.kernel()
    assert solver.converged
    np.testing.assert_array_equal(solver.grids.coords, grid.points)
    np.testing.assert_array_equal(solver.grids.weights, grid.weights)
    return solver


@pytest.mark.parametrize(
    "symbol,charge,multiplicity,energy_oracle",
    (
        ("I", 0, 1, "native"),
        ("I", -1, 2, "independent"),
        ("Cs", 0, 1, "native"),
        ("Cs", -1, 2, "native"),
    ),
)
def test_public_b3lyp_ecp_matches_state_force_and_reconverged_energy(
    symbol: str,
    charge: int,
    multiplicity: int,
    energy_oracle: str,
    ecp_records: tuple,
    record_property: object,
) -> None:
    """Compare all forces and declared energy oracles, including ECP-only motion.

    IH(-) has retained native displaced-SCF failures. Its explicitly selected
    independent energy differences validate the point force, not native geometry
    continuation. No failed native result is consumed or retried as an oracle.
    """
    pytest.importorskip("pyscf")
    _, raw, record = ecp_records
    atoms = molecule_atoms(symbol)
    calculator = make_calculator(record, multiplicity)
    started = perf_counter()
    native = calculator.singlepoint(
        atoms, charge=charge, multiplicity=multiplicity, properties=("energy", "forces")
    )
    record_property("whole_public_endpoint_seconds", perf_counter() - started)
    assert native.converged and native.executed_backend == "cpu_reference"
    diagnostic = native.ks_diagnostic
    assert diagnostic is not None
    assert len(diagnostic.history) == native.iterations <= 240
    assert [row.iteration for row in diagnostic.history] == list(
        range(1, native.iterations + 1)
    )
    assert diagnostic.fock_builds >= native.iterations
    assert native.energy_change < 1e-13
    assert diagnostic.density_change_max < 1e-11
    assert diagnostic.physical_residual_max < 1e-11
    with (
        calculator.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=False
        ) as batch,
        NativeAO(
            atoms,
            basis=record,
            representation="spherical",
            charge=charge,
            multiplicity=multiplicity,
        ) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:
            solver = independent_solver(raw, atoms, charge, multiplicity, state)
            assert basis.nao == solver.mol.nao_nr() < solver.mol.nao_cart()
            cores = tuple(CORES[atom.atomic_number] for atom in basis.atoms)
            assert state._source.ecp_cores == cores
            assert state._source.hamiltonian == "scalar-semilocal-ecp"
            assert (
                solver.mol.nelectron
                == sum(
                    atom.atomic_number - core
                    for atom, core in zip(basis.atoms, cores, strict=True)
                )
                - charge
            )
            expected_counts = (
                [solver.mol.nelectron] if multiplicity == 1 else list(solver.mol.nelec)
            )
            np.testing.assert_allclose(
                np.einsum("spq,pq->s", state.density, state.overlap),
                expected_counts,
                atol=2e-8,
                rtol=0,
            )
            density = np.asarray(solver.make_rdm1()).reshape(state.density.shape)
            np.testing.assert_allclose(state.density, density, atol=2e-7, rtol=0)
            np.testing.assert_allclose(
                state.overlap, solver.mol.intor("int1e_ovlp"), atol=2e-11, rtol=0
            )
            record_property(
                "density_max_error", float(np.max(abs(state.density - density)))
            )
            assert abs(native.energy - solver.e_tot) <= 2e-8
            gradient = solver.nuc_grad_method()
            gradient.grid_response = True
            independent_force = -gradient.kernel()
        finally:
            state._source.close()
    error = float(np.max(abs(native.forces - independent_force)))
    record_property("force_max_error", error)
    assert error <= FORCE_GATE
    np.testing.assert_allclose(native.forces.sum(axis=0), 0, atol=2e-9, rtol=0)
    errors = []
    directions = (
        np.array(((0.17, -0.11, 0.13), (0.0, 0.0, 0.0))),
        np.array(((0.12, 0.03, -0.07), (-0.06, 0.05, 0.16))),
    )
    record_property("displaced_energy_backend", energy_oracle)
    for direction in directions:
        for step in (1e-3, 3e-4):
            energies = []
            for sign in (1, -1):
                moved = [
                    (element, np.asarray(center) + sign * step * motion)
                    for (element, center), motion in zip(atoms, direction, strict=True)
                ]
                if energy_oracle == "native":
                    displaced = calculator.singlepoint(
                        moved,
                        charge=charge,
                        multiplicity=multiplicity,
                        properties=("energy",),
                    )
                    assert displaced.converged
                    assert displaced.physical_residual_rms <= 1e-9
                    energies.append(displaced.energy)
                else:
                    assert energy_oracle == "independent"
                    energies.append(
                        independent_solver(raw, moved, charge, multiplicity).e_tot
                    )
            discrepancy = abs(
                -(energies[0] - energies[1]) / (2 * step)
                - np.vdot(native.forces, direction)
            ) / np.linalg.norm(direction)
            errors.append(float(discrepancy))
    record_property("normalized_reconverged_energy_errors", errors)
    assert max(errors) <= FORCE_GATE


@pytest.mark.parametrize("maximum", (1, 4))
def test_b3lyp_ecp_uks_exhausted_budget_keeps_failure_and_rejects_warm_state(
    ecp_records: tuple, maximum: int
) -> None:
    """A rejected physical state cannot acquire a fresh budget or become a seed."""
    _, _, record = ecp_records
    calculator = make_calculator(record, 2, max_iterations=maximum)
    with calculator.prepare_batch(
        [molecule_atoms("I")], charges=[-1], multiplicities=[2], warm_start=True
    ) as batch:
        for _attempt in range(2):
            result = batch.execute(properties=("energy",), strict=False).items[0]
            assert not result.succeeded and not result.converged
            assert result.iterations <= maximum
            assert not result.warm_start_used
            assert result.ks_diagnostic is not None
            assert len(result.ks_diagnostic.history) == result.iterations


def test_cs_b3lyp_force_covariance_changed_geometry_and_failure_isolation(
    ecp_records: tuple, record_property: object
) -> None:
    """Qualify one discrete grid-preserving rotation, not arbitrary-grid covariance."""
    _, _, record = ecp_records
    atoms = molecule_atoms("Cs")
    calculator = make_calculator(record)
    coordinates = np.asarray([center for _, center in atoms])
    angle = np.pi / 6
    rotation = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        )
    )
    transformed = coordinates @ rotation.T + np.array((0.3, -0.4, 0.2))
    with calculator.prepare_batch([atoms, atoms], warm_start=False) as batch:
        original = batch.execute(strict=True, properties=("energy", "forces"))
        moved = batch.execute(
            [transformed, None], strict=True, properties=("energy", "forces")
        )
        assert abs(moved.items[0].energy - original.items[0].energy) <= 2e-8
        covariance_error = float(
            np.max(abs(moved.items[0].forces - original.items[0].forces @ rotation.T))
        )
        record_property("discrete_rotation_force_error", covariance_error)
        assert covariance_error <= FORCE_GATE
        fresh_atoms = [
            (element, center)
            for (element, _), center in zip(atoms, transformed, strict=True)
        ]
        fresh = calculator.singlepoint(fresh_atoms, properties=("energy", "forces"))
        np.testing.assert_allclose(
            moved.items[0].forces, fresh.forces, atol=2e-9, rtol=0
        )
        failed = batch.execute(
            [np.zeros((1, 3)), None], strict=False, properties=("energy", "forces")
        )
        assert not failed.items[0].succeeded and failed.items[0].forces is None
        assert failed.items[1].succeeded
        np.testing.assert_allclose(
            failed.items[1].forces, original.items[1].forces, atol=2e-9, rtol=0
        )
        restored = batch.execute(
            [coordinates, None], strict=True, properties=("energy", "forces")
        )
        np.testing.assert_allclose(
            restored.items[0].forces, original.items[0].forces, atol=2e-9, rtol=0
        )


def test_cs_global_hybrid_exact_host_budget_and_preparation_rejection(
    ecp_records: tuple, record_property: object
) -> None:
    """An exact native endpoint plan admits work; one byte less rejects preparation."""
    _, _, record = ecp_records
    atoms = molecule_atoms("Cs")
    plan = make_calculator(record).estimate_resources([atoms]).require_feasible()
    required = plan.peak_bytes["host"]
    rejected = make_calculator(
        record, resource_budget=ResourceBudget(host_bytes=required - 1)
    )
    with pytest.raises(MemoryError):
        rejected.prepare_batch([atoms])
    bounded = make_calculator(
        record, resource_budget=ResourceBudget(host_bytes=required)
    )
    with bounded.prepare_batch([atoms]) as batch:
        result = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        assert result.converged and np.isfinite(result.forces).all()
        record_property("native_planned_peak_bytes", dict(plan.peak_bytes))


def test_cs_b3lyp_native_default_grid_matches_independent_force_and_energy(
    ecp_records: tuple, record_property: object
) -> None:
    """Keep ionic Hamiltonian charges distinct from elemental grid-radius indices."""
    pytest.importorskip("pyscf")
    from pyscf.dft import gen_grid

    _, raw, record = ecp_records
    atoms = molecule_atoms("Cs")
    calculator = make_calculator(record, grid=None)
    assert calculator.ks_options.grid.version == 4
    started = perf_counter()
    native = calculator.singlepoint(atoms, properties=("energy", "forces"))
    record_property("whole_default_grid_endpoint_seconds", perf_counter() - started)
    assert native.converged
    with (
        calculator.prepare_batch([atoms], warm_start=False) as batch,
        NativeAO(atoms, basis=record, representation="spherical") as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:
            assert state._source.grid_spec.version == 4
            record_property("native_default_grid_points", len(state.grid.points))
            solver = independent_solver(raw, atoms, 0, 1, state)
            np.testing.assert_array_equal(solver.mol.atom_charges(), (9, 1))
            np.testing.assert_array_equal(solver.grids.mol.atom_charges(), (55, 1))
            points, weights = gen_grid.get_partition(
                solver.grids.mol,
                solver.grids.gen_atomic_grids(),
                radii_adjust=solver.grids.radii_adjust,
                atomic_radii=solver.grids.atomic_radii,
            )
            np.testing.assert_allclose(points, state.grid.points, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                weights, state.grid.weights, atol=1e-11, rtol=1e-13
            )
            assert abs(native.energy - solver.e_tot) <= 2e-8
            np.testing.assert_allclose(
                state.density[0], solver.make_rdm1(), atol=2e-7, rtol=0
            )
            gradient = solver.nuc_grad_method()
            gradient.grid_response = True
            force = -gradient.kernel()
            np.testing.assert_array_equal(solver.mol.atom_charges(), (9, 1))
        finally:
            state._source.close()
    error = float(np.max(abs(native.forces - force)))
    record_property("independent_default_grid_force_error", error)
    assert error <= FORCE_GATE
    direction = np.array(((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)))
    errors = []
    for step in (1e-3, 3e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (symbol, np.asarray(center) + sign * step * motion)
                for (symbol, center), motion in zip(atoms, direction, strict=True)
            ]
            displaced = calculator.singlepoint(moved, properties=("energy",))
            assert displaced.converged
            energies.append(displaced.energy)
        error = abs(
            -(energies[0] - energies[1]) / (2 * step)
            - np.vdot(native.forces, direction)
        )
        errors.append(float(error))
    record_property("native_default_grid_energy_difference_errors", errors)
    assert max(errors) <= FORCE_GATE
