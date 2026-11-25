"""Independent moving-gradient checks for the linked CPU LDA/PBE endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    GridSpec,
    HostResources,
    KsOptions,
    ResourceBudget,
)
from generativeqc._dft_gradient import StationaryKsState
from generativeqc._native_cpu_hessian import linked_method
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from test_dft_complete_cpu import independent_global_hybrid_gradient

if TYPE_CHECKING:
    from collections.abc import Callable

    from generativeqc.batch import PreparedBatch

ATOMS = (
    ("O", (0.01, -0.04, 0.02)),
    ("H", (1.42, 0.13, 1.06)),
    ("H", (-1.37, -0.09, 1.10)),
)
MOTION = np.array(((0.12, -0.07, 0.04), (-0.08, 0.06, 0.15), (-0.04, 0.01, -0.19)))


def test_hvp_panel_admits_exact_host_and_output_capacities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check a count-bound plan and atomic, pre-SCF rejection of short budgets."""

    def make(budget: ResourceBudget | None) -> Calculator:
        return Calculator(
            method="pbe-rks",
            device="cpu",
            basis="sto-3g",
            basis_representation="cartesian",
            ks_options=KsOptions(
                grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
            ),
            resource_budget=budget,
        )

    calculator = make(None)
    if not linked_method(
        calculator._library, calculator.method_ir.identity
    ) or not hasattr(
        calculator._library, "generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1"
    ):
        pytest.skip("linked CPU KS HVP panels are unavailable")
    directions = np.stack((MOTION, MOTION[::-1], -0.5 * MOTION))
    probe = calculator.hessian_vector_products(ATOMS, directions)
    output_bytes = probe.diagnostics["output_bytes"]
    uncapped = make(ResourceBudget()).hessian_vector_products(
        ATOMS, directions, output_budget_bytes=output_bytes
    )
    sizing = uncapped.diagnostics["resource_admission"]
    numeric = uncapped.diagnostics["numeric_reservation_bytes"]
    exact = (
        sizing["plan_host_peak_bytes"] - sizing["derivative_host_cap_bytes"] + numeric
    )
    bounded = make(ResourceBudget(host_bytes=exact)).hessian_vector_products(
        ATOMS, directions, output_budget_bytes=output_bytes
    )
    assert bounded.diagnostics["numeric_reservation_bytes"] == numeric
    assert bounded.diagnostics["resource_admission"]["plan_host_peak_bytes"] == exact
    np.testing.assert_allclose(bounded.value, probe.value, atol=2e-8, rtol=0)

    mutable = directions.copy()
    prepare = calculator.prepare_batch

    def changed_input(*args: Any, **kwargs: Any) -> PreparedBatch:
        mutable.fill(0)
        return prepare(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(calculator, "prepare_batch", changed_input)
        frozen = calculator.hessian_vector_products(ATOMS, mutable)
    np.testing.assert_array_equal(frozen.direction, directions)
    np.testing.assert_array_equal(frozen.value, probe.value)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("an inadmissible HVP panel reached SCF preparation")

    monkeypatch.setattr(Calculator, "prepare_batch", forbidden)
    with pytest.raises(MemoryError, match="no supported plan fits"):
        make(ResourceBudget(host_bytes=exact - 1)).hessian_vector_products(
            ATOMS, directions, output_budget_bytes=output_bytes
        )
    with pytest.raises(ValueError, match="output_budget_bytes"):
        calculator.hessian_vector_products(
            ATOMS, directions, output_budget_bytes=output_bytes - 1
        )
    with pytest.raises(MemoryError, match="numerical reservation exceeded"):
        calculator.hessian_vector_products(
            ATOMS, directions, numeric_budget_bytes=numeric - 1
        )
    for invalid in (
        directions[:0],
        directions[0],
        directions + np.nan,
        directions.astype(complex),
    ):
        with pytest.raises(ValueError, match="directions must be"):
            calculator.hessian_vector_products(ATOMS, invalid)


def test_prepared_hvp_replays_only_the_accepted_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = Calculator(
        method="pbe-d3(bj)-rks",
        ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
    )
    if not hasattr(
        calculator._library, "generativeqc_ks_hvp_cpu_prepared_minimum_bytes_v1"
    ) or not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("linked prepared CPU KS HVP panels are unavailable")
    small = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
    directions = np.stack((MOTION, -0.5 * MOTION))
    moved = np.asarray([atom[1] for atom in ATOMS]) + 0.07 * MOTION
    moved_atoms = tuple((atom[0], xyz) for atom, xyz in zip(ATOMS, moved, strict=True))
    initial_reference = calculator.hessian_vector_products(ATOMS, directions)
    moved_reference = calculator.hessian_vector_products(moved_atoms, directions)

    def no_scf_or_new_system(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("prepared HVP attempted to execute SCF or rebuild its system")

    with calculator.prepare_batch([small, ATOMS]) as batch:
        with pytest.raises(RuntimeError, match="no successful current final state"):
            batch.hessian_vector_products(1, directions)
        batch.execute(strict=True, properties=("energy",))
        with monkeypatch.context() as scoped:
            scoped.setattr(batch, "execute", no_scf_or_new_system)
            scoped.setattr(calculator, "_create_native_system", no_scf_or_new_system)
            initial = batch.hessian_vector_products(1, directions)
            np.testing.assert_array_equal(initial.value, initial_reference.value)
            second_panel = batch.hessian_vector_products(1, directions[:1])
            np.testing.assert_allclose(
                second_panel.value, initial.value[:1], atol=2e-8, rtol=0
            )
        assert initial.diagnostics["scf_fock_builds"] == 0
        assert (
            initial.diagnostics["d3_pair_visits"]
            == initial_reference.diagnostics["d3_pair_visits"]
        )
        batch.execute([None, moved], strict=True, properties=("energy",))
        with pytest.raises(RuntimeError, match="geometry disagrees"):
            batch.hessian_vector_products(1, directions)
        exact = batch.hessian_vector_products(1, directions, coordinates=moved)
        np.testing.assert_allclose(
            exact.value, moved_reference.value, atol=2e-8, rtol=0
        )
        failed = batch.execute(
            [None, np.full_like(moved, np.nan)], properties=("energy",)
        )
        assert failed.failure_indices == (1,)
        with pytest.raises(RuntimeError, match="final state|geometry disagrees"):
            batch.hessian_vector_products(1, directions, coordinates=moved)
        batch.execute([None, moved], strict=True, properties=("energy",))
        np.testing.assert_allclose(
            batch.hessian_vector_products(1, directions, coordinates=moved).value,
            moved_reference.value,
            atol=2e-8,
            rtol=0,
        )
    with pytest.raises(RuntimeError, match="closed"):
        batch.hessian_vector_products(1, directions)


def test_prepared_hvp_admits_exact_fleet_budget() -> None:
    def make(limit: int | None = None) -> Calculator:
        return Calculator(
            method="pbe-rks",
            ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
            resource_budget=ResourceBudget()
            if limit is None
            else ResourceBudget(
                host_bytes=2 * (limit + 1024),
                host_reserve_bytes=1024,
                headroom_fraction=0.5,
            ),
        )

    calculator = make()
    if not hasattr(
        calculator._library, "generativeqc_ks_hvp_cpu_prepared_minimum_bytes_v1"
    ) or not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("linked prepared CPU KS HVP panels are unavailable")
    systems = [(("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))), ATOMS]
    directions = np.stack((MOTION, -MOTION))
    with calculator.prepare_batch(systems) as batch:
        batch.execute(strict=True, properties=("energy",))
        plan = batch.resource_plan
        full = batch.hessian_vector_products(1, directions)
        assert batch.resource_plan is plan
    sizing = full.diagnostics["resource_admission"]
    minimum = full.diagnostics["minimum_numeric_bytes"]
    exact = (
        sizing["plan_host_peak_bytes"] - sizing["derivative_host_cap_bytes"] + minimum
    )
    with make(exact).prepare_batch(systems) as batch:
        batch.execute(strict=True, properties=("energy",))
        plan = batch.resource_plan
        with pytest.raises(MemoryError, match="aggregate numerical reservation"):
            batch.hessian_vector_products(
                1, directions, numeric_budget_bytes=minimum - 1
            )
        result = batch.hessian_vector_products(1, directions)
        admission = result.diagnostics["resource_admission"]
        assert admission["plan_host_peak_bytes"] == exact
        assert admission["effective_host_limit_bytes"] == exact
        assert admission["derivative_host_cap_bytes"] == minimum
        assert result.diagnostics["scf_fock_builds"] == 0
        assert batch.resource_plan is plan
        np.testing.assert_allclose(result.value, full.value, atol=2e-8, rtol=0)
    with make(exact - 1).prepare_batch(systems) as batch:
        batch.execute(strict=True, properties=("energy",))
        plan = batch.resource_plan
        with pytest.raises(MemoryError, match="no supported plan fits"):
            batch.hessian_vector_products(1, directions)
        smaller = batch.hessian_vector_products(1, directions[:1])
        assert smaller.diagnostics["scf_fock_builds"] == 0
        np.testing.assert_allclose(smaller.value, full.value[:1], atol=2e-8, rtol=0)
        assert batch.resource_plan is plan


@pytest.mark.parametrize("method", ("pbe-rks", "pbe-d3(bj)-rks"))
def test_prepared_hvp_shared_host_lease_exact_and_rollback(method: str) -> None:
    calculator = Calculator(
        method=method,
        ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
        initial_guess=None,
    )
    if not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("linked CPU Hessian is unavailable")
    directions = np.stack((MOTION, -MOTION))
    with calculator.prepare_batch([ATOMS]) as unleased:
        unleased.execute(strict=True, properties=("energy",))
        reference = unleased.hessian_vector_products(0, directions)
    floor = reference.diagnostics["minimum_numeric_bytes"]

    def prepare(pool: HostResources, capacity: int = floor) -> PreparedBatch:
        return calculator.prepare_batch(
            [ATOMS],
            host_resources=pool,
            host_hessian_workspace_bytes=capacity,
        )

    with HostResources(1 << 32) as pool:
        with prepare(pool) as batch:
            snapshot = pool.snapshot()
            assert snapshot.owner_count == 1
            assert snapshot.retained_bytes > 0
            assert snapshot.workspace_bytes > floor
        assert pool.snapshot().owner_count == 0
    exact = snapshot.retained_bytes + snapshot.workspace_bytes
    with HostResources(exact - 1) as pool:
        with pytest.raises(MemoryError, match="shared host capacity"):
            prepare(pool)
        assert pool.snapshot().owner_count == 0
        with prepare(pool, floor - 1) as batch:
            batch.execute(strict=True, properties=("energy",))
            with pytest.raises(MemoryError, match="aggregate numerical reservation"):
                batch.hessian_vector_products(0, directions)
            smaller = batch.hessian_vector_products(0, directions[:1])
            np.testing.assert_array_equal(smaller.value, reference.value[:1])
        assert pool.snapshot().owner_count == 0
    with HostResources(exact) as pool, prepare(pool) as batch:
        before = pool.snapshot()
        with pytest.raises(MemoryError, match="shared host capacity"):
            prepare(pool)
        assert pool.snapshot() == before
        batch.execute(strict=True)
        with pytest.raises(NotImplementedError, match="force replay"):
            batch.execute(properties=("energy", "forces"))
        pool.close()
        result = batch.hessian_vector_products(0, directions)
        np.testing.assert_array_equal(result.value, reference.value)
        assert (
            result.diagnostics["resource_admission"]["derivative_host_cap_bytes"]
            == floor
        )
        assert (
            result.diagnostics["d3_pair_visits"]
            == reference.diagnostics["d3_pair_visits"]
        )
        assert bool(result.diagnostics["d3_pair_visits"]) == ("d3" in method)
        assert result.diagnostics["scf_fock_builds"] == 0
        moved = np.asarray([position for _, position in ATOMS]) + 0.1 * MOTION
        batch.execute([moved], strict=True)
        with pytest.raises(RuntimeError, match="geometry disagrees"):
            batch.hessian_vector_products(0, directions)
        moved_reference = calculator.hessian_vector_products(
            tuple(
                (atom, position)
                for (atom, _), position in zip(ATOMS, moved, strict=True)
            ),
            directions,
        )
        np.testing.assert_allclose(
            batch.hessian_vector_products(0, directions, coordinates=moved).value,
            moved_reference.value,
            atol=2e-8,
            rtol=0,
        )


@pytest.mark.parametrize(
    ("representation", "basis_name"),
    (("cartesian", "sto-3g"), ("spherical", "def2-svp")),
)
@pytest.mark.parametrize(("charge", "multiplicity"), ((0, 1), (1, 2)))
def test_linked_pbe_bj_matches_independent_total_curvature(
    representation: str,
    basis_name: str,
    charge: int,
    multiplicity: int,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Differentiate independent PySCF and s-dftd3 moving gradients."""
    grid = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
    spin = "rks" if multiplicity == 1 else "uks"
    molecular_state = {"charge": charge, "multiplicity": multiplicity}
    bare, corrected = (
        Calculator(
            method=method,
            device="cpu",
            basis=basis_name,
            basis_representation=representation,
            ks_options=KsOptions(grid=grid),
            max_iterations=240,
            energy_tolerance=1e-13,
            density_tolerance=1e-11,
        )
        for method in (f"pbe-{spin}", f"pbe-d3(bj)-{spin}")
    )
    if not linked_method(corrected._library, corrected.method_ir.identity):
        pytest.skip("requires linked native PBE-D3(BJ) Hessian support")
    dispersion = pytest.importorskip("dftd3.interface")
    pytest.importorskip("pyscf")
    atoms = tuple(Atom.from_value(atom) for atom in ATOMS)
    numbers = np.array([atom.atomic_number for atom in atoms])
    positions = np.array([atom.position for atom in atoms])
    parameters = dispersion.RationalDampingParam(method="pbe", atm=False)

    def reference_d3(displacement: np.ndarray) -> tuple[float, np.ndarray]:
        result = dispersion.DispersionModel(
            numbers, positions + displacement
        ).get_dispersion(parameters, grad=True)
        return float(result["energy"]), result["gradient"]

    def reference(displacement: np.ndarray) -> tuple[float, np.ndarray]:
        displaced = tuple(
            Atom.from_value((atom.atomic_number, tuple(position)))
            for atom, position in zip(atoms, positions + displacement, strict=True)
        )
        # Share only the discrete quadrature; neither oracle reads native SCF state.
        state = SimpleNamespace(
            grid=MolecularGrid(displaced, grid, charge, multiplicity).explicit(),
            _source=SimpleNamespace(
                grid_spec=grid,
                atomic_weights=np.concatenate(
                    [
                        MolecularGrid((atom,), grid, 0, 1).explicit().weights
                        for atom in displaced
                    ]
                ),
            ),
        )
        with NativeAO(
            displaced,
            basis=basis_name,
            representation=representation,
            **molecular_state,
        ) as basis:
            energy, gradient = independent_global_hybrid_gradient(
                basis, state, f"pbe-{spin}", xc="PBE"
            )
        dispersion_energy, dispersion_gradient = reference_d3(displacement)
        return energy + dispersion_energy, gradient + dispersion_gradient

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("linked corrected Hessian entered the Python scientific owner")

    monkeypatch.setattr(Calculator, "_run_public_rks_second_order", forbidden)
    with pytest.raises(
        NotImplementedError, match="native scalar|only strategy='scalar'"
    ):
        corrected.hessian(atoms, strategy="block", **molecular_state)
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    dense = corrected.hessian(atoms, solver_options=solver, **molecular_state)
    action = corrected.hessian_vector_product(
        atoms, MOTION, solver_options=solver, **molecular_state
    )
    bare_action = bare.hessian_vector_product(
        atoms, MOTION, solver_options=solver, **molecular_state
    )
    for result in (dense, action):
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        assert result.diagnostics["d3_pair_visits"] > 0
        assert result.diagnostics["dispersion"]["electronic_response_solves"] == 0
        assert result.diagnostics["displaced_scf_evaluations"] == 0
    for field in ("response_iterations", "induced_fock_actions"):
        assert action.diagnostics[field] == bare_action.diagnostics[field]
    energy, gradient = reference(np.zeros_like(MOTION))
    point = corrected.singlepoint(
        atoms, properties=("energy", "forces"), **molecular_state
    )
    assert point.converged
    assert point.energy == pytest.approx(energy, abs=2e-9, rel=0)
    np.testing.assert_allclose(point.forces, -gradient, atol=2e-8, rtol=0)
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        dense.matrix @ MOTION.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
    # Use the same fine step as the bare UKS oracle; a coarser step leaves
    # visible central-difference truncation error on the spherical cation.
    step = 1e-4
    axes = np.eye(MOTION.size).reshape((-1, *MOTION.shape))
    reference_matrix = np.column_stack(
        [
            (reference(step * axis)[1] - reference(-step * axis)[1]).ravel()
            / (2 * step)
            for axis in axes
        ]
    )
    error = np.max(np.abs(dense.matrix - reference_matrix))
    record_property("max_total_hessian_error", float(error))
    np.testing.assert_allclose(dense.matrix, reference_matrix, atol=3e-7, rtol=0)
    if spin == "rks" and hasattr(
        corrected._library, "generativeqc_ks_hessian_cpu_sequential_v1"
    ):
        sequential = corrected.hessian(
            atoms, strategy="sequential", solver_options=solver, **molecular_state
        )
        np.testing.assert_allclose(
            sequential.matrix, reference_matrix, atol=3e-7, rtol=0
        )
        assert (
            sequential.diagnostics["d3_pair_visits"]
            == dense.diagnostics["d3_pair_visits"]
        )
        assert sequential.diagnostics["response_concurrent_rhs"] == 1
    dispersion_action = (
        reference_d3(step * MOTION)[1] - reference_d3(-step * MOTION)[1]
    ) / (2 * step)
    assert np.max(np.abs(dispersion_action)) > 1e-7
    d3_error = np.max(np.abs(action.value - bare_action.value - dispersion_action))
    record_property("max_d3_hvp_error", float(d3_error))
    np.testing.assert_allclose(
        action.value - bare_action.value, dispersion_action, atol=2e-9, rtol=0
    )


@pytest.mark.parametrize("correction", ("d3(0)", "d3(bj)-atm", "d4(bj-eeq-atm)"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_unlinked_semilocal_correction_does_not_publish_electronic_hessian(
    correction: str,
    spin: str,
) -> None:
    calculator = Calculator(method=f"pbe-{correction}-{spin}", device="cpu")
    assert not calculator.capabilities.supported_second_order
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        open_shell = spin == "uks"
        calculator.hessian(
            ATOMS, charge=int(open_shell), multiplicity=1 + int(open_shell)
        )


@pytest.mark.parametrize(
    ("method", "xc"), (("lda-rks", "LDA_X,LDA_C_PW"), ("pbe-rks", "PBE"))
)
@pytest.mark.parametrize(
    ("representation", "basis_name"),
    (("cartesian", "sto-3g"), ("spherical", "def2-svp")),
)
def test_linked_hessian_against_reconverged_gradient(
    method: str,
    xc: str,
    representation: str,
    basis_name: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Use the same discrete quadrature but independent PySCF SCF and gradients."""
    calculator = Calculator(
        method=method,
        device="cpu",
        basis=basis_name,
        basis_representation=representation,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
        ),
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    if not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("requires a linked native CPU semilocal Hessian inventory")
    pytest.importorskip("pyscf", reason="independent moving-grid gradient")

    if method == "pbe-rks" and representation == "spherical":
        from functools import partial
        from types import SimpleNamespace

        from pyscf.lib import diis
        from scipy.linalg import eigh

        # Match the PBE0 oracle workaround: change only the DIIS
        # extrapolation eigensolver, not the physical SCF or derivative gates.
        monkeypatch.setattr(
            diis,
            "scipy",
            SimpleNamespace(linalg=SimpleNamespace(eigh=partial(eigh, driver="evd"))),
        )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("linked semilocal Hessian entered the Python scientific owner")

    monkeypatch.setattr(Calculator, "_run_public_rks_second_order", forbidden)
    if representation == "spherical":
        with pytest.raises(
            NotImplementedError, match="requires native scalar or recycled"
        ):
            calculator.hessian(ATOMS, strategy="block", block_size=2)
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    dense = calculator.hessian(ATOMS, solver_options=solver)
    action = calculator.hessian_vector_product(ATOMS, MOTION, solver_options=solver)
    for result in (dense, action):
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        assert not result.diagnostics["runtime_compilation"]
        assert result.diagnostics["displaced_scf_evaluations"] == 0
        assert result.diagnostics["response_residual_norm"] < 2e-10
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        dense.matrix @ np.tile(np.eye(3), (3, 1)), 0, atol=2e-8, rtol=0
    )
    np.testing.assert_allclose(
        dense.matrix @ MOTION.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )

    def gradient(offset: np.ndarray) -> np.ndarray:
        displaced = [
            (element, np.asarray(center) + motion)
            for (element, center), motion in zip(ATOMS, offset, strict=True)
        ]
        with (
            calculator.prepare_batch([displaced], warm_start=False) as batch,
            NativeAO(
                displaced, basis=basis_name, representation=representation
            ) as basis,
        ):
            if representation == "spherical":
                assert any(shell.angular_momentum == 2 for shell in basis.shells)
            point = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            state = StationaryKsState.from_native(batch, basis)
            try:
                energy, reference = independent_global_hybrid_gradient(
                    basis, state, method, xc=xc
                )
                assert point.energy == pytest.approx(energy, abs=2e-9, rel=0)
                np.testing.assert_allclose(point.forces, -reference, atol=2e-8, rtol=0)
                return reference.ravel()
            finally:
                state._source.close()

    errors = []
    for step in (1e-3, 3e-4):
        directions = np.eye(MOTION.size).reshape((-1, *MOTION.shape))
        reference = np.column_stack(
            [
                (gradient(step * axis) - gradient(-step * axis)) / (2 * step)
                for axis in directions
            ]
        )
        error = float(np.max(np.abs(dense.matrix - reference)))
        errors.append(error)
        record_property(f"raw_matrix_max_error_at_{step}", error)
    assert errors[1] < 3e-7, errors
    assert errors[1] < 0.4 * errors[0], errors

    if hasattr(calculator._library, "generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1"):
        seeds = np.stack((MOTION, MOTION[::-1], np.zeros_like(MOTION), -0.5 * MOTION))
        panel = calculator.hessian_vector_products(ATOMS, seeds, solver_options=solver)
        actions = panel.value
        assert panel.diagnostics["response_residual_norm"] < 2e-10
        assert panel.diagnostics["completed_directions"] == len(seeds)
        assert panel.diagnostics["response_strategy"] == "independent"
        assert not actions.flags.writeable and not panel.direction.flags.writeable
        np.testing.assert_array_equal(panel.direction, seeds)
        np.testing.assert_allclose(
            actions.reshape((-1, MOTION.size)),
            seeds.reshape((-1, MOTION.size)) @ reference.T,
            atol=3e-7,
            rtol=0,
        )
        step = 3e-4
        for seed, value in zip(seeds[:2], actions[:2], strict=True):
            expected = (gradient(step * seed) - gradient(-step * seed)) / (2 * step)
            np.testing.assert_allclose(value.ravel(), expected, atol=2e-7, rtol=0)

    if hasattr(calculator._library, "generativeqc_ks_hessian_cpu_sequential_v1"):
        partitions = (
            (None, 1, MOTION.size) if representation == "cartesian" else (None,)
        )
        for partition in partitions:
            sequential = calculator.hessian(
                ATOMS,
                strategy="sequential",
                block_size=partition,
                solver_options=solver,
            )
            diagnostics = sequential.diagnostics
            size = 4 if partition is None else partition
            assert diagnostics["response_strategy"] == "sequential"
            assert diagnostics["response_concurrent_rhs"] == 1
            assert diagnostics["block_size"] == size
            assert diagnostics["block_count"] == (MOTION.size + size - 1) // size
            assert diagnostics["nuclear_response_solves"] == MOTION.size
            assert diagnostics["response_residual_norm"] < 2e-10
            assert not sequential.matrix.flags.writeable
            np.testing.assert_allclose(sequential.matrix, reference, atol=3e-7, rtol=0)
            np.testing.assert_allclose(
                sequential.matrix, dense.matrix, atol=2e-8, rtol=0
            )
        with pytest.raises(ValueError, match="block_size must be between"):
            calculator.hessian(ATOMS, strategy="sequential", block_size=True)
        if representation == "cartesian":
            compatibility_result = object()
            with monkeypatch.context() as patch:
                patch.setattr(
                    Calculator,
                    "_run_public_rks_second_order",
                    lambda *_args, **_kwargs: compatibility_result,
                )
                assert (
                    calculator.hessian(ATOMS, strategy="blocked")
                    is compatibility_result
                )

    if hasattr(calculator._library, "generativeqc_ks_hessian_cpu_recycled_v1"):
        widths = (2, 3, 4) if representation == "cartesian" else (4,)
        for width in widths:
            recycled = calculator.hessian(
                ATOMS, strategy="recycled", block_size=width, solver_options=solver
            )
            assert recycled.diagnostics["response_strategy"] == "recycled"
            assert recycled.diagnostics["block_size"] == width
            assert (
                recycled.diagnostics["block_count"]
                == (MOTION.size + width - 1) // width
            )
            record_property(
                f"recycled_width_{width}_raw_matrix_max_error_at_0.0003",
                float(np.max(np.abs(recycled.matrix - reference))),
            )
            np.testing.assert_allclose(recycled.matrix, reference, atol=3e-7, rtol=0)
            np.testing.assert_allclose(recycled.matrix, dense.matrix, atol=2e-8, rtol=0)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_linked_symmetric_diatomic_default_response(method: str) -> None:
    """Default physical residual gates admit symmetry-zero motion components."""
    calculator = Calculator(
        method=method,
        device="cpu",
        basis="sto-3g",
        basis_representation="cartesian",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    if not linked_method(calculator._library, calculator.method_ir.identity):
        pytest.skip("requires a linked native CPU semilocal Hessian inventory")

    atoms = (("H", (0.0, 0.0, -0.72)), ("H", (0.0, 0.0, 0.72)))
    stretch = np.array(((0.0, 0.0, -0.1), (0.0, 0.0, 0.1)))
    dense = calculator.hessian(atoms)
    action = calculator.hessian_vector_product(atoms, stretch)
    for result in (dense, action):
        assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
        assert result.diagnostics["response_residual_norm"] < 2e-10
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        dense.matrix @ stretch.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
