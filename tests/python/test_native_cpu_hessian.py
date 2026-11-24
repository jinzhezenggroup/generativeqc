"""Actual public CPU batches; compact controls are not heavy-element/TS evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions
from generativeqc._native_cpu_hessian import (
    execute_batch,
    linked_method,
    orbital_frame,
    preflight,
)
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.method import resolve_method
from test_integral_autodiff_second import require_native_library  # noqa: F401

if TYPE_CHECKING:
    from pathlib import Path

GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def calculator(spins: int, bj: bool) -> Calculator:
    """Use canonical total chemistry and only the linked native execution path."""
    return Calculator(
        method=resolve_method(
            "B3LYP-D3(BJ)" if bj else "B3LYP",
            spin="unpolarized" if spins == 1 else "polarized",
        ),
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )


def geometry(spins: int) -> tuple[Atom, ...]:
    return tuple(
        Atom(1, (0.17 * atom * atom, -0.08 * atom, 1.6 * atom))
        for atom in range(spins + 1)
    )


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("bj", [False, True])
def test_public_native_cpu_is_raw_detached_and_never_enters_python_response(
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    bj: bool,
) -> None:
    calc = calculator(spins, bj)
    preflight(calc)
    atoms = geometry(spins)
    with calc.prepare_batch([atoms], multiplicities=[spins], warm_start=False) as batch:
        primal = batch.execute(strict=True, properties=("energy",)).items[0]
    dispersion_energy = 0 if primal.dispersion is None else primal.dispersion.energy

    def reject_legacy(*arguments: Any, **keywords: Any) -> None:
        raise AssertionError("native CPU Hessian entered the legacy numerical owner")

    monkeypatch.setattr(calc, "_run_public_rks_second_order", reject_legacy)
    direction = np.sin(np.arange(3 * len(atoms)) + 1).reshape(-1, 3) * 0.02
    matrix = calc.hessian(atoms, multiplicity=spins)
    product = calc.hessian_vector_product(atoms, direction, multiplicity=spins)
    np.testing.assert_allclose(matrix.matrix, matrix.matrix.T, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        matrix.matrix @ direction.ravel(), product.value.ravel(), atol=2e-8, rtol=0
    )
    for result, values in ((matrix, matrix.matrix), (product, product.value)):
        with pytest.raises(ValueError):
            values.setflags(write=True)
        diagnostic = result.diagnostics
        assert diagnostic["numerical_runtime"] == "native-cpp-cpu"
        assert diagnostic["public_calculator_endpoint"]
        assert diagnostic["endpoint_seconds"] >= diagnostic["assembly_seconds"]
        assert diagnostic["assembly_seconds"] == diagnostic["execution_seconds"]
        assert diagnostic["scf_energy"] == pytest.approx(
            primal.energy - dispersion_energy, abs=2e-11
        )
        assert diagnostic["total_energy"] == pytest.approx(primal.energy, abs=2e-11)
        assert diagnostic["dispersion_energy"] == pytest.approx(
            dispersion_energy, abs=2e-11
        )
        assert diagnostic["max_eri_primitives"] == 100_000_000
        assert not diagnostic["runtime_compilation"]
        assert not diagnostic["displaced_scf_evaluations"]
        assert not diagnostic["posthoc_symmetrization"]
        assert diagnostic["total_method_identity"] == calc.method_ir.identity
        assert diagnostic["source_energy_error"] < 2e-9
        assert (diagnostic["d3_pair_visits"] > 0) == bj
        diagnostic["runtime_compilation"] = True
        assert not result.diagnostics["runtime_compilation"]
    assert (
        matrix.diagnostics["d3_pair_visits"]
        == len(atoms) * 3 * product.diagnostics["d3_pair_visits"]
    )
    calc.clear_cache()


def test_energy_only_owner_cannot_publish_a_derivative_snapshot() -> None:
    """Declining export preserves energy execution, never derivative admission."""
    calc = calculator(1, False)
    atoms = geometry(1)
    try:
        with calc.prepare_batch([atoms], warm_start=False, ks_export=False) as batch:
            assert batch.execute(strict=True, properties=("energy",)).items[0].converged
            with pytest.raises(ValueError, match="ks_export=True"):
                batch.execute(properties=("energy", "forces"))
            with pytest.raises(RuntimeError, match="final state"):
                execute_batch(calc, batch, atoms, charge=0, multiplicity=1)
    finally:
        calc.clear_cache()


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("bj", [False, True])
def test_native_cpu_discrete_grid_covariance_and_atom_order(
    spins: int, bj: bool
) -> None:
    """Rotate within the angular-grid symmetry, not an arbitrary continuum orbit.

    A quarter-turn about z preserves this product angular grid. Reversing mixed
    atomic identities and translating the origin also exercise physical-center
    indexing in every raw Hessian block and in the independently evaluated HVP.
    """
    calc = calculator(spins, bj)
    initial = geometry(spins)
    atoms = (Atom(2, initial[0].position), *initial[1:])
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    permutation = np.arange(len(atoms))[::-1]
    transform = np.kron(np.eye(len(atoms))[permutation], rotation)
    positions = np.array([atom.position for atom in atoms])
    shifted = positions[permutation] @ rotation.T + np.array((0.3, -0.4, 0.2))
    moved = tuple(
        Atom(atoms[index].atomic_number, tuple(position))
        for index, position in zip(permutation, shifted, strict=True)
    )
    context = {"charge": 1, "multiplicity": spins}
    original = calc.hessian(atoms, **context)
    changed = calc.hessian(moved, **context)
    np.testing.assert_allclose(
        changed.matrix, transform @ original.matrix @ transform.T, atol=2e-8, rtol=0
    )
    direction = np.sin(np.arange(3 * len(atoms)) + 1) * 0.02
    product = calc.hessian_vector_product(
        moved, (transform @ direction).reshape(-1, 3), **context
    )
    np.testing.assert_allclose(
        product.value.ravel(),
        transform @ original.matrix @ direction,
        atol=2e-8,
        rtol=0,
    )
    for result in (changed, product):
        diagnostic = result.diagnostics
        assert diagnostic["total_method_identity"] == calc.method_ir.identity
        assert diagnostic["total_energy"] == pytest.approx(
            original.diagnostics["total_energy"], abs=2e-9, rel=0
        )
        assert diagnostic["spin"] == original.diagnostics["spin"]
        assert diagnostic["numerical_runtime"] == "native-cpp-cpu"
        assert not diagnostic["displaced_scf_evaluations"]
        assert not diagnostic["posthoc_symmetrization"]
    calc.clear_cache()


def test_native_cpu_preflight_refuses_missing_electronic_and_total_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calc = calculator(1, True)
    monkeypatch.setattr(
        calc, "prepare_batch", lambda *_, **__: pytest.fail("premature SCF")
    )
    library = calc._library
    with monkeypatch.context() as patch:
        patch.setattr(library, "generativeqc_ks_hessian_cpu_version_v1", None)
        with pytest.raises(NotImplementedError, match="no Python/JIT fallback"):
            calc.hessian(geometry(1))
    from generativeqc import _named_ks

    monkeypatch.setattr(_named_ks, "select_named_program", lambda *_, **__: None)
    with pytest.raises(
        NotImplementedError, match="electronic-only publication is forbidden"
    ):
        calc.hessian(geometry(1))


def test_existing_batch_authenticates_actual_geometry_and_replay() -> None:
    calc = calculator(1, True)
    atoms = geometry(1)
    with calc.prepare_batch([atoms], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        changed = (replace(atoms[0], position=(0.01, 0, 0)), atoms[1])
        for request in (
            {"atoms": changed, "charge": 0, "multiplicity": 1},
            {"atoms": atoms, "charge": 1, "multiplicity": 1},
            {"atoms": atoms, "charge": 0, "multiplicity": 2},
        ):
            with pytest.raises(RuntimeError, match="disagrees with"):
                execute_batch(calc, batch, **request)
        first = execute_batch(calc, batch, atoms, charge=0, multiplicity=1)
        assert not first.diagnostics["public_calculator_endpoint"]
        batch.execute(
            coordinates=[np.array([atom.position for atom in changed])],
            strict=True,
            properties=("energy",),
        )
        with pytest.raises(RuntimeError, match="geometry disagrees"):
            execute_batch(calc, batch, atoms, charge=0, multiplicity=1)
        replay = execute_batch(calc, batch, changed, charge=0, multiplicity=1)
        assert first.identity != replay.identity
        assert np.max(abs(first.matrix - replay.matrix)) > 1e-6
    with pytest.raises(RuntimeError):
        execute_batch(calc, batch, changed, charge=0, multiplicity=1)


def linked_pbe0_calculator(
    *, max_iterations: int = 200, resource_budget: ResourceBudget | None = None
) -> Calculator:
    """Use the corrected linked program, not a legacy numerical fallback."""
    calc = Calculator(
        method="pbe0-d3(bj)-rks",
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=max_iterations,
        resource_budget=resource_budget,
    )
    if not linked_method(calc._library, calc.method_ir.identity):
        pytest.skip("requires a linked CPU PBE0-D3(BJ) inventory")
    return calc


def test_linked_pbe0_whole_budget_refuses_before_primal_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan without space for the primal cannot enter SCF or either derivative owner."""
    calc = linked_pbe0_calculator(resource_budget=ResourceBudget(host_bytes=1))
    atoms = (Atom(2, (0.0, 0.0, -0.72)), Atom(1, (0.11, 0.13, 0.72)))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("infeasible resource plan prepared SCF or used Python response")

    monkeypatch.setattr(calc, "prepare_batch", forbidden)
    monkeypatch.setattr(calc, "_run_public_rks_second_order", forbidden)
    with pytest.raises(MemoryError, match="no supported plan fits"):
        calc.hessian_vector_product(atoms, np.ones((2, 3)), charge=1)
    with pytest.raises(MemoryError, match="no supported plan fits"):
        calc.hessian(atoms, charge=1)


@pytest.mark.parametrize("dense", (False, True), ids=("hvp", "hessian"))
def test_explicit_integral_capacity_is_checked_before_scf(
    monkeypatch: pytest.MonkeyPatch, dense: bool
) -> None:
    """Find the native byte boundary without running SCF or duplicating sizing."""
    from generativeqc._native_cpu_hessian import _minimum_bytes, _options

    calc = linked_pbe0_calculator()
    atoms = (Atom(2, (0.0, 0.0, -0.72)), Atom(1, (0.11, 0.13, 0.72)))
    direction = np.array([[0.02, -0.03, -0.1], [-0.02, 0.03, 0.1]])
    options = _options(64 << 20, 128 << 20, 64 << 20, 256 << 20, None)

    def evaluate(**controls: int) -> Any:
        if dense:
            return calc.hessian(atoms, charge=1, **controls)
        return calc.hessian_vector_product(atoms, direction, charge=1, **controls)

    low, high = 1, options.integral_bytes
    _minimum_bytes(calc, atoms, 1, 1, dense, options)
    while high - low > 1:
        probe = (low + high) // 2
        options.integral_bytes = probe
        try:
            _minimum_bytes(calc, atoms, 1, 1, dense, options)
        except RuntimeError as error:
            assert "integral storage cap exceeded" in str(error)
            low = probe
        else:
            high = probe

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("one byte below the native integral storage minimum prepared SCF")

    with monkeypatch.context() as scoped:
        scoped.setattr(calc, "prepare_batch", forbidden)
        with pytest.raises(RuntimeError, match="integral storage cap exceeded"):
            evaluate(integral_budget_bytes=high - 1)

    reference, exact = evaluate(), evaluate(integral_budget_bytes=high)
    result = (lambda item: item.matrix) if dense else (lambda item: item.value)
    np.testing.assert_allclose(result(exact), result(reference), atol=2e-8, rtol=0)
    assert (
        exact.diagnostics["scf_fock_builds"] == reference.diagnostics["scf_fock_builds"]
    )


@pytest.mark.parametrize(
    "method", ("pbe0-d3(bj)-rks", "b3lyp-d3(bj)-uks", "lda-rks", "pbe-rks")
)
def test_cpu_hessian_shared_host_budget_at_exact_and_below_capacity(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native HVP and scalar dense floors are additive with the live primal plan."""
    semilocal = method in ("lda-rks", "pbe-rks")
    if semilocal or method.startswith("pbe0"):
        atoms = (Atom(2, (0.0, 0.0, -0.72)), Atom(1, (0.11, 0.13, 0.72)))
        charge, multiplicity = 1, 1
    else:
        atoms = geometry(2)
        charge, multiplicity = 0, 2
    system = {"charge": charge, "multiplicity": multiplicity}
    direction = np.sin(1 + np.arange(3 * len(atoms))).reshape(-1, 3) * 0.02

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("budgeted native Hessian used Python response")

    def make(budget: ResourceBudget | None) -> Calculator:
        calc = Calculator(
            method=method,
            device="cpu",
            basis_representation="cartesian" if semilocal else "spherical",
            ks_options=KsOptions(grid=GRID),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            max_iterations=200,
            resource_budget=budget,
        )
        if not linked_method(calc._library, calc.method_ir.identity):
            pytest.skip("requires a linked CPU Hessian inventory for this method")
        monkeypatch.setattr(calc, "_run_public_rks_second_order", forbidden)
        return calc

    unrestricted = make(None)
    action = unrestricted.hessian_vector_product(atoms, direction, **system)
    dense = unrestricted.hessian(atoms, **system)
    probe = make(ResourceBudget())
    for product, oracle in (("hvp", action), ("hessian", dense)):
        is_dense = product == "hessian"

        def evaluate(
            calc: Calculator, is_dense: bool = is_dense, **controls: int
        ) -> Any:
            if is_dense:
                return calc.hessian(atoms, **system, **controls)
            return calc.hessian_vector_product(atoms, direction, **system, **controls)

        sizing = evaluate(probe).diagnostics["resource_admission"]
        # With no optional panels, dense and HVP differ only in the result panel.
        required = (
            action.diagnostics["numeric_reservation_bytes"]
            + oracle.diagnostics["output_bytes"]
            - action.diagnostics["output_bytes"]
        )
        exact = (
            sizing["plan_host_peak_bytes"]
            - sizing["derivative_host_cap_bytes"]
            + required
        )
        # Reserve must be deducted before the native capacity is selected.
        admitted = evaluate(
            make(ResourceBudget(host_bytes=exact + 1024, host_reserve_bytes=1024))
        )
        diagnostic = admitted.diagnostics
        assert diagnostic["numerical_runtime"] == "native-cpp-cpu"
        assert diagnostic["numeric_reservation_bytes"] == required
        assert diagnostic["resource_admission"]["plan_host_peak_bytes"] == exact
        assert diagnostic["resource_admission"]["effective_host_limit_bytes"] == exact
        assert diagnostic["resource_admission"]["derivative_host_cap_bytes"] == required
        assert diagnostic["d3_pair_visits"] == oracle.diagnostics["d3_pair_visits"]
        np.testing.assert_allclose(
            admitted.matrix if is_dense else admitted.value,
            oracle.matrix if is_dense else oracle.value,
            atol=2e-8,
            rtol=0,
        )
        capped = evaluate(unrestricted, numeric_budget_bytes=required)
        assert capped.diagnostics["numeric_reservation_bytes"] == required
        short_global = make(ResourceBudget(host_bytes=exact - 1))
        with monkeypatch.context() as scoped:
            # Failure before preparing a batch also proves that SCF cannot start.
            scoped.setattr(short_global, "prepare_batch", forbidden)
            scoped.setattr(unrestricted, "prepare_batch", forbidden)
            with pytest.raises(MemoryError, match="no supported plan fits"):
                evaluate(short_global)
            with pytest.raises(
                MemoryError, match="aggregate numerical reservation exceeded"
            ):
                evaluate(unrestricted, numeric_budget_bytes=required - 1)

    with probe.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity]
    ) as batch:
        batch.execute(strict=True, properties=("energy",))
        with pytest.raises(
            NotImplementedError, match="composed derivative reservation"
        ):
            execute_batch(probe, batch, atoms, direction=direction, **system)


def test_linked_pbe0_nonconverged_scf_cannot_publish_a_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an unconverged accepted-model request before either derivative owner."""
    calc = linked_pbe0_calculator(max_iterations=1)
    atoms = (Atom(2, (0.0, 0.0, -0.72)), Atom(1, (0.11, 0.13, 0.72)))
    direction = np.array([[0.02, -0.03, -0.1], [-0.02, 0.03, 0.1]])
    with calc.prepare_batch([atoms], charges=[1]) as batch:
        primal = batch.execute(properties=("energy",))
        assert primal.failure_indices == (0,)
        assert not primal.items[0].converged
        with pytest.raises(RuntimeError, match="final state"):
            execute_batch(
                calc, batch, atoms, charge=1, multiplicity=1, direction=direction
            )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("nonconverged SCF entered a derivative owner")

    monkeypatch.setattr("generativeqc._native_cpu_hessian.execute_batch", forbidden)
    monkeypatch.setattr(calc, "_run_public_rks_second_order", forbidden)
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        calc.hessian_vector_product(atoms, direction, charge=1)
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        calc.hessian(atoms, charge=1)


def test_linked_pbe0_ragged_batch_replay_and_failure_isolation() -> None:
    """A failed item cannot lend its derivative state or disturb a moved neighbor."""
    calc = linked_pbe0_calculator()
    heh = (Atom(2, (0.0, 0.0, -0.72)), Atom(1, (0.11, 0.13, 0.72)))
    water = (
        Atom(8, (0.0, 0.0, 0.12)),
        Atom(1, (1.45, 0.04, -1.02)),
        Atom(1, (-1.37, -0.11, -1.07)),
    )
    moved = (*water[:2], replace(water[2], position=(-1.39, -0.12, -1.04)))
    charges = (1, 0)
    directions = [
        np.sin(1 + np.arange(3 * len(atoms))).reshape(-1, 3) * 0.02
        for atoms in (heh, water)
    ]
    with calc.prepare_batch([heh, water], charges=charges, warm_start=True) as batch:

        def action(index: int, atoms: tuple[Atom, ...]) -> Any:
            return execute_batch(
                calc,
                batch,
                atoms,
                charge=charges[index],
                multiplicity=1,
                index=index,
                direction=directions[index],
            )

        with pytest.raises(RuntimeError, match="final state"):
            action(1, water)
        cold = batch.execute(strict=True, properties=("energy",))
        assert all(not item.warm_start_used for item in cold.items)
        cold_actions = [
            action(index, atoms) for index, atoms in enumerate((heh, water))
        ]
        warm = batch.execute(strict=True, properties=("energy",))
        assert all(item.warm_start_used for item in warm.items)
        for index, atoms in enumerate((heh, water)):
            product = action(index, atoms)
            assert product.diagnostics["total_energy"] == pytest.approx(
                warm.items[index].energy, abs=2e-10, rel=0
            )
            np.testing.assert_allclose(
                product.value, cold_actions[index].value, atol=2e-8, rtol=0
            )

        partial = batch.execute(
            [np.zeros((1, 3)), np.array([atom.position for atom in moved])],
            properties=("energy",),
        )
        assert partial.failure_indices == (0,)
        with pytest.raises(RuntimeError, match="final state"):
            action(0, heh)
        with pytest.raises(RuntimeError, match="geometry disagrees"):
            action(1, water)
        product = action(1, moved)
        assert product.diagnostics["total_energy"] == pytest.approx(
            partial.items[1].energy, abs=2e-10, rel=0
        )
        cold_moved = calc.hessian_vector_product(moved, directions[1])
        np.testing.assert_allclose(product.value, cold_moved.value, atol=2e-8, rtol=0)
        assert np.max(np.abs(cold_actions[1].value - product.value)) > 1e-6

        recovered = batch.execute(strict=True, properties=("energy",))
        assert recovered.succeeded
        with pytest.raises(RuntimeError, match="geometry disagrees"):
            action(1, moved)
        for index, atoms in enumerate((heh, water)):
            np.testing.assert_allclose(
                action(index, atoms).value,
                cold_actions[index].value,
                atol=2e-8,
                rtol=0,
            )


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, 1 << 63])
@pytest.mark.parametrize("endpoint", ["hessian", "hvp"])
def test_native_cpu_invalid_work_caps_fail_before_scf(
    monkeypatch: pytest.MonkeyPatch, limit: Any, endpoint: str
) -> None:
    calc = calculator(1, True)
    monkeypatch.setattr(
        calc, "prepare_batch", lambda *_, **__: pytest.fail("premature SCF")
    )
    with pytest.raises(ValueError, match="max_eri_primitives"):
        if endpoint == "hessian":
            calc.hessian(geometry(1), max_eri_primitives=limit)
        else:
            calc.hessian_vector_product(
                geometry(1), np.ones((2, 3)), max_eri_primitives=limit
            )


@pytest.mark.parametrize("endpoint", ["hessian", "hvp"])
def test_native_cpu_work_cap_is_never_ignored_by_another_owner(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    calc = calculator(1, False)
    monkeypatch.setattr(calc, "_native_cpu_second_order", False)
    monkeypatch.setattr(
        calc, "prepare_batch", lambda *_, **__: pytest.fail("premature SCF")
    )
    with pytest.raises(NotImplementedError, match="max_eri_primitives"):
        if endpoint == "hessian":
            calc.hessian(geometry(1), max_eri_primitives=100_000_000)
        else:
            calc.hessian_vector_product(
                geometry(1), np.ones((2, 3)), max_eri_primitives=100_000_000
            )


def test_native_cpu_work_cap_covers_the_whole_matrix_and_recovers() -> None:
    """A per-column allowance must not authorize a complete matrix.

    Measured visits can be below the conservative, direction-independent plan;
    recover using an already admitted capacity rather than inferring that plan.
    """
    calc = calculator(1, True)
    atoms = geometry(1)
    direction = np.ascontiguousarray(np.sin(np.arange(6)).reshape(2, 3))
    with calc.prepare_batch([atoms], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        action = execute_batch(
            calc, batch, atoms, charge=0, multiplicity=1, direction=direction
        )
        per_column = sum(
            action.diagnostics[name]
            for name in ("first_eri_primitives", "curvature_eri_primitives")
        )
        for limit in (1, per_column):
            with pytest.raises(
                (RuntimeError, MemoryError), match="complete nuclear primitive-work cap"
            ):
                execute_batch(
                    calc,
                    batch,
                    atoms,
                    charge=0,
                    multiplicity=1,
                    max_eri_primitives=limit,
                )
        total = action.diagnostics["max_eri_primitives"]
        matrix = execute_batch(
            calc, batch, atoms, charge=0, multiplicity=1, max_eri_primitives=total
        )
        assert matrix.diagnostics["max_eri_primitives"] == total
        assert (
            sum(
                matrix.diagnostics[name]
                for name in ("first_eri_primitives", "curvature_eri_primitives")
            )
            <= total
        )
        assert matrix.diagnostics["completed_directions"] == 6
        np.testing.assert_allclose(
            matrix.matrix @ direction.ravel(), action.value.ravel(), atol=2e-8, rtol=0
        )
    public = calc.hessian(atoms, max_eri_primitives=total)
    assert public.diagnostics["public_calculator_endpoint"]
    assert public.diagnostics["max_eri_primitives"] == total
    np.testing.assert_allclose(public.matrix, matrix.matrix, atol=2e-8, rtol=0)


def test_native_cpu_progress_bounds_failure_recovery_and_best_effort_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Diagnostics distinguish admitted work, completed directions and refusals."""
    calc = calculator(1, False)
    atoms = geometry(1)
    with calc.prepare_batch([atoms], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        refused = tmp_path / "refused.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(refused))
        with pytest.raises(
            (RuntimeError, MemoryError), match="complete nuclear primitive-work cap"
        ):
            execute_batch(
                calc, batch, atoms, charge=0, multiplicity=1, max_eri_primitives=1
            )
        refused_rows = [json.loads(line) for line in refused.read_text().splitlines()]
        assert any(
            row["event"] == "END" and row["status"] == "exception"
            for row in refused_rows
        )
        assert not any(row.get("key") == "completed_directions" for row in refused_rows)

        completed = tmp_path / "completed.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(completed))
        result = execute_batch(calc, batch, atoms, charge=0, multiplicity=1)
        rows = [json.loads(line) for line in completed.read_text().splitlines()]
        admission = next(
            row for row in rows if row.get("key") == "admitted_xc_map_terms"
        )
        values = {
            row["key"]: row["value"]
            for row in rows
            if row["id"] == admission["id"] and row["event"] == "VALUE"
        }
        planned = sum(
            values[name]
            for name in (
                "admitted_first_eri_primitives",
                "admitted_curvature_eri_primitives",
            )
        )
        observed = sum(
            result.diagnostics[name]
            for name in ("first_eri_primitives", "curvature_eri_primitives")
        )
        assert observed <= planned <= result.diagnostics["max_eri_primitives"]
        assert 0 < values["admitted_xc_map_terms"] <= 1 << 40
        assert values["frozen_integral_schedule"] in {
            "column",
            "matrix",
            "projected_matrix",
        }
        matrix_calls = [
            row["value"]
            for row in rows
            if row.get("key") == "completed_eri_matrix_primitives"
        ]
        if values["frozen_integral_schedule"] != "column":
            assert matrix_calls
            assert (
                0 < sum(matrix_calls) <= result.diagnostics["curvature_eri_primitives"]
            )
            assert sum(matrix_calls) <= values["admitted_curvature_eri_primitives"]
            coordinates = sum(
                row["value"]
                for row in rows
                if row.get("key") == "completed_eri_matrix_coordinates"
            )
            partials = sum(
                row["value"]
                for row in rows
                if row.get("key") == "completed_eri_matrix_second_partials"
            )
            assert 3 * sum(matrix_calls) <= coordinates <= 9 * sum(matrix_calls)
            assert 3 * coordinates <= partials <= 81 * sum(matrix_calls)
        else:
            assert not matrix_calls
        assert (
            max(
                row["value"] for row in rows if row.get("key") == "completed_directions"
            )
            == result.diagnostics["completed_directions"]
            == 3 * len(atoms)
        )
        assert any(
            row["id"] == admission["id"]
            and row["event"] == "END"
            and row["status"] == "returned"
            for row in rows
        )

        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(tmp_path))
        unavailable_journal = execute_batch(
            calc, batch, atoms, charge=0, multiplicity=1
        )
        np.testing.assert_allclose(
            unavailable_journal.matrix, result.matrix, atol=3e-12, rtol=0
        )


def test_native_cpu_honors_response_and_output_limits_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calc = calculator(2, True)
    with pytest.raises(
        (RuntimeError, MemoryError), match="workspace|reservation|admission"
    ):
        calc.hessian(
            geometry(2),
            multiplicity=2,
            solver_options=GMRESOptions(max_workspace_bytes=1),
        )
    with monkeypatch.context() as patch:
        patch.setattr(
            calc, "prepare_batch", lambda *_, **__: pytest.fail("premature SCF")
        )
        with pytest.raises(ValueError, match="output_budget_bytes"):
            calc.hessian(geometry(2), multiplicity=2, output_budget_bytes=1)
    recovered = calc.hessian(geometry(2), multiplicity=2)
    assert np.isfinite(recovered.matrix).all()


@pytest.mark.parametrize("spins", [1, 2])
def test_native_orbitals_are_bounded_detached_and_metric_orthonormal(
    spins: int,
) -> None:
    from generativeqc._molecular_saddle import _occupied_blocks, _occupied_overlaps
    from generativeqc.overlap import cross_overlap

    calc = calculator(spins, True)
    atoms = geometry(spins)
    with calc.prepare_batch([atoms], multiplicities=[spins], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        request = {"charge": 0, "multiplicity": spins, "maximum_bytes": 128 << 20}
        frame = orbital_frame(calc, batch, atoms, **request)
        assert frame.identity.spin == calc.ks_options.method_ir.spin
        assert frame.coefficients.shape[0] == spins
        assert not frame.coefficients.flags.writeable
        assert not frame.occupations.flags.writeable
        metric = cross_overlap(calc, calc, atoms, multiplicity=spins)
        for coefficients in frame.coefficients:
            np.testing.assert_allclose(
                coefficients.T @ metric @ coefficients,
                np.eye(metric.shape[0]),
                atol=2e-12,
                rtol=0,
            )
        blocks = _occupied_blocks(frame, calc.ks_options.method_ir.spin)
        assert tuple(block.shape[1] for block in blocks) == (
            (1,) if spins == 1 else (2, 1)
        )
        np.testing.assert_allclose(
            _occupied_overlaps(blocks, blocks, metric, 0.9), 1, atol=2e-12
        )
        with pytest.raises((MemoryError, RuntimeError), match="reservation|budget"):
            orbital_frame(
                calc,
                batch,
                atoms,
                **{**request, "maximum_bytes": frame.numeric_reservation_bytes - 1},
            )
        moved = (replace(atoms[0], position=(0.01, 0, 0)), *atoms[1:])
        with pytest.raises(RuntimeError, match="geometry disagrees"):
            orbital_frame(calc, batch, moved, **request)
        replay = orbital_frame(calc, batch, atoms, **request)
        np.testing.assert_allclose(
            replay.coefficients, frame.coefficients, atol=2e-12, rtol=0
        )
    assert np.isfinite(frame.coefficients).all()
    with pytest.raises(RuntimeError):
        orbital_frame(calc, batch, atoms, **request)


@pytest.mark.parametrize("failure", [None, "response", "root_budget", "eri_work"])
def test_actual_saddle_uses_native_total_batch_without_python_numerical_owner(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from generativeqc.batch import PreparedBatch
    from generativeqc.composite_rks_hessian import CompositeRKSResponse
    from generativeqc.saddle import SaddleOptions, optimize_saddle

    calc = calculator(2, True)

    def reject(*arguments: Any, **keywords: Any) -> None:
        raise AssertionError("saddle entered a Python numerical provider")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", reject)
    monkeypatch.setattr(CompositeRKSResponse, "from_native", reject)
    monkeypatch.setattr("generativeqc.d3_hessian.compose_second_order", reject)
    monkeypatch.setattr("generativeqc_compiler.dft.NativeAO", reject)
    result = optimize_saddle(
        calc,
        geometry(2),
        masses=[1.00782503223] * 3,
        multiplicity=2,
        options=SaddleOptions(
            max_attempts=1, project_rigid_motions=True, raw_symmetry_tolerance=2e-7
        ),
        root_overlap_budget_bytes=1 if failure == "root_budget" else 128 << 20,
        max_eri_primitives=1 if failure == "eri_work" else None,
        solver_options=GMRESOptions(max_workspace_bytes=1)
        if failure == "response"
        else None,
    )
    assert not result.converged
    assert result.optimizer.receipt["attempts"] <= 1
    assert not result.diagnostics["runtime_compilation"]
    assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
    if failure:
        assert result.optimizer.failures
        assert not result.diagnostics["work_counters_complete"]
        assert result.diagnostics["nuclear_response_solves"] == 0
    else:
        assert not result.optimizer.failures
        assert result.diagnostics["nuclear_response_solves"] > 0
        assert result.diagnostics["d3_pair_visits"] > 0
        assert result.optimizer.receipt["has_valid_state"]
        assert result.diagnostics["work_counters_complete"]
    calc.clear_cache()
