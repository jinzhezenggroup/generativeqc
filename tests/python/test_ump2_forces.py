"""Complete native UMP2 forces against independent analytic/reconverged oracles."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from _ump2_force_reference import analytic_gradient
from generativeqc import Calculator, _native
from generativeqc_compiler.dft import NativeAO


def _oracle(atoms: typing.Any, multiplicity: int) -> typing.Any:
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, mp, scf

    with NativeAO(atoms, basis="sto-3g", multiplicity=multiplicity) as ao:
        basis = {}
        for shell in ao.shells:
            element = atoms[shell.atom_index][0]
            row = [
                shell.angular_momentum,
                *[[p.exponent, p.coefficient] for p in shell.primitives],
            ]
            if row not in basis.setdefault(element, []):
                basis[element].append(row)
    mol = gto.M(
        atom=atoms,
        basis=basis,
        spin=multiplicity - 1,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    hf = scf.UHF(mol)
    hf.conv_tol, hf.conv_tol_grad, hf.max_cycle = 1e-13, 1e-11, 200
    hf.kernel()
    assert hf.converged
    return mp.UMP2(hf).run()


@pytest.mark.parametrize(
    "atoms,multiplicity",
    [
        ([("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))], 1),
        ([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5)), ("H", (0.3, 0.0, 3.2))], 2),
        ([("O", (0.0, 0.0, 0.0)), ("H", (0.2, 0.0, 1.8))], 2),
        ([("N", (0.0, 0.0, 0.0)), ("H", (1.4, 0.0, 1.1)), ("H", (-1.1, 0.2, 1.4))], 2),
    ],
)
def test_native_ump2_force_independent_analytic_and_multistep(
    atoms: typing.Any, multiplicity: int
) -> None:
    oracle = _oracle(atoms, multiplicity)
    calc = Calculator(
        method="ump2",
        basis="sto-3g",
        device="cpu",
        initial_guess=None,
        max_iterations=200,
    )
    result = calc.singlepoint(
        atoms, multiplicity=multiplicity, properties=("energy", "forces")
    )
    assert result.converged
    np.testing.assert_allclose(result.energy, oracle.e_tot, atol=2e-8, rtol=0)
    expected = -analytic_gradient(oracle)
    np.testing.assert_allclose(result.forces, expected, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(np.sum(result.forces, axis=0), 0, atol=1e-9)
    direction = np.random.default_rng(1823).normal(size=(len(atoms), 3))
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    positions = np.asarray([p for _, p in atoms])
    expected_direction = -np.vdot(result.forces, direction)
    errors = []
    for step in (1e-3, 3e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (a, p.tolist())
                for (a, _), p in zip(
                    atoms, positions + sign * step * direction, strict=True
                )
            ]
            energies.append(
                calc.singlepoint(
                    moved, multiplicity=multiplicity, properties=("energy",)
                ).energy
            )
        error = abs((energies[0] - energies[1]) / (2 * step) - expected_direction)
        errors.append(error)
    assert errors[-1] < 2e-7
    assert errors[-1] <= errors[0] * 0.5 + 1e-8
    diagnostic = result.correlation
    assert diagnostic.response_absolute_residual < 1e-9
    assert diagnostic.numeric_capacity_bytes <= 256 << 20
    assert diagnostic.planned_endpoint_peak_bytes > 0
    assert diagnostic.response_operator_hash


def test_native_ump2_force_restricted_limit_and_prepared_replay() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    options: dict[str, typing.Any] = {
        "basis": "sto-3g",
        "device": "cpu",
        "initial_guess": None,
    }
    calc = Calculator(method="ump2", **options)
    restricted = Calculator(method="mp2", **options).singlepoint(
        atoms, properties=("energy", "forces")
    )
    with calc.prepare_batch([atoms], warm_start=True) as batch:
        first = batch.execute(properties=("energy", "forces"), strict=True).items[0]
        warm = batch.execute(properties=("energy", "forces"), strict=True).items[0]
        np.testing.assert_allclose(first.forces, restricted.forces, atol=2e-9, rtol=0)
        np.testing.assert_allclose(warm.forces, first.forces, atol=2e-9, rtol=0)
        assert warm.warm_start_used
        moved = [("H", (0.0, 0.0, -0.75)), ("H", (0.0, 0.0, 0.75))]
        changed = batch.execute(
            coordinates=[np.array([p for _, p in moved])],
            properties=("energy", "forces"),
            strict=True,
        ).items[0]
        cold = calc.singlepoint(moved, properties=("energy", "forces"))
        np.testing.assert_allclose(changed.forces, cold.forces, atol=2e-9, rtol=0)


def test_native_ump2_force_budget_failure_is_transactional() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    options: dict[str, typing.Any] = {
        "method": "ump2",
        "basis": "sto-3g",
        "device": "cpu",
        "initial_guess": None,
    }
    with Calculator(**options).prepare_batch([atoms], warm_start=True) as probe:
        probe.execute(properties=("energy",), strict=True)
        energy_capacity = (
            probe.execute(properties=("energy",), strict=True)
            .items[0]
            .correlation.numeric_capacity_bytes
        )
        force_capacity = (
            probe.execute(properties=("energy", "forces"), strict=True)
            .items[0]
            .correlation.numeric_capacity_bytes
        )
    assert energy_capacity < force_capacity
    calc = Calculator(
        **options,
        correlation_memory_budget_bytes=(energy_capacity + force_capacity) // 2,
    )
    with calc.prepare_batch([atoms], warm_start=True) as batch:
        before = batch.execute(properties=("energy",), strict=True).items[0]
        failed = batch.execute(properties=("energy", "forces")).items[0]
        assert (
            not failed.succeeded
            and failed.forces is None
            and failed.correlation is None
        )
        assert failed.status == _native.STATUS_OUT_OF_MEMORY
        after = batch.execute(properties=("energy",), strict=True).items[0]
        assert after.warm_start_used and abs(after.energy - before.energy) < 1e-10


def test_native_ump2_force_empty_rotation_space() -> None:
    result = Calculator(
        method="ump2", basis="sto-3g", device="cpu", initial_guess=None
    ).singlepoint(
        [("H", (0.0, 0.0, 0.0))], multiplicity=2, properties=("energy", "forces")
    )
    assert result.converged
    np.testing.assert_allclose(result.forces, np.zeros((1, 3)), atol=1e-12)
