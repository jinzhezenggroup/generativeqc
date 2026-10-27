"""Independent gates for spin-resolved and joint second-order integral AD."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Primitive, Shell, _native
from generativeqc.experimental.integral_autodiff import (
    IntegralGradient,
    IntegralHF,
    IntegralValues,
)

XYZ = np.array([[-0.2, 0.1, -0.6], [0.1, -0.15, 0.8]])
BASIS = (
    Shell(0, 0, (Primitive(1.2, 0.7), Primitive(0.3, 0.4))),
    Shell(1, 0, (Primitive(0.9, 1.0), Primitive(0.2, 0.0))),
)


@pytest.fixture(scope="module", autouse=True)
def require_native_library() -> None:
    """Source-only runs may omit native builds; configured acceptance may not."""
    if not any(candidate.exists() for candidate in _native._candidate_paths()):
        if os.environ.get("GENERATIVEQC_LIBRARY"):
            pytest.fail("configured GENERATIVEQC_LIBRARY does not exist")
        pytest.skip("integral HF response acceptance requires a native library")


def _flat(value: IntegralGradient) -> np.ndarray:
    return np.concatenate(
        (value.coordinates.ravel(), value.exponents, value.coefficients)
    )


def _direction(model: IntegralHF, seed: int = 22) -> IntegralGradient:
    rng = np.random.default_rng(seed)
    return IntegralGradient(
        rng.normal(size=XYZ.shape) * 0.2,
        rng.normal(size=model.exponents.shape) * 0.1,
        rng.normal(size=model.coefficients.shape) * 0.1,
    )


def _oracle(
    model: IntegralHF, xyz: np.ndarray, alpha: np.ndarray, coefficients: np.ndarray
) -> Any:
    from pyscf import gto, scf
    from pyscf.data.elements import ELEMENTS

    # Supply the same raw Gaussian definition to the independent engine.
    # Built-in STO-3G catalogs differ in retained coefficient digits, enough
    # to obscure a tight absolute energy/derivative acceptance gate.
    labels = tuple(f"{ELEMENTS[z]}{i}" for i, z in enumerate(model.numbers))
    basis: dict[str, list[Any]] = {label: [] for label in labels}
    begin = 0
    for shell in model.shells:
        count = len(shell.primitives)
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    [float(alpha[i]), float(coefficients[i])]
                    for i in range(begin, begin + count)
                ],
            ]
        )
        begin += count
    mol = gto.M(
        atom=list(zip(labels, xyz, strict=True)),
        unit="Bohr",
        basis=basis,
        cart=True,
        charge=model.charge,
        spin=model.spin_multiplicity - 1,
        verbose=0,
    )
    mf = scf.RHF(mol) if model.method == "rhf" else scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope="module", params=["rhf", "uhf"])
def model(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    unrestricted = request.param == "uhf"
    owner = IntegralHF(
        ("He" if unrestricted else "H", "H"),
        method=request.param,
        spin_multiplicity=2 if unrestricted else 1,
        basis=BASIS,
        derivative_order=2,
        cache=tmp_path_factory.mktemp("integral-second"),
    )
    yield owner
    owner.close()


def test_spin_energy_gradient_and_nuclear_hessian_independent(
    model: IntegralHF,
) -> None:
    oracle = _oracle(model, XYZ, model.exponents, model.coefficients)
    value, gradient = model.value_and_grad(XYZ)
    assert value == pytest.approx(oracle.e_tot, abs=3e-11)
    np.testing.assert_allclose(
        gradient.coordinates, oracle.nuc_grad_method().kernel(), atol=2e-9
    )
    direction = _direction(model)
    direction = IntegralGradient(
        direction.coordinates,
        np.zeros_like(direction.exponents),
        np.zeros_like(direction.coefficients),
    )
    actual = model.hvp(XYZ, direction)
    hessian = oracle.Hessian().kernel().transpose(0, 2, 1, 3).reshape(6, 6)
    np.testing.assert_allclose(
        actual.coordinates.ravel(),
        hessian @ direction.coordinates.ravel(),
        atol=2e-8,
        rtol=2e-7,
    )
    assert model.diagnostics()["response_residual"] < 1e-10


def test_joint_hessian_independent_energy_and_symmetry(model: IntegralHF) -> None:
    first, second = _direction(model), _direction(model, 123)
    hv = model.hvp(XYZ, first)
    hu = model.hvp(XYZ, second)
    expected = _flat(second) @ _flat(hv)
    assert expected == pytest.approx(_flat(first) @ _flat(hu), abs=2e-10)
    # Reconverged independent energies verify mixed geometry/alpha/raw-c terms.
    # Two step sizes make a frozen-density or missing normalization response
    # distinguishable from the finite-difference truncation error.
    errors = []
    for step in (0.004, 0.001):
        total = 0.0
        for a in (-1, 1):
            for b in (-1, 1):
                xyz = XYZ + step * (a * first.coordinates + b * second.coordinates)
                alpha = model.exponents + step * (
                    a * first.exponents + b * second.exponents
                )
                coeff = model.coefficients + step * (
                    a * first.coefficients + b * second.coefficients
                )
                total += a * b * _oracle(model, xyz, alpha, coeff).e_tot
        errors.append(abs(total / (4 * step**2) - expected))
    assert errors[-1] < 2e-7, errors
    assert errors[-1] < max(errors[0] * 0.2, 1e-8), errors


def test_integral_jvp_matches_arbitrary_cotangent_transpose(model: IntegralHF) -> None:
    direction = _direction(model)
    tangent = model.integral_jvp(XYZ, direction)
    rng = np.random.default_rng(432)
    n = model.nao
    seeds = IntegralValues(
        rng.normal(size=(n, n)), rng.normal(size=(n, n)), rng.normal(size=(n,) * 4), 0.8
    )
    lhs = sum(
        np.sum(getattr(tangent, name) * getattr(seeds, name))
        for name in ("overlap", "hcore", "eri", "nuclear")
    )
    rhs = _flat(direction) @ _flat(model.integral_vjp(XYZ, seeds))
    assert lhs == pytest.approx(rhs, abs=2e-11)


def test_hessian_translation_and_normalization_scale_identities(
    model: IntegralHF,
) -> None:
    translation = IntegralGradient(
        np.ones_like(XYZ),
        np.zeros_like(model.exponents),
        np.zeros_like(model.coefficients),
    )
    np.testing.assert_allclose(_flat(model.hvp(XYZ, translation)), 0, atol=2e-11)
    # Scale invariance gives H c = -grad_c in coefficient components, and zero
    # in all other components. It does NOT imply H c is the zero vector.
    scale = IntegralGradient(
        np.zeros_like(XYZ), np.zeros_like(model.exponents), model.coefficients
    )
    hv = model.hvp(XYZ, scale)
    gradient = model.value_and_grad(XYZ)[1]
    np.testing.assert_allclose(hv.coordinates, 0, atol=2e-11)
    np.testing.assert_allclose(hv.exponents, 0, atol=2e-11)
    np.testing.assert_allclose(hv.coefficients, -gradient.coefficients, atol=2e-10)


def test_full_hessian_matches_direction_and_replay(model: IntegralHF) -> None:
    hessian = model.hessian(XYZ)
    evidence = model.diagnostics()
    assert evidence["hessian_columns"] == hessian.shape[0]
    assert evidence["primitive_value_evaluations"] == (
        (1 + hessian.shape[0]) * model.primitive_evaluations
    )
    assert evidence["primitive_pullback_evaluations"] == (
        hessian.shape[0] * model.primitive_evaluations
    )
    direction = _direction(model)
    np.testing.assert_allclose(hessian, hessian.T, atol=3e-10)
    np.testing.assert_allclose(
        hessian @ _flat(direction), _flat(model.hvp(XYZ, direction)), atol=3e-10
    )
    with pytest.raises(ValueError):
        hessian.setflags(write=True)
    changed = XYZ.copy()
    changed[1, 2] += 0.13
    assert (
        np.max(
            np.abs(
                _flat(model.hvp(changed, direction)) - _flat(model.hvp(XYZ, direction))
            )
        )
        > 1e-3
    )


def test_failed_hessian_preserves_last_successful_diagnostics(
    model: IntegralHF, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure after a completed column must not publish a partial receipt."""
    model.energy(XYZ)
    previous = model.diagnostics()
    library = model._calculator._library
    solve = library.generativeqc_integral_ad_hf_response_v1
    calls = 0

    def fail_during_hessian(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected response failure")
        return solve(*args)

    with monkeypatch.context() as context:
        context.setattr(
            library, "generativeqc_integral_ad_hf_response_v1", fail_during_hessian
        )
        with pytest.raises(RuntimeError, match="injected response failure"):
            model.hessian(XYZ, parameters=("coordinates",))
    assert model.diagnostics() == previous


def _water_hvp(backend: str, tmp_path: Any) -> None:
    """Exercise p angular components and nontrivial orbital relaxation."""
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 1.4, 1.1], [0.0, -1.4, 1.1]])
    owner = IntegralHF(
        ("O", "H", "H"), backend=backend, derivative_order=2, cache=tmp_path
    )
    try:
        direction = IntegralGradient(
            np.random.default_rng(44).normal(size=xyz.shape) * 0.1,
            np.zeros_like(owner.exponents),
            np.zeros_like(owner.coefficients),
        )
        actual = owner.hvp(xyz, direction)
        mf = _oracle(owner, xyz, owner.exponents, owner.coefficients)
        reference = (
            mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(9, 9)
            @ direction.coordinates.ravel()
        )
        np.testing.assert_allclose(
            actual.coordinates.ravel(), reference, atol=2e-8, rtol=3e-7
        )
        assert owner.diagnostics()["response_residual"] < 1e-10
    finally:
        owner.close()


def test_second_order_p_shell_oracle(tmp_path: Any) -> None:
    _water_hvp("cpu", tmp_path)


def _one_electron_response(backend: str, tmp_path: Any) -> None:
    """A one-AO hydrogen has no orbital rotations and a closed-form energy."""
    alpha = 0.7
    owner = IntegralHF(
        ("H",),
        method="uhf",
        spin_multiplicity=2,
        derivative_order=2,
        basis=(Shell(0, 0, (Primitive(alpha, -1.0),)),),
        backend=backend,
        cache=tmp_path,
    )
    try:
        xyz = np.array([[0.2, -0.3, 0.1]])
        energy, gradient = owner.value_and_grad(xyz)
        assert energy == pytest.approx(
            1.5 * alpha - 2 * np.sqrt(2 * alpha / np.pi), abs=2e-12
        )
        assert gradient.exponents[0] == pytest.approx(
            1.5 - np.sqrt(2 / (np.pi * alpha)), abs=2e-12
        )
        hessian = owner.hessian(xyz)
        expected = np.zeros((5, 5))
        expected[3, 3] = 0.5 * np.sqrt(2 / np.pi) * alpha**-1.5
        np.testing.assert_allclose(hessian, expected, atol=3e-11)
        assert owner.diagnostics()["response_residual"] == 0
        assert owner.diagnostics()["response_operator_actions"] == 0
    finally:
        owner.close()


def test_one_electron_empty_response_space(tmp_path: Any) -> None:
    _one_electron_response("cpu", tmp_path)


def test_retained_response_snapshot_and_recovery(model: IntegralHF) -> None:
    """Owned snapshots replay after caller mutation and reject invalid motion."""
    xyz, alpha = XYZ.copy(), model.exponents.copy()
    direction = _direction(model)
    expected = model.hvp(xyz, direction, exponents=alpha)
    with model.prepare(xyz, exponents=alpha) as state:
        xyz[:] = 99
        alpha[:] = 0.5
        before = state.diagnostics()
        np.testing.assert_allclose(
            _flat(state.hvp(direction)), _flat(expected), atol=3e-10
        )
        after = state.diagnostics()
        assert after["iterations"] == before["iterations"]
        assert after["scf_solves"] == before["scf_solves"]
        bad = IntegralGradient(
            np.full_like(XYZ, np.nan), direction.exponents, direction.coefficients
        )
        with pytest.raises(ValueError):
            state.hvp(bad)
        assert state.diagnostics() == after
        np.testing.assert_allclose(
            _flat(state.hvp(direction)), _flat(expected), atol=3e-10
        )
    state.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.hvp(direction)


def test_hvp_fits_without_full_hessian_budget(tmp_path: Any) -> None:
    """A single response remains usable when all-column work is inadmissible."""
    owner = IntegralHF(
        ("H", "H"),
        basis=BASIS,
        derivative_order=2,
        max_work=4 * (4**4 + 4 * 4**2 + 1),
        cache=tmp_path,
    )
    try:
        with owner.prepare(XYZ) as state:
            direction = _direction(owner)
            value = state.hvp(direction)
            assert np.isfinite(_flat(value)).all()
            with pytest.raises(ValueError, match="max_work"):
                state.hessian()
            owner.close()
            np.testing.assert_allclose(
                _flat(state.hvp(direction)), _flat(value), atol=3e-10
            )
    finally:
        owner.close()
