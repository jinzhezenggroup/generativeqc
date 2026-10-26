"""Streamed HF integral AD: independent numerics, capacity and work admission."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc import Primitive, Shell
from generativeqc.experimental.integral_autodiff import IntegralGradient, IntegralHF
from test_integral_autodiff_second import (
    BASIS,
    XYZ,
    _direction,
    _flat,
    _oracle,
    require_native_library,  # noqa: F401
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_direct_joint_derivatives_match_independent_reference(
    method: str, tmp_path: Path
) -> None:
    owner = IntegralHF(
        ("He" if method == "uhf" else "H", "H"),
        method=method,
        spin_multiplicity=2 if method == "uhf" else 1,
        basis=BASIS,
        derivative_order=2,
        integral_storage="direct",
        cache=tmp_path,
    )
    dense = IntegralHF(
        owner.numbers,
        method=method,
        spin_multiplicity=owner.spin_multiplicity,
        basis=BASIS,
        derivative_order=2,
        cache=tmp_path,
    )
    try:
        energy, gradient = owner.value_and_grad(XYZ)
        oracle = _oracle(owner, XYZ, owner.exponents, owner.coefficients)
        assert energy == pytest.approx(oracle.e_tot, abs=3e-11)
        np.testing.assert_allclose(
            gradient.coordinates, oracle.nuc_grad_method().kernel(), atol=2e-9
        )
        direction = _direction(owner)
        np.testing.assert_allclose(
            _flat(owner.hvp(XYZ, direction)),
            _flat(dense.hvp(XYZ, direction)),
            atol=3e-10,
        )
        expected = dense.integral_jvp(XYZ, direction)
        actual = owner.integral_jvp(XYZ, direction)
        np.testing.assert_allclose(actual.eri, expected.eri, atol=2e-12)
        assert owner.diagnostics()["integral_storage"] == "direct"
    finally:
        owner.close()
        dense.close()


def test_direct_p_shell_endpoint_fits_without_dense_eri(tmp_path: Path) -> None:
    """Sixteen AOs expose the n**4 capacity distinction beyond two-AO fixtures."""
    elements = ("H",) * 4
    basis = tuple(
        Shell(atom, angular, (Primitive(exponent, 1.0),))
        for atom in range(4)
        for angular, exponent in ((0, 1.0), (1, 0.6))
    )
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7], [0.0, 2.6, 0.2], [0.0, 2.6, 1.9]])
    budget = 2 << 20
    with pytest.raises(ValueError, match="max_bytes"):
        IntegralHF(
            elements, basis=basis, derivative_order=2, max_bytes=budget, cache=tmp_path
        )
    owner = IntegralHF(
        elements,
        basis=basis,
        derivative_order=2,
        integral_storage="direct",
        max_bytes=budget,
        cache=tmp_path,
    )
    try:
        energy, gradient = owner.value_and_grad(xyz)
        oracle = _oracle(owner, xyz, owner.exponents, owner.coefficients)
        assert energy == pytest.approx(oracle.e_tot, abs=3e-9)
        np.testing.assert_allclose(
            gradient.coordinates, oracle.nuc_grad_method().kernel(), atol=2e-8
        )
        direction = IntegralGradient(
            np.random.default_rng(11).normal(size=xyz.shape) * 0.1,
            np.zeros_like(owner.exponents),
            np.zeros_like(owner.coefficients),
        )
        actual = owner.hvp(xyz, direction)
        reference = oracle.Hessian().kernel().transpose(0, 2, 1, 3).reshape(12, 12)
        np.testing.assert_allclose(
            actual.coordinates.ravel(),
            reference @ direction.coordinates.ravel(),
            atol=3e-8,
        )
        assert owner.numeric_peak_bytes <= budget
        with pytest.raises(ValueError, match="dense integral publication"):
            owner.integrals(xyz)
    finally:
        owner.close()


def test_direct_fock_work_is_bounded(tmp_path: Path) -> None:
    count = 4**4 + 4 * 4**2 + 1
    owner = IntegralHF(
        ("H", "H"),
        basis=BASIS,
        integral_storage="direct",
        max_work=3 * count + 4**4,
        cache=tmp_path,
    )
    try:
        with pytest.raises(RuntimeError, match="max_work"):
            owner.energy(XYZ)
        assert owner.diagnostics() == {}
    finally:
        owner.close()


def test_direct_hvp_admits_the_complete_endpoint_budget(tmp_path: Path) -> None:
    """SCF and response cannot each spend the same advertised work allowance."""
    owner = IntegralHF(
        ("H", "H"),
        basis=BASIS,
        derivative_order=2,
        integral_storage="direct",
        max_work=4000,
        cache=tmp_path,
    )
    try:
        # This budget admits the primal, but not the complete coupled HVP.
        owner.energy(XYZ)
        previous = owner.diagnostics()
        with pytest.raises(RuntimeError, match="max_work"):
            owner.hvp(XYZ, _direction(owner))
        assert owner.diagnostics() == previous
    finally:
        owner.close()


def test_direct_hessian_shares_one_work_budget_and_recovers(tmp_path: Path) -> None:
    """Exhausted column work must not poison a retained physical state."""
    owner = IntegralHF(
        ("H", "H"),
        basis=BASIS,
        derivative_order=2,
        integral_storage="direct",
        max_work=20 * (4**4 + 4 * 4**2 + 1),
        cache=tmp_path,
    )
    try:
        with owner.prepare(XYZ) as state:
            direction = _direction(owner)
            expected = state.hvp(direction)
            previous = state.diagnostics()
            with pytest.raises((RuntimeError, ValueError), match="max_work"):
                state.hessian(parameters=("exponents",))
            assert state.diagnostics() == previous
            np.testing.assert_allclose(
                _flat(state.hvp(direction)), _flat(expected), atol=3e-10
            )
    finally:
        owner.close()


def test_direct_response_includes_the_shared_gradient_source(tmp_path: Path) -> None:
    """Frontend energy/gradient/response composition has one source allowance."""
    owner = IntegralHF(
        ("H", "H"),
        basis=BASIS,
        derivative_order=2,
        integral_storage="direct",
        max_work=4300,
        cache=tmp_path,
    )
    try:
        direction = _direction(owner)
        owner.hvp(XYZ, direction)
        with owner.prepare(XYZ) as state:
            state.gradient()
            previous = state.diagnostics()
            with pytest.raises(RuntimeError, match="max_work"):
                state.hvp(direction)
            assert state.diagnostics() == previous
    finally:
        owner.close()
