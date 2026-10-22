"""Opt-in real-device qualification of the complete explicit-integral HF chain."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc.experimental.integral_autodiff import IntegralHF, IntegralValues
from test_integral_autodiff_second import BASIS, XYZ, _direction, _flat, _oracle

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_INTEGRAL_AD_CUDA_TEST") != "1",
    reason="set GENERATIVEQC_INTEGRAL_AD_CUDA_TEST=1 inside a finite Slurm GPU job",
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_cuda_integral_primal_pullback_and_coupled_hvp(
    method: str, tmp_path: Path
) -> None:
    """GPU results match a separate CPU chain and independent PySCF endpoints."""
    options = {
        "method": method,
        "spin_multiplicity": 2 if method == "uhf" else 1,
        "basis": BASIS,
        "derivative_order": 2,
        "cache": tmp_path,
    }
    elements = ("He" if method == "uhf" else "H", "H")
    cpu = IntegralHF(elements, **options)
    gpu = IntegralHF(elements, backend="cuda", **options)
    try:
        energy, gradient = gpu.value_and_grad(XYZ)
        reference, expected = cpu.value_and_grad(XYZ)
        oracle = _oracle(cpu, XYZ, cpu.exponents, cpu.coefficients)
        assert energy == pytest.approx(oracle.e_tot, abs=3e-10)
        assert energy == pytest.approx(reference, abs=3e-11)
        np.testing.assert_allclose(
            gradient.coordinates, oracle.nuc_grad_method().kernel(), atol=3e-9
        )
        np.testing.assert_allclose(_flat(gradient), _flat(expected), atol=3e-10)
        direction = _direction(cpu)
        actual = gpu.hvp(XYZ, direction)
        expected = cpu.hvp(XYZ, direction)
        np.testing.assert_allclose(_flat(actual), _flat(expected), atol=5e-9, rtol=1e-8)
        diagnostic = gpu.diagnostics()
        assert diagnostic["integral_execution"] == "cuda"
        assert diagnostic["fock_execution"] == "cuda"
        assert diagnostic["eigen_execution"] == "cuda"
        assert diagnostic["response_residual"] < 1e-9
        # Ordered, nonsymmetric source seeds expose missing task multiplicities.
        rng = np.random.default_rng(98)
        seeds = IntegralValues(
            rng.normal(size=(2, 2)),
            rng.normal(size=(2, 2)),
            rng.normal(size=(2,) * 4),
            0.7,
        )
        np.testing.assert_allclose(
            _flat(gpu.integral_vjp(XYZ, seeds)),
            _flat(cpu.integral_vjp(XYZ, seeds)),
            atol=3e-10,
        )
        for name in ("overlap", "hcore", "eri", "nuclear"):
            np.testing.assert_allclose(
                getattr(gpu.integral_jvp(XYZ, direction), name),
                getattr(cpu.integral_jvp(XYZ, direction), name),
                atol=3e-10,
            )
    finally:
        gpu.close()
        cpu.close()


def test_cuda_second_order_p_shell_oracle(tmp_path: Path) -> None:
    from test_integral_autodiff_second import _water_hvp

    _water_hvp("cuda", tmp_path)


def test_cuda_one_electron_empty_response_space(tmp_path: Path) -> None:
    from test_integral_autodiff_second import _one_electron_response

    _one_electron_response("cuda", tmp_path)


def test_cuda_first_order_p_shell_gradient(tmp_path: Path) -> None:
    """Qualify the default derivative order with its compact device tape."""
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 1.4, 1.1], [0.0, -1.4, 1.1]])
    owner = IntegralHF(("O", "H", "H"), backend="cuda", cache=tmp_path)
    try:
        energy, gradient = owner.value_and_grad(xyz)
        oracle = _oracle(owner, xyz, owner.exponents, owner.coefficients)
        assert energy == pytest.approx(oracle.e_tot, abs=2e-9)
        np.testing.assert_allclose(
            gradient.coordinates, oracle.nuc_grad_method().kernel(), atol=2e-8
        )
    finally:
        owner.close()


def test_cuda_array_api_joint_hvp(tmp_path: Path) -> None:
    """The public GPU forward-over-reverse transform includes outer basis motion."""
    from generativeqc.experimental import array_api as xp

    def build_loss(model: Any) -> Any:
        def loss(r: Any, log_alpha: Any, c: Any) -> Any:
            energy = model(1.03 * r, exponents=xp.exp(log_alpha), coefficients=c)
            return xp.square(energy) + 0.2 * xp.sum(r * r)

        return loss

    owner = IntegralHF(("H", "H"), cache=tmp_path)
    try:
        primals = (XYZ.copy(), np.log(owner.exponents), owner.coefficients.copy())
    finally:
        owner.close()
    results = []
    for backend in ("cpu", "cuda"):
        model = xp.MolecularModel(
            ("H", "H"),
            method="rhf",
            differentiation="integral-ad",
            backend=backend,
        )

        rng = np.random.default_rng(211)
        tangents = tuple(0.1 * rng.normal(size=v.shape) for v in primals)
        derivative = xp.grad(
            build_loss(model), argnums=(0, 1, 2), backend=backend, cache=tmp_path
        )
        try:
            results.append(
                xp.jvp(derivative, primals, tangents, backend=backend, cache=tmp_path)
            )
        finally:
            derivative.close()
    for cpu, gpu in zip(results[0], results[1], strict=True):
        for expected, actual in zip(cpu, gpu, strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
