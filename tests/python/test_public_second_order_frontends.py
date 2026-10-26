"""Qualified native molecular HVPs compose through both public AD frontends."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, KsOptions
from generativeqc.experimental import array_api as xp
from test_hessian_public_rks import GRID
from test_integral_autodiff_second import require_native_library  # noqa: F401

XYZ = np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]])
VECTOR = np.array([[0.03, -0.02, -0.19], [-0.03, 0.02, 0.19]])


def expected_product() -> np.ndarray:
    """Independently compose the two outer-chain Hessian terms and regularizer."""
    calculator = Calculator(
        method="lda-rks",
        basis="sto-3g",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    atoms = list(zip((1, 1), XYZ, strict=True))
    first = calculator.singlepoint(atoms, properties=("energy", "forces"))
    second = calculator.hessian_vector_product(atoms, VECTOR)
    gradient = -first.forces
    return (
        2 * first.energy * second.value
        + 2 * gradient * np.sum(gradient * VECTOR)
        + 0.4 * VECTOR
    )


def test_array_native_second_order_composition(tmp_path: Path) -> None:
    model = xp.MolecularModel(
        ("H", "H"),
        method="lda-rks",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    derivative = xp.grad(
        lambda r: xp.square(model(r)) + 0.2 * xp.sum(r * r), cache=tmp_path
    )
    try:
        _, actual = xp.jvp(derivative, (XYZ,), (VECTOR,), cache=tmp_path)
        np.testing.assert_allclose(actual, expected_product(), atol=3e-8)
    finally:
        derivative.close()


def test_torch_native_second_order_composition() -> None:
    torch = pytest.importorskip("torch")
    from generativeqc.torch import energy

    calculator = Calculator(
        method="lda-rks",
        basis="sto-3g",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    xyz = torch.tensor(XYZ, requires_grad=True)
    loss = energy(xyz, (1, 1), calculator).square() + 0.2 * xyz.square().sum()
    (gradient,) = torch.autograd.grad(loss, xyz, create_graph=True)
    (product,) = torch.autograd.grad(gradient, xyz, grad_outputs=torch.tensor(VECTOR))
    np.testing.assert_allclose(product.detach().numpy(), expected_product(), atol=3e-8)
