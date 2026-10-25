"""Second-order frontend acceptance against independent algebra and HF oracles."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generativeqc.torch import batched_energy, energy, integral_energy


class QuadraticResponse:
    """An exact E=sum(R**2) oracle exposing the public native HVP contract."""

    second_order_capabilities = frozenset(("hvp", "hessian"))
    scale = 1.0

    def _model_signature(self) -> float:
        return self.scale

    def singlepoint(self, atoms: Any, **kwargs: Any) -> Any:
        xyz = np.array([position for _, position in atoms])
        return SimpleNamespace(
            energy=self.scale * float(np.sum(xyz**2)), forces=-2 * self.scale * xyz
        )

    def hessian_vector_product(self, atoms: Any, direction: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(value=2 * self.scale * np.array(direction))

    def batch_singlepoint(self, systems: Any, **kwargs: Any) -> Any:
        items = tuple(self.singlepoint(atoms) for atoms in systems)
        return SimpleNamespace(
            items=items, energies=np.array([item.energy for item in items])
        )


@pytest.mark.parametrize("batch", [False, True])
def test_qualified_native_double_backward_composes_outer_loss(batch: bool) -> None:
    calculator = QuadraticResponse()
    xyz = torch.tensor([[0.7, -0.3, 0.2]], dtype=torch.float64, requires_grad=True)

    def loss(r: Any) -> Any:
        value = (
            batched_energy([r], [[2]], calculator)[0]
            if batch
            else energy(r, [2], calculator)
        )
        return value.square() + 0.3 * r.square().sum()

    actual = torch.autograd.functional.hessian(loss, xyz)
    expected = torch.autograd.functional.hessian(
        lambda r: r.square().sum().square() + 0.3 * r.square().sum(), xyz
    )
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradgradcheck(loss, (xyz,))


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("before_first_backward", [False, True])
def test_model_changes_reject_before_using_a_different_response(
    batch: bool, before_first_backward: bool
) -> None:
    calculator = QuadraticResponse()
    xyz = torch.tensor([[0.7, -0.3, 0.2]], dtype=torch.float64, requires_grad=True)
    value = (
        batched_energy([xyz], [[2]], calculator)[0]
        if batch
        else energy(xyz, [2], calculator)
    )
    if before_first_backward:
        calculator.scale = 3.0
        with pytest.raises(RuntimeError, match="calculator model changed"):
            torch.autograd.grad(value, xyz, create_graph=True)
    else:
        (gradient,) = torch.autograd.grad(value, xyz, create_graph=True)
        calculator.scale = 3.0
        with pytest.raises(RuntimeError, match="calculator model changed"):
            torch.autograd.grad(gradient.sum(), xyz)


@pytest.mark.parametrize("batch", [False, True])
def test_saved_coordinate_mutation_rejects_double_backward(batch: bool) -> None:
    calculator = QuadraticResponse()
    xyz = torch.tensor([[0.7, -0.3, 0.2]], dtype=torch.float64, requires_grad=True)
    value = (
        batched_energy([xyz], [[2]], calculator)[0]
        if batch
        else energy(xyz, [2], calculator)
    )
    (gradient,) = torch.autograd.grad(value, xyz, create_graph=True)
    with torch.no_grad():
        xyz.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(gradient.sum(), xyz)


def test_joint_integral_double_backward_matches_independent_energy(
    tmp_path: Path,
) -> None:
    from generativeqc import _native
    from generativeqc.experimental.integral_autodiff import IntegralHF
    from test_integral_autodiff_second import BASIS, XYZ, _oracle

    # Keep the algebra-only gates runnable from a source checkout. An explicitly
    # configured acceptance library must still fail if the build is missing.
    if not any(candidate.exists() for candidate in _native._candidate_paths()):
        if os.environ.get("GENERATIVEQC_LIBRARY"):
            pytest.fail("configured GENERATIVEQC_LIBRARY does not exist")
        pytest.skip("integral HF response acceptance requires a native library")
    owner = IntegralHF(("H", "H"), basis=BASIS, derivative_order=2, cache=tmp_path)
    xyz = torch.tensor(XYZ, dtype=torch.float64, requires_grad=True)
    log_alpha = torch.tensor(
        np.log(owner.exponents), dtype=torch.float64, requires_grad=True
    )
    coefficients = torch.tensor(
        owner.coefficients.copy(), dtype=torch.float64, requires_grad=True
    )
    inputs = (xyz, log_alpha, coefficients)
    rng = np.random.default_rng(551)
    vectors = tuple(
        torch.tensor(rng.normal(size=tuple(value.shape)) * 0.1, dtype=torch.float64)
        for value in inputs
    )
    try:
        value = integral_energy(1.03 * xyz, log_alpha.exp(), coefficients, owner)
        # The graph owns the converged state independently of the public owner.
        owner.close()
        loss = value.square() + 0.2 * xyz.square().sum()
        gradient = torch.autograd.grad(loss, inputs, create_graph=True)
        product = torch.autograd.grad(gradient, inputs, grad_outputs=vectors)
        actual = sum(
            float((v * h).sum()) for v, h in zip(vectors, product, strict=True)
        )

        def oracle(step: float) -> float:
            r, a, c = tuple(
                value.detach().numpy() + step * vector.numpy()
                for value, vector in zip(inputs, vectors, strict=True)
            )
            e = _oracle(owner, 1.03 * r, np.exp(a), c).e_tot
            return e**2 + 0.2 * float(np.sum(r**2))

        step = 0.001
        expected = (oracle(step) - 2 * oracle(0) + oracle(-step)) / step**2
        assert actual == pytest.approx(expected, abs=3e-7)
    finally:
        owner.close()
