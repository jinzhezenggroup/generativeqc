"""Public Calculator acceptance for the qualified semilocal RKS Hessian slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    Primitive,
    Shell,
    method_capabilities,
)
from generativeqc.response_solver import GMRESOptions

if TYPE_CHECKING:
    from pathlib import Path

ATOMS = [("H", (0.0, 0.0, -0.72)), ("H", (0.0, 0.0, 0.72))]
BASIS = (
    Shell(0, 0, (Primitive(1.2, 1.0),)),
    Shell(1, 0, (Primitive(1.2, 1.0),)),
)
GRID = GridSpec(radial_points=10, angular_polar=4, angular_azimuth=8)


def _calculator(method: str = "lda-rks", **kwargs: object) -> Calculator:
    return Calculator(
        method=method,
        basis=BASIS,
        device="cpu",
        precision="fp64",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        **kwargs,
    )


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_public_rks_second_order_capability_is_context_qualified(method: str) -> None:
    assert method_capabilities(method).supported_second_order == frozenset()
    calculator = _calculator(method)
    assert calculator.capabilities.supported_second_order == frozenset(
        ("hvp", "hessian")
    )
    assert calculator.second_order_capabilities == frozenset(("hvp", "hessian"))


@pytest.mark.parametrize(
    ("method", "kwargs"),
    (
        ("lda-uks", {}),
        ("pbe0-rks", {}),
        ("b3lyp-rks", {}),
        ("r2scan-rks", {}),
        ("pbe-rks", {"basis_representation": "spherical"}),
    ),
)
def test_public_rks_second_order_capability_fails_closed(
    method: str, kwargs: dict[str, object]
) -> None:
    calculator = _calculator(method, **kwargs)
    assert calculator.capabilities.supported_second_order == frozenset()
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(ATOMS)


def test_public_rks_hvp_and_hessian_share_the_installed_owner(tmp_path: Path) -> None:
    calculator = _calculator()
    direction = np.array([[0.03, -0.02, -0.19], [-0.03, 0.02, 0.19]])
    cache = tmp_path / "public-second-order"
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)

    hvp = calculator.hessian_vector_product(
        ATOMS,
        direction,
        cache=cache,
        integral_budget_bytes=64 << 20,
        solver_options=solver,
    )
    hessian = calculator.hessian(
        ATOMS,
        block_size=3,
        cache=cache,
        strategy="recycled",
        output_budget_bytes=64 << 20,
        integral_budget_bytes=64 << 20,
        solver_options=solver,
    )

    assert hvp.diagnostics["public_calculator_endpoint"]
    assert hessian.diagnostics["public_calculator_endpoint"]
    assert not hessian.diagnostics["posthoc_symmetrization"]
    np.testing.assert_allclose(
        hessian.matrix @ direction.reshape(-1),
        hvp.value.reshape(-1),
        atol=2e-8,
        rtol=2e-8,
    )


def test_public_hessian_rejects_output_budget_before_scf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = _calculator()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "SCF preparation must not run after output preflight failure"
        )

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    with pytest.raises(ValueError, match="output_budget_bytes"):
        calculator.hessian(ATOMS, output_budget_bytes=1)


def test_public_second_order_rejects_open_shell_before_scf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = _calculator()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("SCF preparation must not run for open-shell requests")

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    with pytest.raises(NotImplementedError, match="multiplicity=1"):
        calculator.hessian_vector_product(
            ATOMS,
            np.zeros((2, 3)),
            multiplicity=3,
        )
