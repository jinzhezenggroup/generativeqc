"""Complete WB97M-V compiler-AD Hessian gates; finite differences are oracles only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, method_capabilities
from generativeqc.response_solver import GMRESOptions

if TYPE_CHECKING:
    from pathlib import Path

H2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
WATER = [
    ("O", (0.02, -0.03, 0.04)),
    ("H", (0.10, 1.43, 1.10)),
    ("H", (-0.15, -1.45, 1.12)),
]
GRID = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


def _calculator(**kwargs: object) -> Calculator:
    return Calculator(
        method=kwargs.pop("method", "wb97m-v"),
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        **kwargs,
    )


def _force_difference(
    calculator: Calculator, atoms: list, direction: np.ndarray, step: float
) -> np.ndarray:
    forces = []
    for sign in (-1, 1):
        displaced = [
            (element, np.asarray(position) + sign * step * motion)
            for (element, position), motion in zip(atoms, direction, strict=True)
        ]
        result = calculator.singlepoint(displaced, properties=("energy", "forces"))
        assert result.converged
        forces.append(result.forces)
    return -(forces[1] - forces[0]) / (2 * step)


@pytest.mark.parametrize("method", ("wb97m-v", "wb97m-v-rks"))
def test_public_wb97mv_second_order_admission(method: str) -> None:
    assert not method_capabilities(method).supported_second_order
    assert _calculator(method=method).second_order_capabilities == frozenset(
        ("hvp", "hessian")
    )


@pytest.mark.parametrize("kwargs", ({"method": "pbe0-rks"},))
def test_unqualified_wb97mv_second_order_contexts_fail_before_scf(
    kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = _calculator(**kwargs)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported Hessian must not prepare SCF")

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    assert not calculator.second_order_capabilities
    with pytest.raises(NotImplementedError, match="does not expose public hessian"):
        calculator.hessian(H2)


def test_raw_wb97mv_hessian_uses_one_scf_and_matches_all_gradient_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = _calculator()
    prepare = calculator.prepare_batch
    preparations = []

    def counted(*args: object, **kwargs: object) -> object:
        preparations.append(args)
        return prepare(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(calculator, "prepare_batch", counted)
        result = calculator.hessian(H2, cache=tmp_path, solver_options=SOLVER)
    assert len(preparations) == 1
    assert (
        result.diagnostics["derivative_method"] == "compiler-automatic-differentiation"
    )
    assert result.diagnostics["displaced_scf_evaluations"] == 0
    assert result.diagnostics["nuclear_response_solves"] == 6
    assert not result.diagnostics["posthoc_symmetrization"]
    assert not result.matrix.flags.writeable
    np.testing.assert_allclose(result.matrix, result.matrix.T, atol=2e-7, rtol=0)
    translations = np.tile(np.eye(3), (2, 1))
    np.testing.assert_allclose(result.matrix @ translations, 0, atol=2e-8, rtol=0)
    for coordinate in range(6):
        direction = np.zeros((2, 3))
        direction.reshape(-1)[coordinate] = 1
        for step in (1e-3, 3e-4, 1e-4):
            finite = _force_difference(calculator, H2, direction, step)
            np.testing.assert_allclose(
                result.matrix[:, coordinate],
                finite.reshape(-1),
                atol=3e-6 if step == 1e-3 else 4e-7,
                rtol=2e-6,
            )


def test_asymmetric_sp_water_hvp_includes_relaxed_exchange_tau_and_vv10(
    tmp_path: Path,
) -> None:
    calculator = _calculator()
    direction = np.array(
        [[0.02, -0.05, 0.03], [-0.04, 0.09, 0.02], [0.02, -0.04, -0.05]]
    )
    result = calculator.hessian_vector_product(
        WATER, direction, cache=tmp_path, solver_options=SOLVER
    )
    expected = {
        "one_electron",
        "coulomb",
        "exchange_short_range",
        "exchange_long_range",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "nonlocal_ao",
        "nonlocal_grid",
        "nonlocal_weight",
        "overlap_pulay",
        "nuclear",
    }
    assert set(result.components) == expected
    assert result.diagnostics["response_residual_norm"] < 2e-10
    np.testing.assert_allclose(
        result.value, sum(result.components.values()), atol=1e-13
    )
    np.testing.assert_allclose(result.value.sum(axis=0), 0, atol=2e-8, rtol=0)
    for step in (3e-3, 1e-3, 3e-4):
        finite = _force_difference(calculator, WATER, direction, step)
        np.testing.assert_allclose(result.value, finite, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "options",
    (
        {"output_budget_bytes": 1},
        {"grid_budget_bytes": 0},
        {"max_nonlocal_pairs": 0},
        {"block_size": 2},
        {"strategy": "unknown"},
    ),
)
def test_wb97mv_hessian_preflight_options(
    options: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calculator = _calculator()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid Hessian request must not prepare SCF")

    monkeypatch.setattr(calculator, "prepare_batch", forbidden)
    with pytest.raises(ValueError):
        calculator.hessian(H2, **options)
