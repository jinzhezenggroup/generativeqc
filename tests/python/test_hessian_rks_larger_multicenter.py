"""Nonzero multicenter qualification beyond the historical 12-AO RKS gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from generativeqc._stationary_cpu import complete_rks_gradient_diagnostic
from generativeqc_compiler.dft import NativeAO

from tools.generativeqc_hessian import rks_hvp
from tools.generativeqc_response import GMRESOptions, NativeRKSResponse

if TYPE_CHECKING:
    from pathlib import Path

ATOMS = [("He", (0.13, -0.21, -0.8)), ("He", (-0.09, 0.17, 0.86))]
# Seven Cartesian AOs on each center; s/p shells keep this regression smaller
# than a large high-angular-momentum provider compilation campaign.
BASIS = tuple(
    Shell(center, angular, (Primitive(exponent, 1.0),))
    for center in (0, 1)
    for angular, exponent in ((0, 1.5), (1, 0.8), (1, 0.35))
)
GRID = GridSpec(radial_points=10, angular_polar=4, angular_azimuth=8)


def _calculator() -> Calculator:
    return Calculator(
        method="lda-rks",
        basis=BASIS,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


def test_larger_multicenter_rks_hvp_matches_nonzero_gradient_difference(
    tmp_path: Path,
) -> None:
    direction = np.array([[0.17, -0.09, 0.31], [-0.17, 0.09, -0.31]])
    direction /= np.linalg.norm(direction)
    with (
        _calculator().prepare_batch([ATOMS]) as batch,
        NativeAO(ATOMS, basis=BASIS) as basis,
    ):
        assert basis.nao == 14
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as operator:
            result = rks_hvp(
                operator,
                direction,
                cache=tmp_path / "hvp",
                integral_budget_bytes=64 << 20,
                solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            )

    assert result.directional_response.response.solve_result.converged
    assert result.diagnostics["complete_source_coverage"]
    assert tuple(result.components) == (
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    np.testing.assert_allclose(
        sum(result.components.values(), start=np.zeros_like(result.value)),
        result.value,
        atol=2e-13,
        rtol=0,
    )

    errors = []
    # Two separated central-difference steps are sufficient to verify both
    # the nonzero oracle and second-order convergence without paying for a
    # third pair of independently reconverged SCF/gradient calculations.
    for step in (1.2e-3, 1.3e-4):
        gradients = []
        for sign in (1, -1):
            atoms = [
                (symbol, np.asarray(position) + sign * step * displacement)
                for (symbol, position), displacement in zip(
                    ATOMS, direction, strict=True
                )
            ]
            # New calculator/state for every displacement: no reuse of the
            # analytic HVP's solved response or displaced densities.
            with (
                _calculator().prepare_batch([atoms]) as batch,
                NativeAO(atoms, basis=BASIS) as basis,
            ):
                batch.execute(strict=True)
                with NativeRKSResponse.from_native(batch, basis) as current:
                    gradients.append(
                        np.array(
                            complete_rks_gradient_diagnostic(
                                current.state,
                                basis,
                                cache=tmp_path / "gradient",
                                execution="reference",
                            ).gradient,
                            copy=True,
                        )
                    )
        numeric = (gradients[0] - gradients[1]) / (2 * step)
        # A zero-only implementation must not pass this nontranslation gate.
        assert np.max(np.abs(numeric)) > 1e-3
        errors.append(float(np.max(np.abs(result.value - numeric))))

    # Match the existing complete small-domain RKS HVP acceptance thresholds.
    assert errors[-1] < 4e-4, errors
    assert errors[-1] < max(0.35 * errors[0], 2e-5), errors
