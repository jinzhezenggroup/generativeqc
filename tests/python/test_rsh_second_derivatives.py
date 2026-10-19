"""Range-separated generated Hessians against interval-quadrature gradient oracles."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.integral.second_derivatives import build_eri_second_ir
from generativeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    SecondPrimitive,
    compile_second_derivative,
)
from generativeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)

from tools.generativeqc_validation.weighted_eri import primitive_variables

CENTERS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)
EXPONENTS = (0.6, 0.8, 1.1, 0.9)


def _gradient(kernel: object, radial: CoulombKernel, centers: np.ndarray) -> np.ndarray:
    order = kernel.integral.maximum_coulomb_order
    variables = primitive_variables(EXPONENTS, centers, order)
    first_pair, second_pair = sum(EXPONENTS[:2]), sum(EXPONENTS[2:])
    rho = first_pair * second_pair / (first_pair + second_pair)
    difference = np.array([variables[f"difference_{axis}"] for axis in "xyz"])
    argument = rho * np.dot(difference, difference)
    boundary = radial.omega / np.hypot(radial.omega, np.sqrt(rho))
    lower, upper = (
        (0, boundary) if radial.family.value == "long_range" else (boundary, 1)
    )
    nodes, weights = np.polynomial.legendre.leggauss(96)
    points = lower + (upper - lower) * (nodes + 1) / 2
    for index in range(order + 1):
        variables[f"boys_{index}"] = (
            np.dot(weights, points ** (2 * index) * np.exp(-argument * points**2))
            * (upper - lower)
            / 2
        )
    variables["component_weight_0"] = 0.73
    return np.array(
        [
            [kernel.graph.evaluate(root, variables) for root in row]
            for row in kernel.gradients
        ]
    )


@pytest.mark.parametrize("family", ("short_range", "long_range"))
@pytest.mark.parametrize("angular", ((0, 0, 0, 0), (1, 0, 1, 0), (3, 0, 0, 0)))
def test_compiled_rsh_hvp_matches_quadrature_first_gradient_differences(
    family: str,
    angular: tuple,
    tmp_path: Path,
) -> None:
    radial = CoulombKernel(family, 0.3)
    operator = four_center_eri_operator(radial)
    first = build_weighted_eri_kernel(
        build_weighted_eri_ir(angular, operator=operator), (0,)
    )
    second = build_eri_second_ir(angular, operator=operator, output="weighted_hvp")
    direction = np.random.default_rng(1888).normal(size=(4, 3)) * 0.1
    artifact = compile_second_derivative(
        second,
        CppCompilerAdapter(Path(shutil.which("c++") or "c++")),
        tmp_path,
        component_indices=(0,),
        output_indices=tuple(range(12)),
    )
    with PreparedSecondDerivative(artifact, record_capacity=1, tile_capacity=1) as plan:
        primitive = SecondPrimitive(
            EXPONENTS, tuple(map(tuple, CENTERS)), (0.73,), tuple(map(tuple, direction))
        )
        actual = plan.contract((primitive,), tile_count=1).values[0].reshape(4, 3)
    for step in (1e-3, 3e-4, 1e-4):
        finite = (
            _gradient(first, radial, CENTERS + step * direction)
            - _gradient(first, radial, CENTERS - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-12, rtol=0)
