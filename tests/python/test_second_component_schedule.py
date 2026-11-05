"""Bounded second-source partitions and independent complete d/f contractions."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.integral.second_derivatives import (
    build_one_electron_second_ir,
)
from generativeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    SecondPrimitive,
    compile_second_derivative,
)
from generativeqc_compiler.integral.second_order_layout import (
    second_component_tiles,
    second_coordinate_tiles,
)

from tools.generativeqc_validation.second_derivatives import (
    libcint_one_electron_hessian,
)


@pytest.mark.parametrize("count", (1, 9, 60, 10000))
@pytest.mark.parametrize("tile_size", (1, 8, 64))
def test_component_schedule_is_a_bounded_disjoint_cover(
    count: int, tile_size: int
) -> None:
    tiles = second_component_tiles(count, tile_size=tile_size)
    flattened = []
    for tile in tiles:
        assert 1 <= len(tile) <= tile_size
        flattened.extend(tile)
    assert flattened == list(range(count))


@pytest.mark.parametrize(
    ("count", "tile_size", "error"),
    (
        (0, 8, ValueError),
        (-1, 8, ValueError),
        (True, 8, TypeError),
        (1, 0, ValueError),
        (1, 65, ValueError),
        (1, False, TypeError),
        (1.5, 8, TypeError),
        (2**63, 8, OverflowError),
    ),
)
def test_invalid_component_schedule_fails_before_iteration(
    count: object, tile_size: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        second_component_tiles(count, tile_size=tile_size)


@pytest.mark.parametrize("family", ("overlap", "kinetic", "nuclear_attraction"))
def test_full_df_weighted_hvp_matches_independent_analytic_second_derivatives(
    family: str, tmp_path: Path
) -> None:
    """Sum every component/output tile, not just a favorable d/f component.

    Signed asymmetric weights stay fixed. Two primitive records and independent
    output tiling exercise additive contraction, while a common translation
    checks mathematical-center recovery. This primitive gate is not molecular
    endpoint qualification or a compiler wall-clock assertion.
    """
    pytest.importorskip("pyscf")
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("native C++ compiler unavailable")
    ir = build_one_electron_second_ir(family, (2, 3), charge=1.0, output="weighted_hvp")
    compiler = CppCompilerAdapter(Path(executable))
    centers = ((0.13, -0.31, 0.24), (-0.43, 0.27, 0.51), (0.68, -0.14, -0.22))
    center_count = len(ir.operator.centers)
    centers = centers[:center_count]
    dimension = 3 * center_count
    generator = np.random.default_rng(2843)
    weights = generator.normal(size=ir.signature.component_count)
    direction = generator.normal(size=(center_count, 3))
    translation = np.tile((0.17, -0.23, 0.31), (center_count, 1))
    expected_matrix = libcint_one_electron_hessian(
        family, (2, 3), (0.6, 0.8), centers, 1.0
    ).reshape(dimension, dimension, -1)
    expected = np.einsum("ijc,c,j->i", expected_matrix, weights, direction.ravel())
    actual = np.zeros(dimension)
    translated = np.zeros(dimension)
    for components in second_component_tiles(ir.signature.component_count):
        for outputs in second_coordinate_tiles(ir.operator.centers, hvp=True):
            artifact = compile_second_derivative(
                ir,
                compiler,
                tmp_path,
                component_indices=components,
                output_indices=outputs,
            )
            packed_weights = tuple(weights[list(components)])
            primitive = SecondPrimitive(
                (0.6, 0.8), centers, packed_weights, tuple(map(tuple, direction))
            )
            moved = SecondPrimitive(
                (0.6, 0.8), centers, packed_weights, tuple(map(tuple, translation))
            )
            with PreparedSecondDerivative(artifact, record_capacity=2) as plan:
                values = plan.contract((primitive, primitive)).values[0]
                zero = plan.contract((moved,)).values[0]
            actual[list(outputs)] += values
            translated[list(outputs)] += zero
    np.testing.assert_allclose(actual, 2 * expected, atol=2e-10, rtol=5e-11)
    np.testing.assert_allclose(translated, 0, atol=2e-11, rtol=0)
