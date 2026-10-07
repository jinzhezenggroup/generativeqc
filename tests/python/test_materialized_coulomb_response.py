"""Check scalar spatial response against the independent Coulomb IR AD owner."""

import pytest
from generativeqc_compiler.integral.shell_class import (
    build_coulomb_derivative_algebra,
)


@pytest.mark.parametrize("axis", "xyz")
def test_raised_coulomb_state_is_the_ir_first_response(axis: str) -> None:
    """All dddd states share the L+1 prefix under the existing Boys leaf rule."""
    algebra = build_coulomb_derivative_algebra(9)
    graph, roots = algebra.graph, dict(algebra.roots)
    rho = graph.variable("rho")
    coordinate = graph.variable(f"difference_{axis}")
    leaves = {
        f"boys_{n}": -2 * rho * coordinate * graph.variable(f"boys_{n + 1}")
        for n in range(9)
    }
    # Arbitrary independent Boys leaves make this an algebra gate rather than
    # relying on a second special-function approximation or a physical point.
    values = {
        "rho": 0.37,
        "difference_x": 0.3,
        "difference_y": -0.8,
        "difference_z": 0.6,
    }
    values.update({f"boys_{n}": 0.29 / (n + 1) + (-0.07) ** n for n in range(10)})
    axis_index = "xyz".index(axis)
    for orders, root in roots.items():
        if sum(orders) > 8:
            continue
        raised = list(orders)
        raised[axis_index] += 1
        response = graph.differentiate(root, coordinate, leaves)
        assert graph.evaluate(response, values) == pytest.approx(
            graph.evaluate(roots[tuple(raised)], values), abs=2e-12, rel=2e-12
        ), (axis, orders)
