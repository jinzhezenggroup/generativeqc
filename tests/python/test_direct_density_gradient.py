"""Bounded mathematical domains for generated density/Hermite projection."""

from itertools import product

import pytest
from generativeqc_compiler.integral.direct_density_gradient_cuda import (
    density_hermite_states,
)


@pytest.mark.parametrize("maximum", range(5))
def test_density_projection_covers_complete_polynomial_space(maximum: int) -> None:
    """Every supported monomial appears once, including the scalar response."""
    states = density_hermite_states(maximum)
    expected = {
        powers
        for powers in product(range(maximum + 1), repeat=3)
        if sum(powers) <= maximum
    }
    assert set(states) == expected
    assert len(states) == len(expected)
    assert all(sum(state) <= maximum for state in states)


@pytest.mark.parametrize("maximum", (-1, 5, 9))
def test_density_projection_rejects_unbounded_pairs(maximum: int) -> None:
    """A larger shell pair needs an explicitly proved workspace contract."""
    with pytest.raises(ValueError, match="s/p/d"):
        density_hermite_states(maximum)


@pytest.mark.parametrize("maximum", (True, False, 2.0, "2", None))
def test_density_projection_rejects_noninteger_domains(maximum: object) -> None:
    """Boolean or coerced domains cannot silently select a resource frame."""
    with pytest.raises(TypeError, match="integer"):
        density_hermite_states(maximum)
