from __future__ import annotations

from fractions import Fraction

import pytest
from vibeqc_compiler.xc.expression_dispatch import build_energy_expression
from vibeqc_compiler.xc.semilocal_codegen import build_roots
from vibeqc_compiler.xc.spec import FunctionalSpec, UnsupportedXC, functional


def test_automatic_libxc_uses_generic_lowering_without_opt_in() -> None:
    spec = functional("GGA_X_APBE", spin="polarized")
    graph, roots, identity = build_roots(spec, ((), (0,), (1,)))

    assert graph is not None
    assert len(roots) == 3
    assert isinstance(identity, str) and len(identity) == 64


@pytest.mark.parametrize("name", ("GGA_C_AM05", "GGA_X_AK13"))
def test_shared_boundary_cases_still_lower_to_interior_graph(name: str) -> None:
    spec = functional(name, spin="polarized")
    graph, energy, variables = build_energy_expression(spec)

    assert graph is not None
    assert energy is not None
    assert len(variables) == len(spec.features)


def test_automatic_libxc_cannot_mix_with_curated_family() -> None:
    spec = FunctionalSpec(
        "mixed-auto-curated",
        (
            ("GGA_X_APBE", Fraction(1)),
            ("GGA_C_PBE", Fraction(1)),
        ),
        spin="polarized",
    )

    with pytest.raises(UnsupportedXC, match="cannot mix"):
        build_energy_expression(spec)


def test_explicit_pointwise_path_remains_representation_only() -> None:
    spec = functional("GGA_C_AM05", spin="polarized")
    graph, roots, identity = build_roots(
        spec,
        ((),),
        pointwise_bulk=True,
    )

    assert graph is not None
    assert len(roots) == 1
    assert isinstance(identity, str) and len(identity) == 64
