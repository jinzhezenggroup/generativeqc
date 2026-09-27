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


def test_blacklisted_automatic_libxc_is_rejected_by_default_dispatch() -> None:
    spec = functional("GGA_C_AM05", spin="polarized")

    with pytest.raises(UnsupportedXC, match="blacklisted"):
        build_energy_expression(spec)


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


def test_explicit_pointwise_path_can_retest_blacklisted_functional() -> None:
    spec = functional("GGA_C_AM05", spin="polarized")
    graph, roots, identity = build_roots(
        spec,
        ((),),
        pointwise_bulk=True,
    )

    assert graph is not None
    assert len(roots) == 1
    assert isinstance(identity, str) and len(identity) == 64
