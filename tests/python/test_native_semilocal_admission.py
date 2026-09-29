"""Preserve native KS admission while deriving curated families from metadata."""

from fractions import Fraction

import pytest
from generativeqc.ks import (
    KsOptions,
    _native_semilocal_code,
    ks_coefficients,
    ks_range_exchange_parameters,
    resolve_ks_options,
)
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.method import (
    MethodSpec,
    original_nonlocal_correlation,
    resolve_method,
)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ((("GGA_X_PBE", Fraction(1)),), (1.0, 0.0, 0.0)),
        ((("GGA_C_PBE", Fraction(1)),), (0.0, 1.0, 0.0)),
        (
            (
                ("GGA_X_PBE", Fraction(1)),
                ("GGA_X_PBE", Fraction(-1)),
                ("GGA_C_PBE", Fraction(1)),
            ),
            (0.0, 1.0, 0.0),
        ),
    ],
)
def test_public_pbe_options_preserve_missing_or_cancelled_native_scales(
    spin: str,
    components: tuple[tuple[str, Fraction], ...],
    expected: tuple[float, float, float],
) -> None:
    graph = resolve_method(MethodSpec("scaled-fragment", components), spin=spin)
    selector = "pbe-uks" if spin == "polarized" else "pbe-rks"
    options = resolve_ks_options(
        selector, KsOptions(composition=graph, grid=GridSpec())
    )
    assert options.method_ir.identity == graph.identity
    assert options.coefficients == expected
    assert _native_semilocal_code(graph) == 1


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_pbe_point_program_does_not_own_exchange_omega(spin: str) -> None:
    graph = resolve_method(
        MethodSpec(
            "independent-range-exchange",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
            short_range_exchange=Fraction(1, 5),
            long_range_exchange=Fraction(4, 5),
            range_omega=Fraction(3, 10),
        ),
        spin=spin,
    )
    selector = "pbe-uks" if spin == "polarized" else "pbe-rks"
    options = resolve_ks_options(
        selector, KsOptions(composition=graph, grid=GridSpec())
    )
    assert _native_semilocal_code(options.method_ir) == 1
    assert options.coefficients[:2] == (1.0, 1.0)
    assert ks_range_exchange_parameters(options.method_ir) == (0.2, 0.8, 0.3)


@pytest.mark.parametrize(
    "components",
    [
        (("LDA_X", Fraction(1)), ("LDA_C_PW", Fraction(1))),
        (("MGGA_X_R2SCAN", Fraction(1)), ("MGGA_C_R2SCAN", Fraction(1))),
    ],
)
def test_pure_semilocal_family_does_not_admit_new_nonlocal_contributions(
    components: tuple[tuple[str, Fraction], ...],
) -> None:
    graph = resolve_method(
        MethodSpec(
            "unsupported-nonlocal-composition",
            components,
            nonlocal_correlation=original_nonlocal_correlation("vv10"),
        )
    )
    with pytest.raises(NotImplementedError, match="unsupported native KS"):
        ks_coefficients(graph)
