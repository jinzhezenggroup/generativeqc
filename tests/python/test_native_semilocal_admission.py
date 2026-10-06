"""Preserve native KS admission while deriving curated families from metadata."""

from fractions import Fraction
from pathlib import Path

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


def test_cuda_rsh_admission_is_resolved_provider_capability() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src" / "methods" / "dft_method.cpp"
    ).read_text()
    options_begin = source.index("scf::ScfOptions dft_options(")
    options_end = source.index("/** Copy every pointee", options_begin)
    options = source[options_begin:options_end]
    assert (
        "CUDA range-separated KS is qualified only for complete WB97M-V" not in options
    )

    scaled_begin = options.index("if (scaled_or_hybrid")
    scaled_end = options.index("if (execution_plan.nonlocal_correlation", scaled_begin)
    scaled_gate = options[scaled_begin:scaled_end]
    assert "!execution_plan.range_exchange" in scaled_gate

    helper_begin = source.index("bool cuda_rsh_provider_compatible(")
    helper_end = source.index("#endif", helper_begin)
    helper = source[helper_begin:helper_end]
    for fact in (
        "provider.strategy()",
        "prepared_cuda_fock_binding(provider)",
        "FockBackend::Cuda",
        "FockApproximation::Exact",
        "FockOperator::FullRange",
        "FockOperator::LongRange",
        "correction_spec.spin == primary_spec.spin",
        "correction.screening_tolerance == primary.screening_tolerance",
    ):
        assert fact in helper
    assert "Wb97mv" not in helper
    assert "descriptor.method" not in helper

    constructor_begin = source.index("KsPreparedCalculation(")
    constructor_end = source.index("std::size_t atom_count()", constructor_begin)
    constructor = source[constructor_begin:constructor_end]
    assert "cuda_rsh_provider_compatible(fock_, *range_strategy_)" in constructor


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("lda-rks", True),
        ("pbe-rks", True),
        ("lda-uks", False),
        ("pbe-uks", False),
        ("pbe0-rks", False),
        ("r2scan-rks", False),
    ],
)
def test_stationary_second_order_preserves_spin_and_method_domain(
    selector: str, expected: bool
) -> None:
    from generativeqc.ks import stationary_second_order_eligible

    options = resolve_ks_options(selector, KsOptions(grid=GridSpec()))
    assert stationary_second_order_eligible(options.method_ir) is expected


def test_stationary_second_order_rejects_scaled_pbe() -> None:
    from generativeqc.ks import stationary_second_order_eligible

    graph = resolve_method(
        MethodSpec(
            "scaled-second-order",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
        ),
        spin="unpolarized",
    )
    assert not stationary_second_order_eligible(graph)


def test_cpu_stationary_force_preserves_intrinsic_d4_admission() -> None:
    from generativeqc.ks import cpu_stationary_all_electron_force_eligible

    options = resolve_ks_options("pbe-d4-rks", KsOptions(grid=GridSpec()))
    assert cpu_stationary_all_electron_force_eligible(options.method_ir)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_native_device_xc_selector_preserves_both_spin_flows(spin: str) -> None:
    from types import SimpleNamespace

    from generativeqc_compiler.dft.native_semilocal import (
        device_feature_ingredients,
        legacy_grid_xc_selector,
    )
    from generativeqc_compiler.xc.prepared import _native_device_xc
    from generativeqc_compiler.xc.spec import functional

    spec = functional("PBE", spin=spin)
    assert device_feature_ingredients(spec) == ("rho", "gradient")
    assert legacy_grid_xc_selector(spec) == 1
    program = SimpleNamespace(
        spec=spec,
        contract=SimpleNamespace(request=SimpleNamespace(observable="potential")),
    )
    assert _native_device_xc(program, object(), object())
    opposite = "polarized" if spin == "unpolarized" else "unpolarized"
    with pytest.raises(ValueError, match="spin"):
        legacy_grid_xc_selector(spec, spin=opposite)


def test_native_device_xc_selector_falls_back_for_unsupported_graph() -> None:
    from types import SimpleNamespace

    from generativeqc_compiler.xc.prepared import _native_device_xc
    from generativeqc_compiler.xc.spec import functional

    program = SimpleNamespace(
        spec=functional("R2SCAN"),
        contract=SimpleNamespace(request=SimpleNamespace(observable="potential")),
    )
    assert not _native_device_xc(program, object(), object())
