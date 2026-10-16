"""Pinned Libxc hybrid metadata lowers to MethodIR data without C execution."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from generativeqc_compiler.method import METHOD_CATALOG
from generativeqc_compiler.method._generated_libxc_methods import (
    BLOCKED_LIBXC_METHODS,
    LIBXC_METHODS,
)

from tools.libxc_method_metadata import extract_method_registrations

ROOT = Path(__file__).resolve().parents[2]
LIBXC = ROOT / "upstream" / "libxc" / "7.0.0"
FULL_LIBXC = ROOT / "upstream" / "libxc-fulltree" / "7.0.0" / "src"


def _row(filename: str, registration: str) -> dict:
    rows = extract_method_registrations((LIBXC / filename).read_text(), filename)
    return next(row for row in rows if row["registration"] == registration)


def _full_row(filename: str, registration: str) -> dict:
    rows = extract_method_registrations((FULL_LIBXC / filename).read_text(), filename)
    return next(row for row in rows if row["registration"] == registration)


def test_b3lyp_composition_and_exchange_are_source_derived_exactly() -> None:
    row = _row("hyb_gga_xc_b3lyp.c", "HYB_GGA_XC_B3LYP")
    assert row["status"] == "generated"
    assert row["components"] == [
        ["LDA_X", "2/25"],
        ["GGA_X_B88", "18/25"],
        ["LDA_C_VWN_RPA", "19/100"],
        ["GGA_C_LYP", "81/100"],
    ]
    assert row["exact_exchange"] == "1/5"


def test_cam_b3lyp_uses_libxc_cam_convention_for_sr_lr_exchange() -> None:
    row = _row("hyb_gga_xc_cam_b3lyp.c", "HYB_GGA_XC_CAM_B3LYP")
    assert row["status"] == "generated"
    assert row["short_range_exchange"] == "19/100"
    assert row["long_range_exchange"] == "13/20"
    assert row["range_omega"] == "33/100"
    assert row["components"][:2] == [
        ["GGA_X_B88", "7/20"],
        ["GGA_X_ITYH", "23/50"],
    ]


def test_m062x_split_global_hybrid_exchange_is_source_derived() -> None:
    row = _full_row("hyb_mgga_x_m05.c", "HYB_MGGA_X_M06_2X")
    assert row["status"] == "generated"
    assert row["identifier"] == "M06-2X"
    assert row["components"] == [["HYB_MGGA_X_M06_2X", "1"]]
    assert row["split_exchange_component"] == "HYB_MGGA_X_M06_2X"
    assert row["paired_correlation_component"] == "MGGA_C_M06_2X"
    assert row["exact_exchange"] == "27/50"
    assert row["short_range_exchange"] == "0"
    assert row["long_range_exchange"] == "0"


def test_mn15_split_global_hybrid_exchange_is_source_derived() -> None:
    # This owner has registration-dependent initializer control flow. The
    # audited set_ext_params_cpy_exx contract deliberately avoids interpreting
    # that control flow and derives the global exact-exchange fraction from the
    # last external parameter.
    row = _full_row("mgga_x_mn12.c", "HYB_MGGA_X_MN15")
    assert row["status"] == "generated"
    assert row["identifier"] == "MN15"
    assert row["components"] == [["HYB_MGGA_X_MN15", "1"]]
    assert row["paired_correlation_component"] == "MGGA_C_MN15"
    assert row["exact_exchange"] == "11/25"


def test_split_range_hybrid_remains_fail_closed() -> None:
    row = _full_row("mgga_x_mn12.c", "HYB_MGGA_X_MN12_SX")
    assert row["status"] == "blocked"
    assert "global-exchange setter" in row["reason"]


def test_wb97mv_direct_semilocal_and_vv10_metadata_are_source_derived() -> None:
    row = _row("hyb_mgga_xc_wb97mv.c", "HYB_MGGA_XC_WB97M_V")
    assert row["status"] == "generated"
    assert row["direct_semilocal_stem"] == "WB97M_V"
    assert row["short_range_exchange"] == "3/20"
    assert row["long_range_exchange"] == "1"
    assert row["range_omega"] == "3/10"
    assert (row["nonlocal_variant"], row["nlc_b"], row["nlc_c"]) == (
        "vv10",
        "6",
        "1/100",
    )


def test_generated_inventory_drives_existing_method_catalog_entries() -> None:
    for name in (
        "B3P86",
        "B3LYP",
        "B3LYP5",
        "B3PW91",
        "B5050LYP",
        "CAM-B3LYP",
        "CAMH-B3LYP",
        "WB97M-V",
    ):
        assert name in LIBXC_METHODS
        assert name in METHOD_CATALOG
    assert METHOD_CATALOG["B3LYP"].exact_exchange.numerator == 1
    assert METHOD_CATALOG["B3LYP"].exact_exchange.denominator == 5
    assert METHOD_CATALOG["WB97M-V"].nonlocal_correlation is not None


def test_representable_generated_methods_need_no_named_catalog_edit() -> None:
    # These names were not hand-authored MethodSpec entries. Their components are
    # already representable, so pinned Libxc metadata is sufficient for MethodIR.
    for name in ("REVB3LYP", "KMLYP", "QTP17", "RCAM-B3LYP", "TUNED-CAM-B3LYP"):
        assert name in LIBXC_METHODS
        assert name in METHOD_CATALOG


def test_fulltree_nonlocal_methods_use_pinned_source_definitions() -> None:
    # No manually authored B97M-V or WB97X-V exchange/VV10 constants.
    b97mv = extract_method_registrations(
        (FULL_LIBXC / "mgga_xc_b97mv.c").read_text(),
        "mgga_xc_b97mv.c",
        direct_combined=True,
    )
    (pure,) = b97mv
    assert pure["status"] == "generated"
    assert pure["identifier"] == "B97M-V"
    assert pure["components"] == [["MGGA_XC_B97M_V", "1"]]
    assert (pure["nonlocal_variant"], pure["nlc_b"], pure["nlc_c"]) == (
        "vv10",
        "6",
        "1/100",
    )
    assert pure["short_range_exchange"] == pure["long_range_exchange"] == "0"

    wb97 = extract_method_registrations(
        (FULL_LIBXC / "hyb_gga_xc_wb97.c").read_text(),
        "hyb_gga_xc_wb97.c",
        direct_combined=True,
    )
    rows = {row["registration"]: row for row in wb97}
    vv10 = rows["HYB_GGA_XC_WB97X_V"]
    assert vv10["status"] == "generated"
    assert vv10["identifier"] == "WB97X-V"
    assert vv10["direct_semilocal_component"] == "HYB_GGA_XC_WB97X_V"
    assert (vv10["short_range_exchange"], vv10["long_range_exchange"]) == (
        "167/1000",
        "1",
    )
    assert vv10["range_omega"] == "3/10"
    assert (vv10["nonlocal_variant"], vv10["nlc_b"], vv10["nlc_c"]) == (
        "vv10",
        "6",
        "1/100",
    )
    # The shared C initializer enables VV10 only in the -V switch arm.
    for name in (
        "HYB_GGA_XC_WB97",
        "HYB_GGA_XC_WB97X",
        "HYB_GGA_XC_WB97X_D",
        "HYB_GGA_XC_WB97X_D3",
    ):
        assert rows[name]["status"] == "generated"
        assert rows[name]["nonlocal_variant"] is None


def test_nonlocal_libxc_methods_have_representation_only_method_ir() -> None:
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method._generated_libxc_methods import LIBXC_METHODS

    for name in ("B97M-V", "WB97X-V", "GAS22"):
        assert name in LIBXC_METHODS
        spec = METHOD_CATALOG[name]
        assert spec.nonlocal_correlation is not None
        assert spec.nonlocal_correlation.b == Fraction(6)
        assert spec.nonlocal_correlation.c == Fraction(1, 100)
        method = resolve_method(name, spin="unpolarized")
        semilocal = next(
            primitive
            for primitive in method.primitives
            if primitive.kind == "semilocal_xc"
        )
        provenance = semilocal.functional.to_payload()
        assert provenance["qualification"] == "representation-only"
        assert provenance["production_admitted"] is False

    assert METHOD_CATALOG["WB97X-V"].short_range_exchange == Fraction(167, 1000)
    assert METHOD_CATALOG["WB97X-V"].long_range_exchange == 1
    assert METHOD_CATALOG["WB97X-V"].range_omega == Fraction(3, 10)


def test_unmodeled_nonlocal_initializer_control_flow_fails_closed() -> None:
    source = (FULL_LIBXC / "hyb_gga_xc_wb97.c").read_text()
    modified = source.replace(
        "p->nlc_C = 0.01;",
        "p->nlc_C = 0.01; unknown_parameter_mutation(p);",
    )
    rows = extract_method_registrations(
        modified, "hyb_gga_xc_wb97.c", direct_combined=True
    )
    result = next(row for row in rows if row["registration"] == "HYB_GGA_XC_WB97X_V")
    assert result["status"] == "blocked"
    assert "initializer" in result["reason"]


@pytest.mark.parametrize(
    "old, new",
    (
        ("switch(p->info->number)", "switch(p->nspin)"),
        ("case XC_HYB_GGA_XC_WB97X_V:", "default:"),
        ("break;", ""),
        ("p->nlc_b = 6.0;", "if (p->nspin == 1) p->nlc_b = 6.0;"),
        ("p->nlc_b = 6.0;", "return; p->nlc_b = 6.0;"),
        (
            "case XC_HYB_GGA_XC_WB97X_V:",
            "case XC_HYB_GGA_XC_WB97: case XC_HYB_GGA_XC_WB97X_V:",
        ),
        (
            "break;",
            "break; case XC_HYB_GGA_XC_WB97X_V: p->nlc_b = 7.0; break;",
        ),
    ),
)
def test_nonlocal_initializer_rejects_unmodeled_switch_semantics(
    old: str,
    new: str,
) -> None:
    source = (FULL_LIBXC / "hyb_gga_xc_wb97.c").read_text()
    assert old in source
    rows = extract_method_registrations(
        source.replace(old, new), "hyb_gga_xc_wb97.c", direct_combined=True
    )
    result = next(row for row in rows if row["registration"] == "HYB_GGA_XC_WB97X_V")
    assert result["status"] == "blocked"


def test_registration_switch_does_not_admit_setter_control_flow() -> None:
    source = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "p->mix_coef[0] = 1.0 - a0 - ax;"
    new = "switch (p->info->number) { case XC_HYB_GGA_XC_B3LYP: " + old + " break; }"
    assert old in source
    rows = extract_method_registrations(source.replace(old, new))
    result = next(row for row in rows if row["registration"] == "HYB_GGA_XC_B3LYP")
    assert result["status"] == "blocked"
    assert "control flow" in result["reason"]


@pytest.mark.parametrize(
    "name,remove_marker",
    (
        ("src/mgga_xc_b97mv.c", False),
        ("src/mgga_xc_b97mv.c", True),
        ("src/gga_xc_b97.c", False),
    ),
)
def test_nonlocal_discovery_authenticates_before_marker_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    remove_marker: bool,
) -> None:
    import json

    from tools import generate_libxc_methods as generator

    registry = json.loads(generator.REGISTRY.read_text())
    source = registry["sources"][generator.NONLOCAL_SOURCE_ID]
    source["files"] = {name: source["files"][name]}
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    fulltree = tmp_path / "fulltree"
    owner = fulltree / name
    owner.parent.mkdir(parents=True)
    original = (generator.NONLOCAL_SOURCE_ROOT / name).read_bytes()
    altered = (
        original.replace(b"XC_FLAGS_VV10", b"XC_FLAGS_OTHER")
        if remove_marker
        else original + b"\n/* altered source */\n"
    )
    assert altered != original
    owner.write_bytes(altered)
    monkeypatch.setattr(generator, "REGISTRY", registry_path)
    monkeypatch.setattr(generator, "NONLOCAL_SOURCE_ROOT", fulltree)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generator.build_catalog()


def test_complex_dynamic_mix_owners_remain_explicitly_blocked() -> None:
    assert set(BLOCKED_LIBXC_METHODS) == {
        "HYB_GGA_XC_APF",
        "HYB_GGA_XC_WC04",
        "HYB_GGA_XC_WP04",
    }
    assert all("parameter array" in reason for reason in BLOCKED_LIBXC_METHODS.values())


def test_generated_libxc_method_module_is_fresh() -> None:
    from tools.generate_libxc_methods import OUTPUT, build_catalog, render

    generated, blocked = build_catalog()
    assert OUTPUT.read_text(encoding="utf-8") == render(generated, blocked)


@pytest.mark.parametrize(
    "expression", ("0.1234567890123456789012345", "1.234567890123456789e-37", "1e-400")
)
def test_fraction_parser_retains_decimal_source_digits(expression: str) -> None:
    from tools.libxc_method_metadata import _fraction_value

    assert _fraction_value(expression, {}) == Fraction(expression)


@pytest.mark.parametrize(
    "statement",
    (
        "if (p->nspin == 1) p->mix_coef[0] = 1.0 - a0 - ax;",
        "for (int i=0; i<1; ++i) p->mix_coef[0] = 1.0 - a0 - ax;",
        "return; p->mix_coef[0] = 1.0 - a0 - ax;",
        "ax = unknown_scale(ax); p->mix_coef[0] = 1.0 - a0 - ax;",
    ),
)
def test_metadata_rejects_uninterpreted_setter_semantics(statement: str) -> None:
    text = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "p->mix_coef[0] = 1.0 - a0 - ax;"
    assert old in text
    changed = text.replace(old, statement)
    row = next(
        row
        for row in extract_method_registrations(changed)
        if row["registration"] == "HYB_GGA_XC_B3LYP"
    )
    assert row["status"] == "blocked"


def test_metadata_applies_setter_assignments_in_source_order() -> None:
    text = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "p->mix_coef[1] = ax;"
    changed = text.replace(old, old + "\n  ax = 0.5;")
    row = next(
        row
        for row in extract_method_registrations(changed)
        if row["registration"] == "HYB_GGA_XC_B3LYP"
    )
    assert row["status"] == "generated"
    assert row["components"][:2] == [["LDA_X", "2/25"], ["GGA_X_B88", "18/25"]]


@pytest.mark.parametrize(
    "statement",
    (
        "unknown_update(p);",
        "p->cam_alpha = 0.3;",
        "p->nlc_b *= 2.0;",
        "p->params = unknown_allocator();",
    ),
)
def test_metadata_rejects_uninterpreted_initializer_semantics(statement: str) -> None:
    text = (LIBXC / "hyb_mgga_xc_wb97mv.c").read_text()
    changed = text.replace("p->nlc_b = 6.0;", "p->nlc_b = 6.0;\n" + statement)
    (row,) = extract_method_registrations(changed)
    assert row["status"] == "blocked"
    assert "initializer statement" in row["reason"]


def test_metadata_rejects_mix_array_mutation_before_initialization() -> None:
    text = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "xc_mix_init(p, 5, funcs_id, funcs_coef);"
    changed = text.replace(old, "funcs_coef[0] = 0.9;\n" + old)
    row = next(
        row
        for row in extract_method_registrations(changed)
        if row["registration"] == "HYB_GGA_XC_B3P86_NWCHEM"
    )
    assert row["status"] == "blocked"
    assert "initializer statement" in row["reason"]


def test_generated_metadata_component_sequences_are_immutable() -> None:
    components = LIBXC_METHODS["B3LYP"]["components"]
    assert isinstance(components, tuple)
    assert all(isinstance(item, tuple) for item in components)
    with pytest.raises(TypeError):
        components[0][1] = "1/2"


@pytest.mark.parametrize("identifier", sorted(LIBXC_METHODS))
def test_generated_exchange_and_vv10_match_independent_libxc(identifier: str) -> None:
    """Compare parsed C text with the separately compiled Libxc default state."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("metadata oracle requires the pinned Libxc 7.0.0 semantics")
    record = LIBXC_METHODS[identifier]
    # Numeric IDs bypass PySCF aliases (notably the configurable B3LYP alias).
    code = record["libxc_id"]
    omega, alpha, beta = libxc.rsh_coeff(code)

    def number(field: str) -> float:
        return float(Fraction(record[field]))

    assert number("range_omega") == pytest.approx(omega, rel=0, abs=2e-15)
    if omega:
        assert number("short_range_exchange") == pytest.approx(
            alpha + beta, rel=0, abs=2e-15
        )
        assert number("long_range_exchange") == pytest.approx(alpha, rel=0, abs=2e-15)
    else:
        for spin in (0, 1):
            assert number("exact_exchange") == pytest.approx(
                libxc.hybrid_coeff(code, spin), rel=0, abs=2e-15
            )
    if record["nonlocal_variant"]:
        ((parameters, weight),) = libxc.nlc_coeff(code)
        assert weight == 1
        assert (number("nlc_b"), number("nlc_c")) == pytest.approx(
            parameters, rel=0, abs=2e-15
        )
    else:
        assert not libxc.nlc_coeff(code)
