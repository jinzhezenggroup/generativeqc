"""Single-source generation gates for curated native semilocal metadata."""

from pathlib import Path

from generativeqc_compiler.xc._generated_native_semilocal import (
    SCF_DOMAIN_BY_VERSION,
    SEMILOCAL_FAMILIES,
    SEMILOCAL_FAMILY_BY_CODE,
    SEMILOCAL_FAMILY_CODES,
)

from tools import generate_native_semilocal_families as generator

ROOT = Path(__file__).resolve().parents[2]


def test_native_semilocal_generated_outputs_are_current() -> None:
    families = generator.load_manifest()
    assert (ROOT / "src/dft/semilocal_family.hpp").read_text(
        encoding="utf-8"
    ) == generator.emit_cpp(families)
    assert (
        ROOT / "python/generativeqc_compiler/xc/_generated_native_semilocal.py"
    ).read_text(encoding="utf-8") == generator.emit_python(families)


def test_native_semilocal_transport_codes_are_manifest_owned() -> None:
    families = generator.load_manifest()
    assert tuple(item["code"] for item in families) == tuple(range(len(families)))
    assert SEMILOCAL_FAMILY_CODES == frozenset(range(len(families)))
    assert tuple(item["code"] for item in SEMILOCAL_FAMILIES) == tuple(
        item["code"] for item in families
    )


def test_native_semilocal_domain_versions_are_generated_once() -> None:
    assert SCF_DOMAIN_BY_VERSION == {
        item["domain_version"]: item["scf_domain"] for item in SEMILOCAL_FAMILIES
    }


def test_native_semilocal_execution_traits_are_manifest_owned() -> None:
    assert SEMILOCAL_FAMILY_BY_CODE == {
        item["code"]: item for item in SEMILOCAL_FAMILIES
    }
    by_name = {item["name"]: item for item in SEMILOCAL_FAMILIES}
    assert by_name["LDA"]["requires_gradient"] is False
    assert by_name["PBE"]["stationary_kernel"] == "pbe"
    assert by_name["R2SCAN"]["requires_tau"] is True
    assert by_name["R2SCAN"]["stationary_ecp_gradient"] is False
    assert by_name["B3LYP"]["stationary_kernel"] == "composed"
    assert by_name["WB97M-V"]["stationary_kernel"] == "wb97mv"
