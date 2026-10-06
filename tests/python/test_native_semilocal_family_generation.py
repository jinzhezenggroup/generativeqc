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
    assert {
        name for name, record in by_name.items() if record["stationary_second_order"]
    } == {"LDA", "PBE"}
    assert by_name["PBE"]["native_range_exchange"] is True
    assert by_name["PBE"]["cuda_nonlocal_correlation"] is False
    assert by_name["WB97M-V"]["molecular_nonlocal_domain"] is True


CUDA_FAST_PATH_CENSUS = {
    "LDA": {
        "component_scaling": "qualification-required",
        "mixed_ao_precision": "qualified",
        "mixed_density_precision": "qualified",
        "response": "qualified",
        "graph_replay": "qualified",
    },
    "PBE": {
        "component_scaling": "qualified",
        "mixed_ao_precision": "qualified",
        "mixed_density_precision": "qualified",
        "response": "qualified",
        "graph_replay": "qualified",
    },
    "R2SCAN": {
        "component_scaling": "unavailable",
        "mixed_ao_precision": "qualification-required",
        "mixed_density_precision": "qualified",
        "response": "unavailable",
        "graph_replay": "qualification-required",
    },
    "B3LYP": {
        "component_scaling": "unavailable",
        "mixed_ao_precision": "qualification-required",
        "mixed_density_precision": "qualification-required",
        "response": "unavailable",
        "graph_replay": "qualification-required",
    },
    "WB97M-V": {
        "component_scaling": "unavailable",
        "mixed_ao_precision": "qualification-required",
        "mixed_density_precision": "qualification-required",
        "response": "unavailable",
        "graph_replay": "qualification-required",
    },
}


def test_cuda_fast_path_capability_census_is_manifest_owned() -> None:
    families = generator.load_manifest()
    assert {
        item["name"]: item["cuda_fast_paths"] for item in families
    } == CUDA_FAST_PATH_CENSUS


def test_cuda_fast_path_capabilities_ignore_display_names() -> None:
    for item in generator.load_manifest():
        renamed = dict(item)
        renamed["name"] = f"alias-{item['code']}"
        assert generator.cpp_fast_path_capabilities(
            renamed
        ) == generator.cpp_fast_path_capabilities(item)


def test_python_cuda_fast_path_capability_census_is_manifest_owned() -> None:
    assert {
        item["name"]: item["cuda_fast_paths"] for item in SEMILOCAL_FAMILIES
    } == CUDA_FAST_PATH_CENSUS


def test_native_xc_dispatch_consumes_generated_capabilities() -> None:
    from generativeqc_compiler.dft.ao_cuda import emit_native_xc_point_dispatch

    source = emit_native_xc_point_dispatch()
    for item in SEMILOCAL_FAMILIES:
        code = f"semilocal_family_code(SemilocalFamily::{item['symbol']})"
        assert f"return &launch_points<{code}, false>;" in source
        assert (f"return &launch_points<{code}, true>;" in source) == (
            item["cuda_fast_paths"]["response"] == "qualified"
        )


def test_grid_jit_tracks_and_ships_capability_headers() -> None:
    import re

    from generativeqc_compiler.dft.ao_cuda import emit_grid_source

    _, _, headers = emit_grid_source()
    packaged = dict(
        re.findall(
            r'^"([^\"]+)"\s*=\s*"([^\"]+)"$',
            (ROOT / "pyproject.toml").read_text(),
            re.MULTILINE,
        )
    )
    for relative in ("src/dft/semilocal_family.hpp", "src/dft/xc_capabilities.hpp"):
        assert ROOT / relative in headers
        assert packaged[relative] == f"generativeqc_compiler/assets/{relative}"
