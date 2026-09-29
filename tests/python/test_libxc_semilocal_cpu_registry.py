from __future__ import annotations

from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN_VERSION
from generativeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from tools import generate_libxc_semilocal_cpu_registry as registry


def test_registry_is_default_allow_for_structurally_supported_components() -> None:
    entries = registry.registry_entries()
    names = {entry.name for entry in entries}
    codes = {entry.code for entry in entries}

    assert "GGA_X_APBE" in names
    assert "MGGA_X_LTA" in names
    assert "GGA_C_AM05" in names
    assert "GGA_X_AK13" in names
    assert len(codes) == len(entries)
    assert all(name in AUTO_BULK_COMPONENTS for name in names)


def test_generated_gga_point_program_uses_generic_work_boundary() -> None:
    for name in ("GGA_C_AM05", "GGA_X_AK13"):
        entry = next(item for item in registry.registry_entries() if item.name == name)
        source = registry._point_program_source(entry)

        assert f'"{name}"' in source
        assert "total_density <" in source
        assert "work_rho_a = fmax" in source
        assert "work_rho_b = fmax" in source
        assert "work_sigma_aa = fmax" in source
        assert "work_sigma_ab =" in source
        assert "out.energy =" in source
        assert "* total_density / (work_rho_a + work_rho_b)" in source
        assert f"{LIBXC_WORK_DOMAIN_VERSION}U, automatic_" in source
        assert "out.gradient[0][axis]" in source


def test_generated_mgga_point_program_carries_work_tau_and_fhc() -> None:
    entry = next(
        item for item in registry.registry_entries() if item.name == "MGGA_X_LTA"
    )
    source = registry._point_program_source(entry)

    assert "15U," in source
    assert "work_tau_a = fmax" in source
    assert "work_tau_b = fmax" in source
    assert "8.0 * work_rho_a * work_tau_a" in source
    assert "out.kinetic[0] = 0.5" in source
    assert "out.kinetic[1] = 0.5" in source


def test_registry_supports_name_and_stable_code_lookup() -> None:
    header = registry.emit_header()
    dispatcher = registry.emit_registry()

    assert "automatic_libxc_entry(std::string_view name)" in header
    assert "automatic_libxc_entry(std::uint32_t functional_code)" in header
    assert 'automatic_libxc_entry("GGA_X_APBE")' in dispatcher
