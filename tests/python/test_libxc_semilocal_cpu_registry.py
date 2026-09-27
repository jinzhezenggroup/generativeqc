from __future__ import annotations

from vibeqc_compiler.xc.libxc_blacklist import blacklist_reason
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from tools import generate_libxc_semilocal_cpu_registry as registry


def test_registry_is_default_allow_minus_blacklist() -> None:
    entries = registry.registry_entries()
    names = {entry.name for entry in entries}
    codes = {entry.code for entry in entries}

    assert "GGA_X_APBE" in names
    assert "MGGA_X_LTA" in names
    assert "GGA_C_AM05" not in names
    assert len(codes) == len(entries)
    assert all(name in AUTO_BULK_COMPONENTS for name in names)
    assert all(blacklist_reason(name) is None for name in names)


def test_generated_gga_point_program_uses_production_boundary() -> None:
    entry = next(
        item for item in registry.registry_entries() if item.name == "GGA_X_APBE"
    )
    source = registry._point_program_source(entry)

    assert '"GGA_X_APBE"' in source
    assert "total_density <" in source
    assert "7U," in source
    assert "out.gradient[0][axis]" in source


def test_generated_mgga_point_program_carries_tau_pullback() -> None:
    entry = next(
        item for item in registry.registry_entries() if item.name == "MGGA_X_LTA"
    )
    source = registry._point_program_source(entry)

    assert "15U," in source
    assert "out.kinetic[0] = 0.5" in source
    assert "out.kinetic[1] = 0.5" in source


def test_registry_supports_name_and_stable_code_lookup() -> None:
    header = registry.emit_header()
    dispatcher = registry.emit_registry()

    assert "automatic_libxc_entry(std::string_view name)" in header
    assert "automatic_libxc_entry(std::uint32_t functional_code)" in header
    assert 'automatic_libxc_entry("GGA_X_APBE")' in dispatcher
