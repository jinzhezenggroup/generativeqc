from __future__ import annotations

import copy

import pytest
from generativeqc_compiler.common import paths
from generativeqc_compiler.xc import automatic_semilocal, libxc_bulk
from generativeqc_compiler.xc.libxc_maple import MapleImportError
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN_VERSION
from generativeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, UnsupportedXC

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


def test_registry_captures_catalog_and_sources_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_catalog = libxc_bulk.read_catalog
    original_sources = paths.source_hashes
    calls = {"catalog": 0, "sources": 0}

    def read_catalog() -> dict:
        calls["catalog"] += 1
        return original_catalog()

    def source_hashes(*families: str, assets: tuple[str, ...] = ()) -> dict[str, str]:
        calls["sources"] += 1
        return original_sources(*families, assets=assets)

    monkeypatch.setattr(libxc_bulk, "read_catalog", read_catalog)
    monkeypatch.setattr(paths, "source_hashes", source_hashes)
    first = registry.registry_entries()
    assert first
    assert calls == {"catalog": 1, "sources": 1}
    assert registry.registry_entries() == first
    assert calls == {"catalog": 2, "sources": 2}


def test_registry_observes_catalog_override_on_the_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = copy.deepcopy(libxc_bulk.read_catalog())
    name = "LDA_C_VWN_4"
    record = next(row for row in catalog["registrations"] if row["name"] == name)
    monkeypatch.setattr(automatic_semilocal, "AUTO_BULK_COMPONENTS", (name,))
    monkeypatch.setattr(libxc_bulk, "read_catalog", lambda: catalog)
    first = registry.registry_entries()
    assert first[0].libxc_id == record["id"]
    record["id"] = 0xFFFF
    second = registry.registry_entries()
    assert second[0].libxc_id == 0xFFFF
    assert second[0].code == automatic_semilocal.AUTOMATIC_FUNCTIONAL_CODE_BASE | 0xFFFF
    assert first != second


@pytest.mark.parametrize("invalid_id", (-1, 0, 0x10000))
def test_registry_rejects_out_of_range_catalog_ids(
    monkeypatch: pytest.MonkeyPatch, invalid_id: int
) -> None:
    catalog = copy.deepcopy(libxc_bulk.read_catalog())
    name = "LDA_C_VWN_4"
    record = next(row for row in catalog["registrations"] if row["name"] == name)
    record["id"] = invalid_id
    monkeypatch.setattr(automatic_semilocal, "AUTO_BULK_COMPONENTS", (name,))
    monkeypatch.setattr(libxc_bulk, "read_catalog", lambda: catalog)
    with pytest.raises(UnsupportedXC, match="outside the encoded range"):
        registry.registry_entries()


def test_registry_rejects_duplicate_catalog_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = copy.deepcopy(libxc_bulk.read_catalog())
    names = ("LDA_C_VWN_4", "GGA_X_APBE")
    for record in catalog["registrations"]:
        if record["name"] in names:
            record["id"] = 63
    monkeypatch.setattr(automatic_semilocal, "AUTO_BULK_COMPONENTS", names)
    monkeypatch.setattr(libxc_bulk, "read_catalog", lambda: catalog)
    with pytest.raises(ValueError, match="duplicate automatic Libxc functional code"):
        registry.registry_entries()


@pytest.mark.parametrize("state", ("missing", "blocked"))
def test_registry_preserves_unavailable_registration_refusal(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    catalog = copy.deepcopy(libxc_bulk.read_catalog())
    name = "LDA_C_VWN_4"
    record = next(row for row in catalog["registrations"] if row["name"] == name)
    if state == "missing":
        catalog["registrations"].remove(record)
    else:
        record["graph_status"] = "blocked"
        record["reason"] = "test unavailable registration"
    monkeypatch.setattr(automatic_semilocal, "AUTO_BULK_COMPONENTS", (name,))
    monkeypatch.setattr(libxc_bulk, "read_catalog", lambda: catalog)
    with pytest.raises(MapleImportError, match="unknown bulk|not claimable"):
        registry.registry_entries()


def test_registry_keeps_unsupported_ingredient_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        automatic_semilocal, "AUTO_BULK_COMPONENTS", ("LDA_C_VWN_4", "MGGA_X_JK")
    )
    assert tuple(row.name for row in registry.registry_entries()) == ("LDA_C_VWN_4",)
