"""Do not advertise hybrid dry runs before the CLI exposes explicit grids."""

import pytest
from generativeqc import _generated_methods
from generativeqc.__main__ import (
    _automatic_libxc_method_rows,
    _public_method_rows,
    parser,
)


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_resource_cli_does_not_advertise_unusable_hybrid(method: str) -> None:
    # Hybrid resource planning is available through KsOptions(grid=...) in
    # Python. The CLI has no grid control, so accepting these names would only
    # defer an unavoidable failure until scientific model resolution.
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["resources", "unused.xyz", "--method", method])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "method", ("rhf", "uhf", "lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
)
def test_resource_cli_preserves_qualified_method_choices(method: str) -> None:
    arguments = parser().parse_args(["resources", "unused.xyz", "--method", method])
    assert arguments.method == method


def test_method_catalog_combines_native_abi_and_methodir_dft_discovery() -> None:
    rows = _public_method_rows()
    by_name = {row["name"]: row for row in rows}

    # Non-DFT methods still come directly from the stable ABI/provider registry.
    for name, metadata in _generated_methods.METHOD_METADATA.items():
        if metadata["provider"] != "dft":
            assert name in by_name

    # PBE50 is compiler-owned and intentionally has no native ABI manifest row.
    assert "pbe50-rks" not in _generated_methods.METHOD_NAME_TO_ID
    assert by_name["pbe50-rks"]["properties"] == ("energy",)
    assert by_name["pbe50-uks"]["family"] == "density_functional"

    # Existing compatibility selectors remain discoverable without owning the
    # scientific composition.
    assert by_name["pbe0-rks"]["properties"] == ("energy",)
    assert by_name["b3lyp-uks"]["status"] == "available"
    assert by_name["wb97m-v"]["status"] == "available"
    assert by_name["wb97m-v"]["properties"] == ("energy",)

    # Discovery follows the current native lowerer set rather than a stale
    # handwritten whitelist.
    assert by_name["scan-rks"]["status"] == "available"
    assert "cam-b3lyp-rks" not in by_name


def test_methods_command_is_publicly_parseable() -> None:
    assert parser().parse_args(["methods"]).command == "methods"
    args = parser().parse_args(["methods", "--json"])
    assert args.command == "methods"
    assert args.json


def test_methods_command_can_discover_default_allow_libxc() -> None:
    rows = _automatic_libxc_method_rows()
    names = {row["name"] for row in rows}

    for name in ("GGA_X_APBE", "GGA_C_AM05", "GGA_X_AK13"):
        assert f"libxc:{name}" in names
        assert f"libxc-uks:{name}" in names
    assert all(row["properties"] == ("energy",) for row in rows)


def test_methods_libxc_discovery_honors_public_blacklist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.method import bulk_ks

    original = bulk_ks.blacklist_reason

    def blacklist_reason(name: str) -> str | None:
        if name.upper() == "GGA_X_APBE":
            return "test-only blocked registration"
        return original(name)

    monkeypatch.setattr(bulk_ks, "blacklist_reason", blacklist_reason)
    names = {row["name"] for row in _automatic_libxc_method_rows()}

    assert "libxc:GGA_X_APBE" not in names
    assert "libxc-uks:GGA_X_APBE" not in names
    assert "libxc:GGA_X_AK13" in names
    assert "libxc-uks:GGA_X_AK13" in names


def test_methods_libxc_flag_is_publicly_parseable() -> None:
    args = parser().parse_args(["methods", "--libxc", "--json"])
    assert args.command == "methods"
    assert args.libxc
    assert args.json
