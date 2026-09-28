from __future__ import annotations

from pathlib import Path

from tools.render_public_methods_doc import (
    _automatic_libxc_rows,
    _compiler_dft_rows,
    render_public_methods_markdown,
)

ROOT = Path(__file__).resolve().parents[2]


def test_sphinx_public_method_catalog_covers_all_public_entry_classes() -> None:
    rendered = render_public_methods_markdown()

    assert "## Registered native methods" in rendered
    assert "`rhf`" in rendered
    assert "## Compiler-discovered DFT selectors" in rendered
    for selector, _, _ in _compiler_dft_rows():
        assert f"`{selector}`" in rendered
    assert "## Public composite selectors" in rendered
    assert "`r2scan-3c`" in rendered
    assert "`r2scan-3c-rks`" in rendered
    assert "`r2scan-3c-uks`" in rendered
    assert "## Automatic Libxc semilocal MethodIR registrations" in rendered
    assert "`GGA_X_APBE`" in rendered
    blacklist = rendered.split("## Explicit Libxc blacklist", 1)[1]
    _, blocked = _automatic_libxc_rows()
    assert f"**{len(blocked)}**" in blacklist
    for name, _, reason in blocked:
        assert f"`{name}`" in blacklist
        assert reason in blacklist


def test_public_methods_source_is_a_sphinx_shell_not_a_stale_table() -> None:
    source = (ROOT / "docs/public_methods.md").read_text(encoding="utf-8")

    assert "populated at Sphinx build time" in source
    assert "| Method | Family |" not in source
