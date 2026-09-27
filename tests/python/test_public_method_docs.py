from __future__ import annotations

from pathlib import Path

from tools.render_public_methods_doc import render_public_methods_markdown

ROOT = Path(__file__).resolve().parents[2]


def test_sphinx_public_method_catalog_covers_all_public_entry_classes() -> None:
    rendered = render_public_methods_markdown()

    assert "## Registered native methods" in rendered
    assert "`rhf`" in rendered
    assert "## Public composite selectors" in rendered
    assert "`r2scan-3c`" in rendered
    assert "`r2scan-3c-rks`" in rendered
    assert "`r2scan-3c-uks`" in rendered
    assert "## Automatic Libxc semilocal MethodIR registrations" in rendered
    assert "`GGA_X_APBE`" in rendered
    assert "## Explicit Libxc blacklist" in rendered
    assert "`GGA_C_AM05`" in rendered


def test_public_methods_source_is_a_sphinx_shell_not_a_stale_table() -> None:
    source = (ROOT / "docs/public_methods.md").read_text(encoding="utf-8")

    assert "populated at Sphinx build time" in source
    assert "| Method | Family |" not in source
