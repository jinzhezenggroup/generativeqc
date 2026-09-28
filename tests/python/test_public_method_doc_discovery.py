"""Exercise catalog composition and Sphinx dependency publication without a GPU."""

from __future__ import annotations

import sys
import typing
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import render_public_methods_doc as renderer


def _catalog_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    ks = ModuleType("vibeqc.ks")
    method = ModuleType("vibeqc_compiler.method")
    method.METHOD_CATALOG = {"NEW_HYBRID": object(), "UNSUPPORTED": object()}
    method.METHOD_ALIASES = {"NEW_ALIAS": "NEW_HYBRID"}

    def resolve(selector: str) -> tuple[SimpleNamespace, None]:
        if selector.startswith("unsupported-"):
            raise NotImplementedError("missing native primitive")
        stem, spin = selector.rsplit("-", 1)
        return SimpleNamespace(identifier=stem.upper(), spin=spin), None

    ks.resolve_ks_method = resolve
    ks.public_dft_selectors = lambda: (
        "new_hybrid-rks",
        "new_hybrid-uks",
        "new_alias-rks",
        "new_alias-uks",
        "pbe0-d4-rks",
        "unsupported-rks",
        "unsupported-uks",
    )
    monkeypatch.setitem(sys.modules, "vibeqc", ModuleType("vibeqc"))
    monkeypatch.setitem(sys.modules, "vibeqc.ks", ks)
    monkeypatch.setitem(sys.modules, "vibeqc_compiler", ModuleType("vibeqc_compiler"))
    monkeypatch.setitem(sys.modules, "vibeqc_compiler.method", method)


def _manifest() -> dict[str, typing.Any]:
    return {
        "schema_version": 2,
        "methods": [
            {
                "abi_id": 1,
                "name": "rhf",
                "family": "hartree_fock",
                "properties": ["energy", "forces"],
                "supports_batch": True,
                "aliases": [],
            }
        ],
        "composite_methods": [],
    }


def test_new_compiler_names_and_aliases_need_no_abi_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog_modules(monkeypatch)
    monkeypatch.setattr(renderer, "_load_manifest", _manifest)
    monkeypatch.setattr(renderer, "_automatic_libxc_rows", lambda: ([], []))
    text = renderer.render_public_methods_markdown()
    assert "## Compiler-discovered DFT selectors" in text
    for stem in ("new_hybrid", "new_alias"):
        for spin in ("rks", "uks"):
            assert f"|`{stem}-{spin}` |" in text
    assert "|`pbe0-d4-rks` |" in text
    assert "unsupported-rks" not in text
    assert "unsupported-uks" not in text
    assert "|`rhf` |" in text


def test_manifest_compatibility_selectors_are_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog_modules(monkeypatch)
    payload = _manifest()
    payload["methods"][0]["name"] = "new_hybrid-rks"
    payload["methods"][0]["aliases"] = ["new_alias-rks"]
    monkeypatch.setattr(renderer, "_load_manifest", lambda: payload)
    monkeypatch.setattr(renderer, "_automatic_libxc_rows", lambda: ([], []))
    text = renderer.render_public_methods_markdown()
    section = text.split("## Compiler-discovered DFT selectors", 1)[1].split(
        "## Public composite selectors", 1
    )[0]
    assert "new_hybrid-rks" not in section
    assert "new_alias-rks" not in section
    assert "|`new_hybrid-uks` |" in section


@pytest.mark.parametrize("blocked", [[], [("TEST_XC", "GGA", "test defect")]])
def test_blacklist_section_tracks_only_returned_exceptions(
    monkeypatch: pytest.MonkeyPatch, blocked: list[tuple[str, str, str]]
) -> None:
    monkeypatch.setattr(renderer, "_load_manifest", _manifest)
    monkeypatch.setattr(renderer, "_compiler_dft_rows", list)
    monkeypatch.setattr(renderer, "_automatic_libxc_rows", lambda: ([], blocked))
    section = renderer.render_public_methods_markdown().split(
        "## Explicit Libxc blacklist", 1
    )[1]
    assert f"**{len(blocked)}**" in section
    assert ("|`TEST_XC` |" in section) == bool(blocked)


def test_sphinx_tracks_renderer_manifest_and_recursive_catalog_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python"
    owner = python / "vibeqc_compiler/method/nested/spec.py"
    owner.parent.mkdir(parents=True)
    owner.write_text("# catalog owner\n")
    catalog = python / "vibeqc_compiler/xc/libxc_bulk_catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(renderer, "PYTHON", python)
    monkeypatch.setattr(renderer, "MANIFEST", manifest)
    monkeypatch.setattr(renderer, "LIBXC_CATALOG", catalog)
    dependencies = renderer.public_method_doc_dependencies()
    assert owner in dependencies
    assert catalog in dependencies
    assert manifest in dependencies
    assert python / "vibeqc/ks.py" in dependencies
    assert Path(renderer.__file__).resolve() in dependencies

    registered: list[str] = []
    app = SimpleNamespace(env=SimpleNamespace(note_dependency=registered.append))
    monkeypatch.setattr(
        renderer, "render_public_methods_markdown", lambda: "updated catalog"
    )

    text = ["source shell"]
    renderer.render_public_methods_source(app, "public_methods", text)
    assert text == ["updated catalog"]
    assert registered == [str(path) for path in dependencies]

    registered.clear()
    text = ["unrelated page"]
    renderer.render_public_methods_source(app, "index", text)
    assert not registered
    assert text == ["unrelated page"]
