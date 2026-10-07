"""Check automatic Python API module discovery without importing runtime modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import render_python_api_doc as renderer


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_public_modules_are_discovered_recursively_from_all(tmp_path: Path) -> None:
    package = tmp_path / "generativeqc"
    _write(package / "__init__.py", '__all__ = ["Root"]\n')
    _write(package / "future.py", '__all__ = ["Thing", "run"]\n')
    _write(package / "ordinary_internal.py", "def helper(): pass\n")
    _write(package / "_private.py", '__all__ = ["Hidden"]\n')
    _write(package / "__main__.py", '__all__ = ["main"]\n')
    _write(package / "nested/__init__.py", '__all__ = ["feature"]\n')
    _write(package / "nested/feature.py", '__all__ = ["Feature"]\n')

    modules = renderer.public_api_modules(package)
    assert [module.name for module in modules] == [
        "generativeqc",
        "generativeqc.future",
        "generativeqc.nested",
        "generativeqc.nested.feature",
    ]
    rendered = renderer.render_python_api_markdown(package)
    assert "generativeqc.future" in rendered
    assert "generativeqc.nested.feature" in rendered
    assert "ordinary_internal" not in rendered
    assert "_private" not in rendered


def test_nonliteral_public_all_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "generativeqc"
    _write(package / "__init__.py", '__all__ = ["Root"]\n')
    _write(package / "dynamic.py", "__all__ = names()\n")

    with pytest.raises(ValueError, match="literal list or tuple"):
        renderer.public_api_modules(package)


def test_current_reference_covers_declared_public_facades() -> None:
    names = {module.name for module in renderer.public_api_modules()}

    assert "generativeqc" in names
    assert "generativeqc.torch" in names
    assert "generativeqc.extensions" in names
    assert "generativeqc.extensions.method" in names
    assert "generativeqc.extensions.tensor" in names
    assert "generativeqc.extensions.xc" in names


def test_sphinx_source_is_generated_and_tracks_package_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "generativeqc"
    _write(package / "__init__.py", '__all__ = ["Root"]\n')
    _write(package / "future.py", '__all__ = ["Thing"]\n')
    monkeypatch.setattr(renderer, "PACKAGE", package)

    registered: list[str] = []
    app = SimpleNamespace(env=SimpleNamespace(note_dependency=registered.append))
    source = ["source shell"]
    renderer.render_python_api_source(app, "reference/api", source)

    assert "generativeqc.future" in source[0]
    assert str(package) in registered
    assert str(package / "future.py") in registered

    registered.clear()
    source = ["unrelated"]
    renderer.render_python_api_source(app, "index", source)
    assert source == ["unrelated"]
    assert not registered
