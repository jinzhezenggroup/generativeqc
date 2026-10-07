"""Render the Python API reference from explicitly public modules.

A module opts into the generated reference by defining a literal __all__.
Discovery is static, so adding a public module never requires importing it merely
to decide whether it belongs in the documentation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/generativeqc"
BT = chr(96)


@dataclass(frozen=True)
class PublicModule:
    """One statically declared public Python module."""

    name: str
    source: Path
    exports: tuple[str, ...]


def _literal_all(source: Path) -> tuple[str, ...] | None:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    value: ast.expr | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            )
        ):
            value = node.value
    if value is None:
        return None
    try:
        exports = ast.literal_eval(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{source}: public __all__ must be a literal list or tuple of names"
        ) from exc
    if not isinstance(exports, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in exports
    ):
        raise TypeError(f"{source}: public __all__ must contain non-empty strings")
    if len(set(exports)) != len(exports):
        raise ValueError(f"{source}: public __all__ contains duplicate names")
    return tuple(exports)


def _module_name(source: Path, package: Path) -> str:
    relative = source.relative_to(package)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = source.stem
    return ".".join((package.name, *parts))


def public_api_modules(package: Path | None = None) -> tuple[PublicModule, ...]:
    """Discover public modules recursively from literal __all__ declarations."""
    package = PACKAGE if package is None else package
    modules = []
    for source in package.rglob("*.py"):
        relative = source.relative_to(package)
        if source.name == "__main__.py":
            continue
        if any(
            part.startswith("_") and part != "__init__.py" for part in relative.parts
        ):
            continue
        exports = _literal_all(source)
        if exports is None:
            continue
        modules.append(
            PublicModule(
                name=_module_name(source, package),
                source=source,
                exports=exports,
            )
        )
    return tuple(
        sorted(
            modules,
            key=lambda module: (
                module.name != package.name,
                module.name.count("."),
                module.name,
            ),
        )
    )


def python_api_doc_dependencies(package: Path | None = None) -> tuple[Path, ...]:
    """Track the renderer, package directories, and current Python sources."""
    package = PACKAGE if package is None else package
    paths = {Path(__file__).resolve(), package}
    paths.update(path for path in package.rglob("*") if path.is_dir())
    paths.update(package.rglob("*.py"))
    return tuple(sorted(paths))


def render_python_api_markdown(package: Path | None = None) -> str:
    """Render Sphinx directives for every explicitly public module."""
    package = PACKAGE if package is None else package
    modules = public_api_modules(package)
    if not modules or modules[0].name != package.name:
        raise ValueError(f"{package}: package __init__.py must declare public __all__")

    fence = BT * 3
    lines = [
        "# Python API",
        "",
        "This page is generated at Sphinx build time from the Python source tree.",
        f"A module is part of this reference when it has a literal {BT}__all__{BT};",
        "private modules and implementation files without that declaration are skipped.",
        "Adding a new public module therefore does not require editing the documentation",
        "navigation or this page.",
        "",
        "Signatures, type annotations, docstrings, inheritance, and source links are",
        "taken from the importable objects during the Sphinx build.",
        "",
        "## Public modules",
        "",
        f"{fence}{{eval-rst}}",
        ".. autosummary::",
        "",
    ]
    lines.extend(f"   {module.name}" for module in modules)
    lines.extend([fence, ""])

    for module in modules:
        lines.extend(
            [
                f"## {BT}{module.name}{BT}",
                "",
                f"{fence}{{eval-rst}}",
                f".. automodule:: {module.name}",
                "   :members: " + ", ".join(module.exports),
                "   :imported-members:",
                "   :undoc-members:",
                "   :show-inheritance:",
            ]
        )
        # This re-export already has its canonical target in the root facade.
        # Suppress only that duplicate, preserving every unique module/member
        # target for autosummary links, cross-references, and deep links.
        if module.name == "generativeqc.extensions.xc":
            lines.append("   :exclude-members: FunctionalSpec")
        lines.extend([fence, ""])
    return "\n".join(lines)


def render_python_api_source(app: Any, docname: str, source: list[str]) -> None:
    """Populate the API page and register discovery/source dependencies."""
    if docname != "reference/api":
        return
    for dependency in python_api_doc_dependencies():
        app.env.note_dependency(str(dependency))
    source[0] = render_python_api_markdown()


if __name__ == "__main__":
    print(render_python_api_markdown())
