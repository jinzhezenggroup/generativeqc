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

# Autodoc also registers imported classes under their implementation names.
# Describe these shared types once at their established public facade, and link
# the other re-exports there. Keep discovery static and all unique members indexed.
_SHARED_CLASS_TARGETS = {
    "generativeqc.experimental.array_api": {
        name: f"generativeqc.extensions.tensor.{name}"
        for name in ("Index", "IndexSpace", "Program", "TensorSpec")
    },
    "generativeqc.extensions.xc": {
        "FunctionalSpec": "generativeqc.FunctionalSpec",
    },
}


@dataclass(frozen=True)
class PublicModule:
    """One statically declared public Python module."""

    name: str
    source: Path
    exports: tuple[str, ...]


def _uses_all(node: ast.AST) -> bool:
    """Find module-scope uses/bindings, including import aliases and definitions."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Decorators, defaults and annotations are definition-time expressions;
        # the function body is deferred and does not change the import inventory.
        return node.name == "__all__" or any(
            _uses_all(expression)
            for expression in (
                *node.decorator_list,
                node.args,
                node.returns,
                *getattr(node, "type_params", ()),
            )
            if expression is not None
        )
    if isinstance(node, ast.ClassDef):
        # The class body also runs now; nested function bodies remain deferred.
        return node.name == "__all__" or any(
            _uses_all(expression)
            for expression in (
                *node.decorator_list,
                *node.bases,
                *node.keywords,
                *getattr(node, "type_params", ()),
                *node.body,
            )
        )
    if isinstance(node, ast.Lambda):
        return _uses_all(node.args)
    if isinstance(node, ast.ImportFrom):
        return any(
            alias.name == "*" or (alias.asname or alias.name) == "__all__"
            for alias in node.names
        )
    if isinstance(node, ast.Import):
        return any(
            (alias.asname or alias.name.split(".")[0]) == "__all__"
            for alias in node.names
        )
    if isinstance(node, ast.Name):
        return node.id == "__all__"
    return any(_uses_all(child) for child in ast.iter_child_nodes(node))


def _literal_all(source: Path) -> tuple[str, ...] | None:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    value: ast.expr | None = None
    for node in tree.body:
        declares_all = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            or isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        )
        annotation_uses_all = isinstance(node, ast.AnnAssign) and _uses_all(
            node.annotation
        )
        if annotation_uses_all or not declares_all and _uses_all(node):
            raise ValueError(
                f"{source}: dynamic use/mutation of __all__ is unsupported; "
                "public __all__ must be a literal list or tuple of names"
            )
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
            # A bare annotation updates __annotations__, not the runtime value.
            if isinstance(node, ast.AnnAssign) and node.value is None:
                continue
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
    public_members = {
        f"{module.name}.{name}" for module in modules for name in module.exports
    }

    fence = BT * 3
    lines = [
        "# Python API",
        "",
        "This page is generated at Sphinx build time from the Python source tree.",
        f"A module under {BT}python/generativeqc{BT} is part of this reference when it has a literal {BT}__all__{BT};",
        "private modules and implementation files without that declaration are skipped.",
        "Adding a new public module therefore does not require editing the documentation",
        "navigation or this page.",
        "Compiler/research modules are covered only through declarations re-exported by",
        f"this supported facade, including {BT}generativeqc.extensions{BT}.",
        "",
        "Signatures, type annotations, docstrings, inheritance, and source links are",
        "taken from the importable objects during the Sphinx build.",
        "",
        "Shared [calculation, batch, result and Torch contracts](python_execution_contracts.md)",
        "and [extension/family contracts and coverage policy](python_contracts.md)",
        "supply applicable units, shapes, failures, ownership and backend boundaries.",
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
        shared_types = {
            name: target
            for name, target in _SHARED_CLASS_TARGETS.get(module.name, {}).items()
            if name in module.exports
        }
        for name, target in shared_types.items():
            if target not in public_members:
                raise ValueError(
                    f"{module.name}.{name}: shared class target {target} is not public"
                )
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
        if shared_types:
            lines.append("   :exclude-members: " + ", ".join(shared_types))
        lines.extend([fence, ""])
        if shared_types:
            lines.append(
                "Re-exported types (documented at their shared public target):"
            )
            lines.append("")
            lines.extend(
                f"- {{py:class}}{BT}{name} <{target}>{BT}"
                for name, target in shared_types.items()
            )
            lines.append("")
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
