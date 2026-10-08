"""Check exported API documentation statically and enforce Ruff D1xx rules.

The inspected Python modules are the same literal-__all__ modules rendered by
Sphinx. Following local re-exports prevents an undocumented implementation from
appearing documented merely because its facade imports it.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

try:
    from tools.render_python_api_doc import PACKAGE, ROOT, public_api_modules
except ModuleNotFoundError:
    # Support invocation as python tools/check_public_api_docstrings.py.
    from render_python_api_doc import PACKAGE, ROOT, public_api_modules

RUFF_DOC_RULES = "D100,D101,D102,D103,D104,D105,D107"


def _module_source(source_root: Path, name: str) -> Path | None:
    """Locate one first-party Python source without importing its dependencies."""
    prefix = source_root.joinpath(*name.split("."))
    for path in (prefix.with_suffix(".py"), prefix / "__init__.py"):
        if path.is_file():
            return path
    return None


def _resolve_export(
    source_root: Path,
    module_name: str,
    symbol: str,
    seen: set[tuple[str, str]],
) -> tuple[Path, bool] | None:
    """Find an exported function/class through first-party import-from chains."""
    key = (module_name, symbol)
    if key in seen:
        return None
    seen.add(key)
    path = _module_source(source_root, module_name)
    if path is None:
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if definition := definitions.get(symbol):
        return path, bool(ast.get_docstring(definition))

    package = (
        module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    )
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        matches = [
            alias for alias in node.names if (alias.asname or alias.name) == symbol
        ]
        if not matches:
            continue
        imported_module = "." * node.level + (node.module or "")
        if node.level:
            imported_module = importlib.util.resolve_name(imported_module, package)
        result = _resolve_export(source_root, imported_module, matches[0].name, seen)
        if result is not None:
            return result
    return None


def missing_public_docstrings(package: Path | None = None) -> tuple[str, ...]:
    """Return public exports whose first-party defining object lacks a docstring."""
    package = PACKAGE if package is None else package
    source_root = package.parent
    failures = []
    for module in public_api_modules(package):
        for name in module.exports:
            origin = _resolve_export(source_root, module.name, name, set())
            if origin is not None and not origin[1]:
                path = origin[0].relative_to(source_root)
                failures.append(f"{module.name}.{name} -> {path}: missing docstring")
    return tuple(sorted(set(failures)))


def main(argv: list[str] | None = None) -> int:
    """Check public exports and invoke pinned Ruff on every published module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruff", default="ruff", help="path to the Ruff executable")
    args = parser.parse_args(argv)
    errors = missing_public_docstrings()
    for error in errors:
        print(error, file=sys.stderr)
    modules = public_api_modules()
    command = [
        args.ruff,
        "check",
        "--select",
        RUFF_DOC_RULES,
        *(str(module.source.relative_to(ROOT)) for module in modules),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    return 1 if errors or result.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
