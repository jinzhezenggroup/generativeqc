"""Ownership guards for the issue #487 production-module split."""

from __future__ import annotations

import ast
import importlib.util
from graphlib import TopologicalSorter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INTEGRAL = ROOT / "python" / "generativeqc_compiler" / "integral"
PRODUCTION_MODULES = {
    "production_bundle",
    "production_cost",
    "production_emission",
    "production_exchange_queue",
    "production_profile",
    "production_k_block",
    "production_registry",
    "production_selection",
}
INTEGRAL_PACKAGE = "generativeqc_compiler.integral"
EMISSION_IMPORTS = {
    "__future__",
    "capabilities",
    "collections.abc",
    "cuda_emitter",
    "lowering",
    "cuda_schedule",
    "fused_schedule",
    "functools",
    "ir",
    "production_cost",
    "production_exchange_queue",
    "production_profile",
    "production_registry",
    "production_selection",
    "re",
    "shell_spec",
    "signature",
    "typing",
    "generativeqc_compiler.common.cuda_target",
}
EXCHANGE_QUEUE_IMPORTS = {"__future__", "collections.abc", "typing"}


def _imports_from_source(source: str) -> set[str]:
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                target = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), INTEGRAL_PACKAGE
                )
            else:
                target = node.module or ""
            if target == INTEGRAL_PACKAGE:
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif target.startswith(f"{INTEGRAL_PACKAGE}."):
                imports.add(
                    target.removeprefix(f"{INTEGRAL_PACKAGE}.").split(".", 1)[0]
                )
            elif target:
                imports.add(target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{INTEGRAL_PACKAGE}."):
                    imports.add(
                        alias.name.removeprefix(f"{INTEGRAL_PACKAGE}.").split(".", 1)[0]
                    )
                else:
                    imports.add(alias.name)
    return imports


def _imports(module: str) -> set[str]:
    return _imports_from_source((INTEGRAL / f"{module}.py").read_text(encoding="utf-8"))


def _calls(module: str) -> set[str]:
    tree = ast.parse((INTEGRAL / f"{module}.py").read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _production_imports(module: str) -> set[str]:
    """Return direct imports within the production ownership graph."""
    return _imports(module) & PRODUCTION_MODULES


def test_production_ownership_graph_is_cycle_free() -> None:
    graph = {module: _production_imports(module) for module in PRODUCTION_MODULES}
    for module in PRODUCTION_MODULES - {"production"}:
        assert "production" not in graph[module]
    TopologicalSorter(graph).prepare()


def test_import_normalization_covers_equivalent_package_spellings() -> None:
    imports = _imports_from_source(
        """
from . import production_bundle
from .production_registry import emit_registry_source
from .production_exchange_queue import exchange_streaming_worker
from generativeqc_compiler.integral import production_profile
from generativeqc_compiler.integral import production_exchange_queue
from generativeqc_compiler.integral.production_selection import KernelSelection
import generativeqc_compiler.integral.production_cost
"""
    )
    assert imports == {
        "production_bundle",
        "production_cost",
        "production_exchange_queue",
        "production_registry",
        "production_profile",
        "production_selection",
    }


def test_bundle_owner_does_not_import_cuda_emitters() -> None:
    imports = _imports("production_bundle")
    assert not any(
        "cuda_emitter" in name or "cuda_lowering" in name for name in imports
    )
    assert not any("benchmark" in name or "cli" in name for name in imports)


@pytest.mark.parametrize(
    ("module", "allowed_imports"),
    [
        ("production_emission", EMISSION_IMPORTS),
        ("production_exchange_queue", EXCHANGE_QUEUE_IMPORTS),
    ],
)
def test_emission_owner_does_not_import_bundle_or_filesystem_orchestration(
    module: str, allowed_imports: set[str]
) -> None:
    imports = _imports(module)
    assert not any("production_bundle" in name for name in imports)
    assert not any("benchmark" in name or "cli" in name for name in imports)
    assert imports <= allowed_imports
    assert not _calls(module) & {
        "makedirs",
        "mkdir",
        "open",
        "rename",
        "unlink",
        "write_bytes",
        "write_text",
    }


def test_exchange_queue_is_a_registered_leaf_emitter() -> None:
    """Queue scheduling borrows science callbacks without owning another backend."""
    assert "production_exchange_queue" in PRODUCTION_MODULES
    assert "production_exchange_queue" in _production_imports("production_emission")
    assert not _production_imports("production_exchange_queue")
