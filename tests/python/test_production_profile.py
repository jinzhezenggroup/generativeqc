"""Production manifest/profile ownership stays separate from emission orchestration."""

from __future__ import annotations

import ast
from pathlib import Path

from generativeqc_compiler.integral import production_profile


def test_production_profile_avoids_emission_orchestration_dependencies() -> None:
    """Manifest/profile parsing must not depend on emission or bundle owners."""

    source = Path(production_profile.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    absolute_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert relative_modules.isdisjoint(
        {"production", "production_cost", "cuda_emitter", "cuda_lowering"}
    )
    assert absolute_roots.isdisjoint({"benchmarks", "tools", "generativeqc"})


def test_resolved_profile_is_owned_by_profile_module() -> None:
    """The canonical profile result type is no longer defined by the facade."""

    assert production_profile.ResolvedProductionProfile.__module__.endswith(
        ".production_profile"
    )
