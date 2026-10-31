"""Production manifest/profile ownership stays separate from emission orchestration."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
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


@pytest.mark.parametrize("invalid", (0, 1, "false", None))
def test_optional_fock_variants_requires_an_explicit_boolean(
    tmp_path: Path, invalid: object
) -> None:
    """Reject ambiguous build policy instead of silently omitting kernels."""
    manifest = tmp_path / "profile.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "architectures": {
                    "portable_cuda": {
                        "kind": "portable",
                        "kernels": [],
                        "optional_fock_variants": invalid,
                    }
                },
            }
        )
    )
    with pytest.raises(TypeError, match="optional_fock_variants must be a boolean"):
        production_profile.resolve_production_profile(manifest, "sm_80")
