"""Keep the RKS Hessian stack's module and lazy-export boundaries coherent."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HESSIAN = ROOT / "tools/generativeqc_hessian"
INSTALLED = ROOT / "python/generativeqc"
_INSTALLED_OWNER = {
    "rks_directional": "rks_hessian_directional",
    "rks_molecular": "rks_hessian",
}


def _tree(module: str) -> ast.Module:
    owner = _INSTALLED_OWNER.get(module)
    path = (
        (INSTALLED / f"{owner}.py") if owner is not None else (HESSIAN / f"{module}.py")
    )
    return ast.parse(path.read_text(encoding="utf-8"))


def test_molecular_directional_imports_have_definitions() -> None:
    """Catch sync-induced ImportError without loading the optional native stack."""
    declared = {
        node.name
        for node in _tree("rks_directional").body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    requested = {
        alias.name
        for node in _tree("rks_molecular").body
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "rks_hessian_directional"
        for alias in node.names
    }
    assert requested, "molecular RKS lost its directional provider binding"
    assert requested <= declared, (
        f"undefined RKS imports: {sorted(requested - declared)}"
    )
    assert {"DirectionalRKSBatchResponse", "directional_rks_responses"} <= declared


def test_lazy_api_retains_batch_and_full_hessian_exports() -> None:
    values = {
        target.id: ast.literal_eval(node.value)
        for node in _tree("__init__").body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in ("_LAZY", "__all__")
    }
    expected = {
        "DirectionalRKSBatchResponse": "rks_directional",
        "directional_rks_responses": "rks_directional",
        "RKSHVPBatchResult": "rks_molecular",
        "RKSHessianResult": "rks_molecular",
        "rks_hvp_many": "rks_molecular",
        "rks_hessian": "rks_molecular",
    }
    for name, module in expected.items():
        assert values["_LAZY"].get(name) == module
        assert name in values["__all__"]
        declared = {
            node.name
            for node in _tree(module).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert name in declared, f"lazy export {name} has no definition in {module}"
