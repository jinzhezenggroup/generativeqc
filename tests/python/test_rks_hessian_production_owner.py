"""Ownership guards for the installed semilocal RKS Hessian path."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import generativeqc.rks_hessian as production_hessian
import generativeqc.rks_hessian_directional as production_directional
import generativeqc.rks_hessian_integrals as production_integrals

import tools.generativeqc_hessian as tools_api

ROOT = Path(__file__).resolve().parents[2]


def test_tools_rks_hessian_modules_are_production_aliases() -> None:
    assert (
        importlib.import_module("tools.generativeqc_hessian.rks_directional")
        is production_directional
    )
    assert (
        importlib.import_module("tools.generativeqc_hessian.rks_molecular")
        is production_hessian
    )


def test_tools_top_level_rks_hessian_api_reuses_production_objects() -> None:
    for name in (
        "DirectionalRKSBatchResponse",
        "DirectionalRKSResponse",
        "RKSXCHVPComponents",
        "directional_rks_response",
        "directional_rks_responses",
        "native_rks_xc_hvp_components",
    ):
        assert getattr(tools_api, name) is getattr(production_directional, name)
    for name in (
        "RKSHessianResult",
        "RKSHVPBatchResult",
        "RKSHVPResult",
        "rks_hessian",
        "rks_hvp",
        "rks_hvp_many",
    ):
        assert getattr(tools_api, name) is getattr(production_hessian, name)


def test_installed_rks_hessian_modules_never_import_tools() -> None:
    for relative in (
        "python/generativeqc/rks_hessian_integrals.py",
        "python/generativeqc/rks_hessian_directional.py",
        "python/generativeqc/rks_hessian.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == "tools" or name.startswith("tools.") for name in imports)


def test_tools_rks_alias_files_contain_no_scientific_definitions() -> None:
    for relative in (
        "tools/generativeqc_hessian/rks_directional.py",
        "tools/generativeqc_hessian/rks_molecular.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert definitions == set()


def test_rks_integral_topology_is_native_ao_owned() -> None:
    assert (
        production_integrals.RKSIntegralTopology.__module__
        == "generativeqc.rks_hessian_integrals"
    )
    source = (ROOT / "python/generativeqc/rks_hessian_integrals.py").read_text(
        encoding="utf-8"
    )
    assert "NativeSource" not in source
    assert "NativeRHFState" not in source
    assert "NativeAO" in source
