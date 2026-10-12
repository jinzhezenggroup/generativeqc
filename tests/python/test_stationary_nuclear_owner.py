"""Ownership guards for the installed stationary nuclear-response layer."""

from __future__ import annotations

import ast
from pathlib import Path

import generativeqc.stationary_nuclear as production
import numpy as np
import pytest

from tools.generativeqc_hessian import perturbation as perturbation_shim
from tools.generativeqc_hessian import response as response_shim

ROOT = Path(__file__).resolve().parents[2]


def test_real_fp64_validation_does_not_copy_existing_numeric_storage() -> None:
    matrix = np.eye(3, dtype=np.float64)
    energies = np.arange(3.0)
    assert production._finite_matrix(matrix, name="matrix", nmo=3) is matrix
    assert production._orbital_energies(energies, nmo=3) is energies


def test_nuclear_inputs_must_remain_finite_after_fp64_conversion() -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform has no floating dtype wider than FP64")
    matrix = np.full((3, 3), np.finfo(np.longdouble).max, dtype=np.longdouble)
    with pytest.raises(ValueError, match="representable in FP64"):
        production.metric_density_response_mo(matrix, nocc=1)


def test_hessian_shims_reuse_production_stationary_nuclear_objects() -> None:
    assert (
        perturbation_shim.StationaryNuclearResponse
        is production.StationaryNuclearResponse
    )
    assert (
        perturbation_shim.StationaryNuclearBatchResponse
        is production.StationaryNuclearBatchResponse
    )
    assert perturbation_shim.RHFNuclearResponse is production.RHFNuclearResponse
    assert (
        perturbation_shim.RHFNuclearBatchResponse is production.RHFNuclearBatchResponse
    )
    assert (
        response_shim.metric_density_response_mo
        is production.metric_density_response_mo
    )
    assert (
        response_shim.build_stationary_nuclear_rhs
        is production.build_stationary_nuclear_rhs
    )
    assert response_shim.build_rhf_nuclear_rhs is production.build_rhf_nuclear_rhs


def test_production_stationary_nuclear_owner_never_imports_tools() -> None:
    path = ROOT / "python/generativeqc/stationary_nuclear.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "tools" or name.startswith("tools.") for name in imports)


def test_response_shim_contains_no_duplicate_nuclear_algebra() -> None:
    path = ROOT / "tools/generativeqc_hessian/response.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert definitions == set()
