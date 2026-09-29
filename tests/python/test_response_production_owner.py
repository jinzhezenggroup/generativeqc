"""Ownership guards for installed closed-shell response contracts and operators."""

from __future__ import annotations

import ast
from pathlib import Path

import generativeqc.response_operator as production_operator
import generativeqc.response_problem as production_problem
import generativeqc.response_xc as production_xc
import generativeqc.response_solver as production_solver
import generativeqc.rks_response as production_rks

import tools.generativeqc_response as response_api
from tools.generativeqc_response import krylov as solver_shim
from tools.generativeqc_response import native_ks
from tools.generativeqc_response import operators as operator_shim
from tools.generativeqc_response import problem as problem_shim
from tools.generativeqc_response import xc as xc_shim

ROOT = Path(__file__).resolve().parents[2]


def test_response_problem_shim_reuses_installed_objects() -> None:
    for name in (
        "ResponseCompatibilityError",
        "ResponseProblem",
        "ResponseSolveError",
        "ResponseUnsupported",
        "RotationLayout",
    ):
        assert getattr(problem_shim, name) is getattr(production_problem, name)


def test_response_operator_shim_reuses_installed_objects() -> None:
    for name in (
        "CPKSResponseOperator",
        "DenseMatrixResponseOperator",
        "RHFResponseOperator",
        "cpks_operator_identity",
        "rhf_operator_identity",
        "validate_rotation_layout",
    ):
        assert getattr(operator_shim, name) is getattr(production_operator, name)


def test_top_level_tools_response_api_reuses_installed_closed_shell_objects() -> None:
    assert response_api.CPKSResponseOperator is production_operator.CPKSResponseOperator
    assert response_api.RHFResponseOperator is production_operator.RHFResponseOperator
    assert response_api.ResponseProblem is production_problem.ResponseProblem
    assert (
        response_api.FixedDensityXCDerivativeKernel
        is production_xc.FixedDensityXCDerivativeKernel
    )


def test_response_xc_shim_reuses_installed_objects() -> None:
    assert (
        xc_shim.FixedDensityXCDerivativeKernel
        is production_xc.FixedDensityXCDerivativeKernel
    )
    assert xc_shim.density_feature_response is production_xc.density_feature_response


def test_response_solver_and_rks_adapter_use_installed_owners() -> None:
    assert solver_shim is production_solver
    assert native_ks.NativeRKSResponse is production_rks.NativeRKSResponse
    assert response_api.NativeRKSResponse is production_rks.NativeRKSResponse
    assert response_api.GMRESOptions is production_solver.GMRESOptions
    assert response_api.solve is production_solver.solve
    assert response_api.solve_many is production_solver.solve_many


def test_native_rks_adapter_consumes_production_cpks_owner() -> None:
    assert issubclass(
        native_ks.NativeRKSResponse,
        production_operator.CPKSResponseOperator,
    )
    assert (
        native_ks.FixedDensityXCDerivativeKernel
        is production_xc.FixedDensityXCDerivativeKernel
    )


def test_installed_response_owners_never_import_tools() -> None:
    for relative in (
        "python/generativeqc/response_problem.py",
        "python/generativeqc/response_operator.py",
        "python/generativeqc/response_xc.py",
        "python/generativeqc/response_solver.py",
        "python/generativeqc/rks_response.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == "tools" or name.startswith("tools.") for name in imports)


def test_tools_response_shims_contain_no_scientific_definitions() -> None:
    for relative in (
        "tools/generativeqc_response/problem.py",
        "tools/generativeqc_response/operators.py",
        "tools/generativeqc_response/xc.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert definitions == set()


def test_tools_native_ks_no_longer_defines_rks_response() -> None:
    tree = ast.parse(
        (ROOT / "tools/generativeqc_response/native_ks.py").read_text(encoding="utf-8")
    )
    classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "NativeRKSResponse" not in classes
    assert "NativeUKSResponse" in classes
