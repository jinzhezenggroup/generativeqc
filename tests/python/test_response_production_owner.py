"""Ownership guards for installed closed-shell response contracts and operators."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import generativeqc.response_operator as production_operator
import generativeqc.response_problem as production_problem
import generativeqc.response_solver as production_solver
import generativeqc.response_xc as production_xc
import generativeqc.rks_response as production_rks
import pytest

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
    assert native_ks._NativeKSXCKernel is production_rks._NativeRKSXCKernel
    assert issubclass(
        native_ks._NativeKSXCKernel,
        production_xc.FixedDensityXCDerivativeKernel,
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


def _run_isolated(code: str, *arguments: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


_RESPONSE_ALGEBRA = """
from types import SimpleNamespace
import numpy as np

layout = problem.RotationLayout((0,), (1, 2))
np.testing.assert_array_equal(layout.pack([[2.0, -1.0]]), [2.0, -1.0])
np.testing.assert_array_equal(
    layout.density_matrix([2.0, -1.0]),
    [[0.0, 4.0, -2.0], [4.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
)
np.testing.assert_array_equal(
    layout.generator_matrix([2.0, -1.0]),
    [[0.0, 2.0, -1.0], [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
)
reference = SimpleNamespace(
    algorithm="RHF", converged=True, nmo=3, nocc=1,
    identity="synthetic-reference", electron_count=2,
)
snapshot = problem.ResponseProblem(reference, layout, "rhf", "operator", "model")
operator = operators.DenseMatrixResponseOperator(
    snapshot, [[3.0, 1.0], [0.0, 2.0]], backend_identity="synthetic-dense",
)
np.testing.assert_array_equal(operator.apply([2.0, -1.0]), [5.0, -2.0])
np.testing.assert_array_equal(operator.apply_transpose([2.0, -1.0]), [6.0, 0.0])
try:
    layout.validate_vector([1.0])
except ValueError as error:
    assert "shape" in str(error)
else:
    raise AssertionError("response layout accepted a wrong-sized vector")
try:
    problem.RotationLayout((0,), (1,), ("alpha", "beta"))
except problem.ResponseUnsupported:
    pass
else:
    raise AssertionError("restricted response layout accepted two spin blocks")
"""


@pytest.mark.parametrize("first_import", ["production", "shim"])
def test_response_shims_preserve_public_objects_in_both_import_orders(
    first_import: str,
) -> None:
    _run_isolated(
        """
import importlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [str(root / "python"), str(root)]
contracts = (
    ("response_problem", "problem", (
        "ResponseCompatibilityError", "ResponseProblem", "ResponseSolveError",
        "ResponseUnsupported", "RotationLayout",
    )),
    ("response_operator", "operators", (
        "CPKSResponseOperator", "DenseMatrixResponseOperator", "RHFResponseOperator",
        "cpks_operator_identity", "rhf_operator_identity", "validate_rotation_layout",
    )),
    ("response_xc", "xc", (
        "FixedDensityXCDerivativeKernel", "density_feature_response",
    )),
)
for production_name, shim_name, _ in contracts:
    importlib.import_module(
        "generativeqc." + production_name if sys.argv[2] == "production"
        else "tools.generativeqc_response." + shim_name
    )
for production_name, shim_name, names in contracts:
    production_name = "generativeqc." + production_name
    shim_name = "tools.generativeqc_response." + shim_name
    production = importlib.import_module(production_name)
    shim = importlib.import_module(shim_name)
    assert set(names) <= set(shim.__all__)
    for name in names:
        assert getattr(shim, name) is getattr(production, name), name
problem = importlib.import_module("tools.generativeqc_response.problem")
operators = importlib.import_module("tools.generativeqc_response.operators")
"""
        + _RESPONSE_ALGEBRA,
        first_import,
    )


def test_production_response_imports_and_algebra_execute_without_tools() -> None:
    _run_isolated(
        """
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
attempts = []
class BlockTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tools" or fullname.startswith("tools."):
            attempts.append(fullname)
            raise AssertionError("production response imported " + fullname)
        return None
sys.meta_path.insert(0, BlockTools())
from generativeqc import response_problem as problem
from generativeqc import response_operator as operators
from generativeqc import response_solver, response_xc, rks_response
"""
        + _RESPONSE_ALGEBRA
        + "\nassert not attempts, attempts\n"
    )


def test_tools_native_ks_no_longer_defines_rks_response() -> None:
    tree = ast.parse(
        (ROOT / "tools/generativeqc_response/native_ks.py").read_text(encoding="utf-8")
    )
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "NativeRKSResponse" not in classes
    assert "NativeUKSResponse" in classes
