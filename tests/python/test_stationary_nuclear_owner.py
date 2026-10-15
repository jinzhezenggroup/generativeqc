"""Ownership guards for the installed stationary nuclear-response layer."""

from __future__ import annotations

import ast
import subprocess
import sys
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


def _run_isolated(code: str, *arguments: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_stationary_nuclear_public_algebra_executes_without_tools() -> None:
    """Keep production imports and basic RHS/metric use independent of tools."""
    _run_isolated("""
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
attempts = []
class BlockTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tools" or fullname.startswith("tools."):
            attempts.append(fullname)
            raise AssertionError("production nuclear algebra imported " + fullname)
        return None
sys.meta_path.insert(0, BlockTools())
import numpy as np
from generativeqc.stationary_nuclear import (
    metric_density_response_mo,
    build_stationary_nuclear_rhs,
    build_rhf_nuclear_rhs,
)
overlap = np.array([[1.0, 2.0], [2.0, 3.0]])
np.testing.assert_array_equal(
    metric_density_response_mo(overlap, nocc=1), [[-2.0, -2.0], [-2.0, 0.0]]
)
frozen = np.array([[1.0, 2.0], [2.0, 4.0]])
# F_ai - (eps_i + eps_a) * S_ia / 2 = 2 + 0.5.
for build in (build_stationary_nuclear_rhs, build_rhf_nuclear_rhs):
    np.testing.assert_array_equal(
        build(frozen, overlap, np.zeros((2, 2)), [-1.0, 0.5], nocc=1), [[2.5]]
    )
assert not attempts, attempts
""")


@pytest.mark.parametrize("first_import", ["production", "shim"])
def test_response_shim_preserves_public_algebra_in_both_import_orders(
    first_import: str,
) -> None:
    """Public compatibility names are the actual production objects and APIs."""
    _run_isolated(
        """
import importlib
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
sys.path[:0] = [str(root / "python"), str(root)]
production_name = "generativeqc.stationary_nuclear"
shim_name = "tools.generativeqc_hessian.response"
importlib.import_module(production_name if sys.argv[2] == "production" else shim_name)
production = importlib.import_module(production_name)
shim = importlib.import_module(shim_name)
for name in (
    "metric_density_response_mo",
    "build_stationary_nuclear_rhs",
    "build_rhf_nuclear_rhs",
):
    assert getattr(shim, name) is getattr(production, name)
overlap = np.array([[1.0, 2.0], [2.0, 3.0]])
np.testing.assert_array_equal(
    shim.metric_density_response_mo(overlap, nocc=1), [[-2.0, -2.0], [-2.0, 0.0]]
)
frozen = np.array([[1.0, 2.0], [2.0, 4.0]])
for build in (shim.build_stationary_nuclear_rhs, shim.build_rhf_nuclear_rhs):
    np.testing.assert_array_equal(
        build(frozen, overlap, np.zeros((2, 2)), [-1.0, 0.5], nocc=1), [[2.5]]
    )
try:
    shim.metric_density_response_mo(np.zeros((2, 3)), nocc=1)
except ValueError as error:
    assert "square matrix" in str(error)
else:
    raise AssertionError("compatibility API accepted a non-square overlap matrix")
""",
        first_import,
    )
