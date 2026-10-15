"""Ownership guards for the installed semilocal RKS Hessian path."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import generativeqc.rks_hessian as production_hessian
import generativeqc.rks_hessian_directional as production_directional
import generativeqc.rks_hessian_integrals as production_integrals
import pytest

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


def _run_isolated(code: str, *arguments: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


_RKS_RESULT_CONTRACT = """
import numpy as np

components = directional.RKSXCHVPComponents(
    identity="components", xc_ao=np.array([[1.0, 2.0, 3.0]]),
    xc_grid=np.array([[4.0, -2.0, 1.0]]),
    xc_weight=np.array([[-1.0, 3.0, -2.0]]), grid_branch_identity="grid",
)
np.testing.assert_array_equal(components.total, [[4.0, 3.0, 2.0]])
assert not components.total.flags.writeable
result = molecular.RKSHessianResult(
    matrix=np.eye(3), identity="result", _diagnostics={"nested": {"values": [1]}},
)
reported = result.diagnostics
reported["nested"]["values"].append(2)
assert result.diagnostics == {"nested": {"values": [1]}}
for function, arguments in (
    (directional.directional_rks_response, (object(), None)),
    (directional.directional_rks_responses, (object(), None)),
    (directional.native_rks_xc_hvp_components, (object(), None)),
    (molecular.rks_hvp, (object(), None)),
    (molecular.rks_hvp_many, (object(), None)),
    (molecular.rks_hessian, (object(),)),
):
    try:
        function(*arguments)
    except TypeError as error:
        assert "NativeRKSResponse" in str(error), str(error)
    else:
        raise AssertionError("RKS endpoint accepted an invalid response operator")
"""


@pytest.mark.parametrize("first_import", ["production", "alias"])
def test_rks_module_aliases_preserve_contracts_in_both_import_orders(
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
    ("rks_hessian_directional", "rks_directional", (
        "DirectionalRKSBatchResponse", "DirectionalRKSResponse", "RKSXCHVPComponents",
        "directional_rks_response", "directional_rks_responses",
        "native_rks_xc_hvp_components",
    )),
    ("rks_hessian", "rks_molecular", (
        "RKSHessianResult", "RKSHVPBatchResult", "RKSHVPResult",
        "rks_hessian", "rks_hvp", "rks_hvp_many",
    )),
)
for owner, alias, _ in contracts:
    importlib.import_module(
        "generativeqc." + owner if sys.argv[2] == "production"
        else "tools.generativeqc_hessian." + alias
    )
facade = importlib.import_module("tools.generativeqc_hessian")
for owner, alias, names in contracts:
    production = importlib.import_module("generativeqc." + owner)
    compatibility = importlib.import_module("tools.generativeqc_hessian." + alias)
    assert compatibility is production
    for name in names:
        assert name in facade.__all__, name
        assert getattr(facade, name) is getattr(compatibility, name), name
directional = importlib.import_module("tools.generativeqc_hessian.rks_directional")
molecular = importlib.import_module("tools.generativeqc_hessian.rks_molecular")
"""
        + _RKS_RESULT_CONTRACT,
        first_import,
    )


def test_rks_results_and_rejection_execute_without_repository_tools() -> None:
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
            raise AssertionError("production RKS Hessian imported " + fullname)
        return None
sys.meta_path.insert(0, BlockTools())
from generativeqc import rks_hessian_directional as directional
from generativeqc import rks_hessian as molecular
from generativeqc import rks_hessian_integrals
"""
        + _RKS_RESULT_CONTRACT
        + "\nassert not attempts, attempts\n"
    )


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
