"""Exercise real RKS HVP exports in fresh interpreters without native execution."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("order", ["function", "result", "implementation"])
@pytest.mark.parametrize("repeat", [1, 2])
def test_rks_hvp_export_is_independent_of_import_order(order: str, repeat: int) -> None:
    # Each interpreter starts with an empty import cache for the actual packages.
    # No export-table source fragments or fake implementation modules are needed.
    code = textwrap.dedent(
        """
        import importlib
        import sys

        order, repeat = sys.argv[1:]
        package = importlib.import_module("tools.generativeqc_hessian")
        if order == "function":
            from tools.generativeqc_hessian import rks_hvp
        elif order == "result":
            from tools.generativeqc_hessian import RKSHVPResult
        elif order == "implementation":
            importlib.import_module("generativeqc.rks_hessian")

        for _ in range(int(repeat)):
            from tools.generativeqc_hessian import RKSHVPResult, rks_hvp

            assert callable(rks_hvp)
            assert isinstance(RKSHVPResult, type)
            assert package.rks_hvp is rks_hvp
            assert package.RKSHVPResult is RKSHVPResult

            # The tools API exposes the installed production objects.
            owner = importlib.import_module("generativeqc.rks_hessian")
            assert rks_hvp is owner.rks_hvp
            assert RKSHVPResult is owner.RKSHVPResult
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, order, str(repeat)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
