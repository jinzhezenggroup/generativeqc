"""CPU-safe regression checks for explicit CUDA validation targets."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

VALIDATORS = (
    "tools/validate_grid.py",
    "tools/validate_posthf.py",
    "tools/screen_weighted_eri.py",
    "tools/validate_second_ownership.py",
    "tools/validate_xc.py",
    "tools/validate_stationary_cuda_compile.py",
)


@pytest.mark.parametrize("path", VALIDATORS)
def test_validation_tool_has_no_implicit_blackwell_target(path: str) -> None:
    """A generic validator must not compile for 5090 by accident."""
    source = (ROOT / path).read_text(encoding="utf-8")
    ast.parse(source, filename=path)
    assert 'default="sm_120"' not in source
    assert 'cuda_target_info("sm_120")' not in source
    assert '"-arch=sm_120"' not in source
    assert "12 if cuda else 0" not in source


def test_native_second_probe_propagates_both_compute_capability_digits() -> None:
    """The test driver's runtime identity must agree with its NVCC target."""
    source = (ROOT / "tools/validate_second_ownership.py").read_text(encoding="utf-8")
    assert "@MAJOR@, @MINOR@" in source
    assert '"@MAJOR@": target.compute_capability_major if target else 0' in source
    assert '"@MINOR@": target.compute_capability_minor if target else 0' in source
    assert 'f"-arch={target.architecture}"' in source


def test_pinned_ci_resource_jobs_pass_explicit_cuda_target() -> None:
    """Hardware-specific resource receipts remain explicit, not a global default."""
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '--functionals PBE --nvcc "$CUDACXX" --target sm_120' in workflow
    assert "--architecture sm_120" in workflow
