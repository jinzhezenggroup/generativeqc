"""Omitted programs cannot execute from an explicitly bounded CUDA bundle."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from test_libxc_automatic_cuda_xc import native_xc_driver
from test_libxc_automatic_point_native import NAMES

__all__ = ("native_xc_driver",)
if TYPE_CHECKING:
    from pathlib import Path

PACKAGED = ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA")
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_BUNDLE_TEST") != "1",
    reason="requires the exact three-registration CUDA bundle in finite Slurm",
)


@pytest.mark.parametrize("name", tuple(name for name in NAMES if name not in PACKAGED))
@pytest.mark.parametrize("spins", (1, 2))
def test_omitted_registration_has_no_native_xc_fallback(
    name: str, spins: int, native_xc_driver: Path
) -> None:
    """Package omission is not an alias, runtime compile, or CPU science path."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    completed = subprocess.run(
        [
            str(native_xc_driver),
            "--automatic-xc-json",
            str(automatic_functional_code(name)),
            str(spins),
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode != 0
    assert "missing automatic AOT entry" in completed.stderr
