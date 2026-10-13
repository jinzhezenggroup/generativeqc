"""A refused optional shell lease must retain canonical GPU force capability."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)
@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe0-uks"))
def test_canonical_force_survives_shell_stack_refusal(method: str) -> None:
    """Compare separate and combined derivatives against independent Libcint.

    A fresh CUDA process matters: a previously grown context stack would hide
    the admission failure. Reserve memory without touching it, leaving 8 GiB
    for the ordinary SCF owner; the optional 64-KiB/thread shell stack cannot
    fit on sm_120. The admitted canonical derivative buffers still do fit.
    """
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), method],
        timeout=240,
    )
    if result.returncode == 77:
        pytest.skip("the shell stack pressure fixture is qualified on sm_120")
    result.check_returncode()


def _isolated_probe(method: str) -> None:
    import cupy as cp
    from test_cuda_hybrid_snapshot import (
        test_separate_full_range_derivatives_match_libcint,
    )

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    properties = cp.cuda.runtime.getDeviceProperties(0)
    if (properties["major"], properties["minor"]) != (12, 0):
        raise SystemExit(77)
    free, _ = cp.cuda.runtime.memGetInfo()
    available = 8 << 30
    if free <= available:
        raise RuntimeError(
            "the isolated fixture requires more than 8 GiB initially free"
        )
    # Keep the reservation alive across the complete fixed-density oracle test.
    reservation = cp.cuda.alloc(free - available)
    with pytest.MonkeyPatch.context() as patch:
        test_separate_full_range_derivatives_match_libcint(
            method, "spherical", "0", True, "bounded", patch
        )
    del reservation


if __name__ == "__main__":
    _isolated_probe(sys.argv[1])
