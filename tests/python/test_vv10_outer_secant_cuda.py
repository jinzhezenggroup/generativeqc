"""Independent device algebra gates for the optional outer fixed-point proposal."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_outer_secant_rejects_invalid_history_and_matches_scalar_oracle(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Real kernels must reject noise, divergence, large steps and nonfinite data."""
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires a Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import _native

    nvcc = shutil.which("nvcc")
    assert nvcc, "CUDA toolkit required"
    cuda = Path(nvcc).resolve().parents[1]
    library = Path(_native.load_library(device="cuda")._name).resolve()
    executable = tmp_path / "probe"
    native_cxx.build_executable(
        [ROOT / "tests/native/vv10_outer_secant_probe.cpp"],
        executable,
        compile_args=(
            "-std=c++17",
            "-O2",
            f"-I{ROOT / 'src'}",
            f"-I{cuda / 'include'}",
        ),
        link_args=(
            str(library),
            f"-Wl,-rpath,{library.parent}",
            f"-L{cuda / 'lib64'}",
            "-lcudart",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "seven independent secant admission and Fock cases passed" in result.stdout
