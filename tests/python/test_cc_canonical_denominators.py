"""Exercise native admission/identity against an independent FP64 oracle."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner, write_df_cpu_headers

ROOT = Path(__file__).resolve().parents[2]


def test_native_canonical_denominator_admission(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    write_df_cpu_headers(tmp_path)
    executable = tmp_path / "denominators"
    compile_owner(
        compiler,
        tmp_path,
        [ROOT / "tests/native/test_cc_denominators.cpp", ROOT / "src/cc/solver.cpp"],
        executable,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )


def test_native_lambda_derived_preconditioner(tmp_path: Path) -> None:
    """The exact noninteracting response oracle covers both representations."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    write_df_cpu_headers(tmp_path)
    executable = tmp_path / "lambda_denominators"
    compile_owner(
        compiler,
        tmp_path,
        [
            ROOT / "tests/native/test_cc_lambda_preconditioner.cpp",
            ROOT / "src/cc/solver.cpp",
            ROOT / "src/cc/lambda_response.cpp",
            ROOT / "src/response/native_gmres.cpp",
        ],
        executable,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
