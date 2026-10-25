"""Compile the exact host scheduler and inject only optional metadata OOM.

No CUDA scheduling/numerical behavior is simulated or qualified by this test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/one_electron_schedule"


def test_optional_host_metadata_oom_preserves_scalar_plan(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = (ROOT / "src/integrals/cuda_one_electron_response.cu").read_text()
    host = source[: source.index("unsigned blocks(")]
    host = host.replace('#include "response/nuclear_integral_response.cuh"', "")
    host = host.replace("namespace ad = generativeqc_nuclear_integral_ad;", "")
    start = source.index("struct CudaOneElectronResponse::Impl {")
    workspace = source[
        start : source.index("cudaError_t CudaOneElectronResponse::apply(", start)
    ]
    workspace = re.sub(r"<<<.*?>>>", "", workspace, flags=re.DOTALL)
    fixture = tmp_path / "one_electron_schedule.cpp"
    fixture.write_text(
        "#include <cassert>\n#include <cstdlib>\n#include <iostream>\n"
        + host
        + '} }\n#include "runtime.hpp"\nnamespace generativeqc::integrals {\n'
        + workspace
        + "}\n"
        + (FIXTURE / "probe.cpp").read_text()
    )
    executable = native_cxx.build_executable(
        [fixture],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            f"-I{ROOT / 'tests/native/fixtures/hessian_budget'}",
            f"-I{ROOT / 'src'}",
            f"-I{FIXTURE}",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS optional host metadata fallback" in result.stdout
