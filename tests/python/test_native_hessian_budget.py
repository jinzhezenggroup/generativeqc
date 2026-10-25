"""Execute the complete production Hessian preparation with host resource doubles.

Only numerical providers/CUDA are replaced. Admission, ordering, alignment,
cleanup and option forwarding are the real owner's code, not copied expressions.
These are resource/lifetime controls, not real-device numerical qualification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/hessian_budget"


def _translation_unit(source: str) -> str:
    helpers = source[
        source.index("namespace generativeqc::dft {") : source.index("struct Range {")
    ]
    start = source.index("struct CudaKsWb97mvHessian::Impl {")
    owner = source[start : source.index("  cudaError_t authenticate()", start)]
    # Numerical methods are isolated; keep all real storage fields, construction,
    # destruction and resource accounting, then the entire unmodified prepare().
    owner += """
  cudaError_t authenticate() { return cudaSuccess; }
  cudaError_t gate() { return cudaSuccess; }
  cudaError_t sum_weights(const double*, const double*, std::size_t) { return cudaSuccess; }
};
"""
    start = source.index("CudaKsWb97mvHessian::CudaKsWb97mvHessian()")
    prepare = source[
        start : source.index("cudaError_t CudaKsWb97mvHessian::apply(", start)
    ]
    receipt = """
CudaKsWb97mvHessianWork CudaKsWb97mvHessian::work() const noexcept {
  auto result = impl_->receipt;
  result.grid = impl_->grid.work();
  result.one_electron = impl_->one_electron.work();
  return result;
}
} // namespace generativeqc::dft
"""
    return (
        '#include "providers.hpp"\n'
        + helpers
        + "}\n"
        + owner
        + prepare
        + receipt
        + (FIXTURE / "probe.cpp").read_text()
    )


def test_hessian_reserves_every_mandatory_consumer(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = ROOT / "src/dft/cuda_ks_wb97mv_hessian.cu"
    fixture = tmp_path / "hessian_budget.cpp"
    fixture.write_text(_translation_unit(source.read_text()))
    executable = native_cxx.build_executable(
        [fixture],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            f"-I{FIXTURE}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS mandatory-first Hessian admission" in result.stdout
