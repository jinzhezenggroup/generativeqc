"""Execute the complete production Hessian preparation with host resource doubles.

Only numerical providers/CUDA are replaced. Admission, ordering, alignment,
cleanup and option forwarding are the real owner's code, not copied expressions.
These are resource/lifetime controls, not real-device numerical qualification.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.geometry.dispersion_scalar import (
    emit_dispersion_scalar_header,
)

from tools.generativeqc_d3.generate_native_data import render as render_d3_tables

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/hessian_budget"


def _translation_unit(source: str) -> str:
    helpers = source[
        source.index("namespace generativeqc::dft {") : source.index("struct Range {")
    ]
    audit_start = source.index("struct DenseAssemblyAudit {")
    audit = source[audit_start : source.index("\n};", audit_start) + 3]
    start = source.index("struct CudaKsWb97mvHessian::Impl {")
    owner = source[start : source.index("  cudaError_t authenticate()", start)]
    guess_start = owner.index("  static cudaError_t guess_from_cpks_factor(")
    guess_body = owner.index(" noexcept {", guess_start) + len(" noexcept {")
    next_method = owner.index("  maps::Extents extents()", guess_body)
    owner = (
        owner[:guess_body]
        + "\n    return cudaErrorNotSupported;\n  }\n"
        + owner[next_method:]
    )
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
        + audit
        + "\n}\n"
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
    (tmp_path / "generated_dispersion_scalar.hpp").write_text(
        emit_dispersion_scalar_header()
    )
    (tmp_path / "d3_data.hpp").write_text(render_d3_tables())
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_method_parameters.py"),
            "--cpp-output",
            str(tmp_path / "generated_method_parameters.hpp"),
        ],
        check=True,
        capture_output=True,
    )
    executable = native_cxx.build_executable(
        [fixture],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            f"-I{FIXTURE}",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS mandatory-first Hessian admission" in result.stdout
