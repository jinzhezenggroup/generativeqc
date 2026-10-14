"""Execute the method-owned retry gate; never retry driver/token/resource errors."""

import subprocess
from pathlib import Path
from typing import Any

from test_stationary_task_work_budget import _block

ROOT = Path(__file__).resolve().parents[2]


def test_complete_gradient_retry_is_bounded_and_numerical_only(
    tmp_path: Path, native_cxx: Any
) -> None:
    block = _block(
        (ROOT / "src/methods/dft_method.cpp").read_text(),
        "if (status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE && precision &&",
    )
    source = tmp_path / "retry.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "scf/direct_force_precision.hpp"
using generativeqc::scf::DirectForcePrecisionRequest;
struct Controller {
  int calls{};
  generativeqc_status result{};
  generativeqc_status gradient(int,int,int,bool,std::vector<double>& output,std::string&,
                               bool,DirectForcePrecisionRequest* precision) {
    assert(!precision);
    ++calls;
    if (result==GENERATIVEQC_STATUS_SUCCESS) output={2.0};
    return result;
  }
};
bool qualified_spatial_force_default() { return true; }
int main() {
  for (auto initial : {GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                       GENERATIVEQC_STATUS_INVALID_ARGUMENT, GENERATIVEQC_STATUS_CUDA_ERROR,
                       GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_NOT_IMPLEMENTED})
  for (auto retry : {GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                     GENERATIVEQC_STATUS_CUDA_ERROR})
  for (unsigned mode=0;mode<4;++mode) for (bool supplied:{false,true}) {
    generativeqc_status status=initial;
    Controller controller{0,retry};
    auto* stationary_controller_=&controller;
    DirectForcePrecisionRequest request;
    request.mixed_geometry=bool(mode&1);
    request.mixed_grid=bool(mode&2);
    auto* precision=supplied?&request:nullptr;
    std::vector<double> output{123.0};
    std::string detail="original failure";
    int density=0,weights=0,fock_=0;
    bool current=true;
"""
        + block
        + r"""
    const bool expected=initial==GENERATIVEQC_STATUS_NUMERICAL_FAILURE && supplied && mode;
    assert(controller.calls==int(expected));
    assert(request.geometry_retries==unsigned(expected));
    assert(status==(expected?retry:initial));
    assert(output[0]==(expected && retry==GENERATIVEQC_STATUS_SUCCESS?2.0:123.0));
  }
}
"""
    )
    binary = native_cxx.build_executable(
        [source],
        tmp_path / "retry",
        compile_args=(
            "-std=c++17",
            "-O2",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_retry_reset_discards_numeric_status_but_fences_and_preserves_driver_errors(
    tmp_path: Path, native_cxx: Any
) -> None:
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    helper = _block(header, "void discard_geometry(Owner& p)")
    source = tmp_path / "discard.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <stdexcept>
struct Owner { bool geometry_pending{}; void* geometry_stream{}; int synchronizations{}; };
int runtime_error=0, fences=0;
void* expected_stream=nullptr;
int cudaStreamSynchronize(void* stream) { assert(stream==expected_stream); ++fences; return runtime_error; }
void cuda_check(int status) { if (status) throw std::runtime_error("driver failure"); }
"""
        + helper
        + r"""
int main() {
  int storage=0;
  expected_stream=&storage;
  Owner owner{true,expected_stream,0};
  discard_geometry(owner);
  assert(fences==1 && owner.synchronizations==1 && !owner.geometry_pending && !owner.geometry_stream);
  discard_geometry(owner);
  assert(fences==1);
  owner.geometry_pending=true; owner.geometry_stream=expected_stream; runtime_error=1;
  try { discard_geometry(owner); return 1; }
  catch (const std::runtime_error&) {}
  assert(fences==2 && owner.synchronizations==1);
}
"""
    )
    binary = native_cxx.build_executable(
        [source], tmp_path / "discard", compile_args=("-std=c++17", "-O2")
    )
    subprocess.run([str(binary)], check=True, timeout=10)
