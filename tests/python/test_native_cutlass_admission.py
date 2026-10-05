"""Host-only fault injection for CUTLASS admission, without an external SDK."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]

RUNTIME = r"""
#pragma once
#include <map>
#include <mutex>
#include "runtime/allocation_measurement.hpp"
#include "tensor/native_contraction.hpp"
using cudaStream_t = void*;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
constexpr int cudaDevAttrComputeCapabilityMajor = 1, cudaDevAttrComputeCapabilityMinor = 2;
struct cudaFuncAttributes {};
namespace fixture {
enum Failure { None, FirstLoad, SecondLoad, BeforeInfo, AfterInfo, RuntimeVersion };
struct State {
  int device{}, loads{}, infos{};
  Failure failure{None};
  std::size_t retained{};
  std::map<const void*, bool> loaded;
};
inline State state;
}
inline int cudaGetDevice(int* value) { *value = fixture::state.device; return 0; }
inline int cudaSetDevice(int value) { fixture::state.device = value; return 0; }
inline int cudaDeviceGetAttribute(int* value, int attr, int) {
  *value = attr == cudaDevAttrComputeCapabilityMajor ? 12 : 0; return 0;
}
inline int cudaStreamIsCapturing(cudaStream_t, int* value) { *value = 0; return 0; }
inline int cudaStreamSynchronize(cudaStream_t) { return 0; }
inline int cudaGetLastError() { return 0; }
inline int cudaRuntimeGetVersion(int* value) {
  *value = 12090; return fixture::state.failure == fixture::RuntimeVersion;
}
inline int cudaMemGetInfo(std::size_t* free, std::size_t* total) {
  auto& state = fixture::state;
  ++state.infos;
  *total = 1U << 24; *free = *total - state.retained;
  return (state.infos == 1 && state.failure == fixture::BeforeInfo) ||
         (state.infos == 2 && state.failure == fixture::AfterInfo);
}
template<class F> int cudaFuncGetAttributes(cudaFuncAttributes*, F function) {
  auto& state = fixture::state;
  ++state.loads;
  auto& loaded = state.loaded[reinterpret_cast<const void*>(function)];
  if (!loaded) { loaded = true; state.retained += 4096; }
  return (state.loads == 1 && state.failure == fixture::FirstLoad) ||
         (state.loads == 2 && state.failure == fixture::SecondLoad);
}
namespace generativeqc_tensor {
inline void cuda_check(int status) { if (status) throw std::runtime_error("injected CUDA failure"); }
}
namespace generativeqc::tensor {
struct AffineAuditView { std::size_t rank{}, shape[8]{}, strides[8]{}; };
template<class T> void audit_affine_contraction(const T*, AffineAuditView, std::size_t, int*) {}
}
"""

CUTLASS = r"""
#pragma once
#include "cutlass_test_runtime.hpp"
namespace cutlass {
enum class Status { kSuccess };
namespace layout { struct RowMajor {}; struct ColumnMajor {}; }
namespace arch { struct OpClassSimt {}; struct Sm50 {}; }
namespace epilogue::thread {
template<class T, int Count, class Accumulator, class Compute> struct LinearCombination {
  struct Params { T alpha, beta; };
};
}
template<class K> void Kernel() {}
namespace gemm {
template<int M, int N, int K> struct GemmShape {};
namespace threadblock { struct GemmBatchedIdentityThreadblockSwizzle {}; }
namespace device {
template<class A, class LA, class B, class LB, class C, class LC, class Acc,
         class Op, class Sm, class Tile, class Warp, class Instruction, class Epilogue,
         class Swizzle, int Stages, int AlignA, int AlignB>
struct GemmBatched {
  struct GemmKernel { static constexpr int kThreadCount = 128; struct SharedStorage { char bytes[8192]; }; };
  template<class T> struct Ref { T* pointer; int ld; void reset(T* value) { pointer = value; } };
  struct Shape { int m, n, k; };
  struct Arguments {
    Shape shape;
    Ref<const A> ref_A; std::int64_t stride_A;
    Ref<const B> ref_B; std::int64_t stride_B;
    Ref<const C> ref_C; std::int64_t stride_C;
    Ref<C> ref_D; std::int64_t stride_D;
    typename Epilogue::Params epilogue;
    int batches;
  };
  static Status can_implement(const Arguments&) { return Status::kSuccess; }
  static std::size_t get_workspace_size(const Arguments&) { return 0; }
  Status initialize(const Arguments&) { return Status::kSuccess; }
  Status update(const Arguments&) { return Status::kSuccess; }
  Status run(cudaStream_t) { return Status::kSuccess; }
};
}
}
}
"""

PROBE = r"""
#include "cutlass_test_owner.cuh"
#include <iostream>
using namespace generativeqc::tensor;
constexpr std::string_view identity =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
void require(bool value) { if (!value) throw std::runtime_error("CUTLASS admission regression"); }
template<class F> void throws(F action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("expected hard preparation failure");
}
ContractionRequest request(std::size_t m = 2, std::size_t n = 2, std::size_t k = 2,
                           std::size_t batch = 1, bool column = false,
                           PrecisionDtype dtype = PrecisionDtype::Fp64) {
  return {identity, identity, identity,
          {ContractionOperand::dense({3,0,2}, {batch,m,k}, dtype),
           ContractionOperand::dense({3,2,1}, {batch,k,n}, dtype),
           column ? ContractionOperand::dense({3,1,0}, {batch,n,m}, dtype)
                  : ContractionOperand::dense({3,0,1}, {batch,m,n}, dtype)},
          {dtype,dtype,dtype}, dtype};
}
bool prepare(CudaCutlassContraction& owner, const ContractionRequest& r,
             std::size_t bytes = 8192, cudaStream_t stream = nullptr) {
  return owner.prepare(r, stream, identity, 1U << 20, bytes);
}
int main() {
  // Every potentially retaining failure is hard and makes all warm retries hard.
  for (const auto failure : {fixture::FirstLoad, fixture::SecondLoad,
                            fixture::AfterInfo, fixture::RuntimeVersion}) {
    fixture::state = {}; fixture::state.failure = failure;
    CudaCutlassContraction owner;
    throws([&] { prepare(owner, request()); });
    require(owner.module_bytes() == 8192 && owner.preparations() == 0);
    fixture::state.failure = fixture::None;
    const int loads = fixture::state.loads;
    for (const auto reservation : {1U, 8192U, 16384U})
      throws([&] { prepare(owner, request(), reservation); });
    owner.release();
    throws([&] { prepare(owner, request()); });
    require(fixture::state.loads == loads);
  }
  // Failure before loading has not retained a new module and can safely retry.
  {
    fixture::state = {}; fixture::state.failure = fixture::BeforeInfo;
    CudaCutlassContraction owner;
    throws([&] { prepare(owner, request()); });
    require(!owner.module_bytes() && !fixture::state.loads);
    fixture::state.failure = fixture::None;
    require(prepare(owner, request()));
  }
  {
    fixture::state = {};
    CudaCutlassContraction owner;
    throws([&] { prepare(owner, request(), 1); });
    require(owner.module_bytes() == 8192);
    throws([&] { prepare(owner, request(), 16384); });
  }
  // Two successful load increments may individually fit but exceed the total.
  {
    fixture::state = {};
    CudaCutlassContraction owner;
    require(prepare(owner, request())); owner.release();
    throws([&] { prepare(owner, request(2,2,2,1,false,PrecisionDtype::Fp32)); });
    require(owner.module_bytes() == 16384 && owner.preparations() == 1);
    throws([&] { prepare(owner, request(), 32768); });
  }
  {
    fixture::state = {};
    CudaCutlassContraction owner;
    require(prepare(owner, request())); owner.release();
    throws([&] { prepare(owner, request(2,2,2,1,true)); });
    require(owner.module_bytes() == 12288);
  }
  // A sufficient envelope admits different kernels and stream reuse.
  {
    fixture::state = {};
    CudaCutlassContraction owner;
    require(prepare(owner, request(), 16384));
    const auto host = owner.host_bytes(); owner.release();
    require(!owner.prepare(request(), nullptr, identity, host - 1, 16384));
    require(owner.prepare(request(), (void*)1, identity, host, 16384)); owner.release();
    require(prepare(owner, request(2,2,2,1,false,PrecisionDtype::Fp32), 16384));
    owner.release();
    require(owner.module_bytes() == 16384 && fixture::state.retained == 16384);
    require(!prepare(owner, request(), 8192));
    fixture::state.device = 1;
    throws([&] { prepare(owner, request(), 16384); });
    fixture::state.device = 0;
    constexpr std::string_view other =
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    throws([&] { owner.prepare(request(), nullptr, other, 1U << 20, 16384); });
    require(owner.module_bytes() == 16384);
  }
  const auto check_admission = [](const ContractionRequest& r, bool expected) {
    fixture::state = {};
    CudaCutlassContraction owner;
    require(prepare(owner, r) == expected);
    require(expected || (!fixture::state.loads && !owner.module_bytes()));
  };
  // CUTLASS adds the tile before subtracting one, all in signed int.
  const std::size_t limit = std::numeric_limits<int>::max();
  for (bool column : {false, true}) {
    check_admission(request(column ? 2 : limit-32, column ? limit-32 : 2, 1, 1, column), true);
    check_admission(request(column ? 2 : limit-31, column ? limit-31 : 2, 1, 1, column), false);
    check_admission(request(column ? 2 : limit, column ? limit : 2, 1, 1, column), false);
    check_admission(request(2,2,limit-8,1,column), true);
    check_admission(request(2,2,limit-7,1,column), false);
    check_admission(request(2,2,limit,1,column), false);
    const std::size_t bound = 65535 * 64;
    check_admission(request(column ? bound : 2, column ? 2 : bound, 1,1,column), true);
    check_admission(request(column ? bound+1 : 2, column ? 2 : bound+1, 1,1,column), false);
  }
  check_admission(request(1,1,1,65535), true);
  check_admission(request(1,1,1,65536), false);
  std::cout << "CUTLASS retained-load failure, cumulative reservation and signed-boundary gates passed\n";
}
"""


def test_cutlass_host_admission_and_retained_failures(tmp_path: Path) -> None:
    """Exercise the actual owner around fake CUDA load/metadata calls only."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    (tmp_path / "cutlass/gemm/device").mkdir(parents=True)
    (tmp_path / "cutlass/gemm/device/gemm_batched.h").write_text(CUTLASS)
    (tmp_path / "cutlass/version.h").write_text("#define CUTLASS_VERSION 30902\n")
    (tmp_path / "cutlass_test_runtime.hpp").write_text(RUNTIME)
    source = (ROOT / "src/tensor/cuda_cutlass.cuh").read_text()
    source, launches = re.subn(r"<<<.*?>>>", "", source, flags=re.DOTALL)
    assert launches == 1
    source = source.replace(
        '"tensor/cuda_affine_audit.cuh"', '"cutlass_test_runtime.hpp"'
    )
    (tmp_path / "cutlass_test_owner.cuh").write_text(source)
    unit, binary = tmp_path / "cutlass_admission.cpp", tmp_path / "cutlass_admission"
    unit.write_text(PROBE)
    compile_owner(compiler, tmp_path, [unit], binary)
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
