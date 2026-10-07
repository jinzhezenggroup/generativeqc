"""Public KS budgets preserve incumbent capacity before optional point panels.

Compile actual shape queries, native KS policy and allocator ledger against CUDA
allocation doubles. These are host policy/lifetime gates, not GPU endpoint tests.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc import ResourceBudget, resources_ks
from generativeqc.ks import KsOptions
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import plan_resources
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.dft.xc_point_batch_cuda import (
    emit_native_xc_point_batch_plan,
)
from generativeqc_compiler.xc.quadrature_cuda import _LAYOUT

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
CONTROLS = ("GENERATIVEQC_CUDA_XC_BATCH_TILES", "GENERATIVEQC_CUDA_XC_BATCH_BYTES")


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


CUDA_STUB = r"""
#pragma once
#include <cstddef>
#include <cstdint>
using cudaError_t = int;
using cudaStream_t = void*;
constexpr cudaError_t cudaSuccess=0, cudaErrorInvalidValue=1;
constexpr cudaError_t cudaErrorMemoryAllocation=2, cudaErrorInvalidDevice=3;
inline cudaError_t cudaGetDevice(int* d) { *d=0; return 0; }
inline cudaError_t cudaMalloc(void** p,std::size_t) {
  static std::uintptr_t next=4096;
  *p=reinterpret_cast<void*>(next+=4096); return 0;
}
inline cudaError_t cudaFree(void*) { return 0; }
inline cudaError_t cudaMallocAsync(void** p,std::size_t n,cudaStream_t) { return cudaMalloc(p,n); }
inline cudaError_t cudaFreeAsync(void* p,cudaStream_t) { return cudaFree(p); }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
"""


@pytest.fixture(scope="module")
def native_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Any:
    folder = tmp_path_factory.mktemp("ks-point-budget")
    (folder / "cuda_runtime_api.h").write_text(CUDA_STUB)
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    xc = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    direct = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    api = (ROOT / "src/api/c_api_resources.cpp").read_text()
    quadrature = (ROOT / "src/dft/cuda_quadrature.cu").read_text()
    start = ks.index("        const auto point_batch_size =")
    end = ks.index("        prepared_ao_work =", start)
    source = r"""
#include <algorithm>
#include <cassert>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "dft/cuda_ks.hpp"
#include "dft/cuda_ks_kernels.hpp"
#include "dft/cuda_xc.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "scf/generated_shell_task.hpp"
using namespace generativeqc;
namespace generativeqc::scf::cuda_execution {
using ShellPairDensityBounds = detail::GeneratedShellPairDensityBounds;
using GeneratedShellPairStream = detail::GeneratedShellPairStream;
}
namespace generativeqc::scf {
void direct_jk_require(bool yes,const char* message) {
  if (!yes) throw std::invalid_argument(message);
}
std::size_t direct_jk_product(std::size_t a,std::size_t b) { return runtime::size_mul(a,b); }
"""
    for signature in (
        "std::size_t cuda_direct_jk_device_bytes(",
        "std::size_t cuda_direct_coulomb_device_bytes(",
    ):
        source += _definition(direct, signature) + "\n"
    source += "}\n" + _LAYOUT.split("#if defined(__CUDACC__)")[0].replace(
        "#pragma once", ""
    )
    source += r"""
namespace generativeqc::dft {
namespace q = generated::quadrature;
using runtime::size_add;
using runtime::size_mul;
constexpr unsigned kCudaKsChunkCapacity=2, kSmallEigensolverLimit=16;
std::size_t sum(std::size_t a,std::size_t b) { return size_add(a,b); }
std::size_t product(std::size_t a,std::size_t b) { return size_mul(a,b); }
// This probe exercises only the actual PBE shape, without linking GPU programs.
struct Program {
  bool supported{true}, requires_gradient{true}, requires_tau{};
  CudaXcFastPathCapabilities fast_paths{};
};
Program cuda_xc_program_traits(std::uint32_t functional) { assert(functional==1); return {}; }
"""
    source += _definition(ks, "struct KsStateStorage") + ";\n"
    source += _definition(ks, "std::size_t cuda_ks_state_bytes(") + "\n"
    source += _definition(xc, "CudaXcLayout cuda_xc_layout_shape(") + "\n"
    for signature in (
        "std::size_t cuda_resident_grid_bytes(",
        "std::size_t cuda_quadrature_bytes(",
    ):
        source += _definition(quadrature, signature) + "\n"
    source += (
        "namespace cuda_xc_detail {\n" + emit_native_xc_point_batch_plan() + "\n}\n}\n"
    )
    source += 'extern "C" {\n'
    for signature in (
        "void* generativeqc_resource_ledger_create_v1(",
        "void generativeqc_resource_ledger_destroy_v1(",
        "int generativeqc_resource_ledger_bind_v1(",
        "int generativeqc_resource_ledger_read_v1(",
        "int generativeqc_resource_ks_cuda_v1(",
        "int generativeqc_resource_quadrature_cuda_v1(",
    ):
        source += _definition(api, signature) + "\n"
    source += (
        "}\n"
        + r"""
struct Xc {
  dft::CudaXcLayout layout = dft::cuda_xc_layout_shape(2,6,2,49152,1,false,256);
  dft::CudaXcPointBatchPlan batch;
  std::vector<void*>& allocations;
  explicit Xc(std::vector<void*>& owned) : allocations(owned) {}
  void prepare_point_batches(std::size_t tiles,std::size_t budget) {
    const auto plan = dft::cuda_xc_detail::prepare_point_batch_plan(layout,{},tiles,budget);
    if (plan.tiles==1) return;
    void* arena=nullptr;
    const auto status=runtime::resource_cuda_malloc(&arena,plan.device_bytes);
    if (status==cudaErrorMemoryAllocation) return;
    assert(status==cudaSuccess);
    allocations.push_back(arena);
    batch=plan;
  }
  const dft::CudaXcPointBatchPlan& point_batch_plan() const { return batch; }
};
std::size_t prepare_optional(std::vector<void*>& allocations) {
  Xc owner(allocations);
  auto* xc=&owner;
  dft::CudaKsResources resource;
  using dft::sum;
"""
        + ks[start:end]
        + r"""
  return resource.xc_device_bytes;
}
extern "C" void fleet(void* handle,std::size_t count,std::size_t mandatory,
                      std::size_t later_workspace,std::uint64_t* result) {
  using namespace runtime;
  assert(!active_device_resource_ledger);
  auto ledger = handle ? *static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle) : nullptr;
  active_device_resource_ledger=ledger;
  std::vector<void*> allocations;
  std::fill_n(result,5,0);
  for (std::size_t i=0;i<count;++i) {
    void* arena=nullptr;
    if (resource_cuda_malloc(&arena,mandatory)!=cudaSuccess) break;
    allocations.push_back(arena);
    result[1]+=prepare_optional(allocations);
    ++result[0];
  }
  if (result[0]==count && later_workspace) {
    void* arena=nullptr;
    if (resource_cuda_malloc(&arena,later_workspace)==cudaSuccess) {
      allocations.push_back(arena);
      result[2]=later_workspace;
    }
  }
  if (ledger) { result[3]=ledger->live; result[4]=ledger->rejected; }
  for (auto* arena:allocations) assert(resource_cuda_free(arena)==cudaSuccess);
  if (ledger) assert(ledger->live==0);
  active_device_resource_ledger.reset();
}
"""
    )
    unit = folder / "probe.cpp"
    unit.write_text(source)
    library = native_cxx.build_shared(
        [unit],
        folder / "probe.so",
        compile_args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
    )
    probe = ctypes.CDLL(str(library))
    probe.fleet.argtypes = (
        [ctypes.c_void_p] + [ctypes.c_size_t] * 3 + [ctypes.POINTER(ctypes.c_uint64)]
    )
    probe.fleet.restype = None
    return probe


def _request(
    native_probe: Any, monkeypatch: pytest.MonkeyPatch, count: int
) -> tuple[Any, Any]:
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})

    def matrix(nao: int, output: Any) -> int:
        assert nao == 2
        output._obj.value = 0
        return 0

    library = SimpleNamespace(
        generativeqc_ks_resource_inventory_version_v1=lambda: 1,
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_resource_ks_matrix_provider_cuda_v1=matrix,
        **{
            name: getattr(native_probe, name)
            for name in (
                "generativeqc_resource_ks_cuda_v1",
                "generativeqc_resource_quadrature_cuda_v1",
                "generativeqc_resource_ledger_create_v1",
                "generativeqc_resource_ledger_destroy_v1",
                "generativeqc_resource_ledger_bind_v1",
                "generativeqc_resource_ledger_read_v1",
            )
        },
    )
    request = resources_ks.ks_resource_request(
        [H2] * count,
        backend="cuda",
        library=library,
        ks_options=KsOptions(grid=GridSpec()),
    )
    return request, library


@pytest.mark.parametrize("count,workspace", [(1, 512 << 20), (1024, 128 << 20)])
@pytest.mark.parametrize("explicit_limit", [False, True])
def test_public_budget_preserves_fleet_and_later_workspace(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    workspace: int,
    explicit_limit: bool,
) -> None:
    for name in CONTROLS:
        monkeypatch.delenv(name, raising=False)
    request, library = _request(native_probe, monkeypatch, count)
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    if explicit_limit:
        plan = plan_resources(
            [request], ResourceBudget(device_bytes=plan.peak_bytes["device"])
        )
    row = json.loads(dict(request.candidates[0].decisions)["item_device_inventory"])[0]
    # Retained atomic weights add eight bytes/point beyond the legacy XC slot.
    # This test isolates optional panels; it does not certify the legacy inventory.
    mandatory = sum(row[key] for key in ("state", "xc", "coulomb")) + 8 * 49152
    ledger = NativeDeviceLedger(library, plan, owner="ks")
    try:
        if count == 1024:
            assert ledger.limit == 2199167612
        assert count * mandatory + workspace <= ledger.limit
        result = (ctypes.c_uint64 * 5)()
        native_probe.fleet(ledger.handle, count, mandatory, workspace, result)
        assert list(result) == [count, 0, workspace, count * mandatory + workspace, 0]
        assert ledger.to_dict()["live_bytes"] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "tiles,bytes_,expected",
    [(None, None, 1245184), ("0", None, 0), ("1", None, 0), (None, "0", 0)],
)
def test_unbudgeted_default_and_explicit_opt_out(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiles: str | None,
    bytes_: str | None,
    expected: int,
) -> None:
    for name, value in zip(CONTROLS, (tiles, bytes_), strict=True):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    result = (ctypes.c_uint64 * 5)()
    native_probe.fleet(None, 2, 1024, 0, result)
    assert list(result) == [2, 2 * expected, 0, 0, 0]
