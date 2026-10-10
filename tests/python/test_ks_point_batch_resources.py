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
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_point_dispatch
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.dft.xc_point_batch_cuda import (
    emit_native_xc_point_batch_plan,
)
from generativeqc_compiler.xc.quadrature_cuda import _LAYOUT
from tools.generate_xc_split_hybrid_registry import emit_registry

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
inline cudaError_t cudaSetDevice(int) { return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA allocation double"; }
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
    (folder / "cublas_v2.h").write_text("#pragma once\nusing cublasHandle_t = void*;\n")
    (folder / "generated_split_hybrid_registry.cuh").write_text(emit_registry())
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    diis = (ROOT / "src/scf/cuda/scf_diis_kernels.cu").read_text()
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
#include <utility>
#include <vector>
#include "dft/cuda_ks.hpp"
#include "dft/cuda_ks_kernels.hpp"
#include "dft/cuda_xc.hpp"
#include "dft/semilocal_family.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#define __host__
#define __device__
#include "scf/cuda/direct_md_j.hpp"
#include "scf/cuda/matrix_library.hpp"
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
    source += (
        "}\nnamespace generativeqc::scf::cuda_execution {\n"
        + _definition(diis, "bool ordered_incremental_diis_gram_requested(")
        + "\n"
        + _definition(diis, "bool incremental_diis_gram_requested(")
        + "\n}\n"
    )
    source += _LAYOUT.split("#if defined(__CUDACC__)")[0].replace("#pragma once", "")
    source += r"""
namespace generativeqc::dft {
namespace q = generativeqc::generated::quadrature;
using scf::cuda_execution::MatrixLibraryOwner;
using runtime::size_add;
using runtime::size_mul;
constexpr unsigned kCudaKsChunkCapacity=2, kSmallEigensolverLimit=16;
std::size_t sum(std::size_t a,std::size_t b) { return size_add(a,b); }
std::size_t product(std::size_t a,std::size_t b) { return size_mul(a,b); }
"""
    source += _definition(xc, "struct CudaXcProgramTraits") + ";\n"
    source += _definition(xc, "CudaXcProgramTraits cuda_xc_program_traits(") + "\n"
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
    source += "namespace generativeqc::dft { namespace cuda_xc_detail {\n"
    source += (
        _definition(
            emit_native_xc_point_dispatch(),
            "CudaXcPointCapabilities resolve_point_capabilities(",
        )
        + "\n}\n"
    )
    for signature in (
        "CudaXcExecutionCapabilities cuda_xc_execution_capabilities(",
        "CudaXcAoSelectionResources cuda_xc_ao_selection_resources(",
    ):
        source += _definition(xc, signature) + "\n"
    source += '}\nextern "C" {\n'
    for signature in (
        "int generativeqc_resource_tracking_begin_v1(",
        "int generativeqc_resource_tracking_end_v1(",
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
  void prepare_point_batches(std::size_t tiles,std::size_t budget,bool compact=false) {
    const auto plan = dft::cuda_xc_detail::prepare_point_batch_plan(layout,{},tiles,budget,compact);
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
    # Compile the allocation owner and quadrature retention block unchanged.
    owned = (ROOT / "src/runtime/cuda_resources.cuh").read_text()
    source += "namespace generativeqc::runtime {\n"
    for signature in ("inline void cuda_resource_check(", "class CudaDeviceScope"):
        source += _definition(owned, signature) + ";\n"
    for signature in ("struct BorrowedCudaBuffer", "class OwnedCudaBuffer"):
        source += "template<class T>\n" + _definition(owned, signature) + ";\n"
    source += "}\n"
    begin = quadrature.index(
        "  std::shared_ptr<runtime::OwnedCudaBuffer<double>> resident;"
    )
    end = quadrature.index("  runtime::OwnedCudaBuffer<double> storage", begin)
    source += (
        r"""
extern "C" void grid_lifetime(void* handle,std::size_t points,std::uint64_t* out) {
  using namespace runtime;
  assert(!active_device_resource_ledger);
  auto ledger=*static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle);
  active_device_resource_ledger=ledger;
  namespace q=generated::quadrature;
  struct {std::size_t points;} l{points};
  const bool retain_device=true;
  const int device=0;
"""
        + quadrature[begin:end]
        + r"""
  assert(resident_points==resident->get());
  assert(resident_weights==resident_points+3*points);
  assert(resident_atomic_weights==resident_weights+points);
  out[0]=ledger->live;
  std::shared_ptr<const void> borrower=resident;
  resident.reset();
  out[1]=ledger->live;
  borrower.reset();
  out[2]=ledger->live;
  active_device_resource_ledger.reset();
}
extern "C" int grid_shapes(std::size_t points,std::size_t spins,std::size_t functional,
                           std::size_t tile,std::size_t response,std::size_t mixed,
                           std::uint64_t* out) {
  try {
    const auto precision=mixed ? dft::CudaXcAoPrecision::Fp32ComputeFp64Storage
                               : dft::CudaXcAoPrecision::Fp64;
    const auto direct=dft::cuda_xc_layout_shape(2,6,2,points,functional,spins==2,
                                               tile,response,precision);
    const auto borrowed=dft::cuda_xc_layout_shape(2,6,2,points,functional,spins==2,
                                                 tile,response,precision,1,1,true);
    out[0]=direct.device_bytes;
    out[1]=borrowed.device_bytes;
    out[2]=dft::cuda_resident_grid_bytes(points);
    return 0;
  } catch (...) { return 1; }
}
"""
    )
    start = ks.index("    const char* ao_selection = std::getenv(")
    end = ks.index("    if (host_unfused &&", start)
    arena = ks.index("    resource.xc_device_bytes =")
    arena_end = ks.index(";", arena) + 1
    diagnostic = ks.index("        prepared_ao_work.requested = select_ao;")
    diagnostic_end = ks.index("\n      }", diagnostic)
    source += (
        r"""
extern "C" int active_policy(std::size_t spins, bool host_unfused, std::uint64_t* out) {
  using namespace dft;
  try {
    const auto xc_layout=cuda_xc_layout_shape(2,6,2,49152,1,spins==2,256,
                                             false,CudaXcAoPrecision::Fp64,1,1,true);
    CudaKsResources resource;
    CudaXcAoSelectionWork prepared_ao_work;
"""
        + ks[start:end]
        + ks[arena:arena_end]
        + ks[diagnostic:diagnostic_end]
        + r"""
    out[0]=select_ao; out[1]=admit_ao;
    out[2]=xc_layout.device_bytes; out[3]=resource.xc_device_bytes;
    out[4]=prepared_ao_work.requested; out[5]=prepared_ao_work.selected;
    out[6]=prepared_ao_work.discovery_ao_jet_values;
    out[7]=prepared_ao_work.point_ao_square_sum;
    out[8]=prepared_ao_work.reserved_device_bytes;
    return 0;
  } catch (const std::invalid_argument&) { return 1; }
}
extern "C" void active_fleet(void* handle,std::size_t count,std::size_t mandatory,
                             std::size_t force_bytes,std::size_t spins,
                             std::size_t cycles,std::uint64_t* out) {
  using namespace runtime;
  assert(!active_device_resource_ledger);
  auto ledger=*static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle);
  std::fill_n(out,6,0);
  std::vector<void*> arenas(count), panels;
  void* force=nullptr;
  for (std::size_t cycle=0;cycle<cycles;++cycle) {
    assert(generativeqc_resource_tracking_begin_v1(1)==0);
    assert(generativeqc_resource_ledger_bind_v1(handle)==0);
    for (auto*& arena:arenas) {
      if (arena) { assert(resource_cuda_free(arena)==cudaSuccess); arena=nullptr; }
      std::uint64_t policy[9]{};
      assert(active_policy(spins,false,policy)==0);
      const auto extra=policy[3]-policy[2];
      if (resource_cuda_malloc(&arena,mandatory+extra)!=cudaSuccess) break;
      out[1]+=extra+prepare_optional(panels);
      ++out[0];
    }
    if (out[0]==count*(cycle+1) && !force) {
      if (resource_cuda_malloc(&force,force_bytes)==cudaSuccess) out[2]=force_bytes;
    }
    out[3]=ledger->live;
    out[4]+=ledger->rejected;
    std::uint64_t peak{},samples{};
    assert(generativeqc_resource_tracking_end_v1(&peak,&samples)==0);
    assert(!active_device_resource_ledger && !ledger->active);
    assert(ledger->live==out[3]); // Charges survive observation scope changes.
    if (out[0]!=count*(cycle+1) || !force) break;
  }
  for (auto* arena:arenas) if (arena) assert(resource_cuda_free(arena)==cudaSuccess);
  for (auto* arena:panels) assert(resource_cuda_free(arena)==cudaSuccess);
  if (force) assert(resource_cuda_free(force)==cudaSuccess);
  out[5]=ledger->live;
}
"""
    )
    source += r"""
extern "C" std::size_t md_capacity(void* handle,bool reserve) {
  using namespace runtime;
  const auto previous=active_device_resource_ledger;
  active_device_resource_ledger=handle
      ? *static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle) : nullptr;
  const auto bytes=scf::cuda_direct_coulomb_device_bytes(1,24,3,12,30,1,reserve);
  active_device_resource_ledger=previous;
  return bytes;
}
"""
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
    probe.generativeqc_resource_ks_cuda_v1.argtypes = [ctypes.c_size_t] * 9 + [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    probe.generativeqc_resource_ks_cuda_v1.restype = ctypes.c_int
    probe.fleet.argtypes = (
        [ctypes.c_void_p] + [ctypes.c_size_t] * 3 + [ctypes.POINTER(ctypes.c_uint64)]
    )
    probe.fleet.restype = None
    probe.grid_lifetime.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    probe.grid_lifetime.restype = None
    probe.grid_shapes.argtypes = [ctypes.c_size_t] * 6 + [
        ctypes.POINTER(ctypes.c_uint64)
    ]
    probe.grid_shapes.restype = ctypes.c_int
    probe.active_policy.argtypes = [
        ctypes.c_size_t,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    probe.active_policy.restype = ctypes.c_int
    probe.active_fleet.argtypes = (
        [ctypes.c_void_p] + [ctypes.c_size_t] * 5 + [ctypes.POINTER(ctypes.c_uint64)]
    )
    probe.active_fleet.restype = None
    probe.md_capacity.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    probe.md_capacity.restype = ctypes.c_size_t
    return probe


def _request(
    native_probe: Any, monkeypatch: pytest.MonkeyPatch, count: int, **options: Any
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
        ks_options=options.pop("ks_options", KsOptions(grid=GridSpec())),
        **options,
    )
    return request, library


@pytest.mark.parametrize("count,workspace", [(1, 512 << 20), (1024, 128 << 20)])
@pytest.mark.parametrize("explicit_limit", [False, True])
@pytest.mark.parametrize("incremental_gram", ["0", "1"])
@pytest.mark.parametrize("reduction", [None, "cooperative", "ordered"])
def test_public_budget_preserves_fleet_and_later_workspace(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    workspace: int,
    explicit_limit: bool,
    incremental_gram: str,
    reduction: str | None,
) -> None:
    if reduction is None:
        monkeypatch.delenv(
            "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION", raising=False
        )
    else:
        monkeypatch.setenv(
            "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION", reduction
        )
    monkeypatch.setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM", incremental_gram)
    for name in CONTROLS:
        monkeypatch.delenv(name, raising=False)
    request, library = _request(native_probe, monkeypatch, count)
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    if explicit_limit:
        plan = plan_resources(
            [request], ResourceBudget(device_bytes=plan.peak_bytes["device"])
        )
    row = json.loads(dict(request.candidates[0].decisions)["item_device_inventory"])[0]
    mandatory = sum(row[key] for key in ("state", "xc", "coulomb"))
    ledger = NativeDeviceLedger(library, plan, owner="ks")
    try:
        assert count * mandatory + workspace <= ledger.limit
        result = (ctypes.c_uint64 * 5)()
        native_probe.fleet(ledger.handle, count, mandatory, workspace, result)
        assert list(result) == [count, 0, workspace, count * mandatory + workspace, 0]
        assert ledger.to_dict()["live_bytes"] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize("history", [1, 2, 8, 64])
@pytest.mark.parametrize("reduction", [None, "cooperative", "ordered"])
@pytest.mark.parametrize("spins", [1, 2])
def test_incremental_gram_charges_only_optional_physical_cache(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    history: int,
    spins: int,
    reduction: str | None,
) -> None:
    selector = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION"
    if reduction is None:
        monkeypatch.delenv(selector, raising=False)
    else:
        monkeypatch.setenv(selector, reduction)
    inventories = []
    selection = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM"
    for mode in (None, "0", "1"):
        if mode is None:
            monkeypatch.delenv(selection, raising=False)
        else:
            monkeypatch.setenv(selection, mode)
        request, _ = _request(
            native_probe,
            monkeypatch,
            1,
            diis_history=history,
            method="pbe-uks" if spins == 2 else "pbe-rks",
        )
        inventories.append(
            json.loads(dict(request.candidates[0].decisions)["item_device_inventory"])[
                0
            ]
        )
    expected = 8 * history * history if history >= 2 else 0
    assert inventories[0] == inventories[1]
    assert inventories[2]["state"] - inventories[0]["state"] == expected
    for key in ("xc", "coulomb"):
        assert inventories[2][key] == inventories[0][key]
    monkeypatch.setenv(selection, "1")
    monkeypatch.setenv(selector, "invalid")
    output = (ctypes.c_uint64 * 3)()
    status = native_probe.generativeqc_resource_ks_cuda_v1(
        2, 2, 2, 6, 49152, history, spins, 0, 256, output, 3
    )
    assert (status != 0) == (history >= 2)
    monkeypatch.setenv(selection, "0")
    assert (
        native_probe.generativeqc_resource_ks_cuda_v1(
            2, 2, 2, 6, 49152, history, spins, 0, 256, output, 3
        )
        == 0
    )
    monkeypatch.setenv(selection, "invalid")
    assert (
        native_probe.generativeqc_resource_ks_cuda_v1(
            2, 2, 2, 6, 49152, 8, spins, 0, 256, output, 3
        )
        != 0
    )


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
