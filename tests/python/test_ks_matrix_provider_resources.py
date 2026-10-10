"""Actual KS matrix-owner lifetime and private resource ABI with host doubles.

The ABI and owner definitions are compiled unchanged; only CUDA/cuBLAS calls
are doubled. This does not qualify GPU arithmetic, capture, or performance.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc import ResourceBudget, resources_ks
from generativeqc_compiler.common.resources import plan_resources

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
ALLOWANCE = 96 << 20
H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]

PREFIX = r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
using cudaStream_t=void*;
using cublasHandle_t=void*;
using generativeqc_status=int;
constexpr int GENERATIVEQC_STATUS_SUCCESS=0, GENERATIVEQC_STATUS_INVALID_ARGUMENT=1;
constexpr int GENERATIVEQC_STATUS_CUDA_ERROR=2, GENERATIVEQC_STATUS_OUT_OF_MEMORY=3;
constexpr int cudaSuccess=0, CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_ALLOC_FAILED=3;
constexpr int CUBLAS_STATUS_INTERNAL_ERROR=7;
constexpr int cudaStreamCaptureStatusNone=0, CUBLAS_POINTER_MODE_HOST=0, CUBLAS_PEDANTIC_MATH=0;
using cudaStreamCaptureStatus=int;
int device=1, memcalls=0, destroys=0, creates=0, live=0, syncs=0;
int destroy_failures=0, create_status=0, setup_status=0, capture=0;
std::size_t retained=1ULL<<20;
int cudaStreamIsCapturing(cudaStream_t,int* s) {*s=capture; return 0;}
int cudaGetDevice(int* d) {*d=device; return 0;}
int cudaSetDevice(int d) {device=d; return 0;}
int cudaStreamSynchronize(cudaStream_t) {assert(device==1); ++syncs; return 0;}
int cudaGetLastError() {return 0;}
int cudaMemGetInfo(std::size_t* free,std::size_t* total) {
  *total=1024ULL<<20; *free=(512ULL<<20)-(memcalls++%2 ? retained : 0); return 0;
}
int cublasCreate(void** h) {
  ++creates; *h=nullptr;
  if (create_status) return create_status;
  assert(device==1); *h=reinterpret_cast<void*>(2); ++live; return 0;
}
int cublasDestroy(void* h) {
  assert(device==1 && live==1 && h==reinterpret_cast<void*>(2)); ++destroys;
  if (destroy_failures) {--destroy_failures; return CUBLAS_STATUS_INTERNAL_ERROR;}
  --live; return 0;
}
int cublasSetStream(void*,void*) {return setup_status;}
int cublasSetPointerMode(void*,int) {return 0;}
int cublasSetMathMode(void*,int) {return 0;}
int cublasSetWorkspace(void*,void* p,std::size_t n) {assert(!p&&!n); return 0;}
namespace generativeqc::runtime { inline std::mutex allocation_measurement_mutex; }
namespace generativeqc::scf::cuda_execution {
int cuda_status(int s) {return s ? GENERATIVEQC_STATUS_CUDA_ERROR : 0;}
int blas_status(int s) {return s ? GENERATIVEQC_STATUS_CUDA_ERROR : 0;}
}
#define GENERATIVEQC_HAS_CUDA 1
"""

DRIVER = r"""
int main(int argc,char** argv) {
  assert(argc==2);
  const std::string mode=argv[1];
  using generativeqc::scf::cuda_execution::MatrixLibraryOwner;
  const bool fail_destroy=mode=="oversized-destroy" || mode=="setup-destroy";
  if (mode=="oversized" || mode=="oversized-destroy") retained=97ULL<<20;
  if (mode=="setup" || mode=="setup-destroy") setup_status=CUBLAS_STATUS_INTERNAL_ERROR;
  if (mode=="allocation") create_status=CUBLAS_STATUS_ALLOC_FAILED;
  if (mode=="create-error") create_status=CUBLAS_STATUS_INTERNAL_ERROR;
  if (mode=="capture") capture=1;
  if (fail_destroy) destroy_failures=1;
  MatrixLibraryOwner owner;
  const auto stream=reinterpret_cast<void*>(1);
  const int n=mode=="small" ? 16 : 17;
  const auto status=owner.prepare(stream,n);
  const bool error=fail_destroy || mode=="setup" || mode=="create-error";
  assert(status==(mode=="capture" ? GENERATIVEQC_STATUS_INVALID_ARGUMENT :
                 error ? GENERATIVEQC_STATUS_CUDA_ERROR : GENERATIVEQC_STATUS_SUCCESS));
  const bool enabled=mode=="admit" || fail_destroy;
  assert(owner.library_enabled()==enabled);
  assert(live==(enabled?1:0));
  if (fail_destroy) assert(destroys==1 && owner.view().blas_!=nullptr);
  if (mode=="oversized") assert(destroys==1 && owner.retained_bytes()==0);
  if (mode=="admit") assert(owner.retained_bytes()==retained);
  if (mode=="small" || mode=="capture") assert(creates==0);
  if (mode!="capture") assert(owner.prepare(stream,n)==GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  device=2;
  owner.reset();
  assert(device==2 && live==0 && !owner.library_enabled());
  if (fail_destroy) assert(destroys==2 && syncs==1);
  const auto completed_destroys=destroys;
  owner.reset();
  assert(destroys==completed_destroys);
  return 0;
}
"""


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def matrix_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("ks-matrix-provider")
    header = (ROOT / "src/scf/cuda/matrix_library.hpp").read_text()
    source = (ROOT / "src/scf/cuda/matrix_library.cpp").read_text()
    api = (ROOT / "src/api/c_api_resources.cpp").read_text()
    namespace = "namespace generativeqc::scf::cuda_execution {"
    header = header[header.index(namespace) : header.index("/** Use the resolved")]
    source = source[
        source.index(namespace) : source.index(
            "namespace {", source.index(namespace) + len(namespace)
        )
    ]
    unit = folder / "probe.cpp"
    unit.write_text(
        PREFIX
        + header
        + "\n}\n"
        + source
        + '\n}\nextern "C" {\n'
        + _definition(api, "int generativeqc_resource_ks_matrix_provider_cuda_v1(")
        + "\n}\n"
        + DRIVER
    )
    args = ("-std=c++17", "-Wall", "-Wextra", "-Werror")
    executable = native_cxx.build_executable(
        [unit], folder / "probe", compile_args=args
    )
    library = native_cxx.build_shared([unit], folder / "probe.so", compile_args=args)
    return executable, library


@pytest.mark.parametrize(
    "mode",
    [
        "small",
        "admit",
        "allocation",
        "create-error",
        "capture",
        "oversized",
        "oversized-destroy",
        "setup",
        "setup-destroy",
    ],
)
def test_owner_keeps_failed_live_cleanup_owned(
    matrix_probe: tuple[Path, Path], mode: str
) -> None:
    subprocess.run([str(matrix_probe[0]), mode], check=True, timeout=10)


@pytest.fixture(scope="module")
def native_inventory(matrix_probe: tuple[Path, Path]) -> Any:
    library = ctypes.CDLL(str(matrix_probe[1]))
    query = library.generativeqc_resource_ks_matrix_provider_cuda_v1
    query.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    return library


@pytest.mark.parametrize("nao", [1, 16, 17, 24, 1000])
def test_private_abi_uses_owner_dimension_policy(
    native_inventory: Any, nao: int
) -> None:
    output = ctypes.c_uint64()
    assert (
        native_inventory.generativeqc_resource_ks_matrix_provider_cuda_v1(
            nao, ctypes.byref(output)
        )
        == 0
    )
    assert output.value == (ALLOWANCE if nao >= 17 else 0)


@pytest.mark.parametrize("nao", [0, 2**31, 2**64 - 1])
def test_private_abi_rejects_unrepresentable_shapes(
    native_inventory: Any, nao: int
) -> None:
    output = ctypes.c_uint64(123)
    query = native_inventory.generativeqc_resource_ks_matrix_provider_cuda_v1
    assert query(nao, ctypes.byref(output)) == 1
    assert output.value == 123
    assert query(17, None) == 1


def _library(native_inventory: Any) -> SimpleNamespace:
    def shape(*args: Any) -> int:
        args[-2][:] = [1024, 2048, 4096]
        return 0

    def quadrature(atoms: int, points: int, output: Any) -> int:
        output._obj.value = 128
        return 0

    return SimpleNamespace(
        generativeqc_ks_resource_inventory_version_v1=lambda: 1,
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_resource_ks_cuda_v1=shape,
        generativeqc_resource_quadrature_cuda_v1=quadrature,
        generativeqc_resource_ks_matrix_provider_cuda_v1=native_inventory.generativeqc_resource_ks_matrix_provider_cuda_v1,
    )


@pytest.mark.parametrize(
    "systems, count", [([H2], 0), ([WATER], 1), ([WATER, WATER], 2), ([H2, WATER], 1)]
)
def test_python_budget_reserves_each_live_owner(
    native_inventory: Any, monkeypatch: pytest.MonkeyPatch, systems: Any, count: int
) -> None:
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    request = resources_ks.ks_resource_request(
        systems, basis="def2-svp", backend="cuda", library=_library(native_inventory)
    )
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    allowance = next(
        e
        for e in plan.estimates
        if e.name == "all KS matrix-provider retention allowance"
    )
    assert allowance.bytes == count * ALLOWANCE
    assert allowance.kind == "library" and allowance.accounting == "runtime_allowance"
    numeric = sum(
        e.bytes
        for e in plan.estimates
        if e.space.startswith("device:") and e.accounting != "runtime_allowance"
    )
    assert plan.peak_bytes["device"] == numeric + count * ALLOWANCE
    # Execute the actual ledger adapter: opaque bytes must not increase the
    # limit available to explicit cudaMalloc arenas.
    from generativeqc.resources_native import NativeDeviceLedger

    captured = []

    def create(limit: int, device: int) -> int:
        captured.append((limit, device))
        return 7

    ledger_library = SimpleNamespace(
        generativeqc_resource_ledger_create_v1=create,
        generativeqc_resource_ledger_destroy_v1=lambda _: None,
        generativeqc_resource_ledger_bind_v1=lambda _: 0,
        generativeqc_resource_ledger_read_v1=lambda *_: 0,
    )
    ledger = NativeDeviceLedger(ledger_library, plan, owner=request.name)
    assert ledger.limit == numeric and captured == [(numeric, 0)]
    ledger.close()
    exact = ResourceBudget(device_bytes=plan.peak_bytes["device"])
    plan_resources([request], exact).require_feasible()
    assert (
        plan_resources(
            [request], ResourceBudget(device_bytes=exact.device_bytes - 1)
        ).status
        == "infeasible"
    )


@pytest.mark.parametrize("missing", [True, False])
def test_python_rejects_unavailable_matrix_inventory(
    native_inventory: Any, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    library = _library(native_inventory)
    if missing:
        del library.generativeqc_resource_ks_matrix_provider_cuda_v1
    else:
        library.generativeqc_resource_ks_matrix_provider_cuda_v1 = lambda *args: 2
    request = resources_ks.ks_resource_request(
        [WATER], basis="def2-svp", backend="cuda", library=library
    )
    assert not request.candidates and "matrix-provider" in request.unsupported_reason


def test_cpu_budget_does_not_query_matrix_provider() -> None:
    request = resources_ks.ks_resource_request([H2], backend="cpu")
    assert request.candidates
    assert all("matrix-provider" not in e.name for e in request.candidates[0].estimates)
