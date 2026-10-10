"""Execute MD admission/rollback and public planner lifetimes with CUDA doubles."""

from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import test_ks_point_batch_resources as point_budget
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc import ResourceBudget, resources_ks
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import plan_resources
from test_coulomb_optional_allocation import compile_cached_probe
from test_direct_jk_optional_allocation import STUBS

ROOT = Path(__file__).resolve().parents[2]
native_probe = point_budget.native_probe

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("bounded", [False, True])
def test_public_md_default_preserves_normal_budget_and_future_owners(
    native_probe: Any, monkeypatch: pytest.MonkeyPatch, count: int, bounded: bool
) -> None:
    """Unset/opt-out requests share one incumbent allowance across rebuilds."""
    _, library = point_budget._request(native_probe, monkeypatch, 1)

    def matrix(nao: int, output: Any) -> int:
        assert nao == 24
        output._obj.value = 0
        return 0

    library.generativeqc_resource_ks_matrix_provider_cuda_v1 = matrix

    def request() -> Any:
        return resources_ks.ks_resource_request(
            [WATER] * count,
            basis="def2-svp",
            basis_representation="spherical",
            backend="cuda",
            library=library,
        )

    monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
    normal = request()
    incumbent = plan_resources([normal], ResourceBudget()).require_feasible()
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J")
    default = request()
    assert default == normal
    budget = (
        ResourceBudget(device_bytes=incumbent.peak_bytes["device"], headroom_fraction=0)
        if bounded
        else ResourceBudget()
    )
    plan = plan_resources([default], budget).require_feasible()
    assert plan.peak_bytes == incumbent.peak_bytes
    row = json.loads(dict(default.candidates[0].decisions)["item_device_inventory"])[0]
    mandatory = sum(row[key] for key in ("state", "xc", "coulomb"))
    ledger = NativeDeviceLedger(library, plan, owner="ks")
    try:
        normal_capacity = native_probe.md_capacity(None, False)
        for disabled in (None, "1", "0", None):
            if disabled is None:
                monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
            else:
                monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", disabled)
            assert native_probe.md_capacity(None, False) == normal_capacity
            assert native_probe.md_capacity(ledger.handle, True) == normal_capacity
            optional = 0 if disabled == "1" else 128 << 20
            assert native_probe.md_capacity(None, True) == normal_capacity + optional
        force = 512 << 20
        result = (ctypes.c_uint64 * 6)()
        native_probe.active_fleet(ledger.handle, count, mandatory, force, 1, 3, result)
        assert list(result) == [3 * count, 0, force, count * mandatory + force, 0, 0]
    finally:
        ledger.close()


MD_STUBS = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#define __host__
#define __device__
// Use actual MD-J types, constants and Vec3/DeviceBatch ABI from production.
#include "scf/cuda/direct_md_j.hpp"
using namespace generativeqc::scf::cuda_execution;
struct HostBatch {
  std::size_t nbf=8;
  std::vector<unsigned> shell_angular{1,2};
  std::vector<std::int32_t> shell_pair_first{0,1,1}, shell_pair_second{0,0,1};
  std::vector<std::int32_t> shell_pair_systems{0,0,0};
  std::vector<std::int64_t> shell_pair_primitive_offsets{0,1,2,3};
  std::vector<std::int64_t> shell_primitive_offsets{0,1,2}, shell_ao_offsets{0,3,8};
  std::vector<std::int64_t> system_shell_offsets{0,2},system_shell_pair_offsets{0,3};
};
namespace runtime { bool active_device_resource_ledger=false; }
std::size_t direct_jk_product(std::size_t a,std::size_t b) { return a*b; }
int allocation_index=0, failure_stage=-1, failure_kind=1;
void allocation_fault() { if(allocation_index++==failure_stage) fault(failure_kind); }
cudaError_t prepare_md_j(cudaStream_t, int, MdJView&, double*, double) { return cudaSuccess; }
"""

MD_DRIVER = r"""
void exercise(bool ledger,int stage,int kind,const char* disabled) {
  assert(live.empty());
  runtime::active_device_resource_ledger=ledger;
  if(disabled) setenv("GENERATIVEQC_DISABLE_MD_J",disabled,1);
  else unsetenv("GENERATIVEQC_DISABLE_MD_J");
  sticky=synchronization_error=cudaSuccess;
  failure_stage=stage; failure_kind=kind; allocation_index=0;
  {
    CudaDirectJkPlan owner; auto* plan=&owner;
    allocate(owner,8);
    HostBatch host;
    std::size_t budget=128U<<20;
    unsigned derivative_order=1;
    std::string detail="";
    auto upload=[&](const void* source,std::size_t bytes) {
      allocation_fault(); auto* pointer=allocate(owner,bytes);
      std::memcpy(pointer,source,bytes); return pointer;
    };
    auto scratch=[&](std::size_t bytes) {
      allocation_fault(); return static_cast<double*>(allocate(owner,bytes));
    };
    bool propagated=false;
    try {
      MD_PREPARATION
    } catch(const DirectJkFailure& f) {
      propagated=true; assert(f.status!=GENERATIVEQC_STATUS_OUT_OF_MEMORY);
    }
    const bool eligible=!ledger && !(disabled && std::strcmp(disabled,"1")==0);
    const bool faulted=stage>=0 && allocation_index>stage;
    assert(eligible || allocation_index==0);
    if(!eligible || (faulted && kind<=2)) {
      assert(!propagated && !owner.md_j.minimum_bounds);
      assert(owner.allocations.size()==1 && owner.device_bytes==8);
      direct_jk_check(cudaGetLastError());
    } else if(!faulted) {
      assert(!propagated && owner.md_j.minimum_bounds && owner.device_bytes>8);
      const char* counts=std::getenv("GENERATIVEQC_MD_J_WORK_COUNTS");
      assert(bool(owner.md_j.work_counts)==bool(counts && std::strcmp(counts,"1")==0));
    } else {
      assert(propagated);
    }
  }
  assert(live.empty());
}
int main() {
  for(const char* disabled:{static_cast<const char*>(nullptr),"1","0"}) {
    // The same construction path is revisited after ledger binding/unbinding.
    for(bool ledger:{false,true,true,false}) {
      exercise(ledger,-1,1,disabled);
      for(int stage=0;stage<20;++stage)
        for(int kind:{0,1,3,6}) exercise(ledger,stage,kind,disabled);
    }
  }
}
"""


@pytest.mark.parametrize("counted", [False, True])
def test_actual_md_optional_preparation_preserves_ledger_and_cuda_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counted: bool,
) -> None:
    """Optional diagnostic storage obeys the same admission and rollback cap."""
    if counted:
        monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    else:
        monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS", raising=False)
    source = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    # Reuse the existing failure/ownership runtime, adding the real MD layout.
    stubs = STUBS.replace(
        "struct CudaDirectJkPlan {",
        MD_STUBS.split("struct HostBatch", 1)[0] + "\nstruct CudaDirectJkPlan {",
    ).replace(
        "std::size_t device_bytes = 0;",
        "std::size_t device_bytes = 0;\nMdJView md_j; int batch=0; double* bounds=nullptr;"
        " double screening_tolerance=1e-12; struct { const char* schedule; } diagnostic;",
    )
    definitions = [
        "struct HostBatch" + MD_STUBS.split("struct HostBatch", 1)[1],
        cpp_record_definition(source, "MdJHost") + ";",
        cpp_function_definition(source, "direct_jk_check"),
        cpp_function_definition(
            source, "direct_jk_optional_storage", include_template=True
        ),
    ]
    start = source.index("    MdJHost md_host;")
    # Isolate incumbent MD admission from the later optional materialized owner.
    end = source.index("    prepare_materialized_values();", start)
    driver = MD_DRIVER.replace("MD_PREPARATION", source[start:end])
    # The host compiler has no CUDA SDK; only its runtime type declarations
    # are supplied by the controlled shim. All MD-J structs are real headers.
    (tmp_path / "cuda_runtime_api.h").write_text("#pragma once\n")
    cpp, binary = tmp_path / "md.cpp", tmp_path / "md"
    cpp.write_text(stubs + "\n".join(definitions) + driver)
    compile_cached_probe(cpp, binary, include_dirs=(tmp_path,))
    subprocess.run([str(binary)], check=True, timeout=10)
