"""Execute MD admission/rollback and public planner lifetimes with CUDA doubles."""

from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import test_ks_point_batch_resources as point_budget
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc import ResourceBudget, resources_ks
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import plan_resources
from generativeqc_compiler.integral.md_j_masked_cuda import emit_md_j_masked_header
from test_direct_jk_optional_allocation import STUBS

if TYPE_CHECKING:
    from conftest import NativeCxx

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
#include <map>
#include <type_traits>
#define __host__
#define __device__
#include "generated_md_j_masked.cuh"
using cudaEvent_t=void*;
// Use actual MD-J types, constants and Vec3/DeviceBatch ABI from production.
#include "scf/cuda/direct_md_j.hpp"
#include "scf/md_transform_layout.hpp"
#ifndef GENERATIVEQC_MOLECULE_BASIS_HPP
namespace generativeqc::molecule { inline constexpr std::size_t kMaximumAoExpansionTerms=3; }
#endif
using namespace generativeqc;
using namespace generativeqc::scf::cuda_execution;
using namespace generativeqc::scf;
struct HostBatch {
  std::size_t nbf=8;
  std::vector<unsigned> shell_angular{1,2};
  std::vector<std::int32_t> shell_pair_first{0,1,1}, shell_pair_second{0,0,1};
  std::vector<std::int32_t> shell_pair_systems{0,0,0};
  std::vector<std::int64_t> shell_pair_primitive_offsets{0,1,2,3};
  std::vector<std::int64_t> shell_primitive_offsets{0,1,2}, shell_ao_offsets{0,3,8};
  std::vector<std::int64_t> system_shell_offsets{0,2},system_shell_pair_offsets{0,3};
  std::vector<std::uint8_t> ao_term_counts=std::vector<std::uint8_t>(8,1);
  std::vector<std::uint8_t> ao_term_angular=std::vector<std::uint8_t>(8*9,0);
  HostBatch() {
    const unsigned angular[8][3]={{1,0,0},{0,1,0},{0,0,1},{2,0,0},
                                 {1,1,0},{1,0,1},{0,2,0},{0,1,1}};
    for(unsigned ao=0;ao<8;++ao)
      for(unsigned axis=0;axis<3;++axis) ao_term_angular[ao*9+axis]=angular[ao][axis];
  }
};
namespace runtime { bool active_device_resource_ledger=false; }
std::size_t direct_jk_product(std::size_t a,std::size_t b) { return a*b; }
int allocation_index=0, failure_stage=-1, failure_kind=1;
void allocation_fault() { if(allocation_index++==failure_stage) fault(failure_kind); }
cudaError_t prepare_md_j(cudaStream_t, int, MdJView&, double*, double) { return cudaSuccess; }

cudaError_t prepare_md_j_masked(cudaStream_t, int, MdJView, double*) { return cudaSuccess; }
"""

MD_DRIVER = r"""
int baseline_allocations=0;
int exercise(bool resident_source_only,bool ledger,int stage,int kind,const char* disabled) {
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
    const bool eligible=!resident_source_only && !ledger &&
                        !(disabled && std::strcmp(disabled,"1")==0);
    const bool faulted=stage>=0 && allocation_index>stage;
    const bool cache_fault=faulted && stage>=baseline_allocations;
    assert(eligible || allocation_index==0);
    if(!eligible || (faulted && kind<=2 && !cache_fault)) {
      assert(!propagated && !owner.md_j.minimum_bounds);
      assert(owner.allocations.size()==1 && owner.device_bytes==8);
      direct_jk_check(cudaGetLastError());
    } else if(cache_fault && kind<=2) {
      assert(!propagated && owner.md_j.minimum_bounds && owner.device_bytes>8);
      assert(!owner.md_j.masked_pairs && !owner.md_j.masked_suffixes);
    } else if(!faulted) {
      assert(!propagated && owner.md_j.minimum_bounds && owner.device_bytes>8);
      const char* counts=std::getenv("GENERATIVEQC_MD_J_WORK_COUNTS");
      assert(bool(owner.md_j.work_counts)==bool(counts && std::strcmp(counts,"1")==0));
      const char* tiled=std::getenv("GENERATIVEQC_MD_J_RESIDUAL_TILES");
      const char* profile=std::getenv("GENERATIVEQC_MD_J_PROFILE");
      const char* high=std::getenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH");
      const char* reuse=std::getenv("GENERATIVEQC_MD_J_RECIPROCAL");
      assert(owner.md_j.residual_tiled==bool(!tiled || std::strcmp(tiled,"1")==0));
      assert(owner.md_j.profile==bool(profile && std::strcmp(profile,"1")==0));
      assert(owner.md_j.reciprocal_high==bool((!reuse || std::strcmp(reuse,"1")==0) &&
                                             (!high || std::strcmp(high,"1")==0)));
      const char* masked=std::getenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL");
      const bool selected=masked && std::strcmp(masked,"1")==0;
      const bool configured=masked!=nullptr;
      assert(owner.md_j.masked_residual==selected);
      const char* coarse=std::getenv("GENERATIVEQC_MD_J_RESIDUAL_TILES");
      const bool coarse_selected=!coarse || std::strcmp(coarse,"1")==0;
      const bool coarse_configured=coarse!=nullptr;
      assert(owner.md_j.residual_tiled==coarse_selected);
      setenv("GENERATIVEQC_MD_J_RESIDUAL_TILES",coarse_selected ? "0" : "1",1);
      assert(owner.md_j.residual_tiled==coarse_selected);
      if(coarse_configured) setenv("GENERATIVEQC_MD_J_RESIDUAL_TILES",coarse_selected ? "1" : "0",1);
      else unsetenv("GENERATIVEQC_MD_J_RESIDUAL_TILES");
      const char* projection=std::getenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT");
      const bool cached=projection && std::strcmp(projection,"1")==0;
      assert(bool(owner.md_j.masked_pairs)==cached);
      assert(bool(owner.md_j.masked_suffixes)==cached);
      setenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL",selected ? "0" : "1",1);
      assert(owner.md_j.masked_residual==selected);
      if(configured) setenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL",selected ? "1" : "0",1);
      else unsetenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL");
    } else {
      assert(propagated);
    }
  }
  assert(live.empty());
  return allocation_index;
}
int main() {
  const std::string initial_tiled=std::getenv("GENERATIVEQC_MD_J_RESIDUAL_TILES");
  const std::string initial_profile=std::getenv("GENERATIVEQC_MD_J_PROFILE");
  const std::string initial_high=std::getenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH");
  unsetenv("GENERATIVEQC_MD_J_RESIDUAL_TILES");
  unsetenv("GENERATIVEQC_MD_J_PROFILE");
  unsetenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH");
  unsetenv("GENERATIVEQC_MD_J_RECIPROCAL");
  for(const char* control:{"GENERATIVEQC_MD_J_RESIDUAL_TILES","GENERATIVEQC_MD_J_PROFILE",
                          "GENERATIVEQC_MD_J_RECIPROCAL_HIGH"}) {
    unsetenv(control);
    HostBatch host; MdJHost defaults;
    assert(defaults.prepare(host,128U<<20) && defaults.residual_tiled && !defaults.profile &&
           defaults.reciprocal && defaults.reciprocal_high);
    for(const char* invalid:{"", "yes", "2"}) {
      setenv(control,invalid,1);
      MdJHost candidate; bool rejected=false;
      try { candidate.prepare(host,128U<<20); }
      catch(const std::invalid_argument&) { rejected=true; }
      assert(rejected);
    }
    unsetenv(control);
  }
  for(const char* reuse:{"0","1"}) {
    setenv("GENERATIVEQC_MD_J_RECIPROCAL",reuse,1);
    for(const char* high:{static_cast<const char*>(nullptr),"0","1"}) {
      if(high) setenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH",high,1);
      else unsetenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH");
      HostBatch host; MdJHost policy;
      assert(policy.prepare(host,128U<<20));
      assert(policy.reciprocal_high==bool(std::strcmp(reuse,"1")==0 &&
                                         (!high || std::strcmp(high,"1")==0)));
    }
  }
  unsetenv("GENERATIVEQC_MD_J_RECIPROCAL");
  setenv("GENERATIVEQC_MD_J_RESIDUAL_TILES",initial_tiled.c_str(),1);
  setenv("GENERATIVEQC_MD_J_PROFILE",initial_profile.c_str(),1);
  setenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH",initial_high.c_str(),1);
  unsetenv("GENERATIVEQC_DISABLE_MD_J");
  const char* projection=std::getenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT");
  const std::string original_projection=projection ? projection : "";
  unsetenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT");
  exercise(false,false,-1,1,nullptr);
  baseline_allocations=allocation_index;
  for(const char* tiled:{"0","1","2"}) {
    setenv("GENERATIVEQC_MD_J_MASKED_TILES",tiled,1);
    MdJHost metadata; HostBatch host;
    assert(metadata.prepare(host,128U<<20));
    assert(metadata.masked_tiles==static_cast<unsigned>(std::atoi(tiled)));
    exercise(false,false,-1,1,nullptr);
  }
  setenv("GENERATIVEQC_MD_J_MASKED_TILES","invalid",1);
  bool tiled_rejected=false;
  try { MdJHost metadata; HostBatch host; metadata.prepare(host,128U<<20); }
  catch(const std::invalid_argument&) { tiled_rejected=true; }
  assert(tiled_rejected);
  unsetenv("GENERATIVEQC_MD_J_MASKED_TILES");
  for(const char* coarse:{"0","1"}) {
    setenv("GENERATIVEQC_MD_J_RESIDUAL_TILES",coarse,1);
    MdJHost metadata; HostBatch host;
    assert(metadata.prepare(host,128U<<20));
    assert(metadata.residual_tiled==bool(std::atoi(coarse)));
    exercise(false,false,-1,1,nullptr);
  }
  setenv("GENERATIVEQC_MD_J_RESIDUAL_TILES","invalid",1);
  bool coarse_rejected=false;
  try { MdJHost metadata; HostBatch host; metadata.prepare(host,128U<<20); }
  catch(const std::invalid_argument&) { coarse_rejected=true; }
  assert(coarse_rejected);
  unsetenv("GENERATIVEQC_MD_J_RESIDUAL_TILES");
  setenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT","invalid",1);
  bool cache_rejected=false;
  try { MdJHost metadata; HostBatch host; metadata.prepare(host,128U<<20); }
  catch(const std::invalid_argument&) { cache_rejected=true; }
  assert(cache_rejected);
  if(original_projection.empty()) unsetenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT");
  else setenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT",original_projection.c_str(),1);
  if(original_projection=="1") {
    MdJHost metadata; HostBatch host; assert(metadata.prepare(host,128U<<20));
    assert(!metadata.prepare_masked_cache(0));
    assert(metadata.masked_pairs.empty());
    assert(metadata.prepare_masked_cache((128U<<20)-metadata.bytes));
    assert(metadata.masked_bytes+metadata.bytes<=(128U<<20));
    metadata.pairs.push_back(metadata.pairs[0]);
    const auto pair=metadata.pairs[0];
    const auto slots=(pair.primitive_end-pair.primitive_begin)*
                     (2*pair.first_count*pair.second_count+1)*pair.hermites;
    const auto limited=metadata.pairs.size()*sizeof(MdMaskedResidentPair)+slots*sizeof(double);
    assert(metadata.prepare_masked_cache(limited));
    assert(metadata.masked_bytes<=limited);
    assert(metadata.masked_suffixes==slots);
    unsigned cached_pairs=0;
    for(const auto& cached:metadata.masked_pairs)
      if(cached.suffix_offset!=static_cast<std::size_t>(-1)) ++cached_pairs;
    assert(cached_pairs==1);
    MdJHost balanced; balanced.preproject=true;
    auto input=pair;
    input.angular=1; input.hermites=4; input.first_count=3; input.second_count=1;
    balanced.pairs={input,input,pair,pair};
    const auto input_slots=(input.primitive_end-input.primitive_begin)*28;
    const auto balanced_limit=balanced.pairs.size()*sizeof(MdMaskedResidentPair)+
                              (input_slots+slots)*sizeof(double);
    assert(balanced.prepare_masked_cache(balanced_limit));
    assert(balanced.masked_bytes<=balanced_limit);
    assert(balanced.masked_pairs[0].suffix_offset==0);
    assert(balanced.masked_pairs[2].suffix_offset==input_slots);
    assert(balanced.masked_pairs[1].suffix_offset==static_cast<std::size_t>(-1));
    assert(balanced.masked_pairs[3].suffix_offset==static_cast<std::size_t>(-1));
  }
  const char* policy=std::getenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL");
  const std::string original=policy ? policy : "";
  setenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL","invalid",1);
  bool rejected=false;
  try { MdJHost metadata; HostBatch host; metadata.prepare(host,128U<<20); }
  catch(const std::invalid_argument&) { rejected=true; }
  assert(rejected);
  if(original.empty()) unsetenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL");
  else setenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL",original.c_str(),1);
  setenv("GENERATIVEQC_MD_J_RESIDUAL_TILES",initial_tiled.c_str(),1);
  setenv("GENERATIVEQC_MD_J_PROFILE",initial_profile.c_str(),1);
  setenv("GENERATIVEQC_MD_J_RECIPROCAL_HIGH",initial_high.c_str(),1);
  // Resident-only sources must not allocate optional MD storage either.
  for(bool resident_source_only:{false,true}) {
    for(const char* disabled:{static_cast<const char*>(nullptr),"1","0"}) {
      // The same construction path is revisited after ledger binding/unbinding.
      for(bool ledger:{false,true,true,false}) {
        const auto allocations=exercise(resident_source_only,ledger,-1,1,disabled);
        for(int stage=0;stage<allocations;++stage)
          for(int kind:{0,1,3,6}) exercise(resident_source_only,ledger,stage,kind,disabled);
      }
    }
  }
}
"""


@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("masked", [None, "0", "1"])
@pytest.mark.parametrize("preproject", [None, "0", "1"])
def test_actual_md_optional_preparation_preserves_ledger_and_cuda_errors(
    tmp_path: Path,
    native_cxx: NativeCxx,
    monkeypatch: pytest.MonkeyPatch,
    counted: bool,
    diagnostics: bool,
    masked: str | None,
    preproject: str | None,
) -> None:
    """Optional diagnostic storage obeys the same admission and rollback cap."""
    if counted:
        monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    else:
        monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS", raising=False)
    for control in (
        "GENERATIVEQC_MD_J_RESIDUAL_TILES",
        "GENERATIVEQC_MD_J_PROFILE",
        "GENERATIVEQC_MD_J_RECIPROCAL_HIGH",
    ):
        monkeypatch.setenv(control, "1" if diagnostics else "0")
    if masked is None:
        monkeypatch.delenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_MD_J_MASKED_RESIDUAL", masked)
    if preproject is None:
        monkeypatch.delenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_MD_J_MASKED_PREPROJECT", preproject)
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
    (tmp_path / "generated_md_j_masked.cuh").write_text(emit_md_j_masked_header())
    cpp, binary = tmp_path / "md.cpp", tmp_path / "md"
    cpp.write_text(stubs + "\n".join(definitions) + driver)
    native_cxx.build_executable(
        [cpp],
        binary,
        compile_args=[
            "-std=c++20",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize("atoms", [48, 96])
@pytest.mark.parametrize("spherical", [False, True])
def test_canonical_transform_storage_retains_existing_cap(
    tmp_path: Path,
    native_cxx: NativeCxx,
    monkeypatch: pytest.MonkeyPatch,
    atoms: int,
    spherical: bool,
) -> None:
    """Inventory real basis expansions without constructing integrals or a GPU.

    Capacity is the contract, not an exact internal operation count. In
    particular the large-basis case must fit without raising the existing cap
    or changing public-ledger admission. Printed inventories are diagnostic.
    """
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    monkeypatch.delenv("GENERATIVEQC_MD_J_RECIPROCAL", raising=False)
    monkeypatch.delenv("GENERATIVEQC_MD_J_WORK_COUNTS", raising=False)
    basis = json.loads(
        (ROOT / "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json").read_text()
    )
    shells = []
    for atomic_number, count in ((8, atoms // 3), (1, 2 * atoms // 3)):
        element = next(
            row for row in basis["elements"] if row["atomic_number"] == atomic_number
        )
        for _ in range(count):
            for shell in element["shells"]:
                for _ in shell["coefficients"]:
                    shells.append((shell["angular_momentum"], len(shell["exponents"])))
    shapes = ",".join(f"{{{angular},{primitives}}}" for angular, primitives in shells)
    source, binary = tmp_path / "inventory.cpp", tmp_path / "inventory"
    (tmp_path / "generated_md_j_masked.cuh").write_text(emit_md_j_masked_header())
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; using cudaError_t=int;\n"
    )
    production = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    prefix = (
        "#include <cassert>\n#include <cstdio>\n#include <cstdlib>\n"
        "#include <cstring>\n#include <vector>\n#include <stdexcept>\n"
        '#include "molecule/basis.hpp"\n'
        + MD_STUBS.split("namespace runtime", 1)[0]
        + "\nstd::size_t direct_jk_product(std::size_t first,std::size_t second) {\n"
        "if(second && first>std::numeric_limits<std::size_t>::max()/second) throw std::bad_alloc();\n"
        "return first*second; }\n"
        + cpp_record_definition(production, "MdJHost")
        + ";\n"
    )
    driver = r"""
int main() {
  HostBatch host;
  host.nbf=0; host.shell_angular.clear(); host.shell_ao_offsets={0};
  host.shell_primitive_offsets={0}; host.ao_term_counts.clear(); host.ao_term_angular.clear();
  const unsigned shapes[][2]={SHAPES};
  for(const auto& shape:shapes) {
    host.shell_angular.push_back(shape[0]);
    const auto expansion=generativeqc::molecule::ao_expansions(shape[0],REPRESENTATION);
    host.nbf+=expansion.size(); host.shell_ao_offsets.push_back(host.nbf);
    host.shell_primitive_offsets.push_back(host.shell_primitive_offsets.back()+shape[1]);
    for(const auto& component:expansion) {
      host.ao_term_counts.push_back(component.size());
      for(unsigned term=0;term<3;++term) for(unsigned axis=0;axis<3;++axis)
        host.ao_term_angular.push_back(term<component.size() ? component[term].component[axis] : 0);
    }
  }
  host.shell_pair_first.clear(); host.shell_pair_second.clear();
  host.shell_pair_systems.clear(); host.shell_pair_primitive_offsets={0};
  std::size_t dense_values=0;
  for(unsigned first=0;first<host.shell_angular.size();++first)
    for(unsigned second=0;second<=first;++second) {
      host.shell_pair_first.push_back(first); host.shell_pair_second.push_back(second);
      host.shell_pair_systems.push_back(0);
      const auto primitives=(host.shell_primitive_offsets[first+1]-host.shell_primitive_offsets[first])*
                            (host.shell_primitive_offsets[second+1]-host.shell_primitive_offsets[second]);
      host.shell_pair_primitive_offsets.push_back(host.shell_pair_primitive_offsets.back()+primitives);
      dense_values+=primitives*md_j_hermite_count(host.shell_angular[first]+host.shell_angular[second])*
                    (host.shell_ao_offsets[first+1]-host.shell_ao_offsets[first])*
                    (host.shell_ao_offsets[second+1]-host.shell_ao_offsets[second]);
    }
  host.system_shell_offsets={0,static_cast<std::int64_t>(host.shell_angular.size())};
  host.system_shell_pair_offsets={0,static_cast<std::int64_t>(host.shell_pair_first.size())};
  MdJHost planned;
  assert(planned.prepare(host,kMdJResidentCap));
  const auto charged=kMdSourceFixedBytes+planned.pairs.size()*(sizeof(MdJPair)+3*sizeof(double)+sizeof(std::uint32_t))+
      planned.primitives.size()*(sizeof(MdJPrimitive)+sizeof(std::uint32_t))+
      host.system_shell_offsets.size()*3*sizeof(std::int64_t)+
      planned.transform_slots.size()*sizeof(std::uint16_t)+(planned.transforms+2*planned.hermites)*sizeof(double);
  assert(charged==planned.bytes && charged<=kMdJResidentCap);
  const auto dense_bytes=charged-planned.pairs.size()*sizeof(std::uint32_t)-
      planned.transform_slots.size()*sizeof(std::uint16_t)-planned.transforms*sizeof(double)+dense_values*sizeof(double);
  std::printf("nbf=%zu pairs=%zu primitives=%zu dense_values=%zu sparse_values=%zu map_slots=%zu "
              "dense_bytes=%zu sparse_bytes=%zu cap=%zu\n",host.nbf,planned.pairs.size(),
              planned.primitives.size(),dense_values,planned.transforms,planned.transform_slots.size(),
              dense_bytes,charged,kMdJResidentCap);
}
"""
    representation = (
        "GENERATIVEQC_BASIS_SPHERICAL" if spherical else "GENERATIVEQC_BASIS_CARTESIAN"
    )
    source.write_text(
        prefix
        + driver.replace("SHAPES", shapes).replace("REPRESENTATION", representation)
    )
    native_cxx.build_executable(
        [source, ROOT / "src/molecule/basis.cpp"],
        binary,
        compile_args=[
            "-std=c++20",
            "-O2",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    subprocess.run([str(binary)], check=True, timeout=30)
