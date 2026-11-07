"""Execute the private slot/epoch/finite owners without chemical work or CUDA."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
DRIVER = r"""
#include <cassert>
#include <limits>
#include <string>
#include <vector>
#include "dft/vv10_rebase_history.hpp"
#include "solver/diis_coefficients.hpp"
#include "vv10_rebase_cases.hpp"
namespace p = generativeqc::dft::vv10_rebase;
namespace c = generativeqc::tests::vv10_rebase;
void layout() {
  static_assert(sizeof(p::Key)==32 && sizeof(p::SlotTag)==40 && sizeof(p::Transaction)==144);
  auto bound = p::layout(800,8); assert(bound);
  assert(bound->matrix_bytes==153600000 && bound->total_bytes==153600768);
  assert(bound->old_potential + 800*800*8 == bound->matrix_bytes);
  assert(bound->transaction + sizeof(p::Transaction) <= bound->total_bytes);
  assert(!p::layout(0,8) && !p::layout(800,1) && !p::layout(800,65));
  assert(!p::layout(std::numeric_limits<std::size_t>::max(),8));
  assert(!p::layout(100000,64));
  assert(p::optional_state_bytes(p::Request::Off,800,1,8,false)==0);
  assert(p::optional_state_bytes(p::Request::On,800,1,8,false)==bound->total_bytes);
  assert(p::optional_state_bytes(p::Request::On,800,2,8,false)==0);
  assert(p::optional_state_bytes(p::Request::On,800,1,8,true)==0);
  p::Context good{true,false,true,true,true,true,true,true,false,true,true,8,800,bound->total_bytes};
  assert(p::admit(p::Request::On,good)==p::Admission::Active);
  for(int gate=0;gate<13;++gate) {
    auto x=good;
    switch(gate) {
      case 0:x.cold=false;break; case 1:x.imported=true;break;
      case 2:x.restricted=false;break; case 3:x.full_direct=false;break;
      case 4:x.standard_nonlocal=false;break; case 5:x.strict_fp64=false;break;
      case 6:x.outer_solve=false;break; case 7:x.ordinary_width_one=false;break;
      case 8:x.incremental_jk=true;break; case 9:x.raw_ao_metric=false;break;
      case 10:x.ordered_gram=false;break; case 11:x.history=1;break;
      case 12:--x.available_optional_bytes;break;
    }
    assert(p::admit(p::Request::On,x)!=p::Admission::Active);
    assert(p::admit(p::Request::Off,x)==p::Admission::Off);
    assert(p::admit(p::Request::Invalid,x)==p::Admission::InvalidRequest);
  }
  assert(p::parse_request(nullptr)==p::Request::Off);
  for(const char* v:{"","0"}) assert(p::parse_request(v)==p::Request::Off);
  assert(p::parse_request("1")==p::Request::On);
  for(const char* v:{"01","true","1 ","-1","auto"})
    assert(p::parse_request(v)==p::Request::Invalid);
}
void ring() {
  const p::Key old{4,5,6,1}, target{4,5,7,1};
  for(unsigned H:{2U,3U,8U,64U}) for(unsigned head=0;head<H;++head)
    for(unsigned count=1;count<=H;++count) {
      std::vector<p::SlotTag> tags(H,{old,1});
      p::Transaction tx{};tx.enabled=1;
      assert(p::prepare_snapshot(tx,H,count,head,tags.data(),old));
      auto b=std::min(count,H-1);
      assert(tx.bank_count==b);
      for(unsigned i=0;i<b;++i) {
        const unsigned expected=(head+H-b+i)%H;
        assert(p::bank_source_slot(H,count,head,i)==expected);
        assert(p::restored_slot(H,b,i)==H-b+i);
      }
      // A normalized solve can retire oldest entries by reducing count only.
      if(count>1) {
        const auto first=p::chronological_slot(H,count,head,0);
        tags[first].valid=0;
        assert(p::prepare_snapshot(tx,H,count-1,head,tags.data(),old));
      }
      tags[0]={target,1};
      assert(p::prepare_target(tx,H,1,1,tags.data(),target));
      assert(tx.proposed_count==tx.bank_count+1 && tx.proposed_head==1);
      // Slot0 remains target; restored slots exclude it; next insertion is1.
      for(unsigned i=0;i<tx.bank_count;++i) assert(p::restored_slot(H,tx.bank_count,i)!=0);
    }
  for(int error=0;error<6;++error) {
    std::array<p::SlotTag,4> tags;tags.fill({old,1});p::Transaction tx{};tx.enabled=1;
    assert(p::prepare_snapshot(tx,4,3,0,tags.data(),old));
    auto key=target;unsigned count=1,head=1;tags[0]={target,1};
    switch(error) {
      case 0:key.owner++;break;case 1:key.solve++;break;case 2:key.metric++;break;
      case 3:key.anchor=0;tags[0].key=key;break;case 4:count=2;break;case 5:tags[0].valid=0;break;
    }
    assert(!p::prepare_target(tx,4,count,head,tags.data(),key));
    assert(!tx.ready && tx.work.refusals==1);
  }
  std::array<p::SlotTag,4> tags{};p::Transaction missing{};missing.enabled=1;
  assert(!p::prepare_snapshot(missing,4,2,2,tags.data(),old));
  tags.fill({old,1});
  p::Key missingOwner=old;missingOwner.owner=0;
  assert(!p::prepare_snapshot(missing,4,2,2,tags.data(),missingOwner));
  tags[1].key.anchor++;
  assert(!p::prepare_snapshot(missing,4,2,2,tags.data(),old));
}
void phase() {
  for(bool seed:{false,true}) {
    (void)seed; bool bank=true;
    // Both seed and frozen snapshot must survive the temporary target phase.
    bank=p::retain_bank_after_finish(bank,true,false,true,true,false,false);assert(bank);
    const bool bridge=bank;assert(bridge);
    bank=false; // commit consumes bank, not the target DIIS entry
    assert(!p::retain_bank_after_finish(bank,false,true,true,true,false,false));
  }
  assert(!p::retain_bank_after_finish(true,false,false,true,true,false,false));
  assert(!p::retain_bank_after_finish(true,true,false,false,true,false,false));
  assert(!p::retain_bank_after_finish(true,true,false,true,false,false,false));
  assert(!p::retain_bank_after_finish(true,true,false,true,true,true,false));
  assert(!p::retain_bank_after_finish(true,true,false,true,true,false,true));
}
void numeric() {
  for(bool zero:{false,true}) {
    const auto test=c::fixture(zero);c::Matrix delta{};
    for(std::size_t i=0;i<c::m;++i)delta[i]=test.new_potential[i]-test.old_potential[i];
    std::vector<c::Matrix> errors;
    for(unsigned slot=0;slot<c::history;++slot) {
      auto r=c::residual(test.fock[slot],test.density[slot],test.overlap);
      const auto left=c::product(c::product(delta,test.density[slot]),test.overlap);
      const auto right=c::product(c::product(test.overlap,test.density[slot]),delta);
      for(std::size_t i=0;i<c::m;++i) {
        assert(p::accumulate_finite(r[i],left[i],false,r[i]));
        assert(p::accumulate_finite(r[i],right[i],true,r[i]));
      }
      c::close(r,c::residual(c::shifted_fock(test,slot),test.density[slot],test.overlap));
      errors.push_back(r);
    }
    // Full physical-slot Gram oracle is independent of the pending-row cache.
    for(unsigned row=0;row<c::history;++row) for(unsigned col=0;col<c::history;++col)
      assert(std::abs(c::dot(errors[row],errors[col])-c::dot(errors[col],errors[row]))<1e-13);
    const double g00=c::dot(errors[0],errors[0]),g01=c::dot(errors[0],errors[1]),g11=c::dot(errors[1],errors[1]);
    std::vector<double> coefficients;
    generativeqc::solver::detail::DiisCoefficientPolicy policy;
    policy.metric_scaling=generativeqc::solver::detail::DiisMetricScaling::MaximumDiagonal;
    policy.require_affine_constraint=true;
    assert(generativeqc::solver::detail::solve_diis_coefficients({g00,g01,g01,g11},2,policy,coefficients)==
           generativeqc::solver::detail::DiisCoefficientAction::Extrapolate);
    const double expected=(g11-g01)/(g00+g11-2*g01);
    assert(std::abs(coefficients[0]-expected)<1e-11);
    assert(std::abs(coefficients[0]+coefficients[1]-1)<1e-13);
  }
  double value;
  assert(p::accumulate_finite(1e300,-1e300,false,value)&&value==0);
  assert(!p::accumulate_finite(DBL_MAX,DBL_MAX,false,value));
  assert(!p::accumulate_finite(0,std::numeric_limits<double>::infinity(),false,value));
  assert(!p::accumulate_finite(std::numeric_limits<double>::quiet_NaN(),0,false,value));
}
int main(int argc,char** argv) {
  assert(argc==2);const std::string mode=argv[1];
  if(mode=="layout")layout();else if(mode=="ring")ring();else if(mode=="phase")phase();
  else if(mode=="numeric")numeric();else return 2;
}
"""


@pytest.fixture(scope="module")
def probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    folder = tmp_path_factory.mktemp("vv10-exact-history")
    source, binary = folder / "probe.cpp", folder / "probe"
    source.write_text(DRIVER)
    required_native_cxx.build_executable(
        [source],
        binary,
        compile_args=[
            "-std=c++20",
            "-UNDEBUG",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'tests/native'}",
        ],
    )
    return binary


@pytest.mark.parametrize("mode", ("layout", "ring", "phase", "numeric"))
def test_native_transaction_owners(probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(probe), mode], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", (None, "", "0", "1", "01", "auto", "1 "))
def test_cuda_prepared_identity_control(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc.resources_ks import _vv10_rebase_history_control

    if value is None:
        monkeypatch.delenv("GENERATIVEQC_KS_VV10_REBASE_HISTORY", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_KS_VV10_REBASE_HISTORY", value)
    if value not in (None, "", "0", "1"):
        with pytest.raises(ValueError, match="accepts only 0 or 1"):
            _vv10_rebase_history_control()
    else:
        assert _vv10_rebase_history_control() == int(value == "1")


def test_full_gram_preflight_rejects_overflow_and_aliases(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    (leaves / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cstdint>\nusing cudaStream_t=void*;using cudaError_t=int;\n"
        "struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};\n"
    )
    source, binary = tmp_path / "gram.cpp", tmp_path / "gram"
    source.write_text(r"""
#include <cassert>
#include <limits>
#include "scf/cuda/scf_diis_kernels.hpp"
int main() {
  using generativeqc::scf::cuda_execution::ordered_gram_admission::validate_rebuild;
  double residual[36]{},cache[16]{};std::uint32_t finite=1;
  validate_rebuild(9,4,4,1,residual,cache,&finite);
  for(int gate=0;gate<10;++gate) {
    unsigned H=4,count=4,head=1;std::size_t size=9;
    const double* r=residual;double* c=cache;std::uint32_t* f=&finite;
    switch(gate) {
      case 0:size=0;break;case 1:size=std::numeric_limits<std::size_t>::max()/8;break;
      case 2:H=65;break;case 3:count=5;break;case 4:head=4;break;
      case 5:r=nullptr;break;case 6:c=nullptr;break;case 7:f=nullptr;break;
      case 8:c=residual;break;case 9:f=reinterpret_cast<std::uint32_t*>(cache);break;
    }
    bool rejected=false;
    try {validate_rebuild(size,H,count,head,r,c,f);}
    catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
  }
}
""")
    required_native_cxx.build_executable(
        [source],
        binary,
        compile_args=["-std=c++20", "-UNDEBUG", f"-I{leaves}", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_control_binds_even_an_unsupported_cuda_resource_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from generativeqc.resources_ks import ks_resource_request

    systems = [[(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]]
    identities = []
    for value in ("0", "1"):
        monkeypatch.setenv("GENERATIVEQC_KS_VV10_REBASE_HISTORY", value)
        # No ABI means explicit unsupported admission, never a native load.
        request = ks_resource_request(systems, backend="cuda", library=object())
        assert json.loads(request.identity.schedule)["vv10_rebase_history"] == int(
            value
        )
        identities.append(request.identity)
    assert identities[0] != identities[1]
    monkeypatch.setenv("GENERATIVEQC_KS_VV10_REBASE_HISTORY", "01")
    with pytest.raises(ValueError, match="accepts only 0 or 1"):
        ks_resource_request([], backend="cuda", library=object())


def test_linked_fixture_host_interface_syntax(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Compile its real caller against CUDA API type leaves; this is not NVCC/device proof."""
    leaves = tmp_path / "cuda-api-leaves"
    leaves.mkdir()
    (leaves / "cuda_runtime_api.h").write_text(r"""
#pragma once
#include <cstddef>
using cudaStream_t=void*;using cudaError_t=int;
constexpr cudaError_t cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost};
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**,std::size_t);
template<class T> cudaError_t cudaMalloc(T** p,std::size_t n) {return cudaMalloc(reinterpret_cast<void**>(p),n);}
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t*);cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);cudaError_t cudaGetLastError();
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
""")
    (leaves / "cuda_runtime.h").write_text('#include "cuda_runtime_api.h"\n')
    (leaves / "cublas_v2.h").write_text("#pragma once\nusing cublasHandle_t=void*;\n")
    required_native_cxx.compile_object(
        ROOT / "tests/native/vv10_rebase_cuda.cu",
        tmp_path / "fixture-host-interface.o",
        args=[
            "-x",
            "c++",
            "-std=c++20",
            "-UNDEBUG",
            f"-I{leaves}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'tests/native'}",
        ],
    )
