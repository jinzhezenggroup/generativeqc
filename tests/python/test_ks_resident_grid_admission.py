"""Host-execute the actual resident-grid gate with explicit lease stand-ins."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_resident_grid_fallback_preserves_token_and_device_admission(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    begin = source.index("  generativeqc_status resident_grid(")
    end = source.index("  generativeqc_status resident_nonlocal_features(", begin)
    method = source[begin:end]
    prefix = r"""
#include <cassert>
#include <cstddef>
#include <string>
#define GENERATIVEQC_HAS_CUDA 1
using generativeqc_status = int;
constexpr int GENERATIVEQC_STATUS_SUCCESS=0, GENERATIVEQC_STATUS_NOT_IMPLEMENTED=1,
  GENERATIVEQC_STATUS_INTERNAL_ERROR=2, GENERATIVEQC_STATUS_INVALID_ARGUMENT=3;
namespace dft {
struct CudaKsFinalStateToken { int epoch=42; };
struct CudaKsResidentDensityBinding {
  bool valid=false; int device_id=2;
  explicit operator bool() const { return valid; }
};
}
namespace scf { struct ScfOptions {
  enum class XcExecutionSchedule { DeviceFused, HostUnfused };
  XcExecutionSchedule xc_execution_schedule=XcExecutionSchedule::DeviceFused;
}; }
struct DensityOwner {
  bool valid=true; int calls=0;
  int resident_final_density(const dft::CudaKsFinalStateToken& token,
      dft::CudaKsResidentDensityBinding& binding, std::string&) {
    ++calls;
    if(token.epoch!=42)return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    binding.valid=valid; return GENERATIVEQC_STATUS_SUCCESS;
  }
};
struct GridView {
  bool valid=true; int device=2; const double *points,*weights,*atomic_weights;
  std::size_t point_count=3;
  explicit operator bool() const { return valid; }
};
struct GridOwner {
  double data[3]{}; GridView view{true,2,data,data+1,data+2,3}; int calls=0;
  GridView cuda_view() { ++calls;return view; }
};
struct Owner {
  DensityOwner density; DensityOwner* cuda_=&density;
  GridOwner grid_; scf::ScfOptions options_;
"""
    driver = r"""
};
int main() {
  for(bool host:{false,true})for(bool grid:{false,true})
  for(bool density:{false,true})for(bool stale:{false,true})
  for(bool wrong_device:{false,true}) {
    Owner owner;
    if(host)owner.options_.xc_execution_schedule=scf::ScfOptions::XcExecutionSchedule::HostUnfused;
    owner.grid_.view.valid=grid;owner.grid_.view.device=wrong_device?3:2;
    owner.density.valid=density;
    dft::CudaKsFinalStateToken token; if(stale)++token.epoch;
    int device=99;const double *points=owner.grid_.data,*weights=points,*raw=points;
    std::size_t count=99;std::string detail;
    const int result=owner.resident_grid(token,device,points,weights,raw,count,detail);
    const int expected=stale?3:!density?2:!grid&&host?1:!grid||wrong_device?2:0;
    assert(result==expected);
    assert(owner.density.calls==1&&owner.grid_.calls==int(!stale));
    if(result)assert(device==-1&&!points&&!weights&&!raw&&count==0);
    else assert(device==2&&points==owner.grid_.data&&weights==points+1&&raw==points+2&&count==3);
  }
  Owner absent;absent.cuda_=nullptr;
  int device=99;const double *points=nullptr,*weights=nullptr,*raw=nullptr;
  std::size_t count=99;std::string detail;
  assert(absent.resident_grid({},device,points,weights,raw,count,detail)==1);
  assert(absent.density.calls==0&&absent.grid_.calls==0);
}
"""
    unit, binary = tmp_path / "resident.cpp", tmp_path / "resident"
    unit.write_text(prefix + method + driver)
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
