"""Execute the production profiling entry with deterministic CUDA allocation faults."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_failed_event_creation_is_transactional(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Fault every base/Becke event without a CUDA runtime or physical GPU."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/dft/stationary_gradient_cuda.cuh").read_text()
    from test_stationary_task_work_budget import _block

    # Compile only this entry, not unrelated capability ABIs added beside it.
    entry = _block(source, "int stationary_profile(")
    path = tmp_path / "profile.cpp"
    path.write_text(PREFIX + entry + MAIN)
    binary = tmp_path / "profile"
    native_cxx.build_executable(
        [path], binary, compile_args=["-std=c++20"], compile_timeout=30
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <algorithm>
#include <stdexcept>
#include <set>
#include <iostream>
using cudaEvent_t = int;
"""
PREFIX += r"""
int next_id=0, calls=0, fail_at=0;
std::set<int> live;
int cudaEventCreate(cudaEvent_t* event) {
  if (++calls == fail_at) return 1;
  *event=++next_id; live.insert(*event); return 0;
}
int cudaEventDestroy(cudaEvent_t event) { live.erase(event); return 0; }
void cuda_check(int status) { if(status) throw std::runtime_error("injected"); }
namespace generativeqc_stationary_cuda {
struct Context { void check_device() {} };
struct Owner {
  Context context;
  bool profile=false;
  int stage0=0,stage1=0,stage2=0,stage3=0;
  cudaEvent_t becke_events[8]{};
};
template<class F> int guarded(Owner*,char*,size_t,F f) { try { f();return 0; } catch(...) {return 1;} }
}
"""

MAIN = r"""
int main() {
  using namespace generativeqc_stationary_cuda;
  for(int failure=1; failure<=12; ++failure) {
    Owner owner; calls=0; fail_at=failure; char error[128]{};
    if(stationary_profile(&owner,error,sizeof(error))!=1) return 1;
    if(!live.empty() || owner.profile) {std::cerr<<"partial event leak "<<failure;return 2;}
    if(owner.stage0 || owner.stage1 || owner.stage2 || owner.stage3) return 5;
    for(int event:owner.becke_events) if(event) return 6;
    fail_at=0;
    if(stationary_profile(&owner,error,sizeof(error))!=0 || live.size()!=12) return 3;
    const int created=calls;
    if(stationary_profile(&owner,error,sizeof(error))!=0 || calls!=created) return 7;
    for(int event:{owner.stage0,owner.stage1,owner.stage2,owner.stage3}) cudaEventDestroy(event);
    for(int event:owner.becke_events) cudaEventDestroy(event);
    if(!live.empty()) return 4;
  }
}
"""
