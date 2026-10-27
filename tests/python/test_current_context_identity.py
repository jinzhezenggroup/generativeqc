"""Execute the native observation with a HOST-only CUDA-driver callback seam."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

PROBE = r"""
#include <iostream>
#include <stdexcept>
#include "runtime/cuda_provider_identity.hpp"
using namespace generativeqc::runtime;
struct State {std::uintptr_t pointer=17;std::uint64_t id=0;unsigned reads=0;unsigned fail=0;bool change=false;};
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
int main() {
 try {
  State state;
  CurrentContextQueries query{&state,
   [](void* p,std::uintptr_t& output)->std::uint32_t {
    auto& s=*static_cast<State*>(p);output=s.pointer;return 0;
   },
   [](void* p,std::uintptr_t,std::uint64_t& output)->std::uint32_t {
    auto& s=*static_cast<State*>(p);output=s.id+(s.change && s.reads++!=0);return s.fail;
   },
   [](void*,int& output)->std::uint32_t {output=3;return 0;},
   [](void*,int,std::array<unsigned char,16>& output)->std::uint32_t {output[7]=19;return 0;},
   [](void*,int& output)->std::uint32_t {output=13020;return 0;},
   [](void*,int& output)->std::uint32_t {output=12090;return 0;}};
  auto first=observe_current_context(query);
  require(first.status==ContextIdentityStatus::available && first.identity && first.identity->context_id==0,
          "successful CUDA context ID0 was not preserved");
  require(first.identity->device==3 && first.identity->device_uuid[7]==19,
          "device/UUID metadata was not preserved");
  state.id=99;auto changed_id=observe_current_context(query);
  require(changed_id.identity && changed_id.identity->context_id==99,
          "context ID was cached or replaced by its pointer");
  state.fail=700;auto error=observe_current_context(query);
  require(error.status==ContextIdentityStatus::query_error && !error.identity && error.error_status==700 &&
          error.error_domain==IdentityErrorDomain::driver,"actual driver error was swallowed");
  state.fail=0;state.pointer=0;
  require(observe_current_context(query).status==ContextIdentityStatus::unavailable,
          "null current context was fabricated");
  state.pointer=17;state.change=true;state.reads=0;
  require(observe_current_context(query).status==ContextIdentityStatus::changed,
          "identity changed during observation but was accepted");
  state.change=false;
  query.runtime_version=[](void*,int&)->std::uint32_t {return 2;};
  error=observe_current_context(query);
  require(error.status==ContextIdentityStatus::query_error && error.error_status==2 &&
          error.error_domain==IdentityErrorDomain::cuda,"CUDA error lost its original domain/status");
  query.current=nullptr;
  require(observe_current_context(query).status==ContextIdentityStatus::unavailable,
          "missing driver entry point was accepted");
  std::cout<<"HOST context identity availability, changes and typed statuses passed\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
"""


def test_current_context_identity(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    probe = tmp_path / "context.cpp"
    probe.write_text(PROBE)
    try:
        binary = required_native_cxx.build_executable(
            [probe],
            tmp_path / "context",
            compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
            link_args=["-ldl"],
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr, pytrace=False)
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
