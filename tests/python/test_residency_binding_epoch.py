"""Canonical observer binding lifetime behavior through separate native TUs.

These CPU probes do not qualify an installed scientific DSO or old inline clients.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = r"""
#include "runtime/residency_observer.hpp"
using namespace generativeqc::runtime;
extern "C" int api_bind(ResidencyObserver cb,void* ctx){return bind_residency_observer(cb,ctx);}
extern "C" int api_unbind(ResidencyObserver cb,void* ctx,std::uint64_t* e){return unbind_residency_observer(cb,ctx,e);}
extern "C" std::uint64_t epoch(){return residency_binding_epoch.value;}
extern "C" bool admissible(){return residency_binding_epoch_admissible();}
extern "C" bool exhausted(){return residency_binding_epoch.exhausted;}
extern "C" void near_exhaustion(){residency_binding_epoch={std::numeric_limits<std::uint64_t>::max()-1,false};}
"""
SOURCE = r"""
#include "runtime/residency_observer.hpp"
extern "C" void emit(){const std::uint64_t row[]{991,1};generativeqc::runtime::dispatch_residency_source(row,2);}
"""
PROBE = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
using Callback=void(*)(const std::uint64_t*,std::size_t,void*);
extern "C" int api_bind(Callback,void*);
extern "C" int api_unbind(Callback,void*,std::uint64_t*);
extern "C" std::uint64_t epoch();
extern "C" bool admissible();
extern "C" bool exhausted();
extern "C" void near_exhaustion();
extern "C" void emit();
int context=1,foreign=2;bool reentrant=false,throw_callback=false;
unsigned delivered=0;
void callback(const std::uint64_t*,std::size_t,void*){
 ++delivered;if(throw_callback)throw std::runtime_error("observer failure");
 if(reentrant){auto stamp=epoch();std::uint64_t e=99;assert(api_bind(callback,&context)!=0);assert(api_unbind(callback,&context,&e)!=0);assert(epoch()==stamp);}
}
int main(int argc,char**argv){
 assert(argc==2);std::string mode=argv[1];std::uint64_t errors=0;
 assert(epoch()==0&&!admissible()&&!exhausted());
 assert(api_bind(nullptr,&context)!=0&&epoch()==0);
 assert(api_bind(callback,&context)==0&&epoch()>0&&admissible());auto first=epoch();
 if(mode=="mutation"){
  assert(api_bind(callback,&context)!=0&&epoch()==first);
  assert(api_unbind(callback,&foreign,&errors)!=0&&epoch()==first);
  assert(api_unbind(callback,&context,nullptr)!=0&&epoch()==first);
  emit();assert(delivered==1&&epoch()==first);
 }else if(mode=="aba"){
  assert(api_unbind(callback,&context,&errors)==0&&epoch()>first);auto detached=epoch();
  assert(api_bind(callback,&context)==0&&epoch()>detached&&epoch()!=first);
 }else if(mode=="thread"){
  std::thread t([&]{assert(epoch()==0&&!admissible());assert(api_unbind(callback,&context,&errors)!=0);assert(epoch()==0);emit();});t.join();
  assert(epoch()==first&&delivered==0);emit();assert(delivered==1);
 }else if(mode=="reentrant-exception"){
  reentrant=true;emit();assert(epoch()==first);reentrant=false;
  throw_callback=true;emit();throw_callback=false;assert(epoch()==first);
  assert(api_unbind(callback,&context,&errors)==0&&errors==1);assert(api_bind(callback,&context)==0);
 }else if(mode=="exhaustion"){
  near_exhaustion();assert(admissible());
  assert(api_unbind(callback,&context,&errors)==0);assert(epoch()==std::numeric_limits<std::uint64_t>::max()&&!admissible());
  assert(api_bind(callback,&context)==0&&exhausted()&&!admissible());
  assert(epoch()==std::numeric_limits<std::uint64_t>::max());
  emit();assert(delivered==1); // Legacy delivery still works after diagnostic exhaustion.
 }else return 2;
 assert(api_unbind(callback,&context,&errors)==0);emit();return 0;
}
"""


@pytest.fixture(scope="module")
def epoch_probe(
    native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    folder = tmp_path_factory.mktemp("binding-epoch")
    sources = []
    for name, text in (("bridge", BRIDGE), ("source", SOURCE), ("probe", PROBE)):
        source = folder / f"{name}.cpp"
        source.write_text(text)
        sources.append(source)
    output = folder / "probe"
    native_cxx.build_executable(
        sources,
        output,
        compile_args=("-std=c++20", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'src'}"),
        link_args=("-pthread",),
    )
    return output


@pytest.mark.parametrize(
    "mode", ["mutation", "aba", "thread", "reentrant-exception", "exhaustion"]
)
def test_canonical_binding_epoch(epoch_probe: Path, mode: str) -> None:
    subprocess.run(
        [str(epoch_probe), mode], check=True, capture_output=True, text=True, timeout=10
    )
