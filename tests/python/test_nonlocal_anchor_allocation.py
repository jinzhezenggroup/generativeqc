"""Compile the actual optional frozen-potential lease with allocation faults."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_nonlocal_anchor_allocation_and_retirement(
    required_native_cxx: NativeCxx, tmp_path: Path
) -> None:
    """Device refusal falls back; host/runtime faults propagate without leaks."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()

    def method(signature: str) -> str:
        start = source.index(signature)
        first = source.index("{", start)
        depth = 1
        end = first + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]

    methods = "\n".join(
        method(signature)
        for signature in (
            "void prepare_nonlocal_anchor_storage(",
            "std::size_t retire_nonlocal_anchor_storage(",
        )
    )
    unit = tmp_path / "allocation.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <limits>
enum Error { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown };
int fault=0, calls=0, live=0;
Error cudaGetLastError() { return fault==3 ? cudaErrorUnknown : cudaSuccess; }
void check(Error status) { if(status!=cudaSuccess) throw status; }
std::size_t sum(std::size_t a,std::size_t b) {
 if(a>SIZE_MAX-b) throw std::overflow_error("sum"); return a+b;
}
std::size_t product(std::size_t a,std::size_t b) {
 if(b && a>SIZE_MAX/b) throw std::overflow_error("product"); return a*b;
}
namespace runtime {
struct DeviceUnpromisedCapacityScope {};
Error resource_cuda_malloc(void** p,std::size_t n,bool* host) {
 ++calls; *host=fault==2;
 if(fault==1 || fault==2 || fault==3) return cudaErrorMemoryAllocation;
 if(fault==4) return cudaErrorUnknown;
 *p=std::malloc(n); assert(*p); ++live; return cudaSuccess;
}
Error resource_cuda_free(void* p) {
 if(fault==5) return cudaErrorUnknown;
 std::free(p); --live; return cudaSuccess;
}
}
namespace nlc { enum class Vv10Variant { vv10, rvv10 }; }
struct Correlation {
 struct Params { nlc::Vv10Variant variant=nlc::Vv10Variant::vv10; } params;
 Params parameters() const { return params; }
};
struct Owner {
 bool device_nonlocal=true, nonlocal_xc=false;
 unsigned spins=1;
 std::size_t elements=144;
 Correlation value;
 Correlation* nonlocal_correlation=&value;
 void* nonlocal_anchor_arena=nullptr;
 double* nonlocal_anchor_potential=nullptr;
 int* nonlocal_anchor_error=nullptr;
 std::uint64_t nonlocal_anchor_generation=0;
 struct Resource { std::size_t state_device_bytes=73; } resource;
"""
        + methods
        + r"""
};
int main() {
 for(int scenario=0;scenario<5;++scenario) {
  Owner owner;
  if(scenario==0) owner.device_nonlocal=false;
  if(scenario==1) owner.nonlocal_xc=true;
  if(scenario==2) owner.spins=2;
  if(scenario==3) owner.value.params.variant=nlc::Vv10Variant::rvv10;
  calls=0; owner.prepare_nonlocal_anchor_storage(scenario==4 ? 1 : SIZE_MAX);
  assert(calls==0 && live==0 && owner.resource.state_device_bytes==73);
 }
 for(fault=1;fault<=4;++fault) {
  Owner owner; bool threw=false;
  try { owner.prepare_nonlocal_anchor_storage(SIZE_MAX); }
  catch(Error error) { threw=true; assert(error!=cudaSuccess); }
  assert(threw==(fault!=1));
  assert(!owner.nonlocal_anchor_arena && live==0 && owner.resource.state_device_bytes==73);
 }
 fault=0; Owner owner; owner.prepare_nonlocal_anchor_storage(SIZE_MAX);
 const auto bytes=(owner.elements+1)*sizeof(double);
 assert(live==1 && owner.resource.state_device_bytes==73+bytes);
 assert(owner.nonlocal_anchor_error==reinterpret_cast<int*>(owner.nonlocal_anchor_potential+owner.elements));
 owner.nonlocal_anchor_generation=21;
 fault=5; try { owner.retire_nonlocal_anchor_storage(); assert(false); } catch(Error) {}
 assert(live==1 && owner.nonlocal_anchor_generation==21 && owner.resource.state_device_bytes==73+bytes);
 fault=0; assert(owner.retire_nonlocal_anchor_storage()==bytes);
 assert(live==0 && !owner.nonlocal_anchor_arena && !owner.nonlocal_anchor_potential);
 assert(!owner.nonlocal_anchor_error && owner.nonlocal_anchor_generation==0);
 assert(owner.resource.state_device_bytes==73 && owner.retire_nonlocal_anchor_storage()==0);
}
"""
    )
    binary = tmp_path / "allocation"
    required_native_cxx.build_executable(
        [unit], binary, compile_args=("-std=c++20",)
    )
    subprocess.run([str(binary)], check=True, timeout=20)
