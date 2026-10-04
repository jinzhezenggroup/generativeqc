"""Fault-inject the production optional-page block with a host CUDA stand-in.

The real page admission, pointer layout, accounting and exception scopes execute
verbatim. The mandatory owner and allocator are stand-ins; this is not evidence
for device allocation behavior or CUDA concurrency.
"""

import subprocess
from pathlib import Path

import pytest
from test_direct_force_pages import ROOT, compile_probe
from test_direct_shell_derivative_host_lifetime import _extract_function


@pytest.fixture(scope="module")
def allocation_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    start = source.index(
        "  if (force_capability && cuda_policy::compact_bounded_force_requested())"
    )
    block = _extract_function(source, source[start : source.index("\n", start)])
    begin = source.index(
        "std::unique_ptr<GeneratedExchangePlan> prepare_generated_exchange("
    )
    end = source.index("\nnamespace {", begin)
    preparation = source[begin:end]
    catches = preparation[preparation.index("} catch (cudaError_t error)") :]
    probe = (
        PREFIX
        + """
std::unique_ptr<Plan> prepare(std::size_t budget, bool force_capability) try {
  auto plan = std::make_unique<Plan>();
  const int host=0;
"""
        + block
        + "\nreturn plan;\n"
        + catches
        + DRIVER
    )
    return compile_probe(tmp_path_factory.mktemp("force-page-allocation"), probe)


@pytest.mark.parametrize("mode", range(10))
def test_optional_allocation_faults_and_budget(
    allocation_probe: Path, mode: int
) -> None:
    subprocess.run([str(allocation_probe), str(mode)], check=True, timeout=10)


def test_schedule_is_checkpoint_extension() -> None:
    from generativeqc.resources_hf import (
        _CUDA_SCHEDULE_EXTENSION_VARIABLES,
        _CUDA_SCHEDULE_VARIABLES,
    )

    name = "GENERATIVEQC_BOUNDED_FORCE_SCHEDULE"
    assert _CUDA_SCHEDULE_EXTENSION_VARIABLES.count(name) == 1
    assert _CUDA_SCHEDULE_VARIABLES.count(name) == 1


PREFIX = r"""
#include <cassert>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
#include "generated_direct_force_pages.hpp"
#include "scf/direct_block_domain.hpp"
using namespace generativeqc::scf;
using namespace generativeqc::scf::cuda_execution;
enum cudaError_t {cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown};
int mode=0, calls=0, live=0;
cudaError_t pending=cudaSuccess;
bool requested=true;
void check(cudaError_t error) {if(error!=cudaSuccess) throw error;}
cudaError_t cudaGetLastError() {auto result=pending; pending=cudaSuccess; return result;}
namespace cuda_policy { bool compact_bounded_force_requested() {return requested;} }
namespace generativeqc::scf::detail {
constexpr unsigned kDirectQuartetShellClassCount=55, kBoundedDirectShellPairBlockSize=32;
}
namespace runtime {
std::size_t size_add(std::size_t left,std::size_t right) {return left+right;}
cudaError_t resource_cuda_free(void* storage) {std::free(storage); --live; return cudaSuccess;}
cudaError_t resource_cuda_malloc(void** output,std::size_t bytes,bool* host_oom) {
  ++calls;
  if(mode==1 || mode==2 || mode==3 || mode==8 || mode==9) {
    if(mode==9) throw std::bad_alloc();
    *output=nullptr; *host_oom=mode==2;
    pending=mode==8 ? cudaErrorUnknown : cudaErrorMemoryAllocation;
    return mode==3 ? cudaErrorUnknown : cudaErrorMemoryAllocation;
  }
  *output=std::malloc(bytes); assert(*output); ++live; return cudaSuccess;
}
}
struct Allocations: std::vector<void*> {
  void push_back(void* storage) {
    if(mode==4) throw std::bad_alloc();
    std::vector<void*>::push_back(storage);
  }
};
struct ActiveShellQuartetTile {std::uint32_t first_pair,second_pair,tile;};
struct BoundedForcePage {
  std::size_t block_capacity{},candidate_capacity{};
  ActiveShellQuartetTile *input{},*tasks{};
  std::uint8_t* classes{};
  std::uint32_t *counts{},*offsets{},*writes{},*heads{};
  void* profile{};
};
struct Shared {struct {std::size_t total_shell_pair_block_quartets=7;} batch;};
struct Plan {
  std::unique_ptr<Shared> shared=std::make_unique<Shared>();
  detail::BoundedDirectBlockDomain bounded_block_domain{};
  BoundedForcePage force_page{};
  Allocations allocations;
  std::size_t device_bytes=100;
  std::uint64_t force_page_class_mask{};
  ~Plan() {for(auto* storage:allocations) runtime::resource_cuda_free(storage);}
};
std::uint64_t present_direct_shell_class_mask(int) {return 123;}
"""

DRIVER = r"""
int main(int argc,char** argv) {
  assert(argc==2); mode=std::atoi(argv[1]);
  bool propagated=false;
  requested=mode!=6;
  try {
    auto plan=prepare(mode==5 ? 100 : 200000,mode!=7);
    if(mode==2 || mode==4 || mode==9) assert(!plan);
    else {
      assert(plan);
      if(mode==0) {
        const auto layout=plan_force_page(7,199900);
        assert(plan->device_bytes==100+layout.bytes);
        assert(plan->force_page.block_capacity==layout.blocks);
        assert(plan->force_page.candidate_capacity==layout.candidates);
        assert(plan->force_page_class_mask==123 && plan->force_page.profile==nullptr);
        assert(plan->allocations.size()==1);
        const auto* bytes=static_cast<std::byte*>(plan->allocations.front());
        assert(reinterpret_cast<const std::byte*>(plan->force_page.input)==bytes+layout.input);
        assert(reinterpret_cast<const std::byte*>(plan->force_page.tasks)==bytes+layout.tasks);
        assert(reinterpret_cast<const std::byte*>(plan->force_page.classes)==bytes+layout.classes);
        assert(reinterpret_cast<const std::byte*>(plan->force_page.counts)==bytes+layout.counts);
        assert(reinterpret_cast<const std::byte*>(plan->force_page.offsets)==bytes+layout.offsets);
        assert(reinterpret_cast<const std::byte*>(plan->force_page.writes)==bytes+layout.writes);
        assert(reinterpret_cast<const std::byte*>(plan->force_page.heads)==bytes+layout.heads);
      } else {
        assert(!plan->force_page.block_capacity && plan->device_bytes==100);
        assert(plan->allocations.empty());
      }
    }
  } catch(cudaError_t error) {assert(error==cudaErrorUnknown); propagated=true;}
  assert(propagated==(mode==3 || mode==8));
  assert(calls==((mode==5 || mode==6 || mode==7) ? 0 : 1));
  assert(live==0);
  if(mode==1 || mode==2) assert(pending==cudaSuccess);
}
"""
