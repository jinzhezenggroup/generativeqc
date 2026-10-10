"""Fresh CUDA resource queries and the provider fallback, without a real GPU."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_hf_driver_uses_qualified_facts_without_unbatched_grid_query() -> None:
    """Only multi-system compaction retains the full grid-limit provider query."""
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    beginning = source.index("std::vector<RhfBucketItem> execute_hf_cuda_bucket(")
    compaction = source.index("dim3 direct_shell_quartet_compaction_grid", beginning)
    setup = source[beginning:compaction]
    assert (
        "runtime::cuda_device_facts(device_id, direct_target, direct_device_name)"
        in setup
    )
    query = setup.index("cudaGetDeviceProperties(")
    guard = (
        "if (requested_quartet_direct && !requested_bounded_direct_streaming "
        "&& batch_size > 1)"
    )
    assert setup.index(guard) < query
    assert setup.index("first_setup ? requested_bounded_direct_streaming") < query
    assert query < setup.index("++plan.execution_generation")
    assert query < setup.index("SourceSite::hf_positions_input")
    assert setup.count("cudaGetDeviceProperties(") == 1
    assert "compaction_properties.maxGridSize[1]" in setup
    assert "fill_global_failure(outputs, cuda_status(compaction_target_error))" in setup
    grid = source[compaction:]
    assert (
        "if (requested_quartet_direct && !requested_bounded_direct_streaming)" in grid
    )
    assert "cudaGetDeviceProperties(" not in grid
    assert "batch_size <= maximum_compaction_grid_y" in grid


def test_compaction_query_failure_preserves_cached_geometry(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Inject a provider error into actual route/query/upload blocks in source order.

    This host trace models a retained Fleet plan; it does not emulate CUDA or
    establish that a real provider naturally produces this recoverable error.
    """
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text()

    def block_at(start: int) -> str:
        opening = source.index("{", start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]

    query_start = source.index(
        "if (requested_quartet_direct && !requested_bounded_direct_streaming "
        "&& batch_size > 1)"
    )
    route_start = source.index(
        "if (requested_quartet_direct && first_setup && !requested_bounded_direct_streaming)"
    )
    route_else = source.index("else if (requested_quartet_direct)", route_start)
    route_end = route_else + len(block_at(route_else))
    upload_start = source.rfind(
        "if (geometry_changed) {",
        0,
        source.index("const generativeqc_status position_status ="),
    )
    operations = "\n".join(
        text
        for _, text in sorted(
            (
                (route_start, source[route_start:route_end]),
                (query_start, block_at(query_start)),
                (upload_start, block_at(upload_start)),
            )
        )
    )
    probe = tmp_path / "compaction_geometry.cpp"
    probe.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>
using generativeqc_status = int;
constexpr int cudaSuccess = 0, GENERATIVEQC_STATUS_SUCCESS = 0;
constexpr int GENERATIVEQC_STATUS_INVALID_ARGUMENT = 4;
constexpr unsigned kMixedFockMinimumAngularOrder = 2;
struct cudaDeviceProp { int maxGridSize[3]{}; };
struct RhfBucketItem { int status{}; };
struct GeneratedShellTask { char value; };
struct Plan {
  std::vector<double> cached_positions{1.0};
  bool initialized = true, bounded_direct_streaming = false;
  std::size_t total_shell_quartet_tiles = 10;
};
struct Host {
  std::vector<double> positions;
  int shell_direct_ao_offsets{}, shell_angular{}, system_shell_pair_offsets{};
  int shell_pair_first{}, shell_pair_second{};
};
std::size_t layout_tile_count = 10, arena_bytes = 500;
namespace detail {
constexpr std::size_t kDirectFixedTopologyTileLimit = 1000;
struct DirectQuartetTaskLayout {
  std::size_t shell_quartet_count{}, exact_tile_count{};
};
bool make_direct_quartet_task_layout(int, int, int, int, int, unsigned,
                                   DirectQuartetTaskLayout& layout) {
  layout.shell_quartet_count = 8;
  layout.exact_tile_count = layout_tile_count;
  return true;
}
}
int injected_error = 0, queries = 0, uploads = 0;
int cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  assert(device == 7);
  ++queries;
  properties->maxGridSize[1] = 23;
  return injected_error;
}
int cuda_status(int error) { return error == cudaSuccess ? 0 : 1000 + error; }
void fill_global_failure(std::vector<RhfBucketItem>& outputs, int status) {
  for (auto& output : outputs) output.status = status;
}
enum class SourceRole { prepare };
enum class SourceSite { hf_positions_input };
enum class SourcePayload { input_positions };
namespace runtime {
int residency_upload(int execution, SourceRole role, SourceSite site,
                     SourcePayload payload, double* out, const double* in,
                     std::size_t bytes, int stream) {
  assert(execution == 17 && role == SourceRole::prepare);
  assert(site == SourceSite::hf_positions_input);
  assert(payload == SourcePayload::input_positions);
  assert(stream == 9);
  ++uploads;
  std::copy_n(in, bytes / sizeof(double), out);
  return cudaSuccess;
}
}
std::vector<RhfBucketItem> execute(Plan& plan, Host host, double& device_position,
    bool requested_quartet_direct, bool requested_bounded_direct_streaming,
    std::size_t batch_size) {
  std::vector<RhfBucketItem> outputs(batch_size);
  const int device_id = 7;
  const int residency_execution = 17;
  struct { int stream_ = 9; } resources;
  double* positions = &device_position;
  const bool first_setup = !plan.initialized;
  const bool geometry_changed = first_setup || plan.cached_positions != host.positions;
  const std::size_t total_shell_quartets = 8;
  struct {
    struct { std::size_t arena_maximum_bytes; } fixed_topology;
  } direct_schedule{{arena_bytes}};
  detail::DirectQuartetTaskLayout direct_task_layout{};
  std::size_t total_shell_quartet_tiles = 0;
  std::size_t maximum_compaction_grid_y = 0;
"""
        + operations
        + r"""
  assert(maximum_compaction_grid_y ==
      (requested_quartet_direct && !requested_bounded_direct_streaming && batch_size > 1
       ? 23U : 0U));
  plan.cached_positions = host.positions;
  return outputs;
}
int main() {
  Plan plan;
  double device_position = 1.0;
  injected_error = 30;
  auto failed = execute(plan, {{2.0}}, device_position, true, false, 2);
  for (const auto& item : failed) assert(item.status == 1030);
  assert(queries == 1 && uploads == 0);
  assert(plan.cached_positions[0] == 1.0 && device_position == 1.0);
  injected_error = 0;
  auto retry = execute(plan, {{1.0}}, device_position, true, false, 2);
  for (const auto& item : retry) assert(item.status == 0);
  assert(queries == 2 && uploads == 0 && device_position == 1.0);
  execute(plan, {{2.0}}, device_position, true, false, 2);
  assert(queries == 3 && uploads == 1 && device_position == 2.0);
  for (bool quartet : {false, true})
    for (bool bounded : {false, true})
      for (std::size_t batch : {1U, 2U}) {
        const int before = queries;
        plan.bounded_direct_streaming = bounded;
        execute(plan, {{2.0}}, device_position, quartet, bounded, batch);
        assert(queries - before == int(quartet && !bounded && batch > 1));
      }
  // The production admission block can select bounded streaming only after
  // exact tile counts are known. Querying the initial false value is incorrect.
  for (std::size_t tiles : {1001U, 501U}) {
    Plan cold;
    cold.initialized = false;
    layout_tile_count = tiles;
    const int before = queries;
    injected_error = 30;
    auto bounded = execute(cold, {{2.0}}, device_position, true, false, 2);
    for (const auto& item : bounded) assert(item.status == 0);
    assert(queries == before);
  }
  // A captured plan keeps its bounded choice even when current inputs would
  // initially request the fixed-topology route. The reverse is also honored.
  layout_tile_count = 10;
  plan.bounded_direct_streaming = true;
  const int before = queries;
  auto inherited = execute(plan, {{2.0}}, device_position, true, false, 2);
  for (const auto& item : inherited) assert(item.status == 0);
  assert(queries == before);
  plan.bounded_direct_streaming = false;
  injected_error = 0;
  execute(plan, {{2.0}}, device_position, true, true, 2);
  assert(queries == before + 1);
}
"""
    )
    binary = required_native_cxx.build_executable(
        [probe], tmp_path / "compaction_geometry", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(binary)], check=True, timeout=10)


HEADER = r"""
#pragma once
#include <cstddef>
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorNotSupported=801;
enum cudaDeviceAttr {
  cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
  cudaDevAttrWarpSize, cudaDevAttrMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerMultiProcessor,
  cudaDevAttrMaxBlocksPerMultiprocessor, cudaDevAttrMaxRegistersPerMultiprocessor,
  cudaDevAttrMaxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin,
  cudaDevAttrMaxSharedMemoryPerMultiprocessor, cudaDevAttrMultiProcessorCount
};
struct cudaDeviceProp {
  char name[256]{};
  int major{},minor{},warpSize{},maxThreadsPerBlock{},maxThreadsPerMultiProcessor{};
  int maxBlocksPerMultiProcessor{},regsPerMultiprocessor{},multiProcessorCount{};
  std::size_t sharedMemPerBlock{},sharedMemPerBlockOptin{},sharedMemPerMultiprocessor{};
  std::size_t totalGlobalMem{};
};
cudaError_t cudaDeviceGetAttribute(int*,cudaDeviceAttr,int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);
cudaError_t cudaPeekAtLastError();
cudaError_t cudaGetLastError();
"""

HARNESS = r"""
#include "runtime/cuda_device_facts.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
using namespace generativeqc::runtime;
int mode=0, epoch=0, attributes=0, properties=0, opens=0, last_error=0, clears=0;
int values[]={12,0,32,1024,2048,32,65536,49152,101376,102400,100};
int sm(int device) { return 100+5*device+epoch; }
std::size_t memory(int device) { return (std::size_t{1}<<35)+device+epoch; }
void device_name(char* p,int size,int device) { std::snprintf(p,size,"GPU-%d-%d",device,epoch); }
int get_device(int* out,int device) { *out=100+device; return mode==4 ? 3 : 0; }
int get_name(char* out,int size,int device) { device_name(out,size,device-100); return 0; }
int get_memory(std::size_t* out,int device) { *out=memory(device-100); return 0; }
extern "C" void* dlopen(const char* name,int flags) noexcept {
  ++opens;
  assert(std::strcmp(name,"libcuda.so.1")==0 && (flags&RTLD_NOLOAD));
  return mode==2 ? nullptr : reinterpret_cast<void*>(1);
}
extern "C" void* dlsym(void*,const char* name) noexcept {
  if (std::strcmp(name,"cuDeviceGet")==0) return reinterpret_cast<void*>(&get_device);
  if (std::strcmp(name,"cuDeviceGetName")==0) return reinterpret_cast<void*>(&get_name);
  assert(std::strcmp(name,"cuDeviceTotalMem_v2")==0);
  return mode==3 ? nullptr : reinterpret_cast<void*>(&get_memory);
}
cudaError_t cudaDeviceGetAttribute(int* out,cudaDeviceAttr attr,int device) {
  ++attributes;
  if (device<0) return last_error=10;
  if (attr==cudaDevAttrMaxThreadsPerBlock && (mode==1 || mode==6 || mode==10))
    return last_error=cudaErrorNotSupported;
  if (attr==cudaDevAttrMaxThreadsPerBlock && mode==8) return last_error=cudaErrorInvalidValue;
  if (attr==cudaDevAttrMaxThreadsPerBlock && mode==5) return last_error=42;
  *out=attr==cudaDevAttrMultiProcessorCount ? sm(device) : values[int(attr)];
  return cudaSuccess;
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* out,int device) {
  ++properties;
  if (device<0) return last_error=10;
  if (mode==6) return last_error=43;
  *out={};
  device_name(out->name,sizeof(out->name),device);
  out->major=values[0];out->minor=values[1];out->warpSize=values[2];
  out->maxThreadsPerBlock=values[3];out->maxThreadsPerMultiProcessor=values[4];
  out->maxBlocksPerMultiProcessor=values[5];out->regsPerMultiprocessor=values[6];
  out->sharedMemPerBlock=values[7];out->sharedMemPerBlockOptin=values[8];
  out->sharedMemPerMultiprocessor=values[9];out->multiProcessorCount=sm(device);
  out->totalGlobalMem=memory(device);
  return cudaSuccess;
}
cudaError_t cudaPeekAtLastError() { return last_error; }
cudaError_t cudaGetLastError() {
  ++clears;
  const auto result=last_error;
  last_error=0;
  // An asynchronous illegal-address failure can surface during cleanup.
  if (mode==10) return last_error=700;
  return result;
}
void check(int device,int expected_error=0) {
  CudaTargetInfo result{}; result.warp_size=999;
  char name[256];std::memset(name,'x',sizeof(name));
  assert(cuda_device_facts(device,result,name)==expected_error);
  if (expected_error) {
    assert(last_error==expected_error);
    assert(result.warp_size==0 && result.total_global_memory==0);
    for(char c:name) assert(c==0);
    return;
  }
  // Model the next successful kernel's launch check: the handled probe error
  // must not survive the authoritative full-property fallback.
  assert(cudaPeekAtLastError()==cudaSuccess);
  assert(result.compute_capability_major==12 && result.compute_capability_minor==0);
  assert(result.warp_size==32 && result.maximum_threads_per_block==1024);
  assert(result.maximum_threads_per_sm==2048 && result.maximum_blocks_per_sm==32);
  assert(result.registers_per_sm==65536 && result.shared_memory_per_block==49152);
  assert(result.shared_memory_per_block_optin==101376 && result.shared_memory_per_sm==102400);
  assert(result.multiprocessor_count==unsigned(sm(device)) && result.total_global_memory==memory(device));
  char expected[256]{};device_name(expected,sizeof(expected),device);
  assert(std::strcmp(name,expected)==0);
}
int main(int argc,char** argv) {
  assert(argc==2);mode=std::atoi(argv[1]);
  if (mode==9 || mode==11) {
    last_error=mode==9 ? 700 : cudaErrorInvalidValue;
    check(7,last_error);
    assert(attributes==0 && properties==0 && clears==0);
    return 0;
  }
  if (mode==10) {
    check(7,700);
    assert(properties==0 && clears==1);
    return 0;
  }
  if (mode==5 || mode==6) {
    check(7,mode==5 ? 42 : 43);
    assert(properties==(mode==5 ? 0 : 1));
    assert(clears==(mode==5 ? 0 : 1));
    return 0;
  }
  check(7);++epoch;check(7);check(2);check(7);
  // Both repeated and changed ordinals must observe fresh device/resource facts.
  if (mode==0) assert(properties==0 && attributes==44 && opens==1);
  else assert(properties==4);
  assert(clears==((mode==1 || mode==8) ? 4 : 0));
  if (mode==7) assert(attributes==0 && opens==0); // CuMetal bypasses NVIDIA discovery.
  check(-1,10);
}
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ELF loader gate")
@pytest.mark.parametrize("provider", ["nvidia", "cumetal"])
def test_fresh_facts_and_failure_fallback(tmp_path: Path, provider: str) -> None:
    """Compile the production query owner against independent, changing device facts."""
    compiler = shutil.which("c++")
    ccache = shutil.which("ccache")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    if ccache is None:
        pytest.fail("ccache is required for native qualification")
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
    (tmp_path / "cuda_runtime_api.h").write_text(HEADER)
    source = tmp_path / "probe.cpp"
    source.write_text(HARNESS)
    objects = []
    for i, part in enumerate((source, ROOT / "src/runtime/cuda_device_facts.cpp")):
        output = tmp_path / f"part{i}.o"
        subprocess.run(
            [
                ccache,
                compiler,
                "-std=c++20",
                "-O2",
                "-I",
                str(tmp_path),
                "-I",
                str(ROOT / "src"),
                f"-DGENERATIVEQC_CUDA_PROVIDER_CUMETAL={int(provider == 'cumetal')}",
                "-c",
                str(part),
                "-o",
                str(output),
            ],
            check=True,
            timeout=60,
        )
        objects.append(str(output))
    binary = tmp_path / "facts"
    subprocess.run([compiler, *objects, "-ldl", "-o", str(binary)], check=True)
    for mode in [7] if provider == "cumetal" else [*range(7), 8, 9, 10, 11]:
        subprocess.run([str(binary), str(mode)], check=True, timeout=10)


def test_real_cuda_resource_facts(tmp_path: Path) -> None:
    """Require exact agreement with the full query on the allocated CUDA device."""
    if os.environ.get("GENERATIVEQC_TEST_CUDA_DEVICE_FACTS") != "1":
        pytest.skip("explicit CUDA device-fact qualification is disabled")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("CUDA qualification requires a Slurm allocation")
    compiler, ccache = shutil.which("nvcc"), shutil.which("ccache")
    if compiler is None:
        pytest.skip("NVCC required for native CUDA qualification")
    if ccache is None:
        pytest.fail("ccache is required for native qualification")
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
    objects = []
    for i, source in enumerate(
        (
            ROOT / "tests/native/test_cuda_device_facts.cpp",
            ROOT / "src/runtime/cuda_device_facts.cpp",
        )
    ):
        output = tmp_path / f"part{i}.o"
        subprocess.run(
            [
                ccache,
                compiler,
                "-std=c++20",
                "-O2",
                "-I",
                str(ROOT / "src"),
                "-c",
                str(source),
                "-o",
                str(output),
            ],
            check=True,
            timeout=60,
        )
        objects.append(str(output))
    binary = tmp_path / "facts"
    subprocess.run(
        [compiler, *objects, "-ldl", "-o", str(binary)], check=True, timeout=60
    )
    subprocess.run([str(binary)], check=True, timeout=30)
