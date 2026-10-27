"""Exercise production ledger bookkeeping with host-only CUDA allocation doubles.

The real C API translation unit and allocation wrapper are compiled unchanged.
These tests do not establish real-GPU allocation or numerical endpoint gates.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc.resources_native import NativeDeviceJournal, NativeDeviceLedger
from generativeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("standard", ("c++17", "c++20"))
def test_ledger_header_compiles_without_cmake_source_include_path(
    tmp_path: Path, required_native_cxx: NativeCxx, standard: str
) -> None:
    """Offline generated providers include runtime headers without CMake's -I src."""
    source = tmp_path / "ledger_header.cpp"
    source.write_text(f'#include "{ROOT / "src/runtime/resource_ledger.hpp"}"\n')
    required_native_cxx.compile_object(
        source, tmp_path / "ledger_header.o", args=(f"-std={standard}",)
    )


CUDA_STUB = r"""
#pragma once
#include <cstddef>
#include <cstdlib>
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2;
constexpr int cudaErrorInvalidValue = 3, cudaErrorInvalidDevice = 4;
inline int visible_device = 0;
inline int cudaGetDevice(int* device) { *device = visible_device; return cudaSuccess; }
inline int cudaMalloc(void** pointer, std::size_t bytes) {
  *pointer = std::malloc(bytes);
  return *pointer ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline int cudaFree(void* pointer) { std::free(pointer); return cudaSuccess; }
inline int cudaMallocAsync(void** pointer, std::size_t bytes, cudaStream_t) {
  return cudaMalloc(pointer, bytes);
}
inline int cudaFreeAsync(void* pointer, cudaStream_t) { return cudaFree(pointer); }
inline int cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
"""

DRIVER = r"""
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include "runtime/resource_cuda.cuh"
#include "api/device_resources.hpp"

extern "C" {
void* generativeqc_resource_ledger_create_v1(std::size_t, int);
int generativeqc_resource_ledger_create_from_buffers_v1(const std::uint64_t*, std::size_t,
                                                       void**, std::uint64_t*, std::size_t);
void generativeqc_resource_ledger_destroy_v1(void*);
int generativeqc_resource_ledger_bind_v1(void*);
int generativeqc_resource_ledger_read_v1(void*, std::uint64_t*);
int generativeqc_resource_ledger_read_v2(void*, std::uint64_t*, std::size_t);
void* generativeqc_resource_journal_create_v1(void*, std::size_t);
int generativeqc_resource_journal_read_v1(void*, std::uint64_t*, std::size_t, std::uint64_t*);
void generativeqc_resource_journal_destroy_v1(void*);
int generativeqc_resource_tracking_begin_v1(unsigned);
int generativeqc_resource_tracking_end_v1(std::uint64_t*, std::uint64_t*);
}

using namespace generativeqc::runtime;
using Metrics = std::array<std::uint64_t, 5>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Observation {
  void* handle{};
  explicit Observation(bool planned = false) {
    if (planned) {
      const std::array<std::uint64_t, 18> rows{128, 4, 0, 0, 0, 0,
                                               16, 4, 0, 1, 1, 0,
                                             1024, 4, 0, 1, 1, 1};
      std::array<std::uint64_t, 3> receipt{};
      require(generativeqc_resource_ledger_create_from_buffers_v1(
                  rows.data(), rows.size(), &handle, receipt.data(), receipt.size()) == 0 &&
                  receipt == std::array<std::uint64_t, 3>{0, 128, 1040},
              "selected inventory admission");
    } else {
      handle = generativeqc_resource_ledger_create_v1(128, 0);
    }
    require(handle != nullptr, "create");
    bind();
  }
  ~Observation() { end(); generativeqc_resource_ledger_destroy_v1(handle); }
  void bind() {
    require(generativeqc_resource_tracking_begin_v1(0) == 0, "begin");
    require(generativeqc_resource_ledger_bind_v1(handle) == 0, "bind");
  }
  void end() {
    std::uint64_t peak{}, samples{};
    require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0, "end");
  }
  void check(Metrics expected) {
    Metrics observed{};
    require(generativeqc_resource_ledger_read_v2(handle, observed.data(), observed.size()) == 0,
            "v2 read");
    require(observed == expected, "v2 metrics differ");
    observed[4] = 991;
    require(generativeqc_resource_ledger_read_v1(handle, observed.data()) == 0, "v1 read");
    expected[4] = 991;
    require(observed == expected, "v1 ABI changed");
  }
};

void check_journal(void* capture, std::array<std::uint64_t, 4> expected,
                   const std::vector<std::uint64_t>& expected_records) {
  std::array<std::uint64_t, 4> state{};
  require(generativeqc_resource_journal_read_v1(capture, nullptr, 0, state.data()) == 0,
          "journal query");
  require(state == expected, "journal state differs");
  std::vector<std::uint64_t> records(expected_records.size() + 3, 991);
  require(generativeqc_resource_journal_read_v1(capture, records.data(), records.size(),
                                               state.data()) == 0, "journal read");
  require(std::equal(expected_records.begin(), expected_records.end(), records.begin()),
          "journal events differ");
  require(records.back() == 991, "journal wrote past records");
}

void journal_lifetime() {
  auto* handle = generativeqc_resource_ledger_create_v1(128, 0);
  require(handle != nullptr, "journal owner create");
  require(generativeqc_resource_tracking_begin_v1(0) == 0, "journal setup begin");
  require(generativeqc_resource_ledger_bind_v1(handle) == 0, "journal setup bind");
  void* retained{};
  require(resource_cuda_malloc(&retained, 16) == cudaSuccess, "retained upload");
  require(generativeqc_resource_journal_create_v1(handle, 4) == nullptr,
          "capture started during active observation");
  std::uint64_t peak{}, samples{};
  require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0, "journal setup end");
  auto* capture = generativeqc_resource_journal_create_v1(handle, 4);
  require(capture != nullptr, "capture create");
  require(generativeqc_resource_journal_create_v1(handle, 4) == nullptr, "duplicate capture");
  check_journal(capture, {1, 0, 0, 1}, {0, 1, 16});
  require(generativeqc_resource_tracking_begin_v1(0) == 0, "journal replay begin");
  require(generativeqc_resource_ledger_bind_v1(handle) == 0, "journal replay bind");
  void* scratch{};
  require(resource_cuda_malloc_async(&scratch, 32, nullptr) == cudaSuccess, "journal allocation");
  require(resource_cuda_free_async(scratch, nullptr) == cudaSuccess, "journal release");
  require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0, "journal replay end");
  generativeqc_resource_ledger_destroy_v1(handle);
  require(resource_cuda_free(retained) == cudaSuccess, "post-handle buffer release");
  check_journal(capture, {1, 3, 0, 1}, {0, 1, 16, 0, 2, 32, 1, 2, 32, 1, 1, 16});
  generativeqc_resource_journal_destroy_v1(capture);
}

void journal_loss() {
  auto* handle = generativeqc_resource_ledger_create_v1(128, 0);
  for (const auto capacity : {0u, (1u << 20) + 1})
    require(generativeqc_resource_journal_create_v1(handle, capacity) == nullptr,
            "unbounded capture accepted");
  auto* capture = generativeqc_resource_journal_create_v1(handle, 1);
  require(capture != nullptr, "finite capture create");
  require(generativeqc_resource_tracking_begin_v1(0) == 0, "loss begin");
  require(generativeqc_resource_ledger_bind_v1(handle) == 0, "loss bind");
  void* pointer{};
  require(resource_cuda_malloc(&pointer, 8) == cudaSuccess, "loss allocation");
  require(resource_cuda_free(pointer) == cudaSuccess, "loss changed CUDA release");
  check_journal(capture, {0, 1, 1, 1}, {0, 1, 8});
  std::array<std::uint64_t, 4> state{};
  std::array<std::uint64_t, 3> sentinel{991, 991, 991};
  require(generativeqc_resource_journal_read_v1(capture, sentinel.data(), 2, state.data()) == 2,
          "short capture buffer accepted");
  require(sentinel == std::array<std::uint64_t, 3>{991, 991, 991}, "short read changed records");
  require(generativeqc_resource_journal_read_v1(capture, nullptr, 1, state.data()) != 0,
          "null output accepted");
  require(generativeqc_resource_journal_read_v1(nullptr, nullptr, 0, state.data()) != 0,
          "null capture accepted");
  generativeqc_resource_journal_destroy_v1(capture);
  Metrics metrics{};
  require(generativeqc_resource_ledger_read_v2(handle, metrics.data(), metrics.size()) == 0,
          "capture loss changed numeric counters");
  require(metrics == Metrics{0, 8, 1, 0, 8}, "capture changed allocation policy");
  std::uint64_t peak{}, samples{};
  require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0, "loss end");
  generativeqc_resource_ledger_destroy_v1(handle);
}

void journal_reused_address() {
  auto* handle = generativeqc_resource_ledger_create_v1(128, 0);
  auto* capture = generativeqc_resource_journal_create_v1(handle, 4);
  require(capture != nullptr, "reuse capture create");
  require(generativeqc_resource_tracking_begin_v1(0) == 0, "reuse begin");
  require(generativeqc_resource_ledger_bind_v1(handle) == 0, "reuse bind");
  double storage{};
  void* pointer{};
  auto allocate = [&] { pointer = &storage; return cudaSuccess; };
  auto release = [] { return cudaSuccess; };
  require(resource_cuda_allocate(&pointer, 8, allocate, release) == cudaSuccess, "first reuse");
  const auto previous_generation = resource_cuda_generation(pointer);
  require(resource_cuda_allocate(&pointer, 16, allocate, release) == cudaSuccess, "next reuse");
  resource_cuda_forget(pointer, previous_generation);
  resource_cuda_forget(pointer, resource_cuda_generation(pointer));
  check_journal(capture, {0, 4, 0, 1}, {0, 1, 8, 1, 1, 8, 0, 2, 16, 1, 2, 16});
  std::uint64_t peak{}, samples{};
  require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0, "reuse end");
  generativeqc_resource_ledger_destroy_v1(handle);
  generativeqc_resource_journal_destroy_v1(capture);
}

void lifecycle(bool planned = false) {
  Observation observation(planned);
  void* pointer{};
  require(resource_cuda_malloc(&pointer, 24) == cudaSuccess, "first allocation");
  observation.check({24, 24, 1, 0, 24});
  require(resource_cuda_free(pointer) == cudaSuccess, "first release");
  observation.check({0, 24, 1, 0, 24});
  require(resource_cuda_malloc_async(&pointer, 80, nullptr) == cudaSuccess, "async allocation");
  require(resource_cuda_free_async(pointer, nullptr) == cudaSuccess, "async release");
  observation.check({0, 80, 2, 0, 104});
  require(resource_cuda_malloc(&pointer, 64) == cudaSuccess, "retained allocation");
  observation.check({64, 80, 3, 0, 168});
  void* rejected{};
  require(resource_cuda_malloc(&rejected, 65) == cudaErrorMemoryAllocation, "budget rejection");
  require(rejected == nullptr, "rejection pointer");
  observation.check({64, 80, 3, 1, 168});
  observation.end();
  observation.bind();
  observation.check({64, 64, 0, 0, 0});
  require(resource_cuda_free(pointer) == cudaSuccess, "retained release");
  observation.check({0, 64, 0, 0, 0});
}

void failures(bool planned = false) {
  Observation observation(planned);
  void* pointer{};
  bool host_oom = true;
  require(resource_cuda_allocate(&pointer, 8, [] { return cudaErrorMemoryAllocation; },
                                  [] { return cudaSuccess; }, &host_oom) ==
              cudaErrorMemoryAllocation && !host_oom, "driver failure");
  observation.check({0, 0, 0, 1, 0});
  const auto generation = device_allocation_generation;
  device_allocation_generation = std::numeric_limits<std::uint64_t>::max();
  int releases{};
  double storage{};
  require(resource_cuda_allocate(&pointer, 8, [&] { pointer = &storage; return cudaSuccess; },
                                  [&] { ++releases; return cudaSuccess; }, &host_oom) ==
              cudaErrorMemoryAllocation && host_oom, "registry failure");
  device_allocation_generation = generation;
  require(pointer == nullptr && releases == 1, "registry cleanup");
  observation.check({0, 0, 0, 1, 0});
  require(resource_cuda_allocate(&pointer, 8, [] { return cudaSuccess; },
                                  [] { return cudaSuccess; }) == cudaSuccess, "null allocation");
  observation.check({0, 0, 0, 1, 0});
  visible_device = 1;
  require(resource_cuda_malloc(&pointer, 8) == cudaErrorInvalidDevice, "wrong device");
  visible_device = 0;
  observation.check({0, 0, 0, 1, 0});
  Metrics sentinel{991, 991, 991, 991, 991};
  for (const auto count : {0, 4, 6}) {
    require(generativeqc_resource_ledger_read_v2(observation.handle, sentinel.data(), count) != 0,
            "invalid count accepted");
    require(sentinel == Metrics{991, 991, 991, 991, 991}, "invalid read wrote output");
  }
  require(generativeqc_resource_ledger_read_v2(nullptr, sentinel.data(), 5) != 0, "null handle");
  require(generativeqc_resource_ledger_read_v2(observation.handle, nullptr, 5) != 0, "null output");
}

// Exercise the real allocator wrapper and provider allowance with host CUDA
// doubles. Promises must protect future owners, not merely today's live bytes.
void capacity_reservations() {
  generativeqc_device_resource_buffer_v1 row{128, 4, 0, 0, 0, 0};
  generativeqc_device_resource_descriptor descriptor{
      sizeof(descriptor), GENERATIVEQC_ABI_VERSION, &row, 1,
      GENERATIVEQC_DEVICE_RESOURCE_NUMERIC_CAP};
  generativeqc_device_resources* handle{};
  require(generativeqc_device_resources_create(&descriptor, &handle) ==
              GENERATIVEQC_STATUS_SUCCESS, "raw numeric-cap metadata creation failed");
  std::unique_ptr<generativeqc_device_resources, decltype(&generativeqc_device_resources_destroy)>
      owner(handle, generativeqc_device_resources_destroy);
  auto ledger = generativeqc::api::device_resource_owner(handle);
  const auto snapshot = [&](std::size_t numeric, std::size_t provider, std::size_t unused) {
    generativeqc_device_resource_diagnostic value{};
    value.struct_size = sizeof(value);
    value.abi_version = GENERATIVEQC_ABI_VERSION;
    require(generativeqc_device_resources_read(handle, &value) == GENERATIVEQC_STATUS_SUCCESS &&
                value.admission == GENERATIVEQC_DEVICE_RESOURCE_NUMERIC_CAP &&
                value.numeric_live_bytes == numeric && value.provider_reserved_bytes == provider &&
                value.capacity_reserved_bytes == unused,
            "public diagnostic conflated promises, numeric allocations or provider allowances");
  };
  DeviceResourceScope observation(ledger);
  auto first = std::make_unique<DeviceCapacityReservation>(ledger, 80);
  auto second = std::make_unique<DeviceCapacityReservation>(ledger, 32);
  snapshot(0, 0, 112);
  require(device_resource_available(0) == 16 && ledger->live == 0,
          "future promises were reported as free or allocated");
  bool rejected = false;
  try { DeviceCapacityReservation extra(ledger, 17); }
  catch (const std::bad_alloc&) { rejected = true; }
  require(rejected && device_resource_available(0) == 16,
          "over-limit promise modified available capacity");
  require(!DeviceCapacityReservation::try_create(ledger, 17) &&
              device_resource_available(0) == 16,
          "optional promise admission changed capacity on a miss");
  auto exact_capacity = DeviceCapacityReservation::try_create(ledger, 16);
  require(exact_capacity && device_resource_available(0) == 0,
          "optional promise admission refused its exact remaining capacity");
  exact_capacity.reset();
  snapshot(0, 0, 112);
  void* optional{};
  require(resource_cuda_malloc(&optional, 16) == cudaSuccess, "exact optional allowance");
  void* denied{};
  require(resource_cuda_malloc(&denied, 1) == cudaErrorMemoryAllocation && !denied,
          "optional allocation consumed a later owner's promise");
  void* retained{};
  std::unique_ptr<DeviceProviderAllowance> provider;
  {
    DeviceCapacityScope scope(*first);
    require(device_resource_available(0) == 80, "owner cannot use its promised capacity");
    {
      DeviceCapacityScope unpromised_owner(nullptr);
      require(device_resource_available(0) == 0 &&
                  resource_cuda_malloc(&denied, 1) == cudaErrorMemoryAllocation,
              "nested owner without a promise borrowed another owner's capacity");
    }
    require(device_resource_available(0) == 80,
            "unpromised nested owner lost its caller's promise");
    {
      DeviceUnpromisedCapacityScope optional_scope;
      require(device_resource_available(0) == 0 &&
                  resource_cuda_malloc(&denied, 1) == cudaErrorMemoryAllocation,
              "optional scope borrowed its parent's mandatory promise");
    }
    require(device_resource_available(0) == 80, "optional scope lost its parent's promise");
    require(resource_cuda_malloc(&retained, 64) == cudaSuccess, "promised numeric allocation");
    provider = std::make_unique<DeviceProviderAllowance>(0, 16);
    snapshot(80, 16, 32);
    require(device_resource_available(0) == 0 && ledger->capacity_reserved == 32,
            "numeric/provider charge did not convert its promise exactly once");
    require(resource_cuda_malloc(&denied, 1) == cudaErrorMemoryAllocation,
            "owner exceeded its declared bound");
    {
      DeviceCapacityScope nested(*second);
      require(device_resource_available(0) == 32, "nested owner borrowed wrong promise");
      void* temporary{};
      require(resource_cuda_malloc_async(&temporary, 32, nullptr) == cudaSuccess,
              "later owner was starved by optional storage");
      require(resource_cuda_free_async(temporary, nullptr) == cudaSuccess &&
                  device_resource_available(0) == 32, "rebuild capacity was not restored");
    }
    require(device_resource_available(0) == 0, "nested scope did not restore parent promise");
  }
  // Closing one metadata owner preserves both another owner's future and all
  // allocations that outlive the closed promise; frees may happen elsewhere.
  first.reset();
  std::thread release([&] { resource_cuda_free(retained); provider.reset(); });
  release.join();
  snapshot(16, 0, 32);
  require(ledger->live == 16 && ledger->provider_reserved == 0 &&
              ledger->capacity_reserved == 32 && device_resource_available(0) == 80,
          "late close/free leaked or resurrected a closed promise");
  {
    DeviceCapacityScope scope(*second);
    void* pointer{};
    require(resource_cuda_allocate(&pointer, 32, [] { return cudaErrorMemoryAllocation; },
                                    [] { return cudaSuccess; }) == cudaErrorMemoryAllocation &&
                device_resource_available(0) == 32, "driver rollback lost promised capacity");
    const auto generation = device_allocation_generation;
    device_allocation_generation = std::numeric_limits<std::uint64_t>::max();
    bool host_oom = false;
    require(resource_cuda_malloc(&pointer, 32, &host_oom) == cudaErrorMemoryAllocation && host_oom,
            "registry failure was misclassified");
    device_allocation_generation = generation;
    require(!pointer && device_resource_available(0) == 32,
            "registry rollback leaked promise or allocation");
    // Address reuse retires the old generation once. A delayed free for that
    // generation must not credit the new allocation or spend its capacity.
    double storage{};
    auto allocate = [&] { pointer = &storage; return cudaSuccess; };
    auto noop = [] { return cudaSuccess; };
    require(resource_cuda_allocate(&pointer, 8, allocate, noop) == cudaSuccess, "first address");
    const auto old = resource_cuda_generation(pointer);
    require(resource_cuda_allocate(&pointer, 16, allocate, noop) == cudaSuccess, "reused address");
    const auto current = resource_cuda_generation(pointer);
    resource_cuda_forget(pointer, old);
    require(device_resource_available(0) == 16, "stale free restored reused capacity");
    resource_cuda_forget(pointer, current);
    require(device_resource_available(0) == 32, "current free did not restore capacity");
    // A failed foreign-ledger bind leaves the active promise unchanged.
    auto foreign = std::make_shared<DeviceResourceLedger>();
    foreign->limit = 32;
    DeviceCapacityReservation other(foreign, 32);
    rejected = false;
    try { DeviceCapacityScope invalid(other); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && device_resource_available(0) == 32, "foreign scope replaced owner");
  }
  second.reset();
  snapshot(16, 0, 0);
  require(ledger->capacity_reserved == 0 && device_resource_available(0) == 112,
          "reservation close leaked unused capacity");
  require(resource_cuda_free(optional) == cudaSuccess && device_resource_available(0) == 128,
          "optional cleanup changed reservation accounting");
  auto closing = std::make_unique<DeviceCapacityReservation>(ledger, 128);
  {
    DeviceCapacityScope scope(*closing);
    closing.reset();
    require(device_resource_available(0) == 0 &&
                resource_cuda_malloc(&denied, 1) == cudaErrorMemoryAllocation,
            "a closed promise was silently replaced with unbounded capacity");
  }
  require(device_resource_available(0) == 128, "closed scope failed to restore ordinary binding");
  auto maximum = std::make_shared<DeviceResourceLedger>();
  maximum->limit = std::numeric_limits<std::size_t>::max();
  DeviceCapacityReservation near_limit(maximum, maximum->limit - 1);
  rejected = false;
  try { DeviceCapacityReservation overflowed(maximum, 2); }
  catch (const std::bad_alloc&) { rejected = true; }
  require(rejected && maximum->capacity_reserved == maximum->limit - 1,
          "promise sum overflow admitted excess capacity");
}

void overflow() {
  Observation observation;
  auto ledger = active_device_resource_ledger;
  ledger->requested_bytes = std::numeric_limits<std::uint64_t>::max() - 7;
  void* pointer{};
  require(resource_cuda_malloc(&pointer, 8) == cudaSuccess, "overflow changed execution");
  Metrics sentinel{991, 991, 991, 991, 991};
  require(generativeqc_resource_ledger_read_v2(observation.handle, sentinel.data(), 5) != 0,
          "overflow produced usable metrics");
  require(sentinel == Metrics{991, 991, 991, 991, 991}, "overflow wrote output");
  require(resource_cuda_free(pointer) == cudaSuccess, "overflow release");
  observation.end();
  observation.bind();
  observation.check({0, 0, 0, 0, 0});
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "scenario missing");
    const std::string scenario = argv[1];
    if (scenario == "lifecycle") lifecycle();
    else if (scenario == "planned-lifecycle") lifecycle(true);
    else if (scenario == "failures") failures();
    else if (scenario == "planned-failures") failures(true);
    else if (scenario == "overflow") overflow();
    else if (scenario == "journal-lifetime") journal_lifetime();
    else if (scenario == "journal-loss") journal_loss();
    else if (scenario == "journal-reuse") journal_reused_address();
    else if (scenario == "capacity-reservations") capacity_reservations();
    else throw std::runtime_error("unknown scenario");
    require(device_allocation_owners.empty(), "owners leaked");
    std::cout << scenario << " passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


@pytest.fixture(scope="module")
def ledger_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    """Link actual ledger C APIs; discard unrelated CPU shape-query sections."""
    folder = tmp_path_factory.mktemp("device-ledger-requests")
    # The production resource translation unit also owns ECP and D3 queries.
    # Generate their real headers instead of relying on a prior CMake build.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "tools/generate_ecp_kernels.py"),
            "--output",
            str(folder / "generated_ecp_ao.cuh"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "tools/generativeqc_d3/generate_native_data.py"),
            "--output",
            str(folder / "d3_data.hpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (folder / "cuda_runtime_api.h").write_text(CUDA_STUB)
    source = folder / "probe.cpp"
    source.write_text(DRIVER)
    return required_native_cxx.build_executable(
        [
            ROOT / "src/api/c_api_resources.cpp",
            ROOT / "src/api/c_api_resource_ledger_plan.cpp",
            ROOT / "src/api/c_api_device_resources.cpp",
            ROOT / "src/api/error.cpp",
            ROOT / "src/runtime/resource_ledger_plan.cpp",
            ROOT / "src/runtime/resource_planning.cpp",
            source,
        ],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-I",
            str(folder),
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
        link_args=("-Wl,--gc-sections", "-pthread"),
    )


@pytest.mark.parametrize(
    "scenario",
    (
        "lifecycle",
        "planned-lifecycle",
        "failures",
        "planned-failures",
        "overflow",
        "journal-lifetime",
        "journal-loss",
        "journal-reuse",
        "capacity-reservations",
    ),
)
def test_native_requested_byte_accounting(ledger_probe: Path, scenario: str) -> None:
    result = subprocess.run(
        [str(ledger_probe), scenario],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.stdout == f"{scenario} passed\n"


class LedgerFunction:
    """ctypes-compatible host double without a native library or device probe."""

    def __init__(self, result: int | None = 0, values: tuple[int, ...] = ()) -> None:
        self.result = result
        self.values = values
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *arguments: Any) -> int | None:
        self.calls.append(arguments)
        if self.values:
            for index, value in enumerate(self.values):
                arguments[1][index] = value
        return self.result


class JournalReader(LedgerFunction):
    """Expose the exact native wire layout, including loss and bad snapshots."""

    def __init__(self) -> None:
        super().__init__()
        self.state = (1, 2, 0, 1)
        self.records = (0, 11, 64, 0, 12, 16, 1, 12, 16)

    def __call__(self, *arguments: Any) -> int | None:
        self.calls.append(arguments)
        for index, value in enumerate(self.records):
            arguments[1][index] = value
        for index, value in enumerate(self.state):
            arguments[3][index] = value
        return self.result


def test_python_journal_keeps_retained_owners_events_and_loss_separate() -> None:
    reader = JournalReader()
    library = SimpleNamespace(
        generativeqc_resource_journal_create_v1=LedgerFunction(456),
        generativeqc_resource_journal_read_v1=reader,
        generativeqc_resource_journal_destroy_v1=LedgerFunction(None),
    )
    owner = SimpleNamespace(handle=123, library=library, device=0, owner="hf")
    journal = NativeDeviceJournal(owner, capacity=2)
    snapshot = journal.snapshot()
    assert snapshot["initial_live"] == {"allocation-11": 64}
    assert snapshot["event_end"] == 2
    assert [event["kind"] for event in snapshot["events"]] == ["allocate", "release"]
    assert [event["sequence"] for event in snapshot["events"]] == [0, 1]
    assert snapshot["dropped_events"] == 0
    reader.state = (1, 2, 1, 1)
    assert journal.snapshot()["dropped_events"] == 1
    reader.state = (1, 3, 0, 1)
    with pytest.raises(RuntimeError, match="unavailable"):
        journal.snapshot()
    reader.state = (1, 2, 0, 0)
    with pytest.raises(RuntimeError, match="unavailable"):
        journal.snapshot()
    journal.close()
    journal.close()
    assert len(library.generativeqc_resource_journal_destroy_v1.calls) == 1
    with pytest.raises(RuntimeError, match="closed"):
        journal.snapshot()


@pytest.mark.parametrize("capacity", (0, -1, True, 1.0, (1 << 20) + 1))
def test_python_journal_rejects_unbounded_or_untyped_capacity(capacity: Any) -> None:
    with pytest.raises(ValueError, match="capacity"):
        NativeDeviceJournal(SimpleNamespace(handle=123), capacity=capacity)


def test_python_journal_fails_closed_for_old_or_closed_libraries() -> None:
    with pytest.raises(RuntimeError, match="closed"):
        NativeDeviceJournal(SimpleNamespace(handle=None))
    owner = SimpleNamespace(handle=123, library=SimpleNamespace(), device=0, owner="hf")
    with pytest.raises(NotImplementedError, match="journal v1"):
        NativeDeviceJournal(owner)


@pytest.mark.parametrize("version", (1, 2))
def test_python_binding_preserves_v1_and_exports_only_real_v2_metrics(
    version: int,
) -> None:
    identity = ResourceIdentity(
        "hf", "ledger", "cuda", "fp64", '{"nbf": 2}', ("energy",), "fixed"
    )
    candidate = ResourceCandidate(
        "fixed",
        "resident",
        (ResourceEstimate("arena", 128, "device:0", 0, 0),),
    )
    request = ResourceRequest("hf", identity, (candidate,))
    plan = plan_resources([request], ResourceBudget())
    library = SimpleNamespace(
        generativeqc_resource_ledger_create_v1=LedgerFunction(123),
        generativeqc_resource_ledger_destroy_v1=LedgerFunction(None),
        generativeqc_resource_ledger_bind_v1=LedgerFunction(),
        generativeqc_resource_ledger_read_v1=LedgerFunction(values=(64, 64, 3, 0)),
    )
    if version == 2:
        library.generativeqc_resource_ledger_read_v2 = LedgerFunction(
            values=(64, 64, 3, 0, 192)
        )
    ledger = NativeDeviceLedger(library, plan)
    try:
        observed = ledger.to_dict()
        assert observed["allocations"] == 3
        assert observed["peak_bytes"] == 64
        if version == 2:
            assert observed["requested_bytes"] == 192
            assert not library.generativeqc_resource_ledger_read_v1.calls
            reader = library.generativeqc_resource_ledger_read_v2
            assert reader.calls[0][2] == 5
            assert reader.argtypes[-1] is ctypes.c_size_t
        else:
            assert "requested_bytes" not in observed
            reader = library.generativeqc_resource_ledger_read_v1
        reader.result = 1
        with pytest.raises(RuntimeError, match="unavailable"):
            ledger.to_dict()
        with pytest.raises(ValueError, match="CUDA request"):
            NativeDeviceLedger(
                library,
                plan_resources(
                    [replace(request, identity=replace(identity, backend="cpu"))],
                    ResourceBudget(),
                ),
            )
    finally:
        ledger.close()
    with pytest.raises(RuntimeError, match="closed"):
        ledger.to_dict()
    assert len(library.generativeqc_resource_ledger_destroy_v1.calls) == 1
