"""Exercise production ledger bookkeeping with host-only CUDA allocation doubles.

The real C API translation unit and allocation wrapper are compiled unchanged.
These tests do not establish real-GPU allocation or numerical endpoint gates.
"""

from __future__ import annotations

import ctypes
import subprocess
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
#include "runtime/resource_cuda.cuh"

extern "C" {
void* generativeqc_resource_ledger_create_v1(std::size_t, int);
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
  void* handle = generativeqc_resource_ledger_create_v1(128, 0);
  Observation() { require(handle != nullptr, "create"); bind(); }
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

void lifecycle() {
  Observation observation;
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

void failures() {
  Observation observation;
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
    else if (scenario == "failures") failures();
    else if (scenario == "overflow") overflow();
    else if (scenario == "journal-lifetime") journal_lifetime();
    else if (scenario == "journal-loss") journal_loss();
    else if (scenario == "journal-reuse") journal_reused_address();
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
    (folder / "cuda_runtime_api.h").write_text(CUDA_STUB)
    source = folder / "probe.cpp"
    source.write_text(DRIVER)
    return required_native_cxx.build_executable(
        [ROOT / "src/api/c_api_resources.cpp", source],
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
        "failures",
        "overflow",
        "journal-lifetime",
        "journal-loss",
        "journal-reuse",
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
