"""Compile real wrappers/observer with CPU-only CUDA and CUPTI doubles."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools.cupti_source_capture import FIELDS, METADATA

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
#include "runtime/residency_cuda.cuh"
#include "runtime/resource_cuda.cuh"
#include <cupti.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int generativeqc_cupti_source_begin_v1(std::uint64_t);
extern "C" void generativeqc_cupti_residency_observe_v1(const std::uint64_t*, std::size_t, void*);
extern "C" int generativeqc_cupti_source_stop_v1();
extern "C" int generativeqc_cupti_source_read_v1(std::uint64_t*, std::uint64_t, std::uint64_t*, std::uint64_t);

std::string mode;
std::uint64_t correlation{};
int copy_calls{}, sync_calls{}, graph_calls{}, allocation_calls{}, release_calls{};
int event_sync_calls{};
bool ledger_work{};
std::size_t last_bytes{};
void* last_destination{};
const void* last_source{};
cudaMemcpyKind last_kind{};
cudaStream_t last_stream{};
cudaEvent_t last_event{};

extern "C" CUptiResult CUPTIAPI cuptiGetVersion(std::uint32_t* version) {
  *version = mode == "version" ? 27 : 28; return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind, std::uint64_t identity) {
  assert(kind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2 && !correlation);
  if (mode == "push-error") return CUPTI_ERROR_INVALID_PARAMETER;
  correlation = identity; return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind, std::uint64_t* identity) {
  assert(kind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2 && correlation);
  *identity = correlation + (mode == "pop-error" ? 1 : 0); correlation = 0; return CUPTI_SUCCESS;
}
extern "C" void generativeqc_cupti_graph_observe_v1(const std::uint64_t* values, std::size_t count, void*) {
  assert(count == 10 && values[0] == 1); ++graph_calls;
}
cudaError_t fake_copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
  ++copy_calls; last_destination = destination; last_source = source; last_bytes = bytes; last_kind = kind;
  if (mode == "failure" || mode == "upload-failure") return cudaErrorInvalidValue;
  std::memcpy(destination, source, bytes); return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
    std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  last_stream = stream; return fake_copy(destination, source, bytes, kind);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
  return fake_copy(destination, source, bytes, kind);
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  ++sync_calls; last_stream = stream;
  return mode == "failure" || mode == "upload-failure" ||
      (ledger_work && (mode == "ledger-sync-failure" || mode == "ledger-rollback-sync-failure")) ?
      cudaErrorInvalidValue : cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  *device = 0; return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  ++event_sync_calls; last_event = event;
  return mode == "event-failure" ? cudaErrorInvalidValue : cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** output, std::size_t bytes, cudaStream_t) {
  ++allocation_calls; *output = std::malloc(bytes);
  return *output ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* pointer, cudaStream_t) {
  ++release_calls;
  if (mode == "ledger-free-failure") return cudaErrorInvalidValue;
  std::free(pointer); return cudaSuccess;
}
void exercise_ledger(cudaStream_t stream) {
  using namespace generativeqc::runtime;
  ledger_work = true;
  const bool tracked = mode != "ledger-untracked";
  const bool rollback = mode == "ledger-rollback" || mode == "ledger-rollback-sync-failure";
  const bool free_failure = mode == "ledger-free-failure";
  const bool sync_failure = mode == "ledger-sync-failure";
  auto ledger = std::make_shared<DeviceResourceLedger>();
  ledger->limit = 64;
  if (tracked) active_device_resource_ledger = ledger;
  const auto previous_generation = device_allocation_generation;
  if (rollback) device_allocation_generation = std::numeric_limits<std::uint64_t>::max();
  sync_calls = 0;
  void* pointer{};
  const auto allocation_status = resource_cuda_malloc_async(&pointer, 16, stream);
  if (rollback) device_allocation_generation = previous_generation;
  else assert(device_allocation_generation == previous_generation + int(tracked));
  assert(allocation_calls == 1);
  if (rollback) {
    assert(allocation_status == cudaErrorMemoryAllocation && pointer == nullptr);
    assert(ledger->live == 0 && ledger->allocations == 0 && ledger->requested_bytes == 0);
    assert(sync_calls == 1 && release_calls == 1 && last_stream == stream);
  } else {
    assert(allocation_status == cudaSuccess && pointer && sync_calls == 0);
    const auto generation = resource_cuda_generation(pointer);
    assert(bool(generation) == tracked);
    if (mode == "ledger-post-scope") active_device_resource_ledger.reset();
    assert(resource_cuda_free_async(pointer, stream) ==
        (free_failure || sync_failure ? cudaErrorInvalidValue : cudaSuccess));
    assert(release_calls == 1 && sync_calls == int(tracked && !free_failure));
    assert(ledger->live == (free_failure || sync_failure ? 16u : 0u));
    assert(ledger->allocations == std::size_t(tracked) && ledger->requested_bytes == (tracked ? 16u : 0u));
    assert(bool(resource_cuda_generation(pointer)) == (tracked && (free_failure || sync_failure)));
    if (free_failure) std::free(pointer);
    if (free_failure || sync_failure) resource_cuda_forget(pointer, generation);
  }
  active_device_resource_ledger.reset();
  assert(device_allocation_owners.empty() && ledger->live == 0);
  ledger_work = false;
}
void observer(const std::uint64_t* values, std::size_t count, void* context) {
  generativeqc_cupti_residency_observe_v1(values, count, context);
  if (mode == "throw" || mode == "event-throw" || (mode == "ledger-observer-failure" && values[4] == 5))
    throw std::runtime_error("observation only");
  if (mode == "recursion") generativeqc::runtime::dispatch_residency_source(values, count);
}
int main(int count, char** arguments) {
  using namespace generativeqc::runtime;
  assert(count == 2); mode = arguments[1];
  assert(generativeqc_cupti_source_begin_v1(0) == -1);
  if (mode == "version") {
    assert(generativeqc_cupti_source_begin_v1(64) == -2); std::cout << "{}\n"; return 0;
  }
  assert(generativeqc_cupti_source_begin_v1(mode == "overflow" ? 2 : 64) == 0);
  assert(generativeqc_cupti_source_begin_v1(64) == -1);
  const bool disabled = mode == "disabled" || mode == "upload-disabled" || mode == "ledger-disabled" || mode == "event-disabled";
  if (!disabled) assert(bind_residency_observer(observer, nullptr) == 0);
  if (mode == "exhausted") residency_boundary_generation = std::numeric_limits<std::uint64_t>::max();
  std::array<int, 4> input{1, 2, 3, 4}, output{};
  const auto stream = reinterpret_cast<cudaStream_t>(&input);
  const auto bytes = mode == "zero" ? 0u : sizeof(input);
  const auto expected = mode == "failure" || mode == "upload-failure" ? cudaErrorInvalidValue : cudaSuccess;
  {
    ResidencyExecution execution(ResidencyOwner::hf_bucket);
    assert(residency_memcpy_async(execution, ResidencyRole::publication, ResidencySite::hf_results,
        ResidencyPayload::density, output.data(), input.data(), bytes, cudaMemcpyDeviceToHost, stream) == expected);
    assert(copy_calls == 1 && last_destination == output.data() && last_source == input.data() &&
           last_bytes == bytes && last_kind == cudaMemcpyDeviceToHost && last_stream == stream);
    assert(residency_memcpy(execution, ResidencyRole::iteration, ResidencySite::hf_active,
        ResidencyPayload::active, output.data(), input.data(), bytes, cudaMemcpyDeviceToHost) == expected);
    assert(copy_calls == 2 && last_bytes == bytes && last_kind == cudaMemcpyDeviceToHost);
    assert(residency_stream_synchronize(execution, ResidencyRole::publication,
        ResidencySite::hf_results_fence, stream) == expected);
    assert(sync_calls == 1 && last_stream == stream);
    if (mode.rfind("event-", 0) == 0) {
      const ResidencyExecution df(ResidencyOwner::posthf_df_source);
      const auto event = reinterpret_cast<cudaEvent_t>(&output);
      const auto status = mode == "event-failure" ? cudaErrorInvalidValue : cudaSuccess;
      assert(residency_event_synchronize(df, ResidencyRole::compatibility,
          ResidencySite::posthf_df_generation_fence, event) == status);
      assert(residency_event_synchronize(df, ResidencyRole::compatibility,
          ResidencySite::posthf_df_publication_fence, event) == status);
      assert(event_sync_calls == 2 && last_event == event && sync_calls == 1);
      assert(std::string(residency_boundary_name(1, 6)) == "posthf_df_source");
      assert(std::string(residency_boundary_name(3, 29)) == "posthf.df.generation-fence");
      assert(std::string(residency_boundary_name(3, 31)) == "posthf.df.publication-fence");
    }
    if (mode.rfind("ledger-", 0) == 0) exercise_ledger(stream);
    if (mode == "nested-owners") {
      {
        ResidencyExecution graph(ResidencyOwner::hf_graph_setup);
        assert(residency_stream_synchronize(graph, ResidencyRole::prepare,
            ResidencySite::hf_graph_upload_fence, stream) == cudaSuccess);
      }
      {
        ResidencyExecution resources(ResidencyOwner::hf_bucket_resources);
        {
          ResidencyExecution solver(ResidencyOwner::hf_eigensolver_resources);
          assert(residency_stream_synchronize(solver, ResidencyRole::lifetime,
              ResidencySite::hf_eigensolver_release_fence, stream) == cudaSuccess);
        }
        assert(residency_stream_synchronize(resources, ResidencyRole::lifetime,
            ResidencySite::hf_bucket_release_fence, stream) == cudaSuccess);
      }
      assert(sync_calls == 4 && last_stream == stream);
    }
    if (mode.rfind("upload", 0) == 0) {
      const auto upload_bytes = mode == "upload-zero" ? 0u : sizeof(input);
      std::array<int, 4> uploaded{};
      assert(residency_upload(execution, ResidencyRole::prepare, ResidencySite::hf_dynamic_inputs,
          ResidencyPayload::input_warm_density, uploaded.data(), input.data(), upload_bytes, stream)
          == (upload_bytes ? expected : cudaSuccess));
      assert(copy_calls == (upload_bytes ? 3 : 2));
      if (upload_bytes) {
        assert(last_destination == uploaded.data() && last_source == input.data() &&
               last_bytes == upload_bytes && last_kind == cudaMemcpyHostToDevice && last_stream == stream);
      }
      assert((uploaded == input) == (upload_bytes && expected == cudaSuccess));
      assert(sync_calls == 1);
    }
  }
  assert(correlation == 0);
  assert((output == input) == (mode != "zero" && expected == cudaSuccess));
  std::array<std::uint64_t, 10> graph{}; graph[0] = 1;
  generativeqc_cupti_residency_observe_v1(graph.data(), graph.size(), nullptr);
  assert(graph_calls == 1);
  std::uint64_t errors = 0;
  if (!disabled) assert(unbind_residency_observer(observer, nullptr, &errors) == 0);
  assert(generativeqc_cupti_source_stop_v1() == 0);
  assert(generativeqc_cupti_source_stop_v1() == -1);
  std::array<std::uint64_t, 8> metadata{};
  assert(generativeqc_cupti_source_read_v1(nullptr, 0, metadata.data(), metadata.size()) == 0);
  std::vector<std::uint64_t> records(metadata[0] * 14, 99);
  if (metadata[0]) {
    const auto before = metadata;
    assert(generativeqc_cupti_source_read_v1(records.data(), 0, metadata.data(), metadata.size()) == -1);
    assert(metadata == before && std::all_of(records.begin(), records.end(), [](auto value) { return value == 99; }));
    assert(generativeqc_cupti_source_read_v1(records.data(), metadata[0], metadata.data(), metadata.size()) == 0);
  }
  assert(std::string(residency_boundary_name(1, 1)) == "hf_bucket");
  assert(std::string(residency_boundary_name(1, 4)) == "hf_eigensolver_resources");
  assert(std::string(residency_boundary_name(1, 5)) == "device_resource_ledger");
  assert(std::string(residency_boundary_name(3, 27)) == "resource-ledger.release-fence");
  assert(std::string(residency_boundary_name(3, 28)) == "resource-ledger.rollback-fence");
  assert(std::string(residency_boundary_name(2, 7)) == "lifetime");
  assert(std::string(residency_boundary_name(4, 4)) == "density");
  assert(std::string(residency_boundary_name(3,
      static_cast<std::uint64_t>(ResidencySite::hf_dynamic_inputs))) == "hf.dynamic-inputs");
  for (std::uint64_t value = static_cast<std::uint64_t>(ResidencyPayload::input_atom_offsets);
       value <= static_cast<std::uint64_t>(ResidencyPayload::input_previous_energy_seed); ++value) {
    assert(residency_boundary_name(4, value) &&
           std::string(residency_boundary_name(4, value)).rfind("input.", 0) == 0);
  }
  assert(std::string(residency_boundary_name(4,
      static_cast<std::uint64_t>(ResidencyPayload::input_warm_density))) == "input.warm_density");
  assert(residency_boundary_name(1, 0) == nullptr && residency_boundary_name(4, 999) == nullptr);
  std::cout << "{\"dispatch_errors\":" << errors << ",\"metadata\":[";
  for (std::size_t index = 0; index < metadata.size(); ++index) std::cout << (index ? "," : "") << metadata[index];
  std::cout << "],\"records\":[";
  for (std::size_t index = 0; index < records.size(); ++index) std::cout << (index ? "," : "") << records[index];
  std::cout << "]}\n";
}
"""


@pytest.fixture(scope="module")
def source_probe(
    native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Never link a CUDA/CUPTI library or access a GPU in these unit tests."""
    toolkit = Path(
        os.environ.get(
            "GENERATIVEQC_CUPTI_TEST_HEADERS",
            ROOT / ".artifacts/issue1629-cupti/toolkit",
        )
    )
    if (
        not (toolkit / "cupti/cupti.h").is_file()
        or not (toolkit / "cuda/cuda_runtime_api.h").is_file()
    ):
        pytest.skip("requires optional CUPTI 28/CUDA header export")
    folder = tmp_path_factory.mktemp("source-boundaries-double")
    source = folder / "probe.cpp"
    source.write_text(PROBE)
    output = folder / "probe"
    native_cxx.build_executable(
        [ROOT / "tools/cupti_source_capture.cpp", source],
        output,
        compile_args=(
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'src'}",
            f"-I{toolkit / 'cupti'}",
            f"-I{toolkit / 'cuda'}",
        ),
        link_args=("-pthread",),
    )
    return output


@pytest.mark.parametrize(
    "mode",
    [
        "ordinary",
        "version",
        "zero",
        "disabled",
        "failure",
        "throw",
        "recursion",
        "overflow",
        "push-error",
        "pop-error",
        "exhausted",
    ],
)
def test_native_observation_preserves_exactly_once_cuda_work(
    source_probe: Path, mode: str
) -> None:
    result = subprocess.run(
        [str(source_probe), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    raw = json.loads(result.stdout)
    if mode == "version":
        assert raw == {}
        return
    metadata = dict(zip(METADATA, raw["metadata"], strict=True))
    records = [
        dict(zip(FIELDS, raw["records"][offset : offset + len(FIELDS)], strict=True))
        for offset in range(0, len(raw["records"]), len(FIELDS))
    ]
    assert metadata["record_count"] == len(records) <= metadata["capacity"]
    assert metadata["stopped"] == 1 and metadata["cupti_version"] == 28
    assert metadata["outstanding_operations"] == 0
    assert bool(metadata["dropped_records"]) == (mode == "overflow")
    assert bool(metadata["errors"]) == (
        mode in {"push-error", "pop-error", "exhausted"}
    )
    assert raw["dispatch_errors"] == (8 if mode in {"throw", "recursion"} else 0)
    if mode in {"ordinary", "zero", "failure"}:
        assert [row["kind"] for row in records] == [3, 1, 2, 1, 2, 1, 2, 4]
        ends = [row for row in records if row["kind"] == 2]
        assert [row["status"] for row in ends] == [int(mode == "failure")] * 3
        assert [row["bytes"] for row in ends] == (
            [0, 0, 0] if mode == "zero" else [16, 16, 0]
        )
    if mode == "disabled":
        assert records == []


@pytest.mark.parametrize(
    "mode", ["event-success", "event-failure", "event-disabled", "event-throw"]
)
def test_native_event_observation_preserves_exactly_once_cuda_work(
    source_probe: Path, mode: str
) -> None:
    """CPU doubles verify arguments/status without performing any GPU work."""
    result = subprocess.run(
        [str(source_probe), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    raw = json.loads(result.stdout)
    metadata = dict(zip(METADATA, raw["metadata"], strict=True))
    records = [
        dict(zip(FIELDS, raw["records"][offset : offset + len(FIELDS)], strict=True))
        for offset in range(0, len(raw["records"]), len(FIELDS))
    ]
    if mode == "event-disabled":
        assert records == []
        return
    events = [row for row in records if row["kind"] == 2 and row["operation_kind"] == 3]
    assert len(events) == 2
    assert [row["site"] for row in events] == [29, 31]
    assert all(row["owner"] == 6 and row["role"] == 6 for row in events)
    assert [row["status"] for row in events] == [int(mode == "event-failure")] * 2
    assert all(
        not row[field]
        for row in events
        for field in ("bytes", "payload", "payload_instance", "dependency", "direction")
    )
    assert metadata["outstanding_operations"] == 0
    assert metadata["errors"] == 0


@pytest.mark.parametrize(
    "mode", ["upload", "upload-zero", "upload-failure", "upload-disabled"]
)
def test_native_input_upload_preserves_helper_contract(
    source_probe: Path, mode: str
) -> None:
    """Empty HF uploads skip CUDA; nonempty ones are named H2D operations."""
    result = subprocess.run(
        [str(source_probe), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    raw = json.loads(result.stdout)
    metadata = dict(zip(METADATA, raw["metadata"], strict=True))
    records = [
        dict(zip(FIELDS, raw["records"][offset : offset + len(FIELDS)], strict=True))
        for offset in range(0, len(raw["records"]), len(FIELDS))
    ]
    assert raw["dispatch_errors"] == 0
    assert metadata["errors"] == metadata["dropped_records"] == 0
    assert metadata["outstanding_operations"] == 0
    expected_count = (
        0 if mode == "upload-disabled" else 8 if mode == "upload-zero" else 10
    )
    assert metadata["record_count"] == len(records) == expected_count
    uploads = [row for row in records if row["kind"] == 2 and row["direction"] == 1]
    assert len(uploads) == int(mode in {"upload", "upload-failure"})
    if uploads:
        assert uploads[0]["bytes"] == 16
        assert uploads[0]["role"] == 1
        assert uploads[0]["site"] == 19
        assert uploads[0]["payload"] == 75
        assert uploads[0]["status"] == int(mode == "upload-failure")
        assert uploads[0]["dependency"] == 0


def test_native_nested_graph_and_resource_owners_keep_independent_scopes(
    source_probe: Path,
) -> None:
    result = subprocess.run(
        [str(source_probe), "nested-owners"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    raw = json.loads(result.stdout)
    metadata = dict(zip(METADATA, raw["metadata"], strict=True))
    records = [
        dict(zip(FIELDS, raw["records"][offset : offset + len(FIELDS)], strict=True))
        for offset in range(0, len(raw["records"]), len(FIELDS))
    ]
    assert metadata["record_count"] == len(records) == 20
    assert metadata["errors"] == metadata["dropped_records"] == 0
    assert metadata["outstanding_operations"] == raw["dispatch_errors"] == 0
    assert {row["owner"] for row in records} == {1, 2, 3, 4}
    starts = {row["execution_id"] for row in records if row["kind"] == 3}
    ends = {row["execution_id"] for row in records if row["kind"] == 4}
    assert len(starts) == 4 and starts == ends
    fences = [row for row in records if row["kind"] == 2 and row["operation_kind"] == 2]
    assert [(row["owner"], row["role"], row["site"]) for row in fences] == [
        (1, 4, 3),
        (2, 1, 24),
        (4, 7, 26),
        (3, 7, 25),
    ]
    assert all(row["status"] == row["dependency"] == 0 for row in fences)


@pytest.mark.parametrize(
    "mode",
    [
        "ledger-release",
        "ledger-post-scope",
        "ledger-untracked",
        "ledger-free-failure",
        "ledger-sync-failure",
        "ledger-rollback",
        "ledger-rollback-sync-failure",
        "ledger-disabled",
        "ledger-observer-failure",
    ],
)
def test_native_ledger_fences_preserve_conditional_calls_and_status(
    source_probe: Path, mode: str
) -> None:
    """SDK-header CPU doubles check real ledger calls, retirement and correlation."""
    result = subprocess.run(
        [str(source_probe), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    raw = json.loads(result.stdout)
    metadata = dict(zip(METADATA, raw["metadata"], strict=True))
    records = [
        dict(zip(FIELDS, raw["records"][offset : offset + len(FIELDS)], strict=True))
        for offset in range(0, len(raw["records"]), len(FIELDS))
    ]
    assert metadata["errors"] == metadata["dropped_records"] == 0
    assert metadata["outstanding_operations"] == 0
    assert raw["dispatch_errors"] == (4 if mode == "ledger-observer-failure" else 0)
    ledger = [row for row in records if row["owner"] == 5]
    skipped = mode in {"ledger-disabled", "ledger-untracked", "ledger-free-failure"}
    assert len(ledger) == (0 if skipped else 4)
    assert len(records) == (0 if mode == "ledger-disabled" else 8 + len(ledger))
    if skipped:
        return
    assert [row["kind"] for row in ledger] == [3, 1, 2, 4]
    assert ledger[1]["role"] == 7
    assert ledger[1]["site"] == (28 if mode.startswith("ledger-rollback") else 27)
    assert ledger[1]["operation_kind"] == 2
    assert ledger[1]["payload"] == ledger[1]["payload_instance"] == 0
    assert ledger[1]["dependency"] == ledger[1]["bytes"] == 0
    assert ledger[2]["status"] == int(mode.endswith("sync-failure"))
    assert len({row["execution_id"] for row in ledger}) == 1
