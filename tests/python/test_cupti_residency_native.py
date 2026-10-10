"""CPU-double gates for the optional collector's bounded native scalar ABI.

Actual CUPTI 28 headers are required, but all linked CUPTI functions are doubles:
these probes never access a GPU and are not runtime residency qualification.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
#include <cupti.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

extern "C" int generativeqc_cupti_begin_v1(std::uint64_t);
extern "C" int generativeqc_cupti_stop_v1();
extern "C" int generativeqc_cupti_read_v1(std::uint64_t*, std::uint64_t, std::uint64_t*, std::uint64_t);

std::string mode;
CUpti_BuffersCallbackRequestFunc request_callback;
CUpti_BuffersCallbackCompleteFunc complete_callback;
std::array<CUpti_Activity*, 4> activities;
std::size_t cursor = 0;
bool flushed = false;

extern "C" CUptiResult CUPTIAPI cuptiGetVersion(std::uint32_t* version) {
  *version = mode == "version" ? 27 : 28;
  return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc requested, CUpti_BuffersCallbackCompleteFunc completed) {
  request_callback = requested;
  complete_callback = completed;
  return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind) { return CUPTI_SUCCESS; }
extern "C" CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind) { return CUPTI_SUCCESS; }
extern "C" CUptiResult CUPTIAPI cuptiActivityGetNextRecord(
    std::uint8_t*, std::size_t bytes, CUpti_Activity** output) {
  if (mode == "starve" || bytes == 0 || cursor == activities.size()) return CUPTI_ERROR_MAX_LIMIT_REACHED;
  *output = activities[cursor++];
  return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityGetNumDroppedRecords(
    CUcontext, std::uint32_t, std::size_t* count) {
  *count = mode == "dropped" && !flushed ? 3 : 0;
  return mode == "parse" ? CUPTI_ERROR_UNKNOWN : CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityFlushAll(std::uint32_t) {
  if (flushed) return CUPTI_SUCCESS;
  std::array<std::uint8_t*, 17> buffers{};
  const std::size_t requests = mode == "starve" ? buffers.size() : 1;
  for (std::size_t index = 0; index < requests; ++index) {
    std::size_t bytes = 0, records = 0;
    request_callback(&buffers[index], &bytes, &records);
    assert(records == 0);
    assert((buffers[index] == nullptr) == (index == 16));
    assert(bytes == (index == 16 ? 0 : 65536));
  }
  for (auto* buffer : buffers) {
    if (buffer) complete_callback(nullptr, 0, buffer, 65536, mode == "starve" ? 0 : 64);
  }
  flushed = true;
  return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityPushExternalCorrelationId(
    CUpti_ExternalCorrelationKind, std::uint64_t) { return CUPTI_SUCCESS; }
extern "C" CUptiResult CUPTIAPI cuptiActivityPopExternalCorrelationId(
    CUpti_ExternalCorrelationKind, std::uint64_t* value) { *value = 1; return CUPTI_SUCCESS; }
extern "C" CUptiResult CUPTIAPI cuptiGetCallbackName(
    CUpti_CallbackDomain, CUpti_CallbackId, const char** name) { *name = "double"; return CUPTI_SUCCESS; }

int main(int count, char** arguments) {
  assert(count == 2);
  mode = arguments[1];
  if (mode == "version") {
    assert(generativeqc_cupti_begin_v1(8) == -2);
    return 0;
  }
  CUpti_ActivityMemcpy6 copy{};
  copy.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
  copy.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
  copy.bytes = 128;
  copy.start = 10;
  copy.end = 20;
  copy.correlationId = 7;
  copy.runtimeCorrelationId = 8;
  copy.graphNodeId = 9;
  copy.graphId = 10;
  copy.copyCount = 1;
  CUpti_ActivitySynchronization2 sync{};
  sync.kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
  sync.type = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE;
  sync.returnValue = 4;
  CUpti_ActivityExternalCorrelation external{};
  external.kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
  external.externalKind = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;
  external.externalId = 1;
  external.correlationId = 7;
  CUpti_ActivityAPI api{};
  api.kind = mode == "unknown" ? CUPTI_ACTIVITY_KIND_KERNEL : CUPTI_ACTIVITY_KIND_RUNTIME;
  api.cbid = 3;
  api.returnValue = 6;
  activities = {reinterpret_cast<CUpti_Activity*>(&copy), reinterpret_cast<CUpti_Activity*>(&sync),
                reinterpret_cast<CUpti_Activity*>(&external), reinterpret_cast<CUpti_Activity*>(&api)};
  assert(generativeqc_cupti_begin_v1(0) == -1);
  assert(generativeqc_cupti_begin_v1(1u << 21) == -1);
  assert(generativeqc_cupti_begin_v1(mode == "overflow" ? 2 : 8) == 0);
  assert(generativeqc_cupti_begin_v1(8) == -1);
  assert(generativeqc_cupti_stop_v1() == (mode == "parse" ? -5 : 0));
  assert(generativeqc_cupti_stop_v1() == -1);
  std::array<std::uint64_t, 10> metadata;
  metadata.fill(77);
  assert(generativeqc_cupti_read_v1(nullptr, 0, metadata.data(), 9) == -1);
  for (auto value : metadata) assert(value == 77);
  assert(generativeqc_cupti_read_v1(nullptr, 0, metadata.data(), 10) == 0);
  assert(metadata[0] == (mode == "starve" ? 0 : mode == "overflow" ? 2 : mode == "unknown" ? 3 : 4));
  assert(metadata[2] == (mode == "overflow" ? 2 : mode == "dropped" ? 3 : 0));
  assert(metadata[3] == (mode == "starve" ? 1 : 0));
  assert(metadata[4] == (mode == "parse" ? 1 : 0));
  assert(metadata[5] == (mode == "unknown" ? 1 : 0));
  assert(metadata[6] == 0 && metadata[7] == 28 && metadata[8] == 1);
  std::array<std::uint64_t, 64> records;
  records.fill(99);
  if (mode != "starve") {
    auto before = metadata;
    assert(generativeqc_cupti_read_v1(records.data(), 1, metadata.data(), 10) == -1);
    assert(metadata == before);
    for (auto value : records) assert(value == 99);
    assert(generativeqc_cupti_read_v1(records.data(), 4, metadata.data(), 10) == 0);
    assert(records[0] == 1 && records[2] == 128 && records[5] == 7 && records[6] == 8);
    assert(records[11] == 9 && records[12] == 10 && records[15] == 1);
    assert(records[16] == 38 && records[16 + 13] == 4);
  }
}
"""


@pytest.fixture(scope="module")
def native_probe(
    native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Compile against real layouts but link CPU doubles, never libcupti."""
    toolkit = Path(
        os.environ.get(
            "GENERATIVEQC_CUPTI_TEST_HEADERS",
            ROOT / ".artifacts/issue1629-cupti/toolkit",
        )
    )
    if (
        not (toolkit / "cupti/cupti.h").is_file()
        or not (toolkit / "cuda/cuda.h").is_file()
    ):
        pytest.skip("requires optional CUPTI 28/CUDA header export")
    folder = tmp_path_factory.mktemp("cupti-native-double")
    source = folder / "probe.cpp"
    source.write_text(PROBE)
    output = folder / "probe"
    native_cxx.build_executable(
        [ROOT / "tools/cupti_residency_capture.cpp", source],
        output,
        compile_args=(
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{toolkit / 'cupti'}",
            f"-I{toolkit / 'cuda'}",
        ),
        link_args=("-pthread",),
    )
    return output


@pytest.mark.parametrize(
    "mode", ["ordinary", "version", "overflow", "starve", "dropped", "parse", "unknown"]
)
def test_actual_header_layout_and_bounded_native_abi(
    native_probe: Path, mode: str
) -> None:
    subprocess.run(
        [str(native_probe), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
