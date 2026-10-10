"""Actual SDK layout with CPU-only CUDA/CUPTI doubles, not GPU qualification."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools.cupti_graph_inventory import FIELDS, METADATA

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

extern "C" int generativeqc_cupti_graph_begin_v1(std::uint64_t, std::uint64_t, std::uint64_t);
extern "C" void generativeqc_cupti_graph_observe_v1(const std::uint64_t*, std::size_t, void*);
extern "C" int generativeqc_cupti_graph_stop_v1();
extern "C" int generativeqc_cupti_graph_read_v1(std::uint64_t*, std::uint64_t, std::uint64_t*, std::uint64_t);

std::string mode;
std::uint64_t correlation = 0;
int root_storage, child_storage, executable_storage;
std::array<int, 3> node_storage;

extern "C" CUptiResult CUPTIAPI cuptiGetGraphId(CUgraph graph, std::uint32_t* identity) {
  if (mode == "identity-error") return CUPTI_ERROR_INVALID_PARAMETER;
  *identity = reinterpret_cast<void*>(graph) == &root_storage ? 10 : 11;
  return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiGetGraphExecId(CUgraphExec, std::uint32_t* identity) {
  *identity = 20; return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiGetGraphNodeId(CUgraphNode node, std::uint64_t* identity) {
  *identity = 30 + (reinterpret_cast<int*>(node) - node_storage.data());
  return CUPTI_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count) {
  if (mode == "cuda-error") return cudaErrorInvalidValue;
  const bool root = reinterpret_cast<void*>(graph) == &root_storage;
  const auto size = root ? 2u : 1u;
  if (nodes) {
    assert(*count == size);
    for (unsigned index = 0; index < size; ++index)
      nodes[index] = reinterpret_cast<cudaGraphNode_t>(&node_storage[root ? index : 2]);
  }
  *count = size;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type) {
  const auto index = reinterpret_cast<int*>(node) - node_storage.data();
  *type = index == 0 ? cudaGraphNodeTypeWaitEvent : index == 1 ? cudaGraphNodeTypeGraph :
      mode == "conditional" ? cudaGraphNodeTypeConditional :
      mode == "unknown" ? static_cast<cudaGraphNodeType>(99) : cudaGraphNodeTypeExtSemaphoreWait;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t, cudaGraph_t* graph) {
  *graph = reinterpret_cast<cudaGraph_t>(&child_storage); return cudaSuccess;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind, std::uint64_t identity) {
  assert(kind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1);
  assert(correlation == 0);
  correlation = identity; return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind, std::uint64_t* identity) {
  assert(kind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1);
  *identity = correlation; correlation = 0; return CUPTI_SUCCESS;
}
int main(int count, char** arguments) {
  assert(count == 2);
  mode = arguments[1];
  assert(generativeqc_cupti_graph_begin_v1(0, 8, 4) == -1);
  const auto capacity = mode == "overflow" ? 1u : 64u;
  assert(generativeqc_cupti_graph_begin_v1(capacity, mode == "node-bound" ? 1 : 8,
                                         mode == "depth-bound" ? 1 : 4) == 0);
  assert(generativeqc_cupti_graph_begin_v1(capacity, 8, 4) == -1);
  if (mode == "source-layout" || mode == "unknown-source-layout") {
    std::array<std::uint64_t, 14> boundary{};
    boundary[0] = mode == "source-layout" ? 2 : 3;
    generativeqc_cupti_graph_observe_v1(boundary.data(), boundary.size(), nullptr);
  }
  std::array<std::uint64_t, 10> event{1, 1, 1, 1, mode == "device" ? 4u : 0u,
      reinterpret_cast<std::uintptr_t>(&root_storage), reinterpret_cast<std::uintptr_t>(&executable_storage), 0, 0, 0};
  generativeqc_cupti_graph_observe_v1(event.data(), event.size(), nullptr);
  if (mode != "identity-error") {
    event[1] = 2;
    generativeqc_cupti_graph_observe_v1(event.data(), event.size(), nullptr);
    assert(correlation == 1);
    if (mode != "pending") {
      event[1] = 3;
      event[8] = mode == "launch-error" ? 1 : 0;
      generativeqc_cupti_graph_observe_v1(event.data(), event.size(), nullptr);
      assert(correlation == 0);
      event[1] = 4;
      generativeqc_cupti_graph_observe_v1(event.data(), event.size(), nullptr);
    }
  }
  assert(generativeqc_cupti_graph_stop_v1() == 0);
  assert(generativeqc_cupti_graph_stop_v1() == -1);
  std::array<std::uint64_t, 8> metadata{};
  assert(generativeqc_cupti_graph_read_v1(nullptr, 0, metadata.data(), metadata.size()) == 0);
  std::vector<std::uint64_t> records(metadata[0] * 12, 99);
  const auto before = metadata;
  assert(generativeqc_cupti_graph_read_v1(records.data(), 0, metadata.data(), metadata.size()) == -1);
  assert(metadata == before && std::all_of(records.begin(), records.end(), [](auto value) { return value == 99; }));
  assert(generativeqc_cupti_graph_read_v1(records.data(), metadata[0], metadata.data(), metadata.size()) == 0);
  std::cout << "{\"metadata\":[";
  for (std::size_t index = 0; index < metadata.size(); ++index) std::cout << (index ? "," : "") << metadata[index];
  std::cout << "],\"records\":[";
  for (std::size_t index = 0; index < records.size(); ++index) std::cout << (index ? "," : "") << records[index];
  std::cout << "]}\n";
}
"""


@pytest.fixture(scope="module")
def graph_probe(
    native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Link only CPU doubles while compiling the real optional inventory source."""
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
    folder = tmp_path_factory.mktemp("graph-inventory-double")
    source = folder / "probe.cpp"
    source.write_text(PROBE)
    output = folder / "probe"
    native_cxx.build_executable(
        [ROOT / "tools/cupti_graph_inventory.cpp", source],
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
    "mode",
    [
        "ordinary",
        "device",
        "conditional",
        "unknown",
        "overflow",
        "node-bound",
        "depth-bound",
        "cuda-error",
        "identity-error",
        "pending",
        "launch-error",
        "source-layout",
        "unknown-source-layout",
    ],
)
def test_bounded_actual_header_inventory(graph_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(graph_probe), mode], check=True, capture_output=True, text=True, timeout=10
    )
    raw = json.loads(result.stdout)
    metadata = dict(zip(METADATA, raw["metadata"], strict=True))
    records = [
        dict(zip(FIELDS, raw["records"][offset : offset + len(FIELDS)], strict=True))
        for offset in range(0, len(raw["records"]), len(FIELDS))
    ]
    assert metadata["record_count"] == len(records)
    assert metadata["record_count"] <= metadata["capacity"]
    assert metadata["stopped"] == 1
    assert metadata["outstanding_launches"] == int(mode == "pending")
    assert bool(metadata["dropped_records"]) == (mode == "overflow")
    assert bool(metadata["errors"]) == (
        mode
        in {
            "node-bound",
            "depth-bound",
            "cuda-error",
            "identity-error",
            "unknown-source-layout",
        }
    )
    if mode in {"ordinary", "source-layout"}:
        nodes = [row for row in records if row["kind"] == 2]
        assert [
            (row["node"], row["node_type"], row["parent_node"]) for row in nodes
        ] == [(30, 6, 0), (31, 4, 0), (32, 9, 31)]
        assert [row["kind"] for row in records] == [1, 2, 2, 2, 3, 4, 5]
    if mode == "launch-error":
        assert next(row for row in records if row["kind"] == 4)["status"] == 1
