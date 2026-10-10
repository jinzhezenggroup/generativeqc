#include <cuda_runtime_api.h>
#include <cupti.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

static_assert(CUPTI_API_VERSION == 28, "This optional observer requires CUPTI API 28");

namespace {
constexpr std::size_t kFields = 12;
constexpr std::size_t kMetadata = 8;
using Record = std::array<std::uint64_t, kFields>;

/** Separate, process-owned, bounded storage: inspection must not grow the
 * production request or retain graph handles beyond the synchronous callback. */
struct Inventory {
  std::mutex mutex;
  std::vector<Record> records;
  std::vector<cudaGraphNode_t> workspace;
  std::array<cudaGraph_t, 16> ancestors{};
  std::uint64_t count{};
  std::uint64_t dropped{};
  std::uint64_t errors{};
  std::uint64_t launch_sequence{};
  std::uint64_t outstanding{};
  std::size_t node_limit{};
  std::size_t depth_limit{};
  bool begun{};
  bool stopped{};
};
Inventory& inventory = *new Inventory;
thread_local std::uint64_t active_launch{};
thread_local std::uint64_t active_generation{};

void increment(std::uint64_t& value) noexcept {
  if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
}

void append(const Record& record) {
  if (inventory.count == inventory.records.size())
    increment(inventory.dropped);
  else
    inventory.records[inventory.count++] = record;
}

void failure(const Record& source, std::uint64_t stage, std::uint64_t status = 0) {
  auto record = source;
  record[0] = 6;
  record[10] = status;
  record[11] = stage;
  increment(inventory.errors);
  append(record);
}

/** Recursively inspect immutable child graphs at their source owner's boundary.
 * Nodes, total traversal and depth are independently bounded. Conditional nodes
 * are retained but not traversed: a static inventory is not their execution trace.
 * No CUDA query occurs from a CUPTI API/activity callback. */
void inspect(cudaGraph_t graph, Record source, std::uint64_t parent, std::size_t depth,
             std::size_t& visited) {
  if (depth >= inventory.depth_limit) return failure(source, 1);
  for (std::size_t index = 0; index < depth; ++index) {
    if (inventory.ancestors[index] == graph) return failure(source, 2);
  }
  inventory.ancestors[depth] = graph;
  std::uint32_t graph_id = 0;
  const auto identity_status = cuptiGetGraphId(reinterpret_cast<CUgraph>(graph), &graph_id);
  if (identity_status != CUPTI_SUCCESS || !graph_id) return failure(source, 3, identity_status);
  source[4] = graph_id;
  std::size_t size = 0;
  auto status = cudaGraphGetNodes(graph, nullptr, &size);
  if (status != cudaSuccess) return failure(source, 4, status);
  if (size > inventory.node_limit - visited) return failure(source, 5);
  auto* nodes = inventory.workspace.data() + depth * inventory.node_limit;
  auto actual = size;
  status = cudaGraphGetNodes(graph, nodes, &actual);
  if (status != cudaSuccess || actual != size) return failure(source, 6, status);
  visited += size;
  for (std::size_t index = 0; index < size; ++index) {
    cudaGraphNodeType type{};
    status = cudaGraphNodeGetType(nodes[index], &type);
    if (status != cudaSuccess) {
      failure(source, 7, status);
      continue;
    }
    std::uint64_t node_id = 0;
    const auto node_status =
        cuptiGetGraphNodeId(reinterpret_cast<CUgraphNode>(nodes[index]), &node_id);
    if (node_status != CUPTI_SUCCESS || !node_id) {
      failure(source, 8, node_status);
      continue;
    }
    auto record = source;
    record[0] = 2;
    record[6] = node_id;
    record[7] = type;
    record[8] = parent;
    append(record);
    if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = nullptr;
      status = cudaGraphChildGraphNodeGetGraph(nodes[index], &child);
      if (status != cudaSuccess || !child)
        failure(source, 9, status);
      else
        inspect(child, source, node_id, depth + 1, visited);
    }
  }
}
}  // namespace

/** Allocate all storage before production begins. One inventory per process,
 * independently bounded from CUPTI's asynchronous scalar activity records. */
extern "C" int generativeqc_cupti_graph_begin_v1(std::uint64_t capacity, std::uint64_t node_limit,
                                                 std::uint64_t depth_limit) {
  std::lock_guard lock(inventory.mutex);
  if (inventory.begun || capacity == 0 || capacity > (1u << 20) || node_limit == 0 ||
      node_limit > 4096 || depth_limit == 0 || depth_limit > inventory.ancestors.size())
    return -1;
  try {
    inventory.records.resize(capacity);
    inventory.workspace.resize(node_limit * depth_limit);
  } catch (...) {
    return -2;
  }
  inventory.node_limit = node_limit;
  inventory.depth_limit = depth_limit;
  inventory.begun = true;
  return 0;
}

/** Callback matching the production source observer's v1 scalar ABI. IDs from
 * CUPTI are correlated to source lifetime generations, never pointer equality.
 * CUSTOM1 marks the real host launch API independently of outer CUSTOM0 phases;
 * accepted submissions do not establish graph completion or device-tail counts. */
extern "C" void generativeqc_cupti_graph_observe_v1(const std::uint64_t* values, std::size_t count,
                                                    void*) {
  // A graph-only observer does not consume the independent boundary journal.
  // The multiplexer validates layout 2; unknown layouts still fail closed here.
  if (values && count == 14 && values[0] == 2) return;
  std::lock_guard lock(inventory.mutex);
  Record record{};
  if (!inventory.begun || inventory.stopped || !values || count != 10 || values[0] != 1 ||
      values[9] || values[1] < 1 || values[1] > 4 || values[3] < 1 || values[3] > 2) {
    failure(record, 10);
    return;
  }
  record[0] = values[1] == 1 ? 1 : values[1] + 1;
  record[1] = values[2];
  record[2] = values[3];
  record[3] = values[4];
  record[10] = values[8];
  if (!values[2] || !values[5] || !values[6]) {
    failure(record, 11);
    return;
  }
  const auto graph = reinterpret_cast<cudaGraph_t>(static_cast<std::uintptr_t>(values[5]));
  const auto executable = reinterpret_cast<CUgraphExec>(static_cast<std::uintptr_t>(values[6]));
  std::uint32_t graph_id = 0, executable_id = 0;
  const auto graph_status = cuptiGetGraphId(reinterpret_cast<CUgraph>(graph), &graph_id);
  const auto executable_status = cuptiGetGraphExecId(executable, &executable_id);
  if (graph_status != CUPTI_SUCCESS || executable_status != CUPTI_SUCCESS || !graph_id ||
      !executable_id) {
    failure(record, 12, graph_status != CUPTI_SUCCESS ? graph_status : executable_status);
    return;
  }
  record[4] = graph_id;
  record[5] = executable_id;
  if (values[1] == 1) {
    append(record);
    std::size_t visited = 0;
    inspect(graph, record, 0, 0, visited);
    return;
  }
  if (values[1] == 2) {
    if (active_launch || inventory.launch_sequence == std::numeric_limits<std::uint64_t>::max()) {
      failure(record, 13);
      return;
    }
    const auto launch = ++inventory.launch_sequence;
    const auto status =
        cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, launch);
    if (status != CUPTI_SUCCESS) return failure(record, 14, status);
    active_launch = launch;
    active_generation = values[2];
    ++inventory.outstanding;
    record[9] = launch;
  } else if (values[1] == 3) {
    record[9] = active_launch;
    if (!active_launch || active_generation != values[2]) {
      failure(record, 15);
      return;
    }
    std::uint64_t popped = 0;
    const auto status =
        cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, &popped);
    if (status != CUPTI_SUCCESS || popped != active_launch) failure(record, 16, status);
    active_launch = 0;
    active_generation = 0;
    --inventory.outstanding;
  }
  append(record);
}

extern "C" int generativeqc_cupti_graph_stop_v1() {
  std::lock_guard lock(inventory.mutex);
  if (!inventory.begun || inventory.stopped) return -1;
  inventory.stopped = true;
  return 0;
}

/** Read after the source callback is detached. Short buffers leave all output
 * untouched; errors and outstanding launches remain explicit incompleteness. */
extern "C" int generativeqc_cupti_graph_read_v1(std::uint64_t* records, std::uint64_t capacity,
                                                std::uint64_t* metadata, std::uint64_t count) {
  std::lock_guard lock(inventory.mutex);
  if (!inventory.stopped || !metadata || count != kMetadata || (!records && capacity) ||
      (records && capacity < inventory.count))
    return -1;
  const std::array<std::uint64_t, kMetadata> values{
      inventory.count,      inventory.records.size(), inventory.dropped, inventory.errors,
      inventory.node_limit, inventory.depth_limit,    inventory.stopped, inventory.outstanding};
  std::copy(values.begin(), values.end(), metadata);
  if (records) {
    for (std::size_t index = 0; index < inventory.count; ++index) {
      std::copy(inventory.records[index].begin(), inventory.records[index].end(),
                records + index * kFields);
    }
  }
  return 0;
}
