#include <cupti.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

static_assert(CUPTI_API_VERSION == 28, "This optional source collector requires CUPTI API 28");
extern "C" void generativeqc_cupti_graph_observe_v1(const std::uint64_t*, std::size_t, void*);

namespace {
constexpr std::size_t kFields = 14;
constexpr std::size_t kMetadata = 8;
using Record = std::array<std::uint64_t, kFields>;
struct SourceCapture {
  std::mutex mutex;
  std::vector<Record> records;
  std::uint64_t count{};
  std::uint64_t dropped{};
  std::uint64_t errors{};
  std::uint64_t outstanding{};
  std::uint64_t version{};
  bool begun{};
  bool stopped{};
};
/** Retain callback storage for process lifetime; only stopped captures are read.
 * The record array is preallocated before any source operation is observed. */
SourceCapture& source_capture = *new SourceCapture;
thread_local Record active_operation{};

void increment(std::uint64_t& value) {
  if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
}
void append(const Record& record) {
  if (source_capture.count == source_capture.records.size())
    increment(source_capture.dropped);
  else
    source_capture.records[source_capture.count++] = record;
}
void fail(Record record, std::uint64_t stage) {
  record[0] = 2;
  record[1] = 7;
  record[13] = stage;
  increment(source_capture.errors);
  append(record);
}

/** A failed end record must still unwind this observer's correlation stack.
 * Observation errors are retained but must not misattribute later operations. */
void end_operation(const Record& record) {
  if (!active_operation[2]) {
    fail(record, 6);
    return;
  }
  auto expected = active_operation;
  expected[1] = 2;
  expected[13] = record[13];
  if (record != expected) fail(record, 7);
  std::uint64_t actual = 0;
  const auto status =
      cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2, &actual);
  if (status != CUPTI_SUCCESS || actual != active_operation[2]) fail(record, 8);
  active_operation = {};
  --source_capture.outstanding;
  append(record);
}

void observe_source(const std::uint64_t* values, std::size_t count) {
  std::lock_guard lock(source_capture.mutex);
  Record record{};
  if (values && count == kFields) std::copy_n(values, kFields, record.begin());
  if (!source_capture.begun || source_capture.stopped || !values || count != kFields ||
      record[0] != 2 || record[1] < 1 || record[1] > 4) {
    fail(record, 1);
    return;
  }
  if (record[1] == 2) {
    end_operation(record);
    return;
  }
  if (!record[3] || !record[4]) {
    fail(record, 2);
    return;
  }
  if (record[1] == 3 || record[1] == 4) {
    if (record[2] ||
        std::any_of(record.begin() + 5, record.end(), [](auto value) { return value != 0; })) {
      fail(record, 3);
      return;
    }
    append(record);
    return;
  }
  const bool transfer = record[10] == 1;
  const bool sync = record[10] == 2 || record[10] == 3;
  if (!record[2] || record[5] < 1 || record[5] > 7 || !record[6] || record[13] ||
      (!transfer && !sync) ||
      (transfer && (!record[7] || !record[8] || record[11] < 1 || record[11] > 3)) ||
      (sync && (record[7] || record[8] || record[9] || record[11] || record[12]))) {
    fail(record, 4);
    return;
  }
  if (active_operation[2]) {
    fail(record, 5);
    return;
  }
  const auto status =
      cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2, record[2]);
  if (status != CUPTI_SUCCESS) {
    fail(record, 9);
    return;
  }
  active_operation = record;
  ++source_capture.outstanding;
  append(record);
}
}  // namespace

extern "C" int generativeqc_cupti_source_begin_v1(std::uint64_t capacity) {
  std::lock_guard lock(source_capture.mutex);
  if (source_capture.begun || !capacity || capacity > (1u << 20)) return -1;
  std::uint32_t version = 0;
  if (cuptiGetVersion(&version) != CUPTI_SUCCESS || version != CUPTI_API_VERSION) return -2;
  try {
    source_capture.records.resize(capacity);
  } catch (...) {
    return -3;
  }
  source_capture.version = version;
  source_capture.begun = true;
  return 0;
}

/** One production callback, independent graph/source layouts and correlation
 * domains. No CUDA API is called here except by the graph owner's safe observer;
 * actual transfer/sync execution remains exclusively owned by production. */
extern "C" void generativeqc_cupti_residency_observe_v1(const std::uint64_t* values,
                                                        std::size_t count, void* context) {
  if (values && count && values[0] == 1)
    generativeqc_cupti_graph_observe_v1(values, count, context);
  else
    observe_source(values, count);
}

extern "C" int generativeqc_cupti_source_stop_v1() {
  std::lock_guard lock(source_capture.mutex);
  if (!source_capture.begun || source_capture.stopped) return -1;
  source_capture.stopped = true;
  return 0;
}

/** Fixed scalar read ABI: no writes on malformed/short buffers, no truncation.
 * Lost events, pending operations and all callback errors remain explicit. */
extern "C" int generativeqc_cupti_source_read_v1(std::uint64_t* records, std::uint64_t capacity,
                                                 std::uint64_t* metadata, std::uint64_t count) {
  std::lock_guard lock(source_capture.mutex);
  if (!source_capture.stopped || !metadata || count != kMetadata || (!records && capacity) ||
      (records && capacity < source_capture.count))
    return -1;
  const std::array<std::uint64_t, kMetadata> values{
      source_capture.count,       source_capture.records.size(),
      source_capture.dropped,     source_capture.errors,
      source_capture.outstanding, source_capture.stopped,
      source_capture.version,     2};
  std::copy(values.begin(), values.end(), metadata);
  if (records) {
    for (std::size_t index = 0; index < source_capture.count; ++index) {
      std::copy(source_capture.records[index].begin(), source_capture.records[index].end(),
                records + index * kFields);
    }
  }
  return 0;
}
