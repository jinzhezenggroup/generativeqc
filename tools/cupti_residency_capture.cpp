#include <cupti.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

static_assert(CUPTI_API_VERSION == 28, "This optional collector requires CUPTI 12.9 Update 1");

namespace {

constexpr std::size_t kRecordFields = 16;
constexpr std::size_t kMetadataFields = 10;
constexpr std::size_t kBufferBytes = 64 * 1024;
constexpr std::size_t kBufferCount = 16;
using Record = std::array<std::uint64_t, kRecordFields>;

struct alignas(64) ActivityBuffer {
  std::array<std::uint8_t, kBufferBytes> bytes{};
  bool busy = false;
};

/** Process-lifetime storage keeps late CUPTI callbacks from accessing a freed
 * handle. One capture is permitted per fresh process. The record vector and
 * callback buffers are allocated before enabling activities and never grow. */
struct Capture {
  std::mutex mutex;
  std::vector<Record> records;
  std::array<ActivityBuffer, kBufferCount> buffers;
  std::uint64_t count = 0;
  std::uint64_t dropped = 0;
  std::uint64_t starved = 0;
  std::uint64_t parse_errors = 0;
  std::uint64_t unknown = 0;
  std::uint64_t finish_errors = 0;
  std::uint32_t version = 0;
  bool begun = false;
  bool stopped = false;
};

Capture& capture = *new Capture;
constexpr std::array<CUpti_ActivityKind, 7> kKinds{CUPTI_ACTIVITY_KIND_MEMCPY,
                                                   CUPTI_ACTIVITY_KIND_MEMCPY2,
                                                   CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
                                                   CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
                                                   CUPTI_ACTIVITY_KIND_RUNTIME,
                                                   CUPTI_ACTIVITY_KIND_DRIVER,
                                                   CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API};

void add_saturated(std::uint64_t& counter, std::uint64_t value = 1) noexcept {
  counter = value > std::numeric_limits<std::uint64_t>::max() - counter
                ? std::numeric_limits<std::uint64_t>::max()
                : counter + value;
}

void append(const Record& record) {
  std::lock_guard lock(capture.mutex);
  if (capture.count == capture.records.size()) {
    add_saturated(capture.dropped);
  } else {
    capture.records[capture.count++] = record;
  }
}

/** Only scalar fields are copied. Callback-owned pointers cannot survive the
 * completed-buffer callback, and address equality would not prove a payload's
 * scientific identity or a host-transform dependency. */
void consume(const CUpti_Activity* activity) {
  Record record{};
  record[0] = activity->kind;
  switch (activity->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      const auto& source = *reinterpret_cast<const CUpti_ActivityMemcpy6*>(activity);
      record[1] = source.copyKind;
      record[2] = source.bytes;
      record[3] = source.start;
      record[4] = source.end;
      record[5] = source.correlationId;
      record[6] = source.runtimeCorrelationId;
      record[8] = source.contextId;
      record[9] = source.streamId;
      record[10] = source.deviceId;
      record[11] = source.graphNodeId;
      record[12] = source.graphId;
      record[15] = source.copyCount;
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      const auto& source = *reinterpret_cast<const CUpti_ActivityMemcpyPtoP4*>(activity);
      record[1] = source.copyKind;
      record[2] = source.bytes;
      record[3] = source.start;
      record[4] = source.end;
      record[5] = source.correlationId;
      record[8] = source.contextId;
      record[9] = source.streamId;
      record[10] = source.deviceId;
      record[11] = source.graphNodeId;
      record[12] = source.graphId;
      record[15] = 1;
      break;
    }
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
      const auto& source = *reinterpret_cast<const CUpti_ActivitySynchronization2*>(activity);
      record[1] = source.type;
      record[3] = source.start;
      record[4] = source.end;
      record[5] = source.correlationId;
      record[7] = source.cudaEventSyncId;
      record[8] = source.contextId;
      record[9] = source.streamId;
      record[13] = source.returnValue;
      break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
      const auto& source = *reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(activity);
      record[1] = source.externalKind;
      record[5] = source.correlationId;
      record[7] = source.externalId;
      break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    case CUPTI_ACTIVITY_KIND_DRIVER:
    case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API: {
      const auto& source = *reinterpret_cast<const CUpti_ActivityAPI*>(activity);
      record[1] = source.cbid;
      record[3] = source.start;
      record[4] = source.end;
      record[5] = source.correlationId;
      record[13] = source.returnValue;
      record[14] = source.threadId;
      break;
    }
    default: {
      std::lock_guard lock(capture.mutex);
      add_saturated(capture.unknown);
      return;
    }
  }
  append(record);
}

void CUPTIAPI request_buffer(std::uint8_t** buffer, std::size_t* size, std::size_t* max_records) {
  std::lock_guard lock(capture.mutex);
  *buffer = nullptr;
  *size = 0;
  *max_records = 0;
  for (auto& slot : capture.buffers) {
    if (!slot.busy) {
      slot.busy = true;
      *buffer = slot.bytes.data();
      *size = slot.bytes.size();
      return;
    }
  }
  add_saturated(capture.starved);
}

void CUPTIAPI complete_buffer(CUcontext context, std::uint32_t stream, std::uint8_t* buffer,
                              std::size_t, std::size_t valid_bytes) {
  CUpti_Activity* activity = nullptr;
  CUptiResult result;
  while ((result = cuptiActivityGetNextRecord(buffer, valid_bytes, &activity)) == CUPTI_SUCCESS) {
    consume(activity);
  }
  std::size_t dropped = 0;
  const auto drop_status = cuptiActivityGetNumDroppedRecords(context, stream, &dropped);
  std::lock_guard lock(capture.mutex);
  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED || drop_status != CUPTI_SUCCESS) {
    add_saturated(capture.parse_errors);
  }
  add_saturated(capture.dropped, dropped);
  for (auto& slot : capture.buffers) {
    if (slot.bytes.data() == buffer && slot.busy) {
      slot.busy = false;
      return;
    }
  }
  add_saturated(capture.parse_errors);
}

}  // namespace

/** Verify the exact compiled decoder version before touching a CUDA collector.
 * Only the optional profiling library links CUPTI; production stays unchanged. */
extern "C" int generativeqc_cupti_begin_v1(std::uint64_t capacity) {
  if (capacity == 0 || capacity > (1u << 20)) return -1;
  std::uint32_t version = 0;
  auto status = cuptiGetVersion(&version);
  if (status != CUPTI_SUCCESS) return status;
  if (version != CUPTI_API_VERSION) return -2;
  {
    std::lock_guard lock(capture.mutex);
    if (capture.begun) return -1;
    try {
      capture.records.resize(capacity);
    } catch (...) {
      return -3;
    }
    capture.begun = true;
    capture.version = version;
  }
  status = cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
  if (status != CUPTI_SUCCESS) return status;
  for (std::size_t index = 0; index < kKinds.size(); ++index) {
    status = cuptiActivityEnable(kKinds[index]);
    if (status != CUPTI_SUCCESS) {
      while (index > 0) (void)cuptiActivityDisable(kKinds[--index]);
      return status;
    }
  }
  return 0;
}

/** Correlation IDs attach phase roles to the submitting thread, not a timestamp
 * guess. Other threads remain unassigned unless independently annotated. */
extern "C" int generativeqc_cupti_push_v1(std::uint64_t region) {
  return cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, region);
}

extern "C" int generativeqc_cupti_pop_v1(std::uint64_t expected) {
  std::uint64_t actual = 0;
  const auto status =
      cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &actual);
  return status != CUPTI_SUCCESS ? static_cast<int>(status) : (actual == expected ? 0 : -4);
}

/** The caller must fence real CUDA work in an explicitly observer-owned region
 * first. Flush does not synchronize CUDA; forced trailing records are retained,
 * and unknown/unfinished timestamps prevent a complete-trace claim. */
extern "C" int generativeqc_cupti_stop_v1() {
  {
    std::lock_guard lock(capture.mutex);
    if (!capture.begun || capture.stopped) return -1;
  }
  std::uint64_t errors = 0;
  if (cuptiActivityFlushAll(0) != CUPTI_SUCCESS) ++errors;
  for (auto kind : kKinds) {
    if (cuptiActivityDisable(kind) != CUPTI_SUCCESS) ++errors;
  }
  if (cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != CUPTI_SUCCESS) ++errors;
  std::size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped) != CUPTI_SUCCESS) ++errors;
  std::lock_guard lock(capture.mutex);
  add_saturated(capture.dropped, dropped);
  add_saturated(capture.finish_errors, errors);
  capture.stopped = true;
  return errors == 0 ? 0 : -5;
}

/** Fixed scalar ABI, read only after stop. Short buffers fail before any write;
 * a null record buffer with zero capacity queries metadata without truncation. */
extern "C" int generativeqc_cupti_read_v1(std::uint64_t* records, std::uint64_t capacity,
                                          std::uint64_t* metadata, std::uint64_t metadata_size) {
  std::lock_guard lock(capture.mutex);
  if (!capture.stopped || !metadata || metadata_size != kMetadataFields ||
      (!records && capacity != 0) || (records && capacity < capture.count)) {
    return -1;
  }
  const auto pending = std::count_if(capture.buffers.begin(), capture.buffers.end(),
                                     [](const auto& slot) { return slot.busy; });
  const std::array<std::uint64_t, kMetadataFields> values{capture.count,
                                                          capture.records.size(),
                                                          capture.dropped,
                                                          capture.starved,
                                                          capture.parse_errors,
                                                          capture.unknown,
                                                          static_cast<std::uint64_t>(pending),
                                                          capture.version,
                                                          capture.stopped,
                                                          capture.finish_errors};
  std::copy(values.begin(), values.end(), metadata);
  if (records) {
    for (std::size_t index = 0; index < capture.count; ++index) {
      std::copy(capture.records[index].begin(), capture.records[index].end(),
                records + index * kRecordFields);
    }
  }
  return 0;
}

extern "C" const char* generativeqc_cupti_api_name_v1(std::uint64_t kind, std::uint64_t callback) {
  const char* name = nullptr;
  if (kind != CUPTI_ACTIVITY_KIND_RUNTIME && kind != CUPTI_ACTIVITY_KIND_DRIVER &&
      kind != CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API)
    return nullptr;
  const auto domain = kind == CUPTI_ACTIVITY_KIND_RUNTIME ? CUPTI_CB_DOMAIN_RUNTIME_API
                                                          : CUPTI_CB_DOMAIN_DRIVER_API;
  return cuptiGetCallbackName(domain, static_cast<CUpti_CallbackId>(callback), &name) ==
                 CUPTI_SUCCESS
             ? name
             : nullptr;
}
