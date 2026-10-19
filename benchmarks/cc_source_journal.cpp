#include <cupti.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cc/phase_observation_contract.hpp"
#include "cc_trace_io.hpp"

extern "C" {
int generativeqc_cc_cupti_timestamp_v1(std::uint64_t*);
int generativeqc_cc_cupti_push_v1(std::uint64_t);
int generativeqc_cc_cupti_pop_v1(std::uint64_t);
int generativeqc_cupti_graph_begin_v1(std::uint64_t, std::uint64_t, std::uint64_t);
int generativeqc_cupti_graph_stop_v1();
int generativeqc_cupti_graph_read_v1(std::uint64_t*, std::uint64_t, std::uint64_t*, std::uint64_t);
int generativeqc_cupti_source_begin_v1(std::uint64_t);
int generativeqc_cupti_source_stop_v1();
int generativeqc_cupti_source_read_v1(std::uint64_t*, std::uint64_t, std::uint64_t*, std::uint64_t);
void generativeqc_cupti_residency_observe_v1(const std::uint64_t*, std::size_t, void*);
}
namespace {
namespace protocol = generativeqc::cc::phase_observation;
using Row = std::array<std::uint64_t, 19>;  // normalized timestamp, word count, <=17 raw words
struct Journal {
  std::vector<Row> rows;
  std::uint64_t count{}, dropped{}, malformed{}, clocks{}, correlations{};
  std::atomic<std::uint64_t> threads{0};
  std::thread::id owner;
  std::array<std::uint64_t, 24> metadata{};
  bool begun{}, stopped{}, inside{};
};
Journal& journal = *new Journal;  // Pinned fresh-process lifetime, including late callbacks.
std::uint64_t process_start() {
  std::ifstream input("/proc/self/stat");
  std::string text;
  std::getline(input, text);
  const auto end = text.rfind(')');
  if (end == std::string::npos) return 0;
  std::istringstream fields(text.substr(end + 2));
  std::string token;
  for (int i = 0; i < 20; ++i)
    if (!(fields >> token)) return 0;
  try {
    return std::stoull(token);
  } catch (...) {
    return 0;
  }
}
bool shape(const std::uint64_t* words, std::size_t count) {
  return words && count && count <= 17 &&
         ((words[0] == 1 && count == 10) || (words[0] == 2 && count == 14) ||
          (words[0] == protocol::kPhaseLayout && count == protocol::kPhaseWords) ||
          (words[0] == protocol::kAdmissionLayout && count == protocol::kAdmissionWords) ||
          (words[0] == protocol::kTriplesAdmissionLayout &&
           count == protocol::kTriplesAdmissionWords));
}
std::array<std::uint64_t, 24> metadata() {
  auto out = journal.metadata;
  out[0] = journal.count;
  out[1] = journal.rows.size();
  out[2] = journal.dropped;
  out[3] = journal.malformed;
  out[4] = journal.clocks;
  out[5] = journal.correlations;
  out[6] = journal.threads.load();
  out[19] = journal.stopped;
  return out;
}
}  // namespace
extern "C" int generativeqc_cc_source_begin_v1(std::uint64_t capacity, std::uint64_t contract) {
  if (journal.begun || !capacity || capacity > (1u << 18) ||
      contract != protocol::kDetailedContract)
    return -1;
  try {
    journal.rows.resize(capacity);
  } catch (...) {
    return -2;
  }
  journal.begun = true;
  journal.owner = std::this_thread::get_id();
  journal.metadata[7] = getpid();
  journal.metadata[8] = syscall(SYS_gettid);
  CUpti_ActivityThreadIdType type{};
  if (cuptiGetThreadIdType(&type) != CUPTI_SUCCESS || type != CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM)
    return -3;
  journal.metadata[9] = journal.metadata[8];
  journal.metadata[11] = contract;
  journal.metadata[22] = process_start();
  if (!journal.metadata[22]) return -4;
  if (generativeqc_cc_cupti_timestamp_v1(&journal.metadata[17])) return -5;
  if (generativeqc_cupti_graph_begin_v1(capacity, 4096, 16)) return -6;
  if (generativeqc_cupti_source_begin_v1(capacity)) return -7;
  return 0;
}
extern "C" int generativeqc_cc_source_permit_v1(std::uint64_t permit) {
  if (!journal.begun || journal.stopped || !permit || journal.metadata[10]) return -1;
  journal.metadata[10] = permit;
  return 0;
}
extern "C" void generativeqc_cc_source_observe_v1(const std::uint64_t* words, std::size_t count,
                                                  void*) noexcept {
  if (std::this_thread::get_id() != journal.owner) {
    journal.threads.fetch_add(1);
    return;
  }
  if (!journal.begun || journal.stopped || journal.inside) {
    ++journal.malformed;
    return;
  }
  journal.inside = true;
  try {
    if (!shape(words, count)) {
      ++journal.malformed;
      journal.inside = false;
      return;
    }
    std::uint64_t timestamp = 0;
    if (generativeqc_cc_cupti_timestamp_v1(&timestamp)) ++journal.clocks;
    if (words[0] == protocol::kPhaseLayout) {
      if (words[10] != journal.metadata[10]) ++journal.malformed;
      if (words[1] == 1) {
        if (generativeqc_cc_cupti_push_v1(words[2])) ++journal.correlations;
      } else if (words[1] == 2) {
        if (generativeqc_cc_cupti_pop_v1(words[2])) ++journal.correlations;
      } else
        ++journal.malformed;
    } else if (words[0] == 1 || words[0] == 2) {
      generativeqc_cupti_residency_observe_v1(words, count, nullptr);
    } else if (words[0] == protocol::kAdmissionLayout) {
      if (words[12] != journal.metadata[10]) ++journal.malformed;
    } else if (words[0] == protocol::kTriplesAdmissionLayout) {
      if (words[15] != journal.metadata[10]) ++journal.malformed;
    }
    if (journal.count == journal.rows.size())
      ++journal.dropped;
    else {
      auto& row = journal.rows[journal.count++];
      row[0] = timestamp;
      row[1] = count;
      std::copy(words, words + count, row.begin() + 2);
    }
  } catch (...) {
    ++journal.malformed;
  }
  journal.inside = false;
}
extern "C" int generativeqc_cc_source_seal_v1(std::uint64_t finish_result,
                                              std::uint64_t phase_errors,
                                              std::uint64_t unbind_result,
                                              std::uint64_t dispatch_errors,
                                              std::uint64_t science_result) {
  if (!journal.begun || journal.stopped || journal.inside ||
      std::this_thread::get_id() != journal.owner)
    return -1;
  journal.metadata[12] = finish_result;
  journal.metadata[13] = phase_errors;
  journal.metadata[14] = unbind_result;
  journal.metadata[15] = dispatch_errors;
  journal.metadata[16] = science_result;
  if (generativeqc_cc_cupti_timestamp_v1(&journal.metadata[18])) ++journal.clocks;
  journal.metadata[20] = generativeqc_cupti_source_stop_v1();
  journal.metadata[21] = generativeqc_cupti_graph_stop_v1();
  journal.stopped = true;
  return 0;
}
extern "C" int generativeqc_cc_source_read_v1(std::uint64_t* out, std::uint64_t size) {
  if (!journal.stopped || !out || size != 24) return -1;
  const auto data = metadata();
  std::copy(data.begin(), data.end(), out);
  return 0;
}
extern "C" int generativeqc_cc_source_export_v1(int fd) {
  if (!journal.stopped || fd < 0) return -1;
  const auto meta = metadata();
  const std::array<std::uint64_t, 5> header{0x4343534f55524331ULL, 1, 19, 24, journal.count};
  if (auto error = cc_trace_io::write_all(fd, header.data(), sizeof(header))) return error;
  if (auto error = cc_trace_io::write_all(fd, meta.data(), sizeof(meta))) return error;
  return cc_trace_io::write_all(fd, journal.rows.data(), journal.count * sizeof(Row));
}
extern "C" int generativeqc_cc_legacy_export_v1(int fd, std::uint64_t kind) {
  if (fd < 0 || !journal.stopped || (kind != 1 && kind != 2)) return -1;
  const auto read =
      kind == 1 ? generativeqc_cupti_graph_read_v1 : generativeqc_cupti_source_read_v1;
  const std::uint64_t words = kind == 1 ? 12 : 14;
  std::array<std::uint64_t, 8> before{}, after{};
  if (read(nullptr, 0, before.data(), 8) || before[0] > journal.rows.size()) return -2;
  std::vector<std::uint64_t> rows;
  try {
    rows.resize(before[0] * words);
  } catch (...) {
    return -3;
  }
  if (read(rows.data(), before[0], after.data(), 8) || before != after) return -4;
  const std::array<std::uint64_t, 5> header{
      kind == 1 ? 0x4343475241504831ULL : 0x43434c4547414331ULL, 1, words, 8, before[0]};
  if (auto error = cc_trace_io::write_all(fd, header.data(), sizeof(header))) return error;
  if (auto error = cc_trace_io::write_all(fd, before.data(), sizeof(before))) return error;
  return cc_trace_io::write_all(fd, rows.data(), rows.size() * sizeof(std::uint64_t));
}

extern "C" std::uint64_t generativeqc_cc_source_schema_v1(std::uint64_t field) {
  switch (field) {
    case 0:
      return 1;
    case 1:
      return 19;
    case 2:
      return 24;
    case 3:
      return protocol::kDetailedContract;
    default:
      return 0;
  }
}
