#pragma once

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>

#if defined(__CUDACC__)
#define GQC_REBASE_HD __host__ __device__
#else
#define GQC_REBASE_HD
#endif

namespace generativeqc::dft::vv10_rebase {

inline constexpr std::size_t kMaximumBytes = std::size_t{256} << 20;

enum class Request { Off, On, Invalid };
inline Request parse_request(const char* value) noexcept {
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return Request::Off;
  return std::strcmp(value, "1") == 0 ? Request::On : Request::Invalid;
}
inline Request requested() noexcept {
  return parse_request(std::getenv("GENERATIVEQC_KS_VV10_REBASE_HISTORY"));
}

/** Immutable association of an actual stored density/Fock/residual triple. */
struct Key {
  std::uint64_t owner{}, solve{}, anchor{}, metric{};
};
GQC_REBASE_HD inline bool same_key(Key a, Key b) noexcept {
  return a.owner == b.owner && a.solve == b.solve && a.anchor == b.anchor && a.metric == b.metric;
}
GQC_REBASE_HD inline bool valid_key(Key key) noexcept {
  return key.owner && key.solve && key.metric;
}
struct SlotTag {
  Key key{};
  std::uint64_t valid{};
};

enum class Error : std::uint32_t {
  None,
  InvalidWindow,
  MissingDensity,
  OwnerMismatch,
  AnchorMismatch,
  TargetWindow,
  Nonfinite,
  GramNonfinite
};
struct Work {
  std::uint64_t density_writes{}, snapshots{}, successful_rebases{}, entries_transferred{},
      refusals{};
};
struct Transaction {
  Key bank_key{}, target_key{};
  std::uint32_t bank_count{}, proposed_count{}, proposed_head{}, ready{};
  Error error{Error::None};
  std::uint32_t enabled{}, reserved0{}, reserved1{};
  Work work{};
  std::uint64_t serial{};
};

GQC_REBASE_HD inline bool valid_window(unsigned capacity, unsigned count, unsigned head) noexcept {
  return capacity >= 2 && capacity <= 64 && count <= capacity && head < capacity;
}
/** Only the DIIS owner's actual count/head define chronological live slots. */
GQC_REBASE_HD inline unsigned chronological_slot(unsigned capacity, unsigned count, unsigned head,
                                                 unsigned age) noexcept {
  return (head + capacity - count + age) % capacity;
}
GQC_REBASE_HD inline unsigned newest_bank_count(unsigned capacity, unsigned count) noexcept {
  return count < capacity ? count : capacity - 1;
}
GQC_REBASE_HD inline unsigned bank_source_slot(unsigned capacity, unsigned count, unsigned head,
                                               unsigned ordinal) noexcept {
  const unsigned banked = newest_bank_count(capacity, count);
  return chronological_slot(capacity, count, head, count - banked + ordinal);
}
GQC_REBASE_HD inline unsigned restored_slot(unsigned capacity, unsigned banked,
                                            unsigned ordinal) noexcept {
  return capacity - banked + ordinal;
}

/** The snapshot crosses exactly the scheduled fresh target. Its future is
 * decided only by that target's ordinary finish, never by the temporary target
 * phase flag at the preceding handoff. */
inline bool retain_bank_after_finish(bool bank, bool entering_target, bool next_frozen,
                                     bool outer_active, bool active, bool failed,
                                     bool refinement) noexcept {
  return bank && outer_active && active && !failed && !refinement &&
         (entering_target || next_frozen);
}

GQC_REBASE_HD inline bool finite(double value) noexcept {
  return value == value && value <= DBL_MAX && value >= -DBL_MAX;
}
/** Preserve each staged FP64 addition/subtraction, rejecting input, product or
 * result overflow before the separate transaction can become live. */
GQC_REBASE_HD inline bool accumulate_finite(double old, double term, bool subtract,
                                            double& value) noexcept {
  value = subtract ? old - term : old + term;
  return finite(old) && finite(term) && finite(value);
}

/** These scalar decisions are shared by host controls and the CUDA owner.
 * A refusal never changes the incumbent live count/head. */
GQC_REBASE_HD inline void refuse(Transaction& tx, Error reason) noexcept {
  tx.ready = 0;
  tx.error = reason;
  ++tx.work.refusals;
}
GQC_REBASE_HD inline bool prepare_snapshot(Transaction& tx, unsigned capacity, unsigned count,
                                           unsigned head, const SlotTag* tags, Key key) noexcept {
  tx.ready = tx.bank_count = tx.proposed_count = tx.proposed_head = 0;
  tx.target_key = {};
  tx.error = Error::None;
  if (!tx.enabled) return false;
  if (!valid_key(key)) {
    refuse(tx, Error::OwnerMismatch);
    return false;
  }
  if (!valid_window(capacity, count, head) || !count) {
    refuse(tx, Error::InvalidWindow);
    return false;
  }
  const unsigned banked = newest_bank_count(capacity, count);
  for (unsigned i = 0; i < banked; ++i) {
    const auto& tag = tags[bank_source_slot(capacity, count, head, i)];
    if (!tag.valid || !same_key(tag.key, key)) {
      refuse(tx, tag.valid ? Error::AnchorMismatch : Error::MissingDensity);
      return false;
    }
  }
  tx.bank_key = key;
  tx.bank_count = banked;
  tx.ready = 1;
  ++tx.serial;
  return true;
}
GQC_REBASE_HD inline bool prepare_target(Transaction& tx, unsigned capacity, unsigned count,
                                         unsigned head, const SlotTag* tags, Key key) noexcept {
  if (!tx.ready) return false;
  if (!tx.bank_count) {
    refuse(tx, Error::InvalidWindow);
    return false;
  }
  if (!valid_window(capacity, count, head) || count != 1 || head != 1 ||
      tx.bank_count >= capacity || !tags[0].valid || !same_key(tags[0].key, key)) {
    refuse(tx, Error::TargetWindow);
    return false;
  }
  if (tx.bank_key.owner != key.owner || tx.bank_key.solve != key.solve ||
      tx.bank_key.metric != key.metric || !key.anchor) {
    refuse(tx, Error::OwnerMismatch);
    return false;
  }
  tx.target_key = key;
  tx.proposed_count = tx.bank_count + 1;
  tx.proposed_head = 1;
  return true;
}

/** One optional lease. Matrix storage is exactly (4H-2)*N*N doubles. */
struct Layout {
  std::size_t n{}, matrix_elements{}, matrix_bytes{}, total_bytes{};
  unsigned history{};
  std::size_t density{}, bank_density{}, bank_fock{}, bank_residual{}, old_potential{};
  std::size_t density_tags{}, bank_tags{}, transaction{};
};

inline std::optional<Layout> layout(std::size_t n, unsigned history) noexcept {
  if (!n || history < 2 || history > 64) return {};
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (n > maximum / n) return {};
  const auto elements = n * n;
  const std::size_t matrices = 4 * std::size_t{history} - 2;
  if (elements > maximum / sizeof(double) || elements * sizeof(double) > maximum / matrices)
    return {};
  Layout result{};
  result.n = n;
  result.history = history;
  result.matrix_elements = elements;
  const auto one = elements * sizeof(double);
  result.matrix_bytes = matrices * one;
  result.density = 0;
  result.bank_density = history * one;
  result.bank_fock = result.bank_density + (history - 1) * one;
  result.bank_residual = result.bank_fock + (history - 1) * one;
  result.old_potential = result.bank_residual + (history - 1) * one;
  if (result.old_potential + one != result.matrix_bytes) return {};
  std::size_t cursor = result.matrix_bytes;
  const auto reserve = [&](std::size_t bytes, std::size_t& offset) {
    const auto padding =
        (alignof(std::uint64_t) - cursor % alignof(std::uint64_t)) % alignof(std::uint64_t);
    if (cursor > maximum - padding || cursor + padding > maximum - bytes) return false;
    cursor += padding;
    offset = cursor;
    cursor += bytes;
    return true;
  };
  if (!reserve(history * sizeof(SlotTag), result.density_tags) ||
      !reserve((history - 1) * sizeof(SlotTag), result.bank_tags) ||
      !reserve(sizeof(Transaction), result.transaction))
    return {};
  const auto padding = (256 - cursor % 256) % 256;
  if (cursor > maximum - padding) return {};
  result.total_bytes = cursor + padding;
  if (result.total_bytes > kMaximumBytes) return {};
  return result;
}

inline std::size_t optional_state_bytes(Request request, std::size_t n, unsigned spins,
                                        unsigned history, bool incremental) noexcept {
  if (request != Request::On || spins != 1 || incremental) return 0;
  const auto bound = layout(n, history);
  return bound ? bound->total_bytes : 0;
}

enum class Admission {
  Off,
  InvalidRequest,
  Active,
  WarmOrImported,
  Spin,
  Method,
  Precision,
  Topology,
  IncrementalJk,
  History,
  Metric,
  Capacity,
  MissingAnchor
};
struct Context {
  bool cold{}, imported{}, restricted{}, full_direct{}, standard_nonlocal{};
  bool strict_fp64{}, outer_solve{}, ordinary_width_one{}, incremental_jk{};
  bool raw_ao_metric{}, ordered_gram{};
  unsigned history{};
  std::size_t n{}, available_optional_bytes{};
};
inline Admission admit(Request request, const Context& context) noexcept {
  if (request == Request::Invalid) return Admission::InvalidRequest;
  if (request == Request::Off) return Admission::Off;
  if (!context.cold || context.imported) return Admission::WarmOrImported;
  if (!context.restricted) return Admission::Spin;
  if (!context.full_direct || !context.standard_nonlocal) return Admission::Method;
  if (!context.strict_fp64) return Admission::Precision;
  if (!context.outer_solve || !context.ordinary_width_one) return Admission::Topology;
  if (context.incremental_jk) return Admission::IncrementalJk;
  if (context.history < 2 || context.history > 64) return Admission::History;
  if (!context.raw_ao_metric || !context.ordered_gram) return Admission::Metric;
  const auto storage = layout(context.n, context.history);
  if (!storage || storage->total_bytes > context.available_optional_bytes)
    return Admission::Capacity;
  return Admission::Active;
}

}  // namespace generativeqc::dft::vv10_rebase

#undef GQC_REBASE_HD
