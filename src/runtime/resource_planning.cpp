#include "runtime/resource_planning.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::runtime::resources {
namespace {
constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

void checked_value(std::uint64_t value) {
  if (value > maximum) throw std::overflow_error("resource value exceeds signed int64");
}

std::uint64_t checked_sum(std::uint64_t left, std::uint64_t right) {
  if (right > maximum - left)
    throw std::overflow_error("concurrently live resource bytes overflow");
  return left + right;
}

void checked_fraction(double fraction) {
  if (!std::isfinite(fraction) || fraction < 0 || fraction >= 1)
    throw std::invalid_argument("resource headroom must be finite in [0,1)");
}

/** Sums of int64 costs/excesses may exceed int64 without any buffer overflowing.
 * Two limbs cover size_t-counted input rows on every supported ABI; comparing
 * costs never rounds them to floating point or changes lexical ties.
 */
struct WideSum {
  std::uint64_t high{}, low{};
  void add(std::uint64_t value) {
    const auto previous = low;
    low += value;
    if (low < previous && ++high == 0) throw std::overflow_error("resource objective overflows");
  }
  bool operator==(const WideSum&) const = default;
  bool operator<(const WideSum& other) const {
    return high < other.high || (high == other.high && low < other.low);
  }
};

struct Event {
  std::uint64_t phase{}, bytes{};
  std::size_t space{};
  bool add{}, retained{};
};

struct Sweep {
  std::vector<std::uint64_t> live;
  std::vector<Peak> peaks;

  explicit Sweep(std::size_t spaces) : live(spaces), peaks(spaces) {
    for (std::size_t index = 0; index < kFirstDeviceSpace; ++index) peaks[index].present = true;
  }

  void update(const Event& event) {
    // Removals precede all adds at one phase. During adds each touched counter
    // only rises, so its final maximum needs no scan over unrelated devices.
    const auto aggregate = event.space < kFirstDeviceSpace ? kHostSpace : kDeviceSpace;
    for (const auto space : {event.space, aggregate}) {
      if (event.add) {
        live[space] = checked_sum(live[space], event.bytes);
        if (live[space] > peaks[space].bytes) {
          peaks[space].bytes = live[space];
          peaks[space].first_phase = event.phase;
          peaks[space].has_phase = true;
        }
      } else {
        if (live[space] < event.bytes) throw std::logic_error("invalid resource event balance");
        live[space] -= event.bytes;
      }
    }
  }
};

void validate_buffer(const Buffer& buffer, std::size_t spaces) {
  checked_value(buffer.bytes);
  checked_value(buffer.first_phase);
  checked_value(buffer.last_phase);
  if (buffer.first_phase > buffer.last_phase || buffer.space < kPinnedSpace ||
      buffer.space >= spaces)
    throw std::invalid_argument("invalid resource buffer space or lifetime");
}

}  // namespace

std::uint64_t headroom_bytes(std::uint64_t capacity, double fraction) {
  checked_value(capacity);
  checked_fraction(fraction);
  if (fraction == 0 || capacity == 0) return 0;
  int exponent = 0;
  const double normalized = std::frexp(fraction, &exponent);
  const auto numerator = static_cast<std::uint64_t>(std::ldexp(normalized, 53));
  const int shift = 53 - exponent;
  std::uint64_t high = 0, low = 0;
  for (unsigned bit = 0; bit < 53; ++bit) {
    if (!(numerator & (std::uint64_t{1} << bit))) continue;
    const auto previous = low;
    low += capacity << bit;
    high += (bit ? capacity >> (64 - bit) : 0) + (low < previous ? 1 : 0);
  }
  if (shift >= 128) return high || low ? 1 : 0;
  if (shift >= 64) {
    const int upper_shift = shift - 64;
    const auto remainder = upper_shift ? high & ((std::uint64_t{1} << upper_shift) - 1) : 0;
    return (high >> upper_shift) + (remainder || low ? 1 : 0);
  }
  return (low >> shift) + (high << (64 - shift)) +
         ((low & ((std::uint64_t{1} << shift) - 1)) ? 1 : 0);
}

Accounting account(std::span<const Buffer> buffers, std::size_t space_count) {
  if (space_count < kFirstDeviceSpace) throw std::invalid_argument("invalid resource space count");
  Sweep total(space_count), resident(space_count);
  std::vector<Event> events;
  events.reserve(size_mul(2, buffers.size(), "resource event count overflow"));
  for (const auto& buffer : buffers) {
    validate_buffer(buffer, space_count);
    // Portable accounting exposes device labels only for nonzero peaks. A
    // zero-sized placeholder must not lower another candidate's device minimum.
    if (!buffer.bytes) continue;
    total.peaks[buffer.space].present = true;
    if (buffer.retained) resident.peaks[buffer.space].present = true;
    events.push_back({buffer.first_phase, buffer.bytes, buffer.space, true, buffer.retained});
    events.push_back({buffer.last_phase + 1, buffer.bytes, buffer.space, false, buffer.retained});
  }
  std::sort(events.begin(), events.end(), [](const Event& left, const Event& right) {
    return left.phase < right.phase || (left.phase == right.phase && left.add < right.add);
  });
  for (const auto& event : events) {
    total.update(event);
    if (event.retained) resident.update(event);
  }
  return {std::move(total.peaks), std::move(resident.peaks)};
}

std::uint64_t effective_capacity(std::uint64_t capacity, std::uint64_t reserve, double fraction) {
  checked_value(reserve);
  const auto remaining = capacity - headroom_bytes(capacity, fraction);
  return remaining > reserve ? remaining - reserve : 0;
}

Composition compose(std::span<const Request> requests, std::span<const Limit> limits,
                    std::size_t space_count, double headroom_fraction,
                    std::uint64_t maximum_combinations) {
  checked_fraction(headroom_fraction);
  checked_value(maximum_combinations);
  if (requests.empty() || space_count < kFirstDeviceSpace)
    throw std::invalid_argument("resource composition needs requests and memory spaces");
  Composition result;
  result.accounting = account({}, space_count);
  result.limits.resize(space_count);
  result.minima.resize(space_count);
  for (const auto& limit : limits) {
    checked_value(limit.capacity);
    checked_value(limit.reserve);
    if (limit.space >= space_count || result.limits[limit.space])
      throw std::invalid_argument("invalid or duplicate resource limit");
    result.limits[limit.space] =
        effective_capacity(limit.capacity, limit.reserve, headroom_fraction);
  }
  result.combinations = 1;
  for (const auto& request : requests) {
    if (request.candidates.empty()) throw std::invalid_argument("empty resource request");
    std::vector<bool> ranks(request.candidates.size());
    for (const auto& candidate : request.candidates) {
      checked_value(candidate.relative_cost);
      if (candidate.tie_break >= ranks.size() || ranks[candidate.tie_break] ||
          candidate.buffers.empty())
        throw std::invalid_argument("invalid resource candidate rank or buffers");
      ranks[candidate.tie_break] = true;
      for (const auto& buffer : candidate.buffers) validate_buffer(buffer, space_count);
    }
    if (request.candidates.size() > std::numeric_limits<std::uint64_t>::max() / result.combinations)
      result.combinations = std::numeric_limits<std::uint64_t>::max();
    else
      result.combinations *= request.candidates.size();
  }
  if (result.combinations > maximum_combinations) return result;

  std::vector<std::size_t> selected(requests.size());
  WideSum best_excess{}, best_cost{};
  bool has_best = false;
  for (std::uint64_t combination = 0; combination < result.combinations; ++combination) {
    std::vector<Buffer> buffers;
    WideSum cost{}, excess{};
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto& candidate = requests[index].candidates[selected[index]];
      cost.add(candidate.relative_cost);
      buffers.insert(buffers.end(), candidate.buffers.begin(), candidate.buffers.end());
    }
    auto accounting = account(buffers, space_count);
    for (std::size_t index = 0; index < space_count; ++index) {
      const auto peak = accounting.peaks[index].bytes;
      if (accounting.peaks[index].present)
        result.minima[index] = std::min(result.minima[index].value_or(maximum), peak);
      if (result.limits[index] && peak > *result.limits[index])
        excess.add(peak - *result.limits[index]);
    }
    const auto lexical_less = [&]() {
      for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto& candidates = requests[index].candidates;
        const auto current = candidates[selected[index]].tie_break;
        const auto previous = candidates[result.selections[index]].tie_break;
        if (current != previous) return current < previous;
      }
      return false;
    };
    if (!has_best || excess < best_excess ||
        (excess == best_excess && (cost < best_cost || (cost == best_cost && lexical_less())))) {
      has_best = true;
      best_excess = excess;
      best_cost = cost;
      result.selections = selected;
      result.accounting = std::move(accounting);
    }
    for (std::size_t reverse = selected.size(); reverse > 0; --reverse) {
      const auto index = reverse - 1;
      if (++selected[index] < requests[index].candidates.size()) break;
      selected[index] = 0;
    }
  }
  result.status = best_excess == WideSum{} ? Status::feasible : Status::infeasible;
  return result;
}

}  // namespace generativeqc::runtime::resources
