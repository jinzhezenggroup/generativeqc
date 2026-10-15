#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_planning.hpp"

extern "C" {

/** Private metadata-only global composition ABI. Input rows are request
 * candidate counts; candidate (cost, lexical rank, buffer count) triples;
 * buffer (bytes, space, first, last, retained) quintuples; and limit
 * (space, capacity, reserve) triples. Inputs are immutable and disjoint from
 * output until return. No scientific/backend state is created.
 *
 * Output is status, combination count, one chosen index per request, then eight
 * values per space: peak, first-peak phase+1 (zero means no nonzero peak), present,
 * resident peak, resident phase+1, resident present, minimum peak, effective cap.
 * UINT64_MAX marks absent minima/caps or unselected requests. Nonzero return
 * rejects the whole query without touching output: 1 invalid, 2 overflow,
 * 3 metadata allocation failure. Search-limit status is not allocation failure.
 */
int generativeqc_resource_compose_v1(const std::uint64_t* request_counts, std::size_t request_count,
                                     const std::uint64_t* candidate_values,
                                     std::size_t candidate_value_count,
                                     const std::uint64_t* buffer_values,
                                     std::size_t buffer_value_count,
                                     const std::uint64_t* limit_values,
                                     std::size_t limit_value_count, std::size_t space_count,
                                     double headroom_fraction, std::uint64_t maximum_combinations,
                                     std::uint64_t* values, std::size_t value_count) {
  using namespace generativeqc::runtime;
  try {
    if (!request_counts || !request_count || !candidate_values || candidate_value_count % 3 ||
        !buffer_values || buffer_value_count % 5 || limit_value_count % 3 ||
        (!limit_values && limit_value_count) || !values ||
        space_count < resources::kFirstDeviceSpace)
      return 1;
    const auto expected = size_add(size_add(2, request_count), size_mul(8, space_count));
    if (value_count != expected) return 1;
    const auto output_bytes = size_mul(value_count, sizeof(std::uint64_t));
    for (const auto& input :
         {std::pair{request_counts, request_count},
          std::pair{candidate_values, candidate_value_count},
          std::pair{buffer_values, buffer_value_count}, std::pair{limit_values, limit_value_count}})
      if (ranges_overlap(input.first, size_mul(input.second, sizeof(std::uint64_t)), values,
                         output_bytes))
        return 1;

    std::vector<resources::Request> requests;
    std::size_t candidate_cursor = 0, buffer_cursor = 0;
    for (std::size_t request = 0; request < request_count; ++request) {
      const auto count = request_counts[request];
      if (!count || count > (candidate_value_count - candidate_cursor) / 3) return 1;
      resources::Request item;
      for (std::uint64_t candidate = 0; candidate < count; ++candidate) {
        const auto* row = candidate_values + candidate_cursor;
        const auto buffers = row[2];
        if (!buffers || buffers > (buffer_value_count - buffer_cursor) / 5) return 1;
        resources::Candidate choice{row[0], row[1], {}};
        for (std::uint64_t buffer = 0; buffer < buffers; ++buffer) {
          const auto* shape = buffer_values + buffer_cursor;
          if (shape[1] >= space_count || shape[4] > 1) return 1;
          choice.buffers.push_back(
              {shape[0], static_cast<std::size_t>(shape[1]), shape[2], shape[3], bool(shape[4])});
          buffer_cursor += 5;
        }
        item.candidates.push_back(std::move(choice));
        candidate_cursor += 3;
      }
      requests.push_back(std::move(item));
    }
    if (candidate_cursor != candidate_value_count || buffer_cursor != buffer_value_count) return 1;
    std::vector<resources::Limit> limits;
    for (std::size_t index = 0; index < limit_value_count; index += 3) {
      if (limit_values[index] >= space_count) return 1;
      limits.push_back({static_cast<std::size_t>(limit_values[index]), limit_values[index + 1],
                        limit_values[index + 2]});
    }
    const auto result =
        resources::compose(requests, limits, space_count, headroom_fraction, maximum_combinations);
    constexpr auto absent = std::numeric_limits<std::uint64_t>::max();
    values[0] = static_cast<std::uint64_t>(result.status);
    values[1] = result.combinations;
    for (std::size_t index = 0; index < request_count; ++index)
      values[2 + index] = result.selections.empty() ? absent : result.selections[index];
    for (std::size_t index = 0; index < space_count; ++index) {
      auto* row = values + 2 + request_count + 8 * index;
      const auto& peak = result.accounting.peaks[index];
      const auto& resident = result.accounting.resident[index];
      row[0] = peak.bytes;
      row[1] = peak.has_phase ? peak.first_phase + 1 : 0;
      row[2] = peak.present;
      row[3] = resident.bytes;
      row[4] = resident.has_phase ? resident.first_phase + 1 : 0;
      row[5] = resident.present;
      row[6] = result.minima[index].value_or(absent);
      row[7] = result.limits[index].value_or(absent);
    }
    return 0;
  } catch (const std::bad_alloc&) {
    return 3;
  } catch (const std::overflow_error&) {
    return 2;
  } catch (const std::exception&) {
    return 1;
  }
}

}  // extern "C"
