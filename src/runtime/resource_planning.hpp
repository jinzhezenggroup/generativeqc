#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace generativeqc::runtime::resources {

inline constexpr std::size_t kHostSpace = 0;
inline constexpr std::size_t kDeviceSpace = 1;
inline constexpr std::size_t kPinnedSpace = 2;
inline constexpr std::size_t kPageableSpace = 3;
inline constexpr std::size_t kFirstDeviceSpace = 4;

/** Numeric buffer lifetime, inclusive at both ends. Only pinned/pageable and
 * individual device slots own buffers; host/device totals are derived. Device
 * slots are caller-labelled metadata, not probes or allocations on a device.
 */
struct Buffer {
  std::uint64_t bytes{};
  std::size_t space{};
  std::uint64_t first_phase{}, last_phase{};
  bool retained{};
};

/** A supported provider alternative. tie_break is the unique lexical name rank
 * within its request; request order is the caller's lexical owner-name order.
 * Labels stay frontend metadata, while cost and coupled-budget choice are native.
 */
struct Candidate {
  std::uint64_t relative_cost{}, tie_break{};
  std::vector<Buffer> buffers;
};

struct Request {
  std::vector<Candidate> candidates;
};

/** A present cap, including zero. Omitted caps are unlimited. Reserves are
 * withheld after exact IEEE-754 rational headroom, saturating at zero.
 */
struct Limit {
  std::size_t space{};
  std::uint64_t capacity{}, reserve{};
};

struct Peak {
  std::uint64_t bytes{}, first_phase{};
  bool has_phase{}, present{};
};

struct Accounting {
  std::vector<Peak> peaks, resident;
};

enum class Status : std::uint64_t { feasible, infeasible, search_limit };

/** Capacity-only result; no prepared calculation, lease, allocator or backend
 * handle. Minima exclude absent device labels, matching the portable diagnostic
 * contract; absent limits and minima retain explicit optional state.
 */
struct Composition {
  Status status{Status::search_limit};
  std::uint64_t combinations{};
  std::vector<std::size_t> selections;
  Accounting accounting;
  std::vector<std::optional<std::uint64_t>> minima, limits;
};

/** Round up capacity*fraction exactly, including subnormal headroom and int64
 * boundary capacities. Reject nonfinite fractions and out-of-domain integers.
 */
std::uint64_t headroom_bytes(std::uint64_t capacity, double fraction);

/** Effective present cap after exact headroom and fixed reserve. */
std::uint64_t effective_capacity(std::uint64_t capacity, std::uint64_t reserve, double fraction);

/** Sweep sparse endpoints, not phase IDs. Removes at last_phase+1 precede adds
 * at the same endpoint, avoiding spurious transient capacity overflow. Metadata
 * scratch is O(buffers + spaces); numeric/scientific storage is never allocated.
 */
Accounting account(std::span<const Buffer> buffers, std::size_t space_count);

/** Enumerate the finite supported product and minimize (budget excess, cost,
 * lexical selection). All combinations are checked before publication, so a
 * later overflow cannot be hidden by an earlier feasible result. Scratch does
 * not retain the Cartesian product. No alternative is invented on shortage.
 */
Composition compose(std::span<const Request> requests, std::span<const Limit> limits,
                    std::size_t space_count, double headroom_fraction,
                    std::uint64_t maximum_combinations);

}  // namespace generativeqc::runtime::resources
