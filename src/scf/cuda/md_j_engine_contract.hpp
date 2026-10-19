#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "integrals/execution_qualifiers.hpp"

namespace generativeqc::scf::cuda_execution {

inline constexpr unsigned kMdJAngularBuckets = 5;
inline constexpr unsigned kMdJHierarchyLevels = 6;
inline constexpr std::uint64_t kMdJSharedCapacity = 48U << 10;

// Materialize standard-library constants outside HD functions: NVCC treats
// numeric_limits member functions as host-only without relaxed constexpr.
inline constexpr std::uint64_t kMdJIndexMaximum = std::numeric_limits<std::uint64_t>::max();
inline constexpr double kMdJBoundInfinity = std::numeric_limits<double>::infinity();

/** Checked arithmetic has no partially usable result after failure. */
GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_checked_add(std::uint64_t first,
                                                           std::uint64_t second,
                                                           std::uint64_t& result) {
  result = 0;
  if (second > kMdJIndexMaximum - first) return false;
  result = first + second;
  return true;
}

GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_checked_mul(std::uint64_t first,
                                                           std::uint64_t second,
                                                           std::uint64_t& result) {
  result = 0;
  if (first && second > kMdJIndexMaximum / first) return false;
  result = first * second;
  return true;
}

GENERATIVEQC_ERI_SHARED_HD constexpr unsigned md_j_hierarchy_width(unsigned level) {
  return level < kMdJHierarchyLevels ? 1U << level : 0;
}

/** Invalid widths return the sentinel kMdJHierarchyLevels. */
GENERATIVEQC_ERI_SHARED_HD constexpr unsigned md_j_hierarchy_level(unsigned width) {
  for (unsigned level = 0; level < kMdJHierarchyLevels; ++level)
    if (md_j_hierarchy_width(level) == width) return level;
  return kMdJHierarchyLevels;
}

/** All six levels exist, including the level-one leaves and padded tail. The
 * local offset for a width-w group is offsets[log2(w)] + group. Padded leaves
 * contain zero bounds and are never admitted as primitive sources. */
struct MdJHierarchyLayout {
  std::uint64_t count{}, aligned{}, offsets[kMdJHierarchyLevels]{}, entries{};
  bool valid{};
};

GENERATIVEQC_ERI_SHARED_HD constexpr MdJHierarchyLayout md_j_hierarchy_layout(std::uint64_t count) {
  MdJHierarchyLayout layout{};
  std::uint64_t rounded = 0;
  if (!md_j_checked_add(count, 31, rounded)) return {};
  layout.count = count;
  layout.aligned = rounded / 32 * 32;
  for (unsigned level = 0; level < kMdJHierarchyLevels; ++level) {
    layout.offsets[level] = layout.entries;
    const auto previous = layout.entries;
    if (!md_j_checked_add(previous, layout.aligned / md_j_hierarchy_width(level), layout.entries))
      return {};
  }
  layout.valid = true;
  return layout;
}

GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_hierarchy_index(const MdJHierarchyLayout& layout,
                                                               unsigned width, std::uint64_t group,
                                                               std::uint64_t& index) {
  index = 0;
  const unsigned level = md_j_hierarchy_level(width);
  if (!layout.valid || level == kMdJHierarchyLevels || group >= layout.aligned / width)
    return false;
  return md_j_checked_add(layout.offsets[level], group, index) && index < layout.entries;
}

GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_hierarchy_leaf_index(
    const MdJHierarchyLayout& layout, unsigned width, std::uint64_t primitive,
    std::uint64_t& index) {
  index = 0;
  if (!layout.valid || primitive >= layout.count || !width) return false;
  return md_j_hierarchy_index(layout, width, primitive / width, index);
}

/** Linear FP64 hierarchy bounds, never log-compressed public-AO masks. q and
 * minimum are respectively maxima of the public pair's maximum and minimum
 * Schwarz bounds. qd bounds q times that pair's density bound. Internal nodes
 * take maxima of each field; `minimum` is NOT a minimum over the node. These
 * bounds can only reject work conservatively. Density is retained separately
 * so coarse screening preserves the leaf multiplication order (q_out*q_in)*d:
 * qd alone may differ at a rounded threshold or underflow boundary.
 * Exact public-AO/leaf admission
 * remains owned by the source consumer and is not replaced by this hierarchy. */
struct MdJScreenBound {
  double q{}, qd{}, minimum{}, density{};
};

/** Invalid bounds fail open for screening (retain work). A finite zero is
 * exact, including underflow of nonzero factors, which is rounded upward. */
GENERATIVEQC_ERI_SHARED_HD inline double md_j_nonnegative_bound(double value) {
  return value >= 0.0 ? value : kMdJBoundInfinity;
}

GENERATIVEQC_ERI_SHARED_HD inline double md_j_bound_product_up(double first, double second) {
  first = md_j_nonnegative_bound(first);
  second = md_j_nonnegative_bound(second);
  const double infinity = kMdJBoundInfinity;
  // Unknown/infinite bounds must not become zero via 0 * infinity.
  if (first == infinity || second == infinity) return infinity;
  if (first == 0.0 || second == 0.0) return 0.0;
#ifdef __CUDA_ARCH__
  return __dmul_ru(first, second);
#else
  const double product = first * second;
  return product == infinity ? infinity : std::nextafter(product, infinity);
#endif
}

GENERATIVEQC_ERI_SHARED_HD inline MdJScreenBound md_j_screen_bound(double q, double density,
                                                                   double minimum) {
  return {md_j_nonnegative_bound(q), md_j_bound_product_up(q, density),
          md_j_nonnegative_bound(minimum), md_j_nonnegative_bound(density)};
}

GENERATIVEQC_ERI_SHARED_HD inline MdJScreenBound md_j_merge_screen_bounds(
    const MdJScreenBound& first, const MdJScreenBound& second) {
  const auto aq = md_j_nonnegative_bound(first.q);
  const auto bq = md_j_nonnegative_bound(second.q);
  const auto ad = md_j_nonnegative_bound(first.qd);
  const auto bd = md_j_nonnegative_bound(second.qd);
  const auto am = md_j_nonnegative_bound(first.minimum);
  const auto bm = md_j_nonnegative_bound(second.minimum);
  const auto ar = md_j_nonnegative_bound(first.density);
  const auto br = md_j_nonnegative_bound(second.density);
  return {aq > bq ? aq : bq, ad > bd ? ad : bd, am > bm ? am : bm, ar > br ? ar : br};
}

/** Preserve the consumer's left-associated product. Precomputed qd cannot
 * replace this expression at threshold/underflow boundaries. */
GENERATIVEQC_ERI_SHARED_HD inline double md_j_screen_density_bound(const MdJScreenBound& output,
                                                                   const MdJScreenBound& input) {
  return md_j_bound_product_up(md_j_bound_product_up(output.q, input.q), input.density);
}

/** Five independent primitive-angular buckets share one bounded allocation. */
struct MdJHierarchyStorage {
  MdJHierarchyLayout buckets[kMdJAngularBuckets]{};
  std::uint64_t bucket_offsets[kMdJAngularBuckets + 1]{}, entries{}, bytes{};
  bool valid{};
};

GENERATIVEQC_ERI_SHARED_HD constexpr MdJHierarchyStorage md_j_hierarchy_storage(
    const std::uint64_t* counts, std::uint64_t byte_capacity = kMdJIndexMaximum) {
  if (!counts) return {};
  MdJHierarchyStorage storage{};
  for (unsigned bucket = 0; bucket < kMdJAngularBuckets; ++bucket) {
    storage.buckets[bucket] = md_j_hierarchy_layout(counts[bucket]);
    storage.bucket_offsets[bucket] = storage.entries;
    const auto previous = storage.entries;
    if (!storage.buckets[bucket].valid ||
        !md_j_checked_add(previous, storage.buckets[bucket].entries, storage.entries))
      return {};
  }
  storage.bucket_offsets[kMdJAngularBuckets] = storage.entries;
  if (!md_j_checked_mul(storage.entries, sizeof(MdJScreenBound), storage.bytes) ||
      storage.bytes > byte_capacity)
    return {};
  storage.valid = true;
  return storage;
}

GENERATIVEQC_ERI_SHARED_HD constexpr unsigned md_j_contract_hermites(unsigned angular) {
  return angular <= 8 ? (angular + 1) * (angular + 2) * (angular + 3) / 6 : 0;
}

/** Shared workspace contract for the quartet/component thread decomposition.
 * Low-order lanes own independent quartets instead of reserving a whole warp
 * for one scalar radial recurrence. Larger component domains share each source
 * across a power-of-two number of component lanes. The descriptor owner must
 * assert its ABI fits the conservative 96-byte accounting cap. */
inline constexpr unsigned kMdJSourceThreads = 128;
inline constexpr std::uint64_t kMdJPrimitiveDescriptorBytes = 96;
inline constexpr std::uint64_t kMdJSourceControlBytes = 128;
inline constexpr std::uint64_t kMdJSharedOccupancyTarget = 32U << 10;

/** Choose source concurrency from output size and compact radial workspace.
 * These are the upstream single-density register-shape targets (11 outputs,
 * 31 radial states), applied to both independently admitted directions. They
 * provide conservative shape targets, not a register-use guarantee: radial
 * seeds still have one owner per source. They never change scientific screening.
 */
GENERATIVEQC_ERI_SHARED_HD constexpr unsigned md_j_component_lanes(unsigned bra_angular,
                                                                   unsigned ket_angular) {
  const auto bra = md_j_contract_hermites(bra_angular);
  const auto ket = md_j_contract_hermites(ket_angular);
  const auto outputs = bra > ket ? bra : ket;
  const auto radial = md_j_contract_hermites(bra_angular + ket_angular);
  unsigned lanes = 1;
  while (11 * lanes < outputs || 31 * lanes < radial) lanes *= 2;
  // Radial scratch can consume almost the whole shared-memory target before
  // even one full source-lane rectangle fits. In particular (2,3) and (3,4)
  // would otherwise select widths 4 and 1, leaving most quartet lanes idle
  // and multiplying persistent tasks. Share each recurrence over more
  // component lanes until at least one complete rectangle can be retained.
  // This is a utilization invariant, independent of molecular pair counts.
  while (lanes < kMdJSourceThreads) {
    const auto sources = kMdJSourceThreads / lanes;
    unsigned width = 1;
    while (width * width < sources) width *= 2;
    const auto required = sizeof(double) * radial * sources +
                          2 * sizeof(double) * width * (bra + ket) +
                          2 * width * kMdJPrimitiveDescriptorBytes + kMdJSourceControlBytes;
    if (required <= kMdJSharedOccupancyTarget) break;
    lanes *= 2;
  }
  return lanes;
}

GENERATIVEQC_ERI_SHARED_HD constexpr unsigned md_j_source_lanes(unsigned bra_angular,
                                                                unsigned ket_angular) {
  return kMdJSourceThreads / md_j_component_lanes(bra_angular, ket_angular);
}

/** Consecutive bra lanes form shuffle groups within each physical warp. */
GENERATIVEQC_ERI_SHARED_HD constexpr unsigned md_j_bra_lanes(unsigned sources) {
  unsigned lanes = 1;
  while (lanes * lanes < sources) lanes *= 2;
  return lanes;
}

GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_tile_shared_bytes(unsigned bra_angular,
                                                                 unsigned ket_angular,
                                                                 unsigned bra_width,
                                                                 unsigned ket_width,
                                                                 std::uint64_t& bytes) {
  bytes = 0;
  if (bra_angular >= kMdJAngularBuckets || ket_angular >= kMdJAngularBuckets ||
      md_j_hierarchy_level(bra_width) == kMdJHierarchyLevels ||
      md_j_hierarchy_level(ket_width) == kMdJHierarchyLevels)
    return false;
  std::uint64_t radial = 0, bra = 0, ket = 0, vectors = 0, descriptors = 0, total = 0;
  if (!md_j_checked_mul(md_j_source_lanes(bra_angular, ket_angular) * sizeof(double),
                        md_j_contract_hermites(bra_angular + ket_angular), radial) ||
      !md_j_checked_mul(2 * sizeof(double) * bra_width, md_j_contract_hermites(bra_angular), bra) ||
      !md_j_checked_mul(2 * sizeof(double) * ket_width, md_j_contract_hermites(ket_angular), ket) ||
      !md_j_checked_add(bra, ket, vectors) ||
      !md_j_checked_mul(bra_width + ket_width, kMdJPrimitiveDescriptorBytes, descriptors) ||
      !md_j_checked_add(radial, vectors, total))
    return false;
  const auto partial = total;
  if (!md_j_checked_add(partial, descriptors, total)) return false;
  return md_j_checked_add(total, kMdJSourceControlBytes, bytes);
}

struct MdJTileSchedule {
  unsigned bra_width{}, ket_width{}, threads{};
  std::uint64_t shared_bytes{};
  bool valid{};
};

/** Angular/workspace-aware stratum selection, not an upstream thread-tuning
 * claim. Prefer the largest equal power-of-two tiles within 32 KiB to leave
 * occupancy headroom. Smaller supplied capacities are hard limits. All six
 * hierarchy widths are selectable; failure leaves no usable launch shape.
 * This architecture policy is untuned pending real-device endpoint evidence. */
GENERATIVEQC_ERI_SHARED_HD constexpr MdJTileSchedule md_j_tile_schedule(
    unsigned bra_angular, unsigned ket_angular,
    std::uint64_t shared_capacity = kMdJSharedCapacity) {
  const auto target =
      shared_capacity < kMdJSharedOccupancyTarget ? shared_capacity : kMdJSharedOccupancyTarget;
  for (unsigned width = 32; width; width /= 2) {
    std::uint64_t bytes = 0;
    if (md_j_tile_shared_bytes(bra_angular, ket_angular, width, width, bytes) && bytes <= target)
      return {width, width, kMdJSourceThreads, bytes, true};
  }
  return {};
}

/** Divide before multiplying: T(n) remains representable even where n*(n+1)
 * does not. This is the diagonal-inclusive reciprocal inventory. */
GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_triangular_count(std::uint64_t count,
                                                                std::uint64_t& tasks) {
  tasks = 0;
  std::uint64_t next = 0;
  if (!md_j_checked_add(count, 1, next)) return false;
  return count & 1U ? md_j_checked_mul(count, next / 2, tasks)
                    : md_j_checked_mul(count / 2, next, tasks);
}

struct MdJTileInventory {
  std::uint64_t bra_count{}, ket_count{}, bra_tiles{}, ket_tiles{}, tasks{};
  unsigned bra_width{}, ket_width{};
  bool triangular{}, valid{};
};

/** Rectangles retain ordered angular classes; triangles require identical
 * domains and retain only bra_position <= ket_position on diagonal tiles. */
GENERATIVEQC_ERI_SHARED_HD constexpr MdJTileInventory md_j_tile_inventory(std::uint64_t bra_count,
                                                                          std::uint64_t ket_count,
                                                                          unsigned bra_width,
                                                                          unsigned ket_width,
                                                                          bool triangular) {
  if (md_j_hierarchy_level(bra_width) == kMdJHierarchyLevels ||
      md_j_hierarchy_level(ket_width) == kMdJHierarchyLevels ||
      (triangular && (bra_count != ket_count || bra_width != ket_width)))
    return {};
  MdJTileInventory inventory{};
  inventory.bra_count = bra_count;
  inventory.ket_count = ket_count;
  inventory.bra_width = bra_width;
  inventory.ket_width = ket_width;
  inventory.triangular = triangular;
  // The quotient/remainder form cannot overflow on a valid uint64 count.
  inventory.bra_tiles = bra_count / bra_width + (bra_count % bra_width != 0);
  inventory.ket_tiles = ket_count / ket_width + (ket_count % ket_width != 0);
  if (triangular ? !md_j_triangular_count(inventory.bra_tiles, inventory.tasks)
                 : !md_j_checked_mul(inventory.bra_tiles, inventory.ket_tiles, inventory.tasks))
    return {};
  inventory.valid = true;
  return inventory;
}

struct MdJTileTask {
  // Kept trivial for CUDA __shared__ storage; decoders explicitly value-init.
  std::uint64_t bra_tile, ket_tile, bra_begin, ket_begin;
  unsigned bra_count, ket_count;
};

/** Floating sqrt only seeds the row. Integer, overflow-checked triangular
 * comparisons correct both rounding directions, including indices above 2^53.
 * Out-of-inventory cursor values are rejected before division or decoding. */
GENERATIVEQC_ERI_SHARED_HD inline bool md_j_decode_tile(const MdJTileInventory& inventory,
                                                        std::uint64_t index, MdJTileTask& task) {
  task = {};
  if (!inventory.valid || index >= inventory.tasks) return false;
  MdJTileTask decoded{};
  if (inventory.triangular) {
    std::uint64_t row =
        static_cast<std::uint64_t>((::sqrt(8.0 * static_cast<double>(index) + 1.0) - 1.0) * 0.5);
    std::uint64_t start = 0;
    while (!md_j_triangular_count(row, start) || start > index) {
      if (!row) return false;
      --row;
    }
    std::uint64_t next = 0;
    while (md_j_triangular_count(row + 1, next) && next <= index) {
      ++row;
      start = next;
    }
    decoded.bra_tile = index - start;
    decoded.ket_tile = row;
  } else {
    if (!inventory.ket_tiles) return false;
    decoded.bra_tile = index / inventory.ket_tiles;
    decoded.ket_tile = index % inventory.ket_tiles;
  }
  if (decoded.bra_tile >= inventory.bra_tiles || decoded.ket_tile >= inventory.ket_tiles ||
      !md_j_checked_mul(decoded.bra_tile, inventory.bra_width, decoded.bra_begin) ||
      !md_j_checked_mul(decoded.ket_tile, inventory.ket_width, decoded.ket_begin) ||
      decoded.bra_begin >= inventory.bra_count || decoded.ket_begin >= inventory.ket_count)
    return false;
  const auto bra_remaining = inventory.bra_count - decoded.bra_begin;
  const auto ket_remaining = inventory.ket_count - decoded.ket_begin;
  decoded.bra_count = static_cast<unsigned>(
      bra_remaining < inventory.bra_width ? bra_remaining : inventory.bra_width);
  decoded.ket_count = static_cast<unsigned>(
      ket_remaining < inventory.ket_width ? ket_remaining : inventory.ket_width);
  task = decoded;
  return true;
}

/** Diagonal tile padding and the reciprocal half are excluded independently
 * of Schwarz/density admission. Off-diagonal tiles have two distinct owners. */
GENERATIVEQC_ERI_SHARED_HD constexpr bool md_j_tile_source_valid(const MdJTileInventory& inventory,
                                                                 const MdJTileTask& task,
                                                                 unsigned bra_slot,
                                                                 unsigned ket_slot) {
  return inventory.valid && bra_slot < task.bra_count && ket_slot < task.ket_count &&
         (!inventory.triangular || task.bra_tile < task.ket_tile ||
          (task.bra_tile == task.ket_tile && bra_slot <= ket_slot));
}

}  // namespace generativeqc::scf::cuda_execution
