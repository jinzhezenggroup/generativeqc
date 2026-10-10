#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::cuda_execution {

enum class DirectForceClassDomain : unsigned {
  LowOrder,
  WeightedFour,
  WeightedFive,
  Cooperative,
  Materialized,
  Fallback,
};

inline constexpr std::uint32_t kDirectForceClassPageSize = 32U;

/** Scalar consumers fill their 128-lane CTA; shared recurrence consumers
 * process a smaller page serially with all 256 coefficient lanes. */
__host__ __device__ inline constexpr std::uint32_t direct_force_class_page_size(
    DirectForceClassDomain domain) {
  return domain == DirectForceClassDomain::Cooperative ||
                 domain == DirectForceClassDomain::Materialized
             ? kDirectForceClassPageSize
             : 128U;
}

/** The six canonical s/p/d pair classes form 21 disjoint quartet classes.
 * This classification is ownership, not screening or an angular-order shortcut:
 * every f-containing pair stays with the unchanged bounded recurrence. */
__host__ __device__ inline constexpr DirectForceClassDomain direct_force_class_domain(
    unsigned first_class, unsigned second_class) {
  if (first_class > 5U || second_class > first_class) return DirectForceClassDomain::Fallback;
  constexpr unsigned orders[]{0U, 1U, 2U, 2U, 3U, 4U};
  const auto order = orders[first_class] + orders[second_class];
  if (order <= 3U) return DirectForceClassDomain::LowOrder;
  if (order == 4U) return DirectForceClassDomain::WeightedFour;
  if (order == 5U) return DirectForceClassDomain::WeightedFive;
  if (order <= 7U) return DirectForceClassDomain::Cooperative;
  return DirectForceClassDomain::Materialized;
}

struct DirectForceClassPagePosition {
  std::uint32_t system{}, first_class{}, second_class{};
  std::uint64_t page_begin{};
};

struct DirectForceClassPage {
  std::uint32_t system, bra, ket_begin, ket_end;
};

/** Number of pages preceding a row of a same-class triangle.
 * Integer arithmetic avoids float rounding at large physical pair inventories. */
__host__ __device__ inline constexpr std::uint64_t direct_force_triangle_page_prefix(
    std::uint64_t rows, std::uint32_t page_size = kDirectForceClassPageSize) {
  const auto groups = rows / page_size;
  const auto remainder = rows % page_size;
  return groups * (groups + 1U) / 2U * page_size + remainder * (groups + 1U);
}

/** Decode increasing CTA claims directly from borrowed class-major offsets.
 * Rectangles visit each cross-class product once; triangles include their
 * diagonal once. Empty systems/classes consume no pages. No resident domain,
 * host inventory readback or extra full shell-pair scan is introduced. */
template <DirectForceClassDomain Domain>
__host__ __device__ inline bool direct_force_class_pair_page(const std::uint32_t* offsets,
                                                             std::uint32_t batch_size,
                                                             std::uint64_t ordinal,
                                                             DirectForceClassPagePosition& position,
                                                             DirectForceClassPage& page) {
  constexpr auto page_size = direct_force_class_page_size(Domain);
  const auto stride = static_cast<std::size_t>(batch_size) + 1U;
  while (position.system < batch_size) {
    if (direct_force_class_domain(position.first_class, position.second_class) == Domain) {
      const auto bra_index = position.first_class * stride + position.system;
      const auto ket_index = position.second_class * stride + position.system;
      const std::uint64_t bra_count = offsets[bra_index + 1U] - offsets[bra_index];
      const std::uint64_t ket_count = offsets[ket_index + 1U] - offsets[ket_index];
      const bool triangle = position.first_class == position.second_class;
      const auto ket_pages = (ket_count + page_size - 1U) / page_size;
      const auto pages = triangle ? direct_force_triangle_page_prefix(bra_count, page_size)
                                  : bra_count * ket_pages;
      const auto local = ordinal - position.page_begin;
      if (local < pages) {
        std::uint64_t bra = 0U, ket = 0U;
        if (triangle) {
          std::uint64_t lower = 0U, upper = bra_count;
          while (lower + 1U < upper) {
            const auto middle = lower + (upper - lower) / 2U;
            if (direct_force_triangle_page_prefix(middle, page_size) <= local)
              lower = middle;
            else
              upper = middle;
          }
          bra = lower;
          ket = (local - direct_force_triangle_page_prefix(bra, page_size)) * page_size;
        } else {
          bra = local / ket_pages;
          ket = (local % ket_pages) * page_size;
        }
        const auto limit = triangle ? bra + 1U : ket_count;
        const auto end = ket + page_size;
        page = {position.system, static_cast<std::uint32_t>(offsets[bra_index] + bra),
                static_cast<std::uint32_t>(offsets[ket_index] + ket),
                static_cast<std::uint32_t>(offsets[ket_index] + (end < limit ? end : limit))};
        return true;
      }
      position.page_begin += pages;
    }
    if (++position.second_class > position.first_class) {
      position.second_class = 0U;
      if (++position.first_class == 6U) {
        position.first_class = 0U;
        ++position.system;
      }
    }
  }
  return false;
}

}  // namespace generativeqc::scf::cuda_execution
