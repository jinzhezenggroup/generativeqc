#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::cuda_execution {

inline constexpr std::uint32_t kDirectOrderSevenKetPageSize = 32U;
inline constexpr std::uint32_t kDirectDdPairClass = 5U;
inline constexpr std::uint32_t kDirectDpPairClass = 4U;

/** A CTA's monotonically increasing claims never revisit an earlier system.
 * The prefix is reconstructed from immutable class offsets without allocating
 * a second resident domain or reading topology back to the host. */
struct DirectOrderSevenPagePosition {
  std::uint32_t system{};
  std::uint64_t page_begin{};
};

/** Ordered pair indices for one dd bra and at most 32 same-system dp kets. */
struct DirectOrderSevenPairPage {
  std::uint32_t system, bra, ket_begin, ket_end;
};

/** Decode monotonically increasing claims over the ragged dd x dp rectangle.
 * Zero-sized classes/systems consume no pages; 64-bit products and offsets
 * preserve large 32-bit pair inventories. Physical canonicalization belongs
 * to the consumer after loading pair_order, never to this address decoder. */
__host__ __device__ inline bool direct_order_seven_pair_page(const std::uint32_t* class_offsets,
                                                             std::uint32_t batch_size,
                                                             std::uint64_t ordinal,
                                                             DirectOrderSevenPagePosition& position,
                                                             DirectOrderSevenPairPage& page) {
  const std::size_t stride = static_cast<std::size_t>(batch_size) + 1U;
  while (position.system < batch_size) {
    const auto bra_index = kDirectDdPairClass * stride + position.system;
    const auto ket_index = kDirectDpPairClass * stride + position.system;
    const std::uint64_t bra_count = class_offsets[bra_index + 1U] - class_offsets[bra_index];
    const std::uint64_t ket_count = class_offsets[ket_index + 1U] - class_offsets[ket_index];
    const std::uint64_t ket_pages =
        (ket_count + kDirectOrderSevenKetPageSize - 1U) / kDirectOrderSevenKetPageSize;
    const std::uint64_t system_pages = bra_count * ket_pages;
    const std::uint64_t local = ordinal - position.page_begin;
    if (local < system_pages) {
      const std::uint64_t ket_begin =
          class_offsets[ket_index] + (local % ket_pages) * kDirectOrderSevenKetPageSize;
      const std::uint64_t ket_end = ket_begin + kDirectOrderSevenKetPageSize;
      page = {
          position.system, static_cast<std::uint32_t>(class_offsets[bra_index] + local / ket_pages),
          static_cast<std::uint32_t>(ket_begin),
          static_cast<std::uint32_t>(
              ket_end < class_offsets[ket_index + 1U] ? ket_end : class_offsets[ket_index + 1U])};
      return true;
    }
    position.page_begin += system_pages;
    ++position.system;
  }
  return false;
}

}  // namespace generativeqc::scf::cuda_execution
