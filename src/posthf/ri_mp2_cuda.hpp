#pragma once

#include <cstddef>
#include <cstdint>

#include "hf/reference.hpp"
#include "posthf/raw_source.hpp"
#include "tensor/metrics.hpp"

namespace generativeqc::mp2 {

struct RiMp2CudaBlockPlan {
  std::size_t virtual_block{};
  std::size_t j_batch{};
  std::size_t peak_bytes{};
  bool full_resident{};
};

RiMp2CudaBlockPlan plan_ri_mp2_cuda_blocks(std::size_t fixed_bytes, std::size_t budget_bytes,
                                           std::size_t nbf, std::size_t occupied,
                                           std::size_t virtuals, std::size_t auxiliaries);

struct RiMp2CudaEnergy {
  double opposite_spin{};
  double same_spin{};
  std::size_t numeric_capacity_bytes{};
  std::size_t logical_tiles{};
  std::size_t transfer_bytes{};
  std::size_t virtual_block{};
  std::size_t source_passes{};
  generativeqc_tensor::Metrics metrics;
};

RiMp2CudaEnergy density_fitted_energy_cuda(const hf::PhysicalReference& reference,
                                           const posthf::RawSource& source, std::size_t budget,
                                           double metric_relative_threshold, int device);

}  // namespace generativeqc::mp2
