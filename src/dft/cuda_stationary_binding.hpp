#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "scf/cuda_derivative_sources.hpp"

namespace generativeqc::dft {
struct GridTaskView;

/** Build-time binding to one compiler-owned stationary plan.
 * No runtime compiler, Python module, or unauthenticated artifact lookup is
 * involved. Profile availability is not public force-method admission: the
 * enclosing calculation still authenticates model identity and its final SCF
 * token before source submission and terminal publication. */
struct CudaStationaryBinding {
  const char *profile{}, *plan_identity{}, *method_identity{};
  const char* const* source_names{};
  std::size_t source_count{}, spin_blocks{};
  double full_range_exchange{};
  unsigned nuclear_kind{};
  std::size_t geometry_max_lanes{}, geometry_max_scratch_bytes{}, geometry_max_threads{};
  unsigned ao_jets{}, density_jets{}, feature_mask{};
  std::size_t (*numeric_bytes)(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                               std::size_t, std::size_t, bool){};
  /** Executor statuses: 0 success, 1 invalid/runtime failure, 2 device OOM,
   * 3 host OOM. These are not the public generativeqc_status enumeration. */
  int (*create)(int, int, int, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, void**, char*,
                std::size_t){};
  int (*topology)(void*, const double*, const std::int64_t*, const double*, const std::int64_t*,
                  char*, std::size_t){};
  int (*reset_geometry)(void*, const double*, double, char*, std::size_t){};
  int (*nuclear)(void*, const double*, std::size_t, char*, std::size_t){};
  int (*consume_integrals)(void*, const scf::CudaDerivativeSourceView*, char*, std::size_t){};
  int (*geometry)(void*, const GridTaskView*, const double*, std::size_t, std::size_t,
                  const double*, const double*, std::uint64_t, std::uint64_t, char*, std::size_t){};
  int (*finish_gradient)(void*, std::size_t, double*, std::size_t, char*, std::size_t){};
  int (*source_span)(void*, std::size_t, std::size_t, double*, std::size_t, char*, std::size_t){};
  int (*metrics)(void*, std::uint64_t*, std::size_t){};
  void (*destroy)(void*){};
};

/** Return only statically linked profiles, or nullptr for an unavailable one. */
const CudaStationaryBinding* cuda_stationary_binding(std::string_view profile) noexcept;

}  // namespace generativeqc::dft
