#pragma once

#include <cuda_runtime_api.h>

namespace generativeqc::dft::hessian_aot::contracted {

inline constexpr unsigned active_coordinates = 9;
inline constexpr unsigned hessian_fields = active_coordinates * (active_coordinates + 1) / 2;
// 81 primitive quartets and all Hermite states of total degree <= 6.
inline constexpr int max_quartet_shared_bytes = 81 * 84 * sizeof(double);

/** Caller-owned device slots for one Cartesian s/p pair class. A valid pair
 * has nine primitive pairs in layout order, each with 6 geometry fields,
 * 13 fields per weighted Hermite state, and 4 fields per unweighted
 * component/state. Launches borrow every buffer through stream completion. */
struct PairJets {
  double *geometry{}, *weighted{}, *unweighted{};
};

using PairLaunch = cudaError_t (*)(const double*, unsigned, PairJets, int*, cudaStream_t);
using QuartetLaunch = cudaError_t (*)(PairJets, PairJets, const unsigned*, unsigned, double*, int*,
                                      cudaStream_t);
using QuartetConfigure = cudaError_t (*)();

#define GQC_CONTRACTED_PAIR(L, R) \
  cudaError_t prepare_##L##_##R(const double*, unsigned, PairJets, int*, cudaStream_t) noexcept
GQC_CONTRACTED_PAIR(0, 0);
GQC_CONTRACTED_PAIR(0, 1);
GQC_CONTRACTED_PAIR(1, 0);
GQC_CONTRACTED_PAIR(1, 1);
#undef GQC_CONTRACTED_PAIR

cudaError_t prepare_aux_0_0(const double*, unsigned, PairJets, int*, cudaStream_t) noexcept;
cudaError_t prepare_aux_1_0(const double*, unsigned, PairJets, int*, cudaStream_t) noexcept;

#define GQC_CONTRACTED_QUARTET(L, R)                                                         \
  cudaError_t configure_##L##_##R() noexcept;                                                \
  cudaError_t launch_##L##_##R(PairJets, PairJets, const unsigned*, unsigned, double*, int*, \
                               cudaStream_t) noexcept
GQC_CONTRACTED_QUARTET(0, 0);
GQC_CONTRACTED_QUARTET(1, 0);
GQC_CONTRACTED_QUARTET(1, 1);
GQC_CONTRACTED_QUARTET(2, 0);
GQC_CONTRACTED_QUARTET(2, 1);
GQC_CONTRACTED_QUARTET(2, 2);
#undef GQC_CONTRACTED_QUARTET

#define GQC_CONTRACTED_FITTED(N, L, R)                                                             \
  cudaError_t configure_fitted##N##_##L##_##R() noexcept;                                          \
  cudaError_t launch_fitted##N##_##L##_##R(PairJets, PairJets, const unsigned*, unsigned, double*, \
                                           int*, cudaStream_t) noexcept
GQC_CONTRACTED_FITTED(2, 0, 0);
GQC_CONTRACTED_FITTED(2, 1, 0);
GQC_CONTRACTED_FITTED(2, 1, 1);
GQC_CONTRACTED_FITTED(3, 0, 0);
GQC_CONTRACTED_FITTED(3, 0, 1);
GQC_CONTRACTED_FITTED(3, 1, 0);
GQC_CONTRACTED_FITTED(3, 1, 1);
GQC_CONTRACTED_FITTED(3, 2, 0);
GQC_CONTRACTED_FITTED(3, 2, 1);
#undef GQC_CONTRACTED_FITTED

}  // namespace generativeqc::dft::hessian_aot::contracted
