#ifndef GENERATIVEQC_XTB_MODEL_COMMON_INTEGRALS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GENERATIVEQC_XTB_MODEL_COMMON_INTEGRALS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/common/basis.hpp"
#include "runtime/cpu_isa.hpp"
#include "runtime/types.hpp"

namespace generativeqc::xtb::detail::common {

inline constexpr std::size_t kIntegralMaximumCartesianFunctions = 6u;
inline constexpr std::size_t kIntegralMaximumSphericalFunctions = 5u;
inline constexpr std::size_t kIntegralMaximumCartesianBlock =
    kIntegralMaximumCartesianFunctions * kIntegralMaximumCartesianFunctions;
inline constexpr std::size_t kIntegralMaximumSphericalBlock =
    kIntegralMaximumSphericalFunctions * kIntegralMaximumSphericalFunctions;
inline constexpr std::size_t kIntegralDipoleComponents = 3u;
inline constexpr std::size_t kIntegralQuadrupoleComponents = 6u;
inline constexpr std::size_t kIntegralMultipoleComponents =
    kIntegralDipoleComponents + kIntegralQuadrupoleComponents;

/*
 * The fixed-size shell-pair scratch is shared by the molecular evaluator and
 * the periodic image evaluator.  Keeping the layout in the internal header
 * lets the latter reuse exactly the same recurrence and spherical transform
 * without changing the molecular loop or adding per-image allocations.
 */
struct alignas(double) IntegralWorkspace {
  std::array<double, kIntegralMaximumCartesianBlock> cartesian;
  std::array<double, 3u * kIntegralMaximumCartesianBlock> cartesian_gradient;
  std::array<double, kIntegralMaximumSphericalBlock> spherical;
  std::array<double, 3u * kIntegralMaximumSphericalBlock> spherical_gradient;
  /* Components are [x,y,z,xx,xy,yy,xz,yz,zz], one shell block each. */
  std::array<double, kIntegralMultipoleComponents * kIntegralMaximumCartesianBlock>
      cartesian_multipole;
  std::array<double, kIntegralMultipoleComponents * kIntegralMaximumSphericalBlock>
      spherical_multipole;
  /* Derivative layout is [coordinate][component][shell-block element]. */
  std::array<double, 3u * kIntegralMultipoleComponents * kIntegralMaximumCartesianBlock>
      cartesian_multipole_gradient;
  std::array<double, 3u * kIntegralMultipoleComponents * kIntegralMaximumSphericalBlock>
      spherical_multipole_gradient;
};

/* Geometry-independent packed layout for ragged one-electron matrices. */
struct IntegralPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  double integral_cutoff = 0.0;
  std::size_t workspace_size_bytes = 0;
  std::vector<std::int64_t> matrix_offsets;
};

/* tblite's default dimensionless Gaussian-product cutoff at accuracy 1.0. */
inline constexpr double kDefaultIntegralCutoff = 25.0;

/*
 * Context-selected implementation of the multipole-gradient shell-pair
 * arithmetic. The opaque workspace points to the evaluator's preallocated
 * integral scratch buffer; keeping its private layout out of this header gives
 * each ISA variant a narrow, allocation-free boundary without expanding the
 * model interface.
 */
using MultipoleGradientShellPairKernel = void (*)(const BasisPlan&, std::size_t, std::size_t,
                                                  const double*, double, void*) noexcept;

struct IntegralKernelTable {
  MultipoleGradientShellPairKernel multipole_gradient_shell_pair = nullptr;
  generativeqc::runtime::CpuIsa isa = generativeqc::runtime::CpuIsa::kBaseline;
};

[[nodiscard]] const IntegralKernelTable& integral_baseline_kernels() noexcept;
[[nodiscard]] const IntegralKernelTable& integral_avx2_fma_kernels() noexcept;
[[nodiscard]] const IntegralKernelTable& integral_kernels_for_cpu_isa(
    generativeqc::runtime::CpuIsa isa) noexcept;

generativeqc_xtb_status_t make_integral_plan(const BasisPlan& basis, IntegralPlan& plan, std::string& error,
                                    double integral_cutoff = kDefaultIntegralCutoff);

generativeqc_xtb_status_t evaluate_overlap_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                      const double* positions, double* overlap, void* workspace,
                                      std::size_t workspace_size, std::string& error);

generativeqc_xtb_status_t evaluate_multipole_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                        const double* positions, double* dipole, double* quadrupole,
                                        void* workspace, std::size_t workspace_size,
                                        std::string& error);

generativeqc_xtb_status_t add_multipole_gradient_cpu(
    const BasisPlan& basis, const IntegralPlan& plan, const double* positions,
    const double* dE_ddipole, const double* dE_dquadrupole, double* gradients, void* workspace,
    std::size_t workspace_size, std::string& error,
    generativeqc::runtime::CpuIsa cpu_isa = generativeqc::runtime::CpuIsa::kBaseline);

generativeqc_xtb_status_t add_overlap_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                          const double* positions, const double* dE_doverlap,
                                          double* gradients, void* workspace,
                                          std::size_t workspace_size, std::string& error);

}  // namespace generativeqc::xtb::detail::common

#endif  // GENERATIVEQC_XTB_MODEL_COMMON_INTEGRALS_HPP
