#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "response/native_gmres.hpp"

namespace generativeqc::posthf {
class RawSource;
struct BorrowedMOBlock;
}  // namespace generativeqc::posthf
namespace generativeqc::integrals {
class ElectronInteractionSource;
}
namespace generativeqc::hf {
struct PhysicalReference;
}

namespace generativeqc::mp2 {

struct ConventionalForceResult {
  std::vector<double> forces;
  response::GmresResult response;
  double stationarity_residual{};
  std::size_t weighted_eri_shell_tiles{};
  std::size_t derivative_workspace_bytes{};
  std::size_t planned_endpoint_peak_bytes{};
  std::size_t measured_endpoint_peak_bytes{};
};

/** Stationary RHF force replay from an already converged canonical reference.
 * Uses the same bounded analytic derivative consumer as MP2, without another
 * SCF solve or orbital response. Source/reference ownership stays with caller. */
ConventionalForceResult reference_force(const hf::PhysicalReference& reference,
                                        const integrals::ElectronInteractionSource& source,
                                        std::size_t budget_bytes, bool cuda = false,
                                        int device_id = 0);

/** Complete conventional canonical RHF-MP2 analytic force on the CPU.
 *
 * The result is unpublished. The caller remains responsible for transactionally
 * copying it only after the full energy/force endpoint succeeds.
 */
ConventionalForceResult conventional_force_cpu(
    const hf::PhysicalReference& reference, const integrals::ElectronInteractionSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, std::span<const double> ovov = {},
    std::span<const posthf::BorrowedMOBlock> blocks = {});

/** Complete conventional canonical RHF-MP2 analytic force on CUDA.
 * The source may borrow the exact RHF device integrals or generate bounded
 * device tiles. Its owner must survive all response and derivative consumers.
 */
ConventionalForceResult conventional_force_cuda(
    const hf::PhysicalReference& reference, const integrals::ElectronInteractionSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, int device_id,
    std::span<const double> ovov = {}, std::span<const posthf::BorrowedMOBlock> blocks = {});

/** Complete CPU RI-MP2 analytic force for the same fitted RHF Hamiltonian.
 *
 * The response uses the factorized MO provider, the metric reverse delegates
 * to the shared fixed-rank matrix-function VJP, and raw A/M cotangents are
 * consumed by the bounded generated DF derivative path.
 */
ConventionalForceResult density_fitted_force_cpu(
    const hf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold, double metric_relative_threshold,
    double same_space_threshold, const response::GmresOptions& response_options);

}  // namespace generativeqc::mp2
