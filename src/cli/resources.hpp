#ifndef GENERATIVEQC_CLI_RESOURCES_HPP
#define GENERATIVEQC_CLI_RESOURCES_HPP

#include <span>

#include "cli/native_basis.hpp"
#include "cli/options.hpp"

namespace generativeqc::cli {
struct HessianInvocation;

struct CpuHessianHostAdmission {
  std::uint64_t minimum_peak_bytes, selected_peak_bytes, effective_limit_bytes;
  std::uint64_t derivative_cap_bytes;
};

struct CpuHostAdmission {
  std::uint64_t peak_bytes, effective_limit_bytes;
};

/** Apply the dry-run inventory before SCF setup. An explicit initial guess can
 * borrow the already normalized CPU system rather than creating it again. */
CpuHostAdmission admit_cpu_host_budget(const RunOptions& options,
                                       std::span<const generativeqc_atom> atoms,
                                       const NativeBasisData& expanded_basis,
                                       Context* context = nullptr, const System* system = nullptr);

/** Verify that an absent automatic execution receipt is a native target-domain
 * decline, not an unexpected loss of the provider diagnostic. */
bool automatic_preliminary_target_declined(const RunOptions& options,
                                           std::span<const generativeqc_atom> atoms,
                                           const NativeBasisData& expanded_basis);

/** Admit a CPU endpoint against the same aggregate inventory as the CLI dry-run.
 * Borrow the basis and derivative floor already authenticated by that endpoint. */
CpuHessianHostAdmission admit_cpu_hessian_host_budget(const RunOptions& options,
                                                      std::span<const generativeqc_atom> atoms,
                                                      const HessianInvocation& derivative,
                                                      const NativeBasisData& expanded_basis,
                                                      std::size_t minimum_bytes);

/** Metadata-only planning; a linked derivative may normalize a CPU system but never prepares
 * a calculation or probes a device. */
int resources_command(const RunOptions& options, std::span<const generativeqc_atom> atoms,
                      const HessianInvocation* derivative = nullptr);
}  // namespace generativeqc::cli
#endif
