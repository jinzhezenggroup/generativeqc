#ifndef GENERATIVEQC_SCF_PRELIMINARY_GUESS_HPP
#define GENERATIVEQC_SCF_PRELIMINARY_GUESS_HPP

#include <functional>
#include <optional>

#include "scf/types.hpp"

namespace generativeqc::scf {
class PreparedFockPlan;
}

namespace generativeqc::scf::initial_guess {

std::optional<PreliminaryOptions> preliminary_options(
    const generativeqc_initial_guess_options* descriptor);
void validate_preliminary_options(const PreliminaryOptions& options);
/** Conservative numeric payload bound, not process RSS or allocator overhead. */
std::size_t preliminary_numeric_capacity(const core::System& system,
                                         const PreliminaryOptions& options);
void validate_preliminary_target(const core::System& system, const ResolvedFockBuild& strategy,
                                 const ScfOptions& options);

/** Strict same-basis admission. No symmetry/trace repair or metric transport.
 * Ownership moves only after the raw candidate passes the existing ensemble gate. */
std::vector<double> admit_preliminary_density(const PreparedFockPlan& target,
                                              std::vector<double> density);

/** Prepare one bounded cold seed without executing the immutable target. */
std::optional<std::vector<double>> prepare_preliminary_density(
    const PreparedFockPlan& target, const PreliminaryOptions& policy,
    PreliminaryDiagnostic& diagnostic);

using TargetSolve = std::function<ScfResult(const std::vector<double>*)>;
/** The callback executes immutable target equations with fresh iterative state.
 * Existing explicit/imported/retained density is authoritative. A cold seed
 * gets at most one preparation and one core retry. Allocation failures propagate.
 * TargetSolve must disable this policy in its nested call to avoid recursion.
 */
ScfResult run_with_preliminary_guess(const PreparedFockPlan& target, const ScfOptions& options,
                                     const std::vector<double>* initial_density,
                                     std::size_t retained_target_bytes, const TargetSolve& solve);

}  // namespace generativeqc::scf::initial_guess
#endif
