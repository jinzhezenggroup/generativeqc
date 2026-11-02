#ifndef GENERATIVEQC_SCF_PRELIMINARY_GUESS_HPP
#define GENERATIVEQC_SCF_PRELIMINARY_GUESS_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/types.hpp"

namespace generativeqc::integrals {
struct IntegralData;
}

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
/** Shape-only MINAO capacity query reusing the production numeric bound and
 * native occupied-ANO records; does not allocate a target basis/SCF. */
std::size_t preliminary_minao_numeric_capacity(std::size_t target_aos,
                                               const std::int32_t* atomic_numbers,
                                               std::size_t atom_count);
void validate_preliminary_target(const core::System& system, const ResolvedFockBuild& strategy,
                                 const ScfOptions& options);

/** Resolve AUTO_CPU/AUTO_HUCKEL_CPU to their provider or no policy for this topology.
 * Controls are validated before fallback. Explicit choices remain unchanged.
 * Call before both cold execution and warm-retry diagnostics so unsupported
 * neighbors cannot affect another item's policy or retry/allocation contract.
 */
void resolve_automatic_preliminary_guess(const core::System& system,
                                         const ResolvedFockBuild& strategy, ScfOptions& options);

/** Prepare Huckel against the target's actual overlap without constructing a
 * second molecular Fock owner. The caller validates the target domain and
 * supplies its idle eigen operation; failed supplied operations never retry
 * with a different backend. Preparation is bounded before atomic allocation. */
std::optional<std::vector<double>> prepare_huckel_initial_density(
    const core::System& system, const integrals::IntegralData& target,
    const PreliminaryOptions& policy, PreliminaryDiagnostic& diagnostic,
    const EigenOperation& eigen, const MatrixProductOperation& products = {});

/** Strict same-basis admission. No symmetry/trace repair or metric transport.
 * Ownership moves only after the raw candidate passes the existing ensemble gate. */
std::vector<double> admit_preliminary_density(const PreparedFockPlan& target,
                                              std::vector<double> density,
                                              const EigenOperation& eigen = {});

/** Prepare one bounded cold seed without executing the immutable target.
 * MINAO borrows the target's idle eigen operation for overlap, occupation
 * construction and strict admission. Host-scalar HF supplies its native owner;
 * other callers retain their existing default operation when none is supplied.
 * Huckel/MINAO strict admission may also borrow synchronous square products;
 * the provider-specific density construction remains unchanged. A failing
 * supplied operation never silently retries on the reference backend.
 */
std::optional<std::vector<double>> prepare_preliminary_density(
    const PreparedFockPlan& target, const PreliminaryOptions& policy,
    PreliminaryDiagnostic& diagnostic, const EigenOperation& eigen = {},
    const MatrixProductOperation& products = {});

using TargetSolve = std::function<ScfResult(const std::vector<double>*)>;
/** The callback executes immutable target equations with fresh iterative state.
 * Existing explicit/imported/retained density is authoritative. A cold seed
 * gets at most one preparation and one core retry. Allocation failures propagate.
 * TargetSolve must disable this policy in its nested call to avoid recursion.
 */
ScfResult run_with_preliminary_guess(const PreparedFockPlan& target, const ScfOptions& options,
                                     const std::vector<double>* initial_density,
                                     std::size_t retained_target_bytes, const TargetSolve& solve,
                                     const EigenOperation& eigen = {});

}  // namespace generativeqc::scf::initial_guess
#endif
