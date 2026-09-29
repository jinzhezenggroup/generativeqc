#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::scf {
struct CudaDirectJkPlan;

/** Persistent source/scratch sizes for the generic direct provider. No molecular
 * four-index ERI or derivative tensor is retained. CUDA module/context and
 * compiler-managed recurrence stack storage are outside these buffer counts.
 */
struct CudaDirectJkDiagnostic {
  std::size_t batch_size{}, nbf{}, coordinates_per_item{};
  std::size_t device_bytes{}, host_bytes{}, host_preparation_bytes{};
  unsigned derivative_order{};
  double screening_tolerance{};
  const char* schedule{"generic-contracted-eri-public-ao"};
};

/** Exact explicit device capacity for homogeneous public-AO items. Atom,
 * shell and primitive counts are totals across the batch. Shape-only query:
 * no host packing, density allocation, CUDA initialization or free-memory probe.
 */
std::size_t cuda_direct_jk_device_bytes(std::size_t batch, std::size_t nao, std::size_t atoms,
                                        std::size_t shells, std::size_t primitives,
                                        unsigned derivative_order);

/** Conservative shape-only capacity for the optional generated shell owner
 * plus the exact bounded generic fallback. A retained derivative order also
 * reserves the full-range shell J/K density/force lease used by stationary
 * consumers. Counts are totals as above.
 */
std::size_t cuda_direct_coulomb_device_bytes(std::size_t batch, std::size_t nao, std::size_t atoms,
                                             std::size_t shells, std::size_t primitives,
                                             unsigned derivative_order = 0);

/** Bind normalized, homogeneous public AO dimensions and coordinate counts.
 * Each item retains its own shell/geometry metadata. A geometry or basis change
 * requires a new plan. A positive budget bounds explicit device allocations;
 * no host integrals are evaluated or uploaded by this preparation.
 */
generativeqc_status create_cuda_direct_jk_plan(
    int device_id, const std::vector<core::System>& systems, unsigned derivative_order,
    double screening_tolerance, std::size_t device_budget_bytes, CudaDirectJkPlan** output,
    CudaDirectJkDiagnostic& diagnostic, std::string& detail);
void destroy_cuda_direct_jk_plan(CudaDirectJkPlan* plan) noexcept;
/** Null handles return an empty diagnostic. */
CudaDirectJkDiagnostic cuda_direct_jk_plan_diagnostic(const CudaDirectJkPlan* plan) noexcept;

/** Return raw unscaled J/K in row-major [item,AO,AO] order. Densities use that
 * same layout; beta is empty for restricted spin. Absent outputs are empty.
 * Only density and requested raw matrices cross the host/device boundary.
 */
generativeqc_status execute_cuda_direct_jk(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                           const std::vector<double>& density,
                                           const std::vector<double>& beta,
                                           std::vector<double>& coulomb,
                                           std::vector<double>& alpha_exchange,
                                           std::vector<double>& beta_exchange, std::string& detail);

/** Fixed-density two-electron gradient [item,coordinate], using the same
 * operators, coefficients and screened quartet set as the raw value provider.
 * Derivatives reuse the existing contracted-ERI Dual evaluator; nuclear,
 * one-electron and Pulay terms belong to the surrounding method.
 */
generativeqc_status execute_cuda_direct_energy_derivative(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const std::vector<double>& density,
    const std::vector<double>& beta, std::vector<double>& derivative, std::string& detail);

/** Item-level counterparts preserve neighboring batch scratch and upload only
 * this item's densities. Returned arrays have no batch dimension. */
generativeqc_status execute_cuda_direct_jk_item(
    CudaDirectJkPlan* plan, std::size_t item, FockBuildSpec spec,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& coulomb, std::vector<double>& alpha_exchange,
    std::vector<double>& beta_exchange, std::string& detail);
generativeqc_status execute_cuda_direct_energy_derivative_item(CudaDirectJkPlan* plan,
                                                               std::size_t item, FockBuildSpec spec,
                                                               const std::vector<double>& density,
                                                               const std::vector<double>& beta,
                                                               std::vector<double>& derivative,
                                                               std::string& detail);

/** Fused RSH fixed-density derivative for one item. Output is source-major:
 * [J, short-range K, long-range K], each with coordinates_per_item values.
 * Short-range derivatives are formed from Full - Long inside one quartet pass.
 */
generativeqc_status execute_cuda_direct_rsh_energy_derivatives_item(
    CudaDirectJkPlan* plan, std::size_t item, FockSpin spin, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient, double omega,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& derivatives, std::string& detail);

/** Single-item fused RSH derivative borrowing already-resident row-major
 * densities on the Direct owner's device. The caller owns both pointers and
 * must keep them live through completion. Qualified native KS final-state
 * publication establishes the physical symmetric-density precondition. */
generativeqc_status execute_cuda_direct_rsh_energy_derivatives_device(
    CudaDirectJkPlan* plan, FockSpin spin, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient, double omega,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail);

/** Full-range fixed-density [J',K'] energy derivatives through the retained
 * shell topology. The resident public-AO density is transformed once and
 * shell Schwarz+density screening drives the bounded production scheduler. */
generativeqc_status execute_cuda_direct_shell_full_range_derivatives_device(
    CudaDirectJkPlan* plan, FockSpin spin, double coulomb_coefficient, double exchange_coefficient,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail);

/** Shell-scheduled RSH fixed-density derivatives [J', SR-K', LR-K'] with
 * explicit omega. The retained owner performs one density transform and three
 * screened shell traversals; no public-AO quartet traversal is used. */
generativeqc_status execute_cuda_direct_shell_rsh_energy_derivatives_device(
    CudaDirectJkPlan* plan, FockSpin spin, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient, double omega,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail);
}  // namespace generativeqc::scf
