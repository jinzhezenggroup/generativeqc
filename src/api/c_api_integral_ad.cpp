/** Private installed binding for the explicit-integral RHF AD primitive.
 * Mathematical integral pullbacks and stationary source weights are generated
 * by the compiler. This adapter owns only validated native SCF execution and
 * a transactionally published final density/energy-weighted density snapshot.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "api/handles.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/mean_field_driver.hpp"

extern "C" GENERATIVEQC_API int generativeqc_integral_ad_rhf_v1(
    const generativeqc_system* system, const double* overlap, const double* hcore,
    const double* eri, std::size_t n, double nuclear, unsigned iterations, unsigned diis_history,
    double energy_tolerance, double density_tolerance, std::size_t budget, double* density,
    double* weighted, double* receipt, char* detail, std::size_t detail_size) {
  try {
    if (!system || !overlap || !hcore || !eri || !density || !weighted || !receipt || !detail ||
        !detail_size || !n || n > 256 || iterations == 0 || iterations > 10000 ||
        diis_history == 0 || diis_history > 32 || !(energy_tolerance > 0) ||
        !std::isfinite(energy_tolerance) || !(density_tolerance > 0) ||
        !std::isfinite(density_tolerance))
      throw std::invalid_argument("invalid integral AD SCF arguments");
    const auto& molecule = system->data;
    if (molecule.electron_count <= 0 || molecule.electron_count % 2 || molecule.multiplicity != 1 ||
        !molecule.ecp_terms.empty() ||
        molecule.basis_representation != generativeqc::core::kCartesianBasis ||
        static_cast<std::size_t>(molecule.electron_count / 2) > n ||
        generativeqc::molecule::ao_count(molecule) != n)
      throw std::invalid_argument("integral AD requires all-electron Cartesian RHF");
    // Includes the caller's dense integrals and this owner's copy, DIIS, final
    // validation, and publication. This bounds numeric arrays, not process RSS.
    if (budget < sizeof(double) * (2 * n * n * n * n + 128 * n * n + 32 * n))
      throw std::invalid_argument("integral AD SCF exceeds numeric byte budget");
    using namespace generativeqc;
    integrals::IntegralData values;
    values.nbf = n;
    values.overlap.assign(overlap, overlap + n * n);
    values.hcore.assign(hcore, hcore + n * n);
    values.eri.assign(eri, eri + n * n * n * n);
    values.nuclear_repulsion = nuclear;
    scf::PreparedFockPlan plan(molecule, std::move(values));
    scf::ScfOptions options;
    options.max_iterations = iterations;
    options.diis_history = diis_history;
    options.energy_tolerance = energy_tolerance;
    options.density_tolerance = density_tolerance;
    options.screening_tolerance = 0;
    options.compute_forces = false;
    options.resolved_fock_build = plan.strategy();
    auto result =
        scf::solver::run_rhf_host_plan(molecule, options, plan.one_electron(), plan, nullptr);
    if (!result.converged) throw std::runtime_error("integral AD SCF did not converge");
    const auto fock =
        scf::assemble_fock(plan.strategy(), plan.one_electron().hcore, plan.build(result.density))
            .alpha;
    const auto residual =
        scf::reference::commutator_residual(fock, result.density, plan.one_electron().overlap, n);
    const double maximum = scf::reference::residual_max_abs(residual);
    if (!std::isfinite(maximum) || maximum > std::max(1e-9, 10 * density_tolerance))
      throw std::runtime_error("integral AD final physical stationarity check failed");
    const auto weight = scf::reference::physical_weighted_density(result.density, fock, n);
    if (!std::isfinite(result.energy) ||
        !std::all_of(weight.begin(), weight.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("integral AD final physical state is nonfinite");
    std::copy(result.density.begin(), result.density.end(), density);
    std::copy(weight.begin(), weight.end(), weighted);
    receipt[0] = result.energy;
    receipt[1] = result.iterations;
    receipt[2] = maximum;
    receipt[3] = result.fock_builds + 1;
    receipt[4] = result.energy_change;
    receipt[5] = result.density_rms;
    return 0;
  } catch (const std::exception& error) {
    if (detail && detail_size) std::snprintf(detail, detail_size, "%s", error.what());
    return 1;
  } catch (...) {
    if (detail && detail_size) std::snprintf(detail, detail_size, "integral AD SCF failed");
    return 1;
  }
}
