/** Private installed binding for the explicit-integral HF AD primitive.
 * Mathematical integral pullbacks and stationary source weights are generated
 * by the compiler. This adapter owns only validated native SCF execution and
 * a transactionally published final density/energy-weighted density snapshot.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "api/handles.hpp"
#include "api/integral_ad_response.hpp"
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

/** Versioned explicit-integral HF boundary. State arrays are [Da,Db,Wa,Wb]
 * with unit occupation in each spin; optional directions have the same layout.
 * The caller's ordered source pack is [S,h,ERI,Enuc]. Outputs commit only after
 * the primal and (when requested) true-residual response gates both pass. */
extern "C" GENERATIVEQC_API int generativeqc_integral_ad_hf_v2(
    const generativeqc_system* system, const double* values, const double* direction, std::size_t n,
    int unrestricted, int backend, int device, unsigned iterations, unsigned diis_history,
    double energy_tolerance, double density_tolerance, std::size_t budget, double* state,
    double* state_direction, double* receipt, char* detail, std::size_t detail_size) {
  try {
    using namespace generativeqc;
    if (!system || !values || !state || !receipt || !detail || !detail_size || !n || n > 256 ||
        (unrestricted != 0 && unrestricted != 1) || (backend != 0 && backend != 1) || device < 0 ||
        !iterations || iterations > 10000 || !diis_history || diis_history > 32 ||
        !std::isfinite(energy_tolerance) || energy_tolerance <= 0 ||
        !std::isfinite(density_tolerance) || density_tolerance <= 0 ||
        (direction && !state_direction))
      throw std::invalid_argument("invalid integral AD HF arguments");
    const auto& molecule = system->data;
    const int electrons = molecule.electron_count, spin = molecule.multiplicity - 1;
    if (electrons <= 0 || spin < 0 || spin > electrons || (electrons - spin) % 2 ||
        (!unrestricted && spin != 0) || !molecule.ecp_terms.empty() ||
        molecule.basis_representation != core::kCartesianBasis ||
        molecule::ao_count(molecule) != n || std::size_t((electrons + spin) / 2) > n)
      throw std::invalid_argument("integral AD requires all-electron Cartesian HF occupations");
    const auto n2 = n * n, n4 = n2 * n2;
    // Reserve shared solver workspace separately from all live integral/state
    // copies; a response solve cannot borrow the caller's array allowance.
    const auto device_budget = backend ? sizeof(double) * (n4 + 41 * n2 + n) + (2u << 20) + 5 : 0;
    const auto numeric = sizeof(double) * (4 * n4 + 400 * n2 + 64 * n) + 3 * device_budget;
    const auto selected_backend = backend ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    if (budget < numeric) throw std::invalid_argument("integral AD HF exceeds numeric byte budget");
    const auto unpack = [&](const double* data) {
      integrals::IntegralData ints;
      ints.nbf = n;
      ints.overlap.assign(data, data + n2);
      ints.hcore.assign(data + n2, data + 2 * n2);
      ints.eri.assign(data + 2 * n2, data + 2 * n2 + n4);
      ints.nuclear_repulsion = data[2 * n2 + n4];
      return ints;
    };
    scf::PreparedFockPlan plan(molecule, unpack(values), scf::FockSpin::Unrestricted,
                               selected_backend, device, device_budget);
    scf::ScfOptions options;
    options.max_iterations = iterations;
    options.diis_history = diis_history;
    options.energy_tolerance = energy_tolerance;
    options.density_tolerance = density_tolerance;
    options.screening_tolerance = 0;
    options.compute_forces = false;
    options.resolved_fock_build = plan.strategy();
    api::integral_ad::SpinMatrices density, fock, weighted, dd, dw;
    scf::ScfResult result;
    if (unrestricted) {
      result =
          scf::solver::run_uhf_host_plan(molecule, options, plan.one_electron(), plan, nullptr);
      if (result.density.size() != 2 * n2)
        throw std::runtime_error("integral AD spin density extent mismatch");
      density[0].assign(result.density.begin(), result.density.begin() + n2);
      density[1].assign(result.density.begin() + n2, result.density.end());
    } else {
      scf::PreparedFockPlan restricted_plan(molecule, unpack(values), scf::FockSpin::Restricted,
                                            selected_backend, device, device_budget);
      options.resolved_fock_build = restricted_plan.strategy();
      result = scf::solver::run_rhf_host_plan(molecule, options, restricted_plan.one_electron(),
                                              restricted_plan, nullptr);
      density[0] = result.density;
      for (auto& value : density[0]) value *= 0.5;
      density[1] = density[0];
    }
    if (!result.converged) throw std::runtime_error("integral AD SCF did not converge");
    auto physical = scf::assemble_fock(plan.strategy(), plan.one_electron().hcore,
                                       plan.build(density[0], density[1]));
    fock = {std::move(physical.alpha), std::move(physical.beta)};
    double maximum = 0;
    for (unsigned s = 0; s < 2; ++s) {
      maximum =
          std::max(maximum, scf::reference::residual_max_abs(scf::reference::commutator_residual(
                                fock[s], density[s], plan.one_electron().overlap, n)));
      weighted[s] = scf::reference::physical_weighted_density(density[s], fock[s], n, 1);
    }
    if (!std::isfinite(result.energy) || !std::isfinite(maximum) ||
        maximum > std::max(1e-9, 10 * density_tolerance))
      throw std::runtime_error("integral AD final physical stationarity check failed");
    double evidence[8] = {
        result.energy,        double(result.iterations), maximum, double(result.fock_builds + 1),
        result.energy_change, result.density_rms,        0,       0};
    if (direction) {
      scf::PreparedFockPlan direction_plan(molecule, unpack(direction), scf::FockSpin::Unrestricted,
                                           selected_backend, device, device_budget);
      api::integral_ad::response(
          plan, direction_plan, density, fock,
          {std::size_t((electrons + spin) / 2), std::size_t((electrons - spin) / 2)}, !unrestricted,
          budget - numeric, dd, dw, evidence);
    }
    const auto check = [](const auto& matrices) {
      for (const auto& matrix : matrices)
        if (!std::all_of(matrix.begin(), matrix.end(), [](double v) { return std::isfinite(v); }))
          throw std::runtime_error("integral AD physical state is nonfinite");
    };
    check(density);
    check(weighted);
    check(dd);
    check(dw);
    for (unsigned s = 0; s < 2; ++s) {
      std::copy(density[s].begin(), density[s].end(), state + s * n2);
      std::copy(weighted[s].begin(), weighted[s].end(), state + (2 + s) * n2);
      if (direction) {
        std::copy(dd[s].begin(), dd[s].end(), state_direction + s * n2);
        std::copy(dw[s].begin(), dw[s].end(), state_direction + (2 + s) * n2);
      }
    }
    std::copy(evidence, evidence + 8, receipt);
    return 0;
  } catch (const std::exception& error) {
    if (detail && detail_size) std::snprintf(detail, detail_size, "%s", error.what());
    return 1;
  } catch (...) {
    if (detail && detail_size) std::snprintf(detail, detail_size, "integral AD HF failed");
    return 1;
  }
}
