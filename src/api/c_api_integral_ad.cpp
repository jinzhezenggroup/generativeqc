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

namespace {
using namespace generativeqc;

/** Own exactly one converged determinant and its immutable integral provider.
 * Response actions borrow this state; they never rerun SCF. The owner is
 * synchronous and non-reentrant, including on CUDA's retained provider stream.
 */
struct IntegralHFState {
  using SourceFock = int (*)(const double*, const double*, const double*, const double*, double*);
  struct Source {
    SourceFock fock;
    std::vector<double> parameters;
    std::size_t builds{}, limit;
  };
  std::shared_ptr<Source> source;
  core::System molecule;
  std::size_t n, n2, budget, device_budget{}, numeric{};
  bool unrestricted;
  int device;
  scf::FockBackend selected_backend;
  std::unique_ptr<scf::PreparedFockPlan> plan;
  api::integral_ad::SpinMatrices density, fock, weighted;
  scf::ScfResult result;
  std::array<double, 8> evidence{};

  integrals::IntegralData unpack(const double* data) const {
    integrals::IntegralData ints;
    ints.nbf = n;
    ints.overlap.assign(data, data + n2);
    ints.hcore.assign(data + n2, data + 2 * n2);
    if (!source) ints.eri.assign(data + 2 * n2, data + 2 * n2 + n2 * n2);
    ints.nuclear_repulsion = data[2 * n2 + (source ? 0 : n2 * n2)];
    return ints;
  }

  scf::PreparedFockPlan::ExternalBuilder builder(const double* motion = nullptr) const {
    if (!source) return {};
    std::vector<double> direction;
    if (motion) direction.assign(motion, motion + source->parameters.size());
    return [source = source, direction = std::move(direction), n = n, n2 = n2](
               const scf::FockBuildSpec& spec, const std::vector<double>& da,
               const std::vector<double>& db) {
      const bool spin = spec.spin == scf::FockSpin::Unrestricted;
      if (da.size() != n2 || (spin && db.size() != n2))
        throw std::invalid_argument("streamed integral density shape mismatch");
      if (source->builds >= source->limit)
        throw std::runtime_error("direct integral AD exceeds max_work during Fock/response");
      ++source->builds;
      const std::vector<double> zero(n2, 0);
      std::vector<double> output(3 * n2);
      if (source->fock(source->parameters.data(), direction.empty() ? nullptr : direction.data(),
                       da.data(), spin ? db.data() : zero.data(), output.data()) ||
          !std::all_of(output.begin(), output.end(), [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("generated direct integral Fock action failed");
      scf::DirectJkMatrices result;
      result.nbf = n;
      if (spec.coulomb.present) result.coulomb.assign(output.begin(), output.begin() + n2);
      if (spec.exchange.present) {
        result.exchange_alpha.assign(output.begin() + n2, output.begin() + 2 * n2);
        if (spin) result.exchange_beta.assign(output.begin() + 2 * n2, output.end());
      }
      return result;
    };
  }

  IntegralHFState(const generativeqc_system* system, const double* values, std::size_t n,
                  int unrestricted, int backend, int device, unsigned iterations,
                  unsigned diis_history, double energy_tolerance, double density_tolerance,
                  std::size_t budget, SourceFock source_fock = nullptr,
                  const double* parameters = nullptr, std::size_t parameter_count = 0,
                  std::size_t max_source_builds = 0)
      : n(n), n2(n * n), budget(budget), unrestricted(unrestricted), device(device) {
    if (!system || !values || !n || n > 256 || (unrestricted != 0 && unrestricted != 1) ||
        (backend != 0 && backend != 1) || device < 0 || !iterations || iterations > 10000 ||
        !diis_history || diis_history > 32 || !std::isfinite(energy_tolerance) ||
        energy_tolerance <= 0 || !std::isfinite(density_tolerance) || density_tolerance <= 0)
      throw std::invalid_argument("invalid integral AD HF arguments");
    molecule = system->data;
    const int electrons = molecule.electron_count, spin = molecule.multiplicity - 1;
    if (electrons <= 0 || spin < 0 || spin > electrons || (electrons - spin) % 2 ||
        (!unrestricted && spin != 0) || !molecule.ecp_terms.empty() ||
        molecule.basis_representation != core::kCartesianBasis ||
        molecule::ao_count(molecule) != n || std::size_t((electrons + spin) / 2) > n)
      throw std::invalid_argument("integral AD requires all-electron Cartesian HF occupations");
    if (source_fock) {
      if (backend || !parameters || !parameter_count || parameter_count > 1000000 ||
          !max_source_builds)
        throw std::invalid_argument("invalid direct integral AD provider");
      source = std::make_shared<Source>(
          Source{source_fock, std::vector<double>(parameters, parameters + parameter_count), 0,
                 max_source_builds});
    }
    const auto n4 = n2 * n2;
    // Reserve shared solver workspace separately from all live integral/state
    // copies; a response solve cannot borrow the caller's array allowance.
    device_budget = backend ? sizeof(double) * (n4 + 41 * n2 + n) + (2u << 20) + 5 : 0;
    numeric = sizeof(double) * ((source ? 4 * parameter_count : 4 * n4) + 400 * n2 + 64 * n) +
              3 * device_budget;
    selected_backend = backend ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    if (budget < numeric) throw std::invalid_argument("integral AD HF exceeds numeric byte budget");
    plan = std::make_unique<scf::PreparedFockPlan>(molecule, unpack(values),
                                                   scf::FockSpin::Unrestricted, selected_backend,
                                                   device, device_budget, builder());
    scf::ScfOptions options;
    options.max_iterations = iterations;
    options.diis_history = diis_history;
    options.energy_tolerance = energy_tolerance;
    options.density_tolerance = density_tolerance;
    options.screening_tolerance = 0;
    options.compute_forces = false;
    options.resolved_fock_build = plan->strategy();
    if (unrestricted) {
      result =
          scf::solver::run_uhf_host_plan(molecule, options, plan->one_electron(), *plan, nullptr);
      if (result.density.size() != 2 * n2)
        throw std::runtime_error("integral AD spin density extent mismatch");
      density[0].assign(result.density.begin(), result.density.begin() + n2);
      density[1].assign(result.density.begin() + n2, result.density.end());
    } else {
      scf::PreparedFockPlan restricted_plan(molecule, unpack(values), scf::FockSpin::Restricted,
                                            selected_backend, device, device_budget, builder());
      options.resolved_fock_build = restricted_plan.strategy();
      result = scf::solver::run_rhf_host_plan(molecule, options, restricted_plan.one_electron(),
                                              restricted_plan, nullptr);
      density[0] = result.density;
      for (auto& value : density[0]) value *= 0.5;
      density[1] = density[0];
    }
    if (!result.converged) throw std::runtime_error("integral AD SCF did not converge");
    auto physical = scf::assemble_fock(plan->strategy(), plan->one_electron().hcore,
                                       plan->build(density[0], density[1]));
    fock = {std::move(physical.alpha), std::move(physical.beta)};
    double maximum = 0;
    for (unsigned s = 0; s < 2; ++s) {
      maximum =
          std::max(maximum, scf::reference::residual_max_abs(scf::reference::commutator_residual(
                                fock[s], density[s], plan->one_electron().overlap, n)));
      weighted[s] = scf::reference::physical_weighted_density(density[s], fock[s], n, 1);
    }
    if (!std::isfinite(result.energy) || !std::isfinite(maximum) ||
        maximum > std::max(1e-9, 10 * density_tolerance))
      throw std::runtime_error("integral AD final physical stationarity check failed");
    evidence = {
        result.energy,        double(result.iterations), maximum, double(result.fock_builds + 1),
        result.energy_change, result.density_rms,        0,       0};
    if (source) evidence[3] = double(source->builds);
    check(density);
    check(weighted);
  }

  static void check(const api::integral_ad::SpinMatrices& matrices) {
    for (const auto& matrix : matrices)
      if (!std::all_of(matrix.begin(), matrix.end(), [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("integral AD physical state is nonfinite");
  }

  void publish(double* state, double* receipt) const {
    for (unsigned s = 0; s < 2; ++s) {
      std::copy(density[s].begin(), density[s].end(), state + s * n2);
      std::copy(weighted[s].begin(), weighted[s].end(), state + (2 + s) * n2);
    }
    std::copy(evidence.begin(), evidence.end(), receipt);
  }

  void respond(const double* direction, double* output, double* receipt,
               const double* parameter_motion = nullptr, std::size_t max_source_builds = 0) const {
    if (!direction || !output || !receipt)
      throw std::invalid_argument("invalid retained integral AD response buffers");
    if (source && !parameter_motion)
      throw std::invalid_argument("direct integral response requires parameter motion");
    if (source) {
      // The caller reserves primal/column and pullback work before admitting
      // this response. Enforce the remaining budget inside every Fock action.
      source->builds = 0;
      source->limit = max_source_builds;
    }
    auto counters = evidence;
    counters[1] = counters[3] = 0;  // This action performs no primal SCF iterations.
    scf::PreparedFockPlan direction_plan(molecule, unpack(direction), scf::FockSpin::Unrestricted,
                                         selected_backend, device, device_budget,
                                         builder(parameter_motion));
    api::integral_ad::SpinMatrices dd, dw;
    const int electrons = molecule.electron_count, spin = molecule.multiplicity - 1;
    api::integral_ad::response(
        *plan, direction_plan, density, fock,
        {std::size_t((electrons + spin) / 2), std::size_t((electrons - spin) / 2)}, !unrestricted,
        budget - numeric, dd, dw, counters.data());
    check(dd);
    check(dw);
    for (unsigned s = 0; s < 2; ++s) {
      std::copy(dd[s].begin(), dd[s].end(), output + s * n2);
      std::copy(dw[s].begin(), dw[s].end(), output + (2 + s) * n2);
    }
    std::copy(counters.begin(), counters.end(), receipt);
  }
};

template <class Function>
int integral_ad_status(Function&& function, char* detail, std::size_t detail_size) {
  try {
    if (!detail || !detail_size) throw std::invalid_argument("missing integral AD error buffer");
    function();
    return 0;
  } catch (const std::exception& error) {
    if (detail && detail_size) std::snprintf(detail, detail_size, "%s", error.what());
  } catch (...) {
    if (detail && detail_size) std::snprintf(detail, detail_size, "integral AD HF failed");
  }
  return 1;
}
}  // namespace

/** Prepare an owned, immutable stationary state. No handle or numerical output
 * is published on failure. The caller may release its system/inputs afterwards.
 */
extern "C" GENERATIVEQC_API int generativeqc_integral_ad_hf_prepare_v1(
    const generativeqc_system* system, const double* values, std::size_t n, int unrestricted,
    int backend, int device, unsigned iterations, unsigned diis_history, double energy_tolerance,
    double density_tolerance, std::size_t budget, void** owner, double* state, double* receipt,
    char* detail, std::size_t detail_size) {
  return integral_ad_status(
      [&] {
        if (!owner || !state || !receipt) throw std::invalid_argument("missing integral AD output");
        auto prepared = std::make_unique<IntegralHFState>(
            system, values, n, unrestricted, backend, device, iterations, diis_history,
            energy_tolerance, density_tolerance, budget);
        prepared->publish(state, receipt);
        *owner = prepared.release();
      },
      detail, detail_size);
}

/** CPU streamed source: the compiler callback owns J/K mathematics, while
 * this owner retains the copied parameter identity and enforces source work. */
extern "C" GENERATIVEQC_API int generativeqc_integral_ad_direct_prepare_v1(
    const generativeqc_system* system, const double* values, std::size_t n, int unrestricted,
    int backend, int device, unsigned iterations, unsigned diis_history, double energy_tolerance,
    double density_tolerance, std::size_t budget, IntegralHFState::SourceFock fock,
    const double* parameters, std::size_t parameter_count, std::size_t max_source_builds,
    void** owner, double* state, double* receipt, char* detail, std::size_t detail_size) {
  return integral_ad_status(
      [&] {
        if (!owner || !state || !receipt) throw std::invalid_argument("missing integral AD output");
        auto prepared = std::make_unique<IntegralHFState>(
            system, values, n, unrestricted, backend, device, iterations, diis_history,
            energy_tolerance, density_tolerance, budget, fock, parameters, parameter_count,
            max_source_builds);
        prepared->publish(state, receipt);
        *owner = prepared.release();
      },
      detail, detail_size);
}

extern "C" GENERATIVEQC_API int generativeqc_integral_ad_direct_response_v1(
    const void* owner, const double* direction, const double* parameter_motion,
    std::size_t max_source_builds, double* state_direction, double* receipt, char* detail,
    std::size_t detail_size) {
  return integral_ad_status(
      [&] {
        if (!owner) throw std::invalid_argument("missing integral AD stationary owner");
        static_cast<const IntegralHFState*>(owner)->respond(direction, state_direction, receipt,
                                                            parameter_motion, max_source_builds);
      },
      detail, detail_size);
}

/** A response is transactional and may be retried after an invalid direction. */
extern "C" GENERATIVEQC_API int generativeqc_integral_ad_hf_response_v1(
    const void* owner, const double* direction, double* state_direction, double* receipt,
    char* detail, std::size_t detail_size) {
  return integral_ad_status(
      [&] {
        if (!owner) throw std::invalid_argument("missing integral AD stationary owner");
        static_cast<const IntegralHFState*>(owner)->respond(direction, state_direction, receipt);
      },
      detail, detail_size);
}

extern "C" GENERATIVEQC_API void generativeqc_integral_ad_hf_destroy_v1(void* owner) {
  delete static_cast<IntegralHFState*>(owner);
}

/** Compatibility execution keeps the v2 ABI and its complete-call work counts. */
extern "C" GENERATIVEQC_API int generativeqc_integral_ad_hf_v2(
    const generativeqc_system* system, const double* values, const double* direction, std::size_t n,
    int unrestricted, int backend, int device, unsigned iterations, unsigned diis_history,
    double energy_tolerance, double density_tolerance, std::size_t budget, double* state,
    double* state_direction, double* receipt, char* detail, std::size_t detail_size) {
  return integral_ad_status(
      [&] {
        if (!state || !receipt || (direction && !state_direction))
          throw std::invalid_argument("missing integral AD output");
        IntegralHFState prepared(system, values, n, unrestricted, backend, device, iterations,
                                 diis_history, energy_tolerance, density_tolerance, budget);
        std::vector<double> motion(4 * n * n);
        auto evidence = prepared.evidence;
        if (direction) {
          prepared.respond(direction, motion.data(), evidence.data());
          evidence[1] += prepared.evidence[1];
          evidence[3] += prepared.evidence[3];
        }
        prepared.publish(state, receipt);
        if (direction) std::copy(motion.begin(), motion.end(), state_direction);
        std::copy(evidence.begin(), evidence.end(), receipt);
      },
      detail, detail_size);
}
