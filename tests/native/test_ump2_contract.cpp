#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/handles.hpp"
#include "cc/uccsd_solver.hpp"
#include "generativeqc/generativeqc.hpp"
#include "methods/ump2_method.hpp"
#include "methods/unrestricted_df.hpp"
#include "methods/unrestricted_response.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/df_interaction_source.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "posthf/ump2_cpu_generated.hpp"
#include "posthf/ump2_energy.hpp"
#include "posthf/ump2_force.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace {

void require(bool valid, const char* detail) {
  if (!valid) throw std::runtime_error(detail);
}

generativeqc_method_descriptor ump2_descriptor() {
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = GENERATIVEQC_METHOD_UMP2;
  return descriptor;
}

generativeqc::core::System h2() {
  generativeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<generativeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "H2 setup failed");
  return system;
}

generativeqc::core::System h14() {
  generativeqc::core::System system;
  const std::vector<generativeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  for (unsigned atom = 0; atom < 14; ++atom) {
    system.atoms.push_back({1, {3.0 * atom, 0, 0}});
    system.shells.push_back({atom, 0, primitives});
  }
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "14-H resource fixture setup failed");
  return system;
}

void energy_only_reference_admission() {
  const auto system = h14();
  const auto spec = generativeqc::methods::detail::UnrestrictedFitting{}.reference_spec(false);
  const auto strategy =
      generativeqc::scf::resolve_fock_build(spec, generativeqc::scf::FockBackend::Cpu, 0);
  const generativeqc::scf::PreparedFockPlan plan(system, nullptr, strategy);
  const auto budget = generativeqc::posthf::uhf_reference_capacity(system, 8, true);
  if (plan.cpu_observation_capacity() > budget)
    throw std::runtime_error("UMP2 energy-only Fock plan exceeds admitted reference budget: " +
                             std::to_string(plan.cpu_observation_capacity()) + " > " +
                             std::to_string(budget));
  std::cout << "14-H energy-only Fock capacity " << plan.cpu_observation_capacity()
            << " <= admitted " << budget << '\n';
  require(spec.derivative_order == 0, "UMP2 energy-only reference retained derivative order");
  for (const bool full_reference : {false, true}) {
    generativeqc::methods::detail::UnrestrictedFitting fitting;
    fitting.full_reference = full_reference;
    require(fitting.reference_spec(false).derivative_order == 0,
            "energy-only fitted policy retained derivative resources");
    require(fitting.reference_spec(true).derivative_order == 1,
            "force fitted policy lost derivative resources");
  }
}

void one_electron_response_admission() {
  auto system = h2();
  system.shells.resize(1024, system.shells.front());
  const auto basis_count = generativeqc::molecule::ao_count(system);
  const auto two_weights = 2 * basis_count * basis_count * sizeof(double);
  require(two_weights > generativeqc::posthf::source_scratch_bytes,
          "large response fixture does not exceed the old fixed scratch allowance");
  require(generativeqc::posthf::one_electron_response_capacity(system) > two_weights,
          "one-electron response admission omits the two device weight matrices");
  auto ecp_system = h2();
  const auto ordinary = generativeqc::posthf::one_electron_response_capacity(ecp_system);
  ecp_system.ecp_terms.reserve(32);
  ecp_system.ecp_terms.push_back({0, -1, 2, 0.5, 0.2});
  const auto ecp_workspace = generativeqc::integrals::ecp_workspace_capacity(ecp_system);
  const auto metadata = 2 * sizeof(generativeqc::core::EcpTerm) * ecp_system.ecp_terms.capacity();
  require(ecp_workspace > 0 && generativeqc::posthf::one_electron_response_capacity(ecp_system) ==
                                   ordinary + metadata + ecp_workspace,
          "one-electron response admission omits ECP quadrature or retained metadata");
}

// Query a synthetic topology only: no AO tensor or physical endpoint is evaluated.
void full_fitted_reference_admission() {
  auto system = h2();
  system.shells.resize(1024, system.shells.front());
  generativeqc::methods::detail::UnrestrictedFitting full;
  full.full_reference = true;
  full.auxiliary = h2();
  full.budget = 4ULL << 30;
  const auto fitted_capacity = full.reference_capacity(system, 8);
  const auto conventional_capacity =
      generativeqc::methods::detail::UnrestrictedFitting{}.reference_capacity(system, 8);
  require(fitted_capacity < full.budget,
          "full DF reference reserves a nonexistent dense four-index tensor");
  require(conventional_capacity > full.budget,
          "conventional reference lost its dense ERI reservation");
  auto correlation_only = full;
  correlation_only.full_reference = false;
  require(correlation_only.reference_capacity(system, 8) == conventional_capacity,
          "correlation-only fitting changed conventional reference admission");
  full.budget = 1024;
  bool rejected = false;
  try {
    full.reference_capacity(system, 8);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "full DF reference stopped enforcing fitted storage admission");
}

class CountingSource final : public generativeqc::integrals::ElectronInteractionSource {
 public:
  explicit CountingSource(const generativeqc::core::System& system) : raw_(system) {}
  const generativeqc::core::System& orbital() const override { return raw_.orbital(); }
  std::size_t nbf() const override { return raw_.nbf(); }
  std::size_t naux() const override { return raw_.naux(); }
  std::size_t retained_numeric_bytes() const override { return raw_.retained_numeric_bytes(); }
  bool supports(Operator op) const noexcept override { return raw_.supports(op); }
  void read(Operator op, const std::array<std::size_t, 4>& begin,
            const std::array<std::size_t, 4>& count, double* output,
            std::size_t elements) const override {
    ++reads;
    if (fail) throw std::runtime_error("injected source failure");
    raw_.read(op, begin, count, output, elements);
  }
  mutable std::size_t reads{};
  bool fail{};

 private:
  generativeqc::posthf::RawSource raw_;
};

void cuda_channel_admission() {
  auto system = h2();
  system.shells.resize(64, system.shells.front());
  generativeqc::hf::UnrestrictedPhysicalReference reference;
  reference.nbf = 64;
  reference.overlap.resize(64 * 64);
  reference.hcore.resize(64 * 64);
  for (unsigned spin = 0; spin < 2; ++spin) {
    reference.coefficients[spin].resize(64 * 64);
    reference.fock[spin].resize(64 * 64);
    reference.density[spin].resize(64 * 64);
    reference.orbital_energies[spin].resize(64);
  }
  reference.source_identity.emplace(system);
  reference.source_charge = system.charge;
  reference.source_electrons = system.electron_count;
  reference.source_multiplicity = system.multiplicity;
  CountingSource source(system);
  for (const auto occupations :
       std::array<std::array<std::size_t, 2>, 3>{{{1, 1}, {1, 0}, {2, 1}}}) {
    reference.nocc = occupations;
    generativeqc::posthf::NativeBlockProvider provider(source, reference, 1ULL << 30, 8,
                                                       generativeqc::posthf::AOTileDomain::Basis);
    auto expected = provider.batch_bytes({1, 1, 1, 1}, 1, true);
    for (const auto spins : std::array<std::array<unsigned, 2>, 3>{{{0, 0}, {1, 1}, {0, 1}}}) {
      const auto occupied_left = occupations[spins[0]], occupied_right = occupations[spins[1]];
      if (!occupied_left || !occupied_right) continue;
      const std::array<std::size_t, 4> shape{occupied_left, 64 - occupied_left, occupied_right,
                                             64 - occupied_right};
      const auto count = shape[0] * shape[1] * shape[2] * shape[3];
      const auto feeds = (spins[0] == spins[1] ? 2 : 1) * count + 2 * 64 + 1;
      expected = std::max(expected, provider.batch_bytes(shape, 1, true) + sizeof(double) * feeds);
    }
    const auto actual =
        generativeqc::mp2::unrestricted_cuda_energy_host_capacity(reference, provider);
    require(actual == expected, "CUDA UMP2 admission does not match active channel payloads");
    bool dense_rejected = false;
    try {
      (void)provider.batch_bytes({64, 64, 64, 64}, 1, true);
    } catch (const std::length_error&) {
      dense_rejected = true;
    }
    require(dense_rejected,
            "synthetic budget does not distinguish active channels from a dense molecular block");
  }
  for (const auto occupations : std::array<std::array<std::size_t, 2>, 2>{{{32, 32}, {31, 33}}}) {
    reference.nocc = occupations;
    generativeqc::posthf::NativeBlockProvider provider(source, reference, 1ULL << 30, 8,
                                                       generativeqc::posthf::AOTileDomain::Basis);
    const auto maximum = std::max(occupations[0], occupations[1]);
    const auto expected = provider.batch_bytes({maximum, maximum, maximum, maximum}, 1, true);
    require(generativeqc::cc::unrestricted_cuda_solver_provider_capacity(reference, provider) ==
                expected,
            "CUDA UCCSD admission charges an unexecuted full molecular MO block");
  }
  require(source.reads == 0, "shape-only CUDA admission read AO integrals");
}

void native_equations() {
  constexpr double g = 0.7, x = 0.2, ei = -1.1, ej = -0.8, ea = 0.3, eb = 0.5;
  const double denominator = ei + ej - ea - eb;
  for (unsigned channel = 0; channel < 3; ++channel) {
    const auto program = generativeqc::mp2::generated::ump2_cpu_plan(channel);
    double value{};
    program.run(g, x, ei, ej, ea, eb, &value);
    const double expected =
        channel == 2 ? g * g / denominator : 0.25 * (g - x) * (g - x) / denominator;
    require(std::abs(value - expected) < 1e-14, "native spin TensorIR factor mismatch");
    std::array<double, 6> bars{};
    generativeqc::mp2::generated::ump2_adjoint_cpu_plan(channel).run(g, x, ei, ej, ea, eb,
                                                                     bars.data());
    const double numerator = channel == 2 ? g * g : 0.25 * (g - x) * (g - x);
    const double dg = channel == 2 ? 2 * g / denominator : 0.5 * (g - x) / denominator;
    require(std::abs(bars[0] - dg) < 1e-14, "native direct cotangent factor mismatch");
    require(std::abs(bars[1] - (channel == 2 ? 0 : -dg)) < 1e-14,
            "native exchange cotangent factor mismatch");
    for (unsigned axis = 2; axis < 6; ++axis)
      require(std::abs(bars[axis] - (axis < 4 ? -1 : 1) * numerator / (denominator * denominator)) <
                  1e-14,
              "native epsilon cotangent sign mismatch");
  }
}

using namespace generativeqc;

void finite_density_equations() {
  for (unsigned channel = 0; channel < 3; ++channel) {
    const double left = -0.71, right = 0.23;
    std::array<double, 4> density{};
    generativeqc::mp2::generated::ump2_density_cpu_plan(channel).run(left, right, density.data());
    std::array<double, 6> plus{}, minus{};
    const auto adjoint = generativeqc::mp2::generated::ump2_adjoint_cpu_plan(channel);
    adjoint.run(-(left + right) / 2, (left + right) / 2, -1, 0, 0, 0, plus.data());
    adjoint.run(-(left - right) / 2, (left - right) / 2, -1, 0, 0, 0, minus.data());
    for (unsigned leg = 0; leg < 4; ++leg) {
      require(std::abs(density[leg] - (plus[leg + 2] - minus[leg + 2]) / 4) < 1e-14,
              "finite density disagrees with source VJP polarization");
      require(std::abs(density[leg] - (leg < 2 ? -1 : 1) * left * right / 4) < 1e-14,
              "finite density spin factor/sign mismatch");
    }
    bool rejected = false;
    try {
      generativeqc::mp2::generated::ump2_density_cpu_plan(channel).run(
          std::numeric_limits<double>::infinity(), right, density.data());
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "finite density accepted a nonfinite cotangent");
  }
}

void degenerate_occupied_gauge() {
  core::System system;
  const std::vector<core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  const double b = 1.2646147821786555;
  for (double y : {-b, b})
    for (double x : {-2.0, 0.0, 2.0}) {
      system.atoms.push_back({1, {x, y, 0}});
      system.shells.push_back({static_cast<unsigned>(system.atoms.size() - 1), 0, primitives});
    }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "degenerate H6 setup failed");
  const auto strategy = scf::resolve_fock_build(
      methods::detail::UnrestrictedFitting{}.reference_spec(false), scf::FockBackend::Cpu, 0);
  scf::PreparedFockPlan plan(system, nullptr, strategy);
  scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-13;
  options.max_iterations = 500;
  options.reference_memory_budget_bytes = 256ULL << 20;
  options.resolved_fock_build = strategy;
  auto solved = scf::run_prepared_fock_strategy(plan, options);
  require(solved.converged && solved.unrestricted_reference, "degenerate H6 UHF failed");
  const auto original = *solved.unrestricted_reference;
  for (unsigned spin = 0; spin < 2; ++spin)
    require(
        std::abs(original.orbital_energies[spin][1] - original.orbital_energies[spin][2]) < 1e-9,
        "H6 no longer exercises the same-space degenerate domain");
  scf::PreparedFockInteractionSourceView source(plan);
  std::vector<double> baseline;
  double baseline_energy = 0;
  for (unsigned mask = 0; mask < 4; ++mask) {
    auto ref = original;
    const auto n = ref.nbf;
    const double c = std::cos(0.3), s = std::sin(0.3);
    for (unsigned spin = 0; spin < 2; ++spin)
      if (mask & (1U << spin))
        for (std::size_t row = 0; row < n; ++row) {
          const auto first = original.coefficients[spin][row * n + 1];
          const auto second = original.coefficients[spin][row * n + 2];
          ref.coefficients[spin][row * n + 1] = c * first + s * second;
          ref.coefficients[spin][row * n + 2] = -s * first + c * second;
        }
    auto problem = methods::detail::unrestricted_response_problem(ref, plan);
    const auto force = mp2::unrestricted_force_cpu(ref, &problem, source, 256ULL << 20, 1e-10);
    const auto energy = mp2::conventional_unrestricted_energy(ref, source, 256ULL << 20, 1e-10);
    const double total = ref.energy + energy.channels[0] + energy.channels[1] + energy.channels[2];
    require(force.response.converged() && force.stationarity_residual < 1e-8,
            "degenerate H6 response/stationarity failed");
    require(std::abs(force.forces[3]) < 2e-8, "H6 middle-atom reflection force is nonzero");
    if (!mask) {
      baseline = force.forces;
      baseline_energy = total;
    } else {
      require(std::abs(total - baseline_energy) < 1e-11, "occupied gauge changed energy");
      for (std::size_t k = 0; k < baseline.size(); ++k)
        require(std::abs(force.forces[k] - baseline[k]) < 1e-9,
                "same-space occupied gauge changed nuclear force");
    }
  }

  // Corrupt the physical reference in its degenerate occupied block without
  // changing its claimed scalar residual. Its same-space residual lies between
  // the strict 1e-9 compatibility gate and final 1e-7 gate. Two independent
  // symmetric directions avoid accidentally commuting with the density when
  // another eigensolver chooses a different gauge of this degenerate block.
  const auto n = original.nbf;
  std::vector<double> first(n), second(n);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t col = 0; col < n; ++col) {
      first[row] += original.overlap[row * n + col] * original.coefficients[0][col * n + 1];
      second[row] += original.overlap[row * n + col] * original.coefficients[0][col * n + 2];
    }
  bool rejected = false;
  for (unsigned direction = 0; direction < 2; ++direction) {
    auto inconsistent = original;
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t col = 0; col < n; ++col)
        inconsistent.hcore[row * n + col] +=
            1e-6 * (direction == 0 ? first[row] * second[col] + second[row] * first[col]
                                   : first[row] * first[col] - second[row] * second[col]);
    auto incompatible_problem = methods::detail::unrestricted_response_problem(inconsistent, plan);
    try {
      (void)mp2::unrestricted_force_cpu(inconsistent, &incompatible_problem, source, 256ULL << 20,
                                        1e-10);
    } catch (const std::runtime_error& error) {
      require(std::string(error.what()).find("nonstationary degenerate") != std::string::npos,
              "inconsistent H6 reference failed outside strict compatibility gate");
      rejected = true;
    }
  }
  require(rejected, "degenerate compatibility failure reached force publication");
}

void sparse_coordinate_admission() {
  core::System system;
  system.charge = 599;
  system.multiplicity = 2;
  for (unsigned i = 0; i < 600; ++i) system.atoms.push_back({1, {5.0 * i, 0, 0}});
  system.shells = {{0, 0, {{1.0, 1.0}}}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "sparse-coordinate setup failed");
  core::ContextState context;
  context.requested_backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  methods::Capabilities capabilities;
  capabilities.method = GENERATIVEQC_METHOD_UMP2;
  auto descriptor = ump2_descriptor();
  descriptor.correlation_memory_budget_bytes = 20ULL << 20;
  auto calculation =
      methods::detail::prepare_ump2_calculation(capabilities, context, system, descriptor);
  bool rejected = false;
  try {
    (void)calculation->execute(true);
  } catch (const methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  require(rejected && !calculation->correlation_diagnostic(),
          "sparse-coordinate force exceeded capacity or published a failed diagnostic");
  const auto energy = calculation->execute(false);
  require(energy.convergence.converged && calculation->correlation_diagnostic(),
          "sparse-coordinate force failure poisoned energy-only replay");
}

void force_contract() {
  auto system = h2();
  system.atoms.push_back({1, {0.3, 0, 2.7}});
  system.shells.push_back({2, 0, system.shells[0].primitives});
  system.multiplicity = 2;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "open-shell force fixture setup failed");
  const auto strategy = generativeqc::scf::resolve_fock_build(
      generativeqc::methods::detail::UnrestrictedFitting{}.reference_spec(false),
      generativeqc::scf::FockBackend::Cpu, 0);
  generativeqc::scf::PreparedFockPlan plan(system, nullptr, strategy);
  generativeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.energy_tolerance = options.density_tolerance = 1e-12;
  options.max_iterations = 200;
  options.reference_memory_budget_bytes = 256ULL << 20;
  options.resolved_fock_build = strategy;
  options.screening_tolerance = 0;
  auto solved = generativeqc::scf::run_prepared_fock_strategy(plan, options);
  require(solved.converged && solved.unrestricted_reference, "force UHF did not converge");
  auto ref = *solved.unrestricted_reference;
  const auto hf_force =
      generativeqc::methods::detail::unrestricted_reference_force_cpu(ref, plan, 256ULL << 20);
  require(generativeqc::methods::detail::unrestricted_reference_force_cpu(ref, plan, 256ULL << 20,
                                                                          32ULL << 20) == hf_force,
          "reserving a live fitted source changes the UHF component force");
  for (const auto retained : {256ULL << 20, 257ULL << 20}) {
    bool refused = false;
    try {
      (void)generativeqc::methods::detail::unrestricted_reference_force_cpu(ref, plan, 256ULL << 20,
                                                                            retained);
    } catch (const std::length_error&) {
      refused = true;
    }
    require(refused, "UHF component ignores retained fitted source or underflows its budget");
  }
  CountingSource source(system);
  const auto evaluate = [&](const auto& state, std::size_t budget) {
    auto problem = generativeqc::methods::detail::unrestricted_response_problem(state, plan);
    return generativeqc::mp2::unrestricted_force_cpu(state, &problem, source, budget, 1e-10);
  };
  const auto first = evaluate(ref, 256ULL << 20);
  {
    generativeqc::posthf::DensityFittedInteractionSource fitted(source, ref, system, 1e-10,
                                                                256ULL << 20, false, 0, true);
    generativeqc::posthf::NativeBlockProvider provider(fitted, ref, 256ULL << 20);
    const auto n = ref.nbf;
    const generativeqc::posthf::SpinMOSlots request{{{{0, 1}, {n - 1}, {0}, {n - 1}}},
                                                    {0, 0, 1, 1}};
    const auto block = provider.get_spin(request);
    std::vector<double> ao(n * n * n * n);
    fitted.read(generativeqc::integrals::ElectronInteractionOperator::eri, {0, 0, 0, 0},
                {n, n, n, n}, ao.data(), ao.size());
    for (std::size_t selected = 0; selected < block.size(); ++selected) {
      double expected = 0;
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          for (std::size_t rho = 0; rho < n; ++rho)
            for (std::size_t sigma = 0; sigma < n; ++sigma)
              expected += ref.coefficients[0][mu * n + selected] *
                          ref.coefficients[0][nu * n + n - 1] * ref.coefficients[1][rho * n] *
                          ref.coefficients[1][sigma * n + n - 1] *
                          ao[((mu * n + nu) * n + rho) * n + sigma];
      require(std::abs(block[selected] - expected) < 1e-11,
              "three-index MO block differs from independent AO contraction");
    }
    auto stale_frame = ref;
    stale_frame.coefficients[0][0] += 1e-6;
    bool stale_rejected = false;
    try {
      generativeqc::posthf::NativeBlockProvider stale(fitted, stale_frame, 256ULL << 20);
      (void)stale.get_spin(request);
    } catch (const std::invalid_argument&) {
      stale_rejected = true;
    }
    require(stale_rejected, "three-index cache replayed in a stale orbital frame");
  }
  require(first.response.converged() && first.stationarity_residual < 1e-8,
          "UMP2 force did not qualify full response");
  auto swapped = ref;
  std::swap(swapped.nocc[0], swapped.nocc[1]);
  std::swap(swapped.fock[0], swapped.fock[1]);
  std::swap(swapped.coefficients[0], swapped.coefficients[1]);
  std::swap(swapped.orbital_energies[0], swapped.orbital_energies[1]);
  std::swap(swapped.density[0], swapped.density[1]);
  const auto permuted = evaluate(swapped, 256ULL << 20);
  for (std::size_t k = 0; k < first.forces.size(); ++k)
    require(std::abs(first.forces[k] - permuted.forces[k]) < 1e-10,
            "alpha/beta permutation changed the force");
  auto unbound_copy = ref;
  auto old_problem = generativeqc::methods::detail::unrestricted_response_problem(ref, plan);
  bool stale_problem = false;
  try {
    (void)generativeqc::mp2::unrestricted_force_cpu(unbound_copy, &old_problem, source,
                                                    256ULL << 20, 1e-10);
  } catch (const std::invalid_argument&) {
    stale_problem = true;
  }
  require(stale_problem, "same-shape response from another reference was accepted");
  const auto reads = source.reads;
  bool rejected = false;
  try {
    (void)evaluate(ref, 1);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected && source.reads == reads, "force budget refusal read the source");
  const auto exact = evaluate(ref, first.planned_endpoint_peak_bytes);
  require(exact.forces == first.forces, "inclusive force capacity changed the result");
  const auto bounded = evaluate(ref, first.planned_endpoint_peak_bytes - 1);
  require(bounded.planned_endpoint_peak_bytes < first.planned_endpoint_peak_bytes,
          "optional resident storage did not retain the bounded fallback");
  for (std::size_t coordinate = 0; coordinate < first.forces.size(); ++coordinate)
    require(std::abs(bounded.forces[coordinate] - first.forces[coordinate]) < 1e-10,
            "bounded orbital-transform fallback changed UMP2 forces");
  source.fail = true;
  rejected = false;
  try {
    (void)evaluate(ref, 256ULL << 20);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "source failure published a UMP2 force");
  source.fail = false;
  const auto recovered = evaluate(ref, 256ULL << 20);
  require(recovered.forces == first.forces, "failed force poisoned subsequent replay");
  auto nonstationary = ref;
  nonstationary.commutator_residual = 1e-3;
  rejected = false;
  try {
    (void)evaluate(nonstationary, 256ULL << 20);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "nonstationary force reference accepted");
  auto changed = system;
  changed.atoms[0].position[0] += 0.01;
  CountingSource stale(changed);
  rejected = false;
  try {
    auto problem = generativeqc::methods::detail::unrestricted_response_problem(ref, plan);
    (void)generativeqc::mp2::unrestricted_force_cpu(ref, &problem, stale, 256ULL << 20, 1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected && stale.reads == 0, "stale derivative source was read");
}

void unsupported_requests() {
  generativeqc::core::ContextState context;
  context.requested_backend = GENERATIVEQC_BACKEND_HYBRID_CUDA;
  generativeqc::methods::Capabilities capabilities;
  capabilities.method = GENERATIVEQC_METHOD_UMP2;
  auto descriptor = ump2_descriptor();
  bool rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_calculation(capabilities, context, h2(),
                                                                  descriptor);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(rejected, "native UMP2 silently admitted an unsupported hybrid backend");

  // CUDA is an admitted request, not a promise of a device in this host test.
  context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
  context.device_id = 0;
  auto cuda_calculation = generativeqc::methods::detail::prepare_ump2_calculation(
      capabilities, context, h2(), descriptor);
  require(static_cast<bool>(cuda_calculation), "native UMP2 rejected CUDA preparation");
#if !GENERATIVEQC_HAS_CUDA
  for (bool forces : {false, true}) {
    rejected = false;
    try {
      (void)cuda_calculation->execute(forces);
    } catch (const generativeqc::Error& error) {
      rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    require(rejected && !cuda_calculation->correlation_diagnostic(),
            "CPU-only UMP2 executed CUDA or published a failed diagnostic");
  }
#endif

  context.requested_backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  descriptor.energy_tolerance = std::numeric_limits<double>::quiet_NaN();
  rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_calculation(capabilities, context, h2(),
                                                                  descriptor);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "native UMP2 silently reset a nonfinite SCF tolerance");

  descriptor = ump2_descriptor();
  generativeqc_initial_guess_options preliminary{};
  descriptor.initial_guess = &preliminary;
  rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_calculation(capabilities, context, h2(),
                                                                  descriptor);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(rejected, "native UMP2 silently ignored an initial-guess request");

  descriptor = ump2_descriptor();
  descriptor.density_fitting_memory_budget_bytes = 1024;
  rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_calculation(capabilities, context, h2(),
                                                                  descriptor);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "native UMP2 silently ignored a DF budget without density fitting");

  descriptor = ump2_descriptor();
  generativeqc_system auxiliary{};
  auxiliary.data = h2();
  descriptor.density_fitting_auxiliary_basis = &auxiliary;
  // Registry admission checks the ABI before family-specific options. Keep
  // malformed-prefix rejection separate from the valid auxiliary-basis case.
  for (unsigned fault = 0; fault < 3; ++fault) {
    auto malformed = descriptor;
    if (fault == 0)
      malformed.abi_version = GENERATIVEQC_ABI_VERSION + 1;
    else
      malformed.struct_size = fault == 1 ? 0 : sizeof(malformed) - 1;
    rejected = false;
    try {
      (void)generativeqc::methods::prepare_batch(context, {h2()}, malformed, 0);
    } catch (const generativeqc::methods::MethodError& error) {
      rejected = error.status() == GENERATIVEQC_STATUS_ABI_MISMATCH;
    }
    require(rejected, "native UMP2 batch did not reject an invalid descriptor prefix");
  }
  rejected = false;
  try {
    (void)generativeqc::methods::prepare_batch(context, {h2()}, descriptor, 0);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
               std::string(error.what()).find("auxiliary-basis") != std::string::npos;
  }
  require(rejected, "native UMP2 batch silently erased an auxiliary basis");

  descriptor = ump2_descriptor();
  generativeqc_ks_options ks{};
  descriptor.ks_options = &ks;
  rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_batch(capabilities, context, {h2()},
                                                            descriptor, 0);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(rejected, "native UMP2 batch silently erased KS options");
}

void source_and_slots() {
  const auto system = h2();
  generativeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto solved = generativeqc::scf::run_uhf(system, options);
  require(solved.converged && solved.unrestricted_reference, "native physical UHF export failed");
  auto reference = *solved.unrestricted_reference;
  require(reference.source_identity && reference.source_identity->matches(system),
          "physical UHF reference lost exact source identity");
  CountingSource source(system);
  const auto result =
      generativeqc::mp2::conventional_unrestricted_energy(reference, source, 256ULL << 20, 1e-10);
  require(std::isfinite(result.channels[2]) && result.tiles != 0 && source.reads != 0,
          "native UMP2 energy did not read all-electron AO source");

  auto changed = system;
  changed.atoms[1].position[2] += 0.1;
  CountingSource stale(changed);
  bool rejected = false;
  try {
    (void)generativeqc::mp2::conventional_unrestricted_energy(reference, stale, 256ULL << 20,
                                                              1e-10);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("stale UMP2") != std::string::npos;
  }
  require(rejected && stale.reads == 0, "stale source read or accepted");
  rejected = false;
  try {
    generativeqc::posthf::NativeBlockProvider stale_provider(stale, reference, 256ULL << 20, 1);
    (void)stale_provider;
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("identity mismatch") != std::string::npos;
  }
  require(rejected && stale.reads == 0, "spin-owned provider accepted a stale source");

  CountingSource near(system);
  rejected = false;
  try {
    (void)generativeqc::mp2::conventional_unrestricted_energy(reference, near, 256ULL << 20, 100.0);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("near-zero") != std::string::npos;
  }
  require(rejected && near.reads == 0, "near denominator read AO source");

  auto expanded = system;
  expanded.shells.push_back({0, 0, {{0.05, 1.0}}});
  expanded.shells.push_back({1, 0, {{0.06, 1.0}}});
  std::string detail;
  require(generativeqc::molecule::validate_and_normalize(expanded, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "four-AO denominator fixture setup failed");
  auto far = reference;
  far.nbf = 4;
  far.nocc = {1, 1};
  far.overlap.assign(16, 0.0);
  far.hcore.assign(16, 0.0);
  for (unsigned spin = 0; spin < 2; ++spin) {
    far.fock[spin].assign(16, 0.0);
    far.coefficients[spin].assign(16, 0.0);
    far.density[spin].assign(16, 0.0);
    far.orbital_energies[spin] = {-1.0, 0.1, 0.2, 1e308};
  }
  far.source_identity.emplace(expanded);
  CountingSource far_source(expanded);
  rejected = false;
  try {
    (void)generativeqc::mp2::conventional_unrestricted_energy(far, far_source, 256ULL << 20, 1e-10);
  } catch (const std::invalid_argument& error) {
    rejected =
        std::string(error.what()).find("nonfinite UMP2 denominator extrema") != std::string::npos;
  }
  require(rejected && far_source.reads == 0, "far denominator overflow read AO source");

  // Rotate only beta columns. This is a provider-specific slot-owner probe;
  // the physical UHF state above, not this modified frame, qualifies energy.
  const double angle = 0.3, c = std::cos(angle), s = std::sin(angle);
  const auto original = reference.coefficients[1];
  for (std::size_t row = 0; row < reference.nbf; ++row) {
    reference.coefficients[1][row * 2] = c * original[row * 2] + s * original[row * 2 + 1];
    reference.coefficients[1][row * 2 + 1] = -s * original[row * 2] + c * original[row * 2 + 1];
  }
  generativeqc::posthf::NativeBlockProvider provider(source, reference, 256ULL << 20, 1);
  require(provider.reference_bytes() >= reference.source_identity->storage_bytes(),
          "spin-owned provider omitted retained source identity capacity");
  const generativeqc::posthf::SpinMOSlots block{
      generativeqc::posthf::MOSlots{std::vector<std::size_t>{0}, std::vector<std::size_t>{1},
                                    std::vector<std::size_t>{0}, std::vector<std::size_t>{1}},
      {0, 0, 1, 1}};
  const double actual = provider.get_spin(block).at(0);
  std::array<double, 16> ao{};
  source.read(generativeqc::integrals::ElectronInteractionOperator::eri, {0, 0, 0, 0}, {2, 2, 2, 2},
              ao.data(), ao.size());
  double expected{};
  for (std::size_t u = 0; u < 2; ++u)
    for (std::size_t v = 0; v < 2; ++v)
      for (std::size_t w = 0; w < 2; ++w)
        for (std::size_t z = 0; z < 2; ++z)
          expected += ao[((u * 2 + v) * 2 + w) * 2 + z] * reference.coefficients[0][u * 2] *
                      reference.coefficients[0][v * 2 + 1] * reference.coefficients[1][w * 2] *
                      reference.coefficients[1][z * 2 + 1];
  require(std::abs(actual - expected) < 1e-12,
          "native MO provider ignored a beta coefficient slot");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--force-contract") {
      force_contract();
      std::cout << "native UMP2 force contract passed\n";
      return 0;
    }
    native_equations();
    finite_density_equations();
    degenerate_occupied_gauge();
    sparse_coordinate_admission();
    energy_only_reference_admission();
    one_electron_response_admission();
    full_fitted_reference_admission();
    cuda_channel_admission();
    unsupported_requests();
    source_and_slots();
    force_contract();
    std::cout << "native UMP2 contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
