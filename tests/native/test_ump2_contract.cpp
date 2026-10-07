#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/handles.hpp"
#include "methods/ump2_method.hpp"
#include "molecule/basis.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "posthf/ump2_cpu_generated.hpp"
#include "posthf/ump2_energy.hpp"
#include "scf/mean_field.hpp"

namespace {

void require(bool valid, const char* detail) {
  if (!valid) throw std::runtime_error(detail);
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
    raw_.read(op, begin, count, output, elements);
  }
  mutable std::size_t reads{};

 private:
  generativeqc::posthf::RawSource raw_;
};

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
  }
}

void unsupported_requests() {
  generativeqc::core::ContextState context;
  context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
  generativeqc::methods::Capabilities capabilities;
  capabilities.method = GENERATIVEQC_METHOD_UMP2;
  generativeqc_method_descriptor descriptor{};
  bool rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_calculation(capabilities, context, h2(),
                                                                  descriptor);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(rejected, "native UMP2 silently admitted CUDA backend");

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

  descriptor = {};
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

  descriptor = {};
  descriptor.density_fitting_memory_budget_bytes = 1024;
  rejected = false;
  try {
    (void)generativeqc::methods::detail::prepare_ump2_calculation(capabilities, context, h2(),
                                                                  descriptor);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(rejected, "native UMP2 silently ignored a DF memory request");

  descriptor = {};
  generativeqc_system auxiliary{};
  auxiliary.data = h2();
  descriptor.method = GENERATIVEQC_METHOD_UMP2;
  descriptor.density_fitting_auxiliary_basis = &auxiliary;
  rejected = false;
  try {
    (void)generativeqc::methods::prepare_batch(context, {h2()}, descriptor, 0);
  } catch (const generativeqc::methods::MethodError& error) {
    rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(rejected, "native UMP2 batch silently erased an auxiliary basis");

  descriptor = {};
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

int main() {
  try {
    native_equations();
    unsupported_requests();
    source_and_slots();
    std::cout << "native UMP2 contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
