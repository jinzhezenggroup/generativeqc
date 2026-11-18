#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cli/molecular_system.hpp"

namespace {
using generativeqc::Batch;
using generativeqc::Context;
using generativeqc::System;
using generativeqc::cli::make_molecular_system;
using generativeqc::cli::NativeBasisData;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Fixture {
  std::array<generativeqc_atom, 2> atoms{{{11, 0, 0, 0}, {1, 0.2, -0.1, 3.0}}};
  NativeBasisData basis;

  Fixture() {
    basis.primitives = {{0.55, 1.0}, {0.7, 1.0}};
    basis.shells = {{0, 0, 0, 1}, {1, 0, 1, 1}};
    basis.core_electrons = {10, 0};
    basis.ecp_terms = {{0, -1, 2, 0.8, -2.0}};
    basis.representation = GENERATIVEQC_BASIS_SPHERICAL;
  }

  generativeqc_system_descriptor geometry() const {
    return {sizeof(generativeqc_system_descriptor),
            GENERATIVEQC_ABI_VERSION,
            atoms.data(),
            static_cast<std::uint32_t>(atoms.size()),
            basis.shells.data(),
            static_cast<std::uint32_t>(basis.shells.size()),
            basis.primitives.data(),
            static_cast<std::uint32_t>(basis.primitives.size()),
            0,
            1,
            basis.representation};
  }
};

generativeqc::BatchItemResult execute(Context& context, const System& system) {
  auto method = generativeqc::default_method_descriptor("rhf");
  method.max_iterations = 100;
  method.energy_tolerance = 1e-12;
  method.density_tolerance = 1e-11;
  const std::array<const System*, 1> systems{&system};
  Batch batch(context, systems, method, 0);
  auto result = batch.execute().front();
  require(result.status == GENERATIVEQC_STATUS_SUCCESS && result.converged,
          "molecular system did not preserve the executable two-electron Hamiltonian");
  return result;
}

void same_result(const generativeqc::BatchItemResult& actual,
                 const generativeqc::BatchItemResult& expected) {
  require(std::abs(actual.energy - expected.energy) < 1e-11,
          "molecular Hamiltonian energy changed");
  require(actual.forces.size() == expected.forces.size() && actual.forces.size() == 6,
          "molecular force layout changed");
  for (std::size_t index = 0; index < actual.forces.size(); ++index)
    require(std::abs(actual.forces[index] - expected.forces[index]) < 1e-10,
            "molecular Hamiltonian force changed");
}

/** Ownership contracts only: these minimal potentials do not qualify chemistry. */
void check_ecp_frames(Context& context) {
  Fixture fixture;
  auto system = make_molecular_system(context, fixture.geometry(), fixture.basis);
  System explicit_ecp(context, fixture.geometry(), fixture.basis.core_electrons,
                      fixture.basis.ecp_terms);
  const auto expected = execute(context, explicit_ecp);
  same_result(execute(context, system), expected);
  fixture.basis.core_electrons[0] = 8;
  fixture.basis.ecp_terms[0].coefficient = -9.0;
  same_result(execute(context, system), expected);
  fixture.basis.core_electrons[0] = 10;
  fixture.basis.ecp_terms[0].coefficient = -2.0;
  fixture.atoms[1].z += 0.3;
  auto moved = make_molecular_system(context, fixture.geometry(), fixture.basis);
  System explicit_moved(context, fixture.geometry(), fixture.basis.core_electrons,
                        fixture.basis.ecp_terms);
  const auto changed = execute(context, moved);
  same_result(changed, execute(context, explicit_moved));
  require(std::abs(changed.energy - expected.energy) > 1e-6,
          "a fresh molecular frame ignored the changed geometry");
  same_result(execute(context, system), expected);
}

void check_all_electron_and_invalid_core_layout(Context& context) {
  Fixture fixture;
  fixture.atoms[0].atomic_number = 1;
  fixture.basis.ecp_terms.clear();
  fixture.basis.core_electrons = {0, 0};
  auto system = make_molecular_system(context, fixture.geometry(), fixture.basis);
  System explicit_all_electron(context, fixture.geometry());
  same_result(execute(context, system), execute(context, explicit_all_electron));
  fixture.basis.core_electrons.resize(1);
  bool refused = false;
  try {
    auto invalid = make_molecular_system(context, fixture.geometry(), fixture.basis);
  } catch (const generativeqc::Error& error) {
    refused = error.status() == GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  require(refused, "malformed molecular core counts were silently discarded");
  fixture.atoms[0].atomic_number = 2;
  fixture.atoms[1].atomic_number = 2;
  fixture.basis.core_electrons = {1, 1};
  System valid_all_electron(context, fixture.geometry());
  require(valid_all_electron.get() != nullptr, "invalid all-electron refusal control");
  refused = false;
  try {
    auto invalid = make_molecular_system(context, fixture.geometry(), fixture.basis);
  } catch (const generativeqc::Error& error) {
    refused = error.status() == GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  require(refused, "ECP cores without required potentials became an all-electron system");
}
}  // namespace

int main() {
  try {
    const generativeqc_context_descriptor descriptor{sizeof(descriptor), GENERATIVEQC_ABI_VERSION,
                                                     -1, GENERATIVEQC_BACKEND_CPU_REFERENCE};
    Context context(descriptor);
    check_ecp_frames(context);
    check_all_electron_and_invalid_core_layout(context);
    std::cout << "CLI molecular system ownership contracts passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
