#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "generativeqc/generativeqc.hpp"

int main() {
  try {
    generativeqc_context_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
    generativeqc::Context context(descriptor);
    auto method = generativeqc::default_method_descriptor("rhf");
    method.energy_tolerance = 1e-12;
    method.density_tolerance = 1e-10;
    method.max_iterations = 100;
    std::unique_ptr<generativeqc::Calculation> retained;
    {
      std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
      generativeqc::System system(context, atoms, "STO_3G");
      retained = std::make_unique<generativeqc::Calculation>(context, system, method);
      atoms[0].z = -100;
    }
    const auto result =
        retained->execute(GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES);
    if (std::abs(result.energy - (-1.1167143251757694)) > 2e-10 ||
        result.executed_backend != GENERATIVEQC_BACKEND_CPU_REFERENCE || !result.forces ||
        result.forces->size() != 6 || std::abs((*result.forces)[2] - 0.028454057162517143) > 2e-9 ||
        std::abs((*result.forces)[5] + 0.028454057162517143) > 2e-9)
      throw std::runtime_error("bundled SDK endpoint differs from independent H2 reference");
    const auto replay = retained->execute(GENERATIVEQC_PROPERTY_ENERGY);
    if (std::abs(replay.energy - result.energy) > 2e-10 || replay.forces)
      throw std::runtime_error("retained bundled SDK replay failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
