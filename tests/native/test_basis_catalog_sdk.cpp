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
    std::unique_ptr<generativeqc::Calculation> custom;
    {
      std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
      std::array<double, 3> exponents{{3.425250914, 0.6239137298, 0.1688554040}};
      std::array<double, 3> coefficients{{0.1543289673, 0.5353281423, 0.4446345422}};
      const std::array<generativeqc_basis_shell_template_v1, 1> shells{{{0, 0, 3, 0, 1}}};
      const std::array<generativeqc_basis_element_template_v1, 1> elements{{{1, 0, 0, 1, 0, 0}}};
      generativeqc_basis_template_descriptor_v1 basis{};
      basis.struct_size = sizeof(basis);
      basis.abi_version = GENERATIVEQC_ABI_VERSION;
      basis.elements = elements.data();
      basis.element_count = elements.size();
      basis.shells = shells.data();
      basis.shell_count = shells.size();
      basis.exponents = exponents.data();
      basis.exponent_count = exponents.size();
      basis.coefficients = coefficients.data();
      basis.coefficient_count = coefficients.size();
      generativeqc::System system(context, atoms, basis);
      custom = std::make_unique<generativeqc::Calculation>(context, system, method);
      exponents.fill(99);
      coefficients.fill(0);
      atoms[0].z = -100;
    }
    const auto custom_result =
        custom->execute(GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES);
    if (custom_result.energy != result.energy || custom_result.forces != result.forces)
      throw std::runtime_error("element-template SDK changed the retained H2 model");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
