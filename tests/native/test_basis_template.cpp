#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "api/handles.hpp"
#include "generativeqc/generativeqc.hpp"
#include "molecule/basis_template.hpp"

struct TemplateExpansionView {
  const generativeqc_shell* shells;
  std::size_t shell_count;
  const generativeqc_primitive* primitives;
  std::size_t primitive_count;
  const std::int32_t* cores;
  std::size_t core_count;
  const generativeqc_ecp_term* terms;
  std::size_t term_count;
  void* owner;
};

extern "C" int test_expand_basis_template(const generativeqc_basis_template_descriptor_v1* basis,
                                          const generativeqc_atom* atoms, std::size_t atom_count,
                                          int representation, TemplateExpansionView* output) {
  try {
    auto owner = std::make_unique<generativeqc::molecule::ExpandedBasis>(
        generativeqc::molecule::expand_basis_template(
            *basis, std::span(atoms, atom_count),
            static_cast<generativeqc_basis_representation>(representation)));
    *output = {owner->shells.data(),     owner->shells.size(),         owner->primitives.data(),
               owner->primitives.size(), owner->core_electrons.data(), owner->core_electrons.size(),
               owner->ecp_terms.data(),  owner->ecp_terms.size(),      owner.get()};
    owner.release();
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" void test_free_basis_template(void* owner) {
  delete static_cast<generativeqc::molecule::ExpandedBasis*>(owner);
}

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc::molecule::BasisTemplateData fixture() {
  generativeqc::molecule::BasisTemplateData result;
  result.elements = {{1, 0, 0, 1, 0, 0}, {3, 2, 1, 1, 0, 3}};
  result.shells = {{0, 0, 2, 0, 2}, {0, 2, 1, 4, 1}};
  result.exponents = {1.1, 0.4, 0.7};
  result.coefficients = {1, 0, 0.3, -0.5, 1};
  result.ecp_terms = {{2, 2, 1.0, 0.7}, {0, 0, 0.6, 0.2}, {1, 1, 0.8, -0.1}};
  return result;
}

void run() {
  generativeqc_context_descriptor context_descriptor{};
  context_descriptor.struct_size = sizeof(context_descriptor);
  context_descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  context_descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context context(context_descriptor);
  auto source = fixture();
  auto basis = source.descriptor();
  std::array<generativeqc_atom, 2> atoms{{{3, 0, 0, 0}, {1, 0, 0, 1.4}}};
  generativeqc_system_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.atoms = atoms.data();
  descriptor.atom_count = atoms.size();
  descriptor.multiplicity = 1;
  descriptor.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;

  const auto raw =
      generativeqc::molecule::expand_basis_template(basis, atoms, GENERATIVEQC_BASIS_CARTESIAN);
  require(raw.shells.size() == 3 && raw.primitives.size() == 5 && raw.ecp_terms.size() == 3,
          "general row or ECP counts changed");
  require(raw.shells[0].atom_index == 0 && raw.shells[1].atom_index == 1 &&
              raw.shells[2].primitive_offset == 3 && raw.primitives[2].coefficient == 0,
          "atom/source row order or zero coefficient changed");
  require(raw.ecp_terms[0].channel == -1 && raw.ecp_terms[1].channel == 0 &&
              raw.ecp_terms[2].channel == 1 && raw.ecp_terms[0].atom_index == 0,
          "highest ECP channel must be local independently of record order");
  generativeqc::System system(context, atoms, basis);
  require(system.get()->data.atoms[0].ecp_core == 2 && system.get()->data.electron_count == 2 &&
              system.get()->data.charge == 0 && system.get()->data.ecp_terms.size() == 3,
          "template constructor changed existing ECP bookkeeping");

  auto rejected = [&](generativeqc_status expected) {
    auto* output = system.get();
    const auto status =
        generativeqc_system_create_basis_template_v1(context.get(), &descriptor, &basis, &output);
    require(status == expected && output == nullptr,
            "invalid template must reject without publication");
    require(system.get()->data.electron_count == 2,
            "failed creation must preserve an existing owner");
  };
  const auto original = basis;
  basis.struct_size -= 1;
  rejected(GENERATIVEQC_STATUS_ABI_MISMATCH);
  basis = original;
  basis.abi_version += 1;
  rejected(GENERATIVEQC_STATUS_ABI_MISMATCH);
  basis = original;
  basis.exponents = nullptr;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  basis = original;
  basis.coefficient_count -= 1;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  basis = original;
  source.elements[1].atomic_number = 1;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  source.elements[1].atomic_number = 3;
  source.shells[0].contraction_count = std::numeric_limits<std::uint32_t>::max();
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  source.shells[0].contraction_count = 2;
  source.coefficients[0] = 0;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  source.coefficients[0] = 1;
  source.exponents[0] = std::numeric_limits<double>::infinity();
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  source.exponents[0] = 1.1;
  source.ecp_terms[0].power = 5;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  source.ecp_terms[0].power = 2;
  source.elements[1].ecp_core_electrons = 0;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  source.elements[1].ecp_core_electrons = 2;
  descriptor.basis_representation = 99;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  descriptor.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  atoms[0].atomic_number = 8;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  atoms[0].atomic_number = 3;
  descriptor.primitive_count = 1;
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  descriptor.primitive_count = 0;

  std::vector<generativeqc_atom> many_atoms(65536, {1, 0, 0, 0});
  generativeqc::molecule::BasisTemplateData huge;
  huge.elements = {{1, 0, 0, 1, 0, 0}};
  huge.shells = {{0, 0, 1, 0, 65536}};
  huge.exponents = {1};
  huge.coefficients.assign(65536, 1);
  basis = huge.descriptor();
  descriptor.atoms = many_atoms.data();
  descriptor.atom_count = many_atoms.size();
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require(std::string_view(generativeqc_context_get_last_detail(context.get())).find("uint32") !=
              std::string_view::npos,
          "expanded dimension overflow must fail before allocation/normalization");
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
