#include "integrals/minao_basis.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "integrals/minao_basis_data.hpp"
#include "molecule/basis.hpp"

namespace generativeqc::integrals {
namespace {

using minao_data::kElementShellOffsets;
using minao_data::kPrimitives;
using minao_data::kShells;

void validate_domain(const core::System& system) {
  if (!system.ecp_terms.empty() || std::any_of(system.atoms.begin(), system.atoms.end(),
                                               [](const auto& atom) { return atom.ecp_core != 0; }))
    throw std::invalid_argument("MINAO initial guess is qualified only for all-electron systems");
  for (const auto& atom : system.atoms)
    if (atom.atomic_number < 1 || atom.atomic_number > 18)
      throw std::invalid_argument("MINAO initial guess is currently qualified for H-Ar");
}

}  // namespace

std::size_t minao_basis_ao_count(const core::System& system) {
  validate_domain(system);
  std::size_t count = 0;
  for (const auto& atom : system.atoms) {
    const auto z = static_cast<std::size_t>(atom.atomic_number);
    for (std::size_t shell_index = kElementShellOffsets[z - 1];
         shell_index < kElementShellOffsets[z]; ++shell_index) {
      const auto l = static_cast<std::size_t>(kShells[shell_index].angular_momentum);
      count += 2 * l + 1;
    }
  }
  return count;
}

std::size_t minao_basis_primitive_count(const core::System& system) {
  validate_domain(system);
  std::size_t count = 0;
  for (const auto& atom : system.atoms) {
    const auto z = static_cast<std::size_t>(atom.atomic_number);
    for (std::size_t shell_index = kElementShellOffsets[z - 1];
         shell_index < kElementShellOffsets[z]; ++shell_index)
      count += kShells[shell_index].primitive_count;
  }
  return count;
}

MinaoBasisSource make_minao_basis_source(const core::System& target) {
  validate_domain(target);
  MinaoBasisSource result;
  result.system.atoms.reserve(target.atoms.size());
  for (const auto& atom : target.atoms)
    result.system.atoms.push_back({atom.atomic_number, atom.position, 0});
  result.system.charge = 0;
  result.system.multiplicity = 1;
  result.system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;

  for (std::size_t atom_index = 0; atom_index < target.atoms.size(); ++atom_index) {
    const auto z = static_cast<std::size_t>(target.atoms[atom_index].atomic_number);
    const auto begin = kElementShellOffsets[z - 1];
    const auto end = kElementShellOffsets[z];
    for (std::size_t shell_index = begin; shell_index < end; ++shell_index) {
      const auto& record = kShells[shell_index];
      core::Shell shell;
      shell.atom_index = static_cast<std::uint32_t>(atom_index);
      shell.angular_momentum = record.angular_momentum;
      shell.primitives.reserve(record.primitive_count);
      for (std::size_t p = record.primitive_begin;
           p < static_cast<std::size_t>(record.primitive_begin) + record.primitive_count; ++p)
        shell.primitives.push_back({kPrimitives[p].exponent, kPrimitives[p].coefficient});
      result.primitive_count += shell.primitives.size();
      result.system.shells.push_back(std::move(shell));
      for (unsigned component = 0; component < 2U * record.angular_momentum + 1U; ++component)
        result.occupations.push_back(record.occupation);
    }
  }

  std::string detail;
  const auto status = molecule::validate_and_normalize(result.system, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail.empty() ? "MINAO source normalization failed" : detail);
  if (molecule::ao_count(result.system) != result.occupations.size())
    throw std::logic_error("MINAO source AO/occupation shape mismatch");
  return result;
}

}  // namespace generativeqc::integrals
