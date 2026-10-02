#include "cli/native_basis.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cli_basis_pack.hpp"

namespace generativeqc::cli {
namespace {

std::string canonical_basis_name(std::string_view name) {
  std::string result(name);
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
    if (value == '_') return '-';
    return static_cast<char>(std::tolower(value));
  });
  return result;
}

const basis_generated::BasisRow* find_basis(std::string_view name) {
  const std::string canonical = canonical_basis_name(name);
  for (const auto& basis : basis_generated::kBases)
    if (basis.name == canonical) return &basis;
  return nullptr;
}

}  // namespace

std::vector<std::string_view> bundled_basis_names() {
  std::vector<std::string_view> result;
  result.reserve(basis_generated::kBases.size());
  for (const auto& basis : basis_generated::kBases) result.push_back(basis.name);
  return result;
}

bool bundled_basis_supports(std::string_view name, int atomic_number) {
  const auto* basis = find_basis(name);
  if (basis == nullptr) return false;
  const auto begin = basis_generated::kShells.begin() + basis->shell_offset;
  const auto end = begin + basis->shell_count;
  return std::any_of(begin, end, [atomic_number](const auto& shell) {
    return shell.atomic_number == atomic_number;
  });
}

NativeBasisData expand_bundled_basis(std::string_view name,
                                     std::span<const generativeqc_atom> atoms,
                                     generativeqc_basis_representation representation) {
  const auto* basis = find_basis(name);
  if (basis == nullptr) throw std::invalid_argument("unknown bundled basis: " + std::string(name));

  NativeBasisData result;
  result.representation = representation;
  const auto begin = basis_generated::kShells.begin() + basis->shell_offset;
  const auto end = begin + basis->shell_count;

  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
    const int atomic_number = atoms[atom_index].atomic_number;
    bool found = false;
    for (auto shell = begin; shell != end; ++shell) {
      if (shell->atomic_number != atomic_number) continue;
      found = true;
      const auto primitive_offset = static_cast<std::uint32_t>(result.primitives.size());
      for (std::uint32_t index = 0; index < shell->primitive_count; ++index) {
        const auto& primitive = basis_generated::kPrimitives[shell->primitive_offset + index];
        result.primitives.push_back({primitive.exponent, primitive.coefficient});
      }
      result.shells.push_back({static_cast<std::uint32_t>(atom_index), shell->angular_momentum,
                               primitive_offset, shell->primitive_count});
    }
    if (!found) {
      throw std::invalid_argument(std::string(basis->name) + " is not bundled for atomic number " +
                                  std::to_string(atomic_number));
    }
  }
  return result;
}

}  // namespace generativeqc::cli
