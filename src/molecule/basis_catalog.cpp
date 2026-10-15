#include "molecule/basis_catalog.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

#include "basis_catalog_data.hpp"
#include "molecule/basis.hpp"

namespace generativeqc::molecule::catalog {
namespace {

const basis_generated::BasisRow* find_basis(std::string_view name) {
  std::string canonical(name);
  std::transform(canonical.begin(), canonical.end(), canonical.begin(), [](unsigned char value) {
    if (value == '_') return '-';
    return static_cast<char>(std::tolower(value));
  });
  for (const auto& basis : basis_generated::kBases)
    if (basis.name == canonical) return &basis;
  return nullptr;
}

void add(std::size_t& count, std::size_t increment, std::size_t maximum) {
  if (increment > maximum || count > maximum - increment)
    throw std::invalid_argument("expanded basis exceeds native dimensions");
  count += increment;
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

Capacity bundled_basis_capacity(std::string_view name, std::span<const generativeqc_atom> atoms,
                                generativeqc_basis_representation representation) {
  constexpr auto maximum_index = std::numeric_limits<std::uint32_t>::max();
  constexpr auto maximum_size = std::numeric_limits<std::size_t>::max();
  if (atoms.empty() || atoms.size() > maximum_index)
    throw std::invalid_argument("bundled basis requires a bounded nonempty atom array");
  if (representation != GENERATIVEQC_BASIS_CARTESIAN &&
      representation != GENERATIVEQC_BASIS_SPHERICAL)
    throw std::invalid_argument("unknown basis representation");
  const auto* basis = find_basis(name);
  if (basis == nullptr) throw std::invalid_argument("unknown bundled basis: " + std::string(name));
  const auto begin = basis_generated::kShells.begin() + basis->shell_offset;
  const auto end = begin + basis->shell_count;
  Capacity result;
  for (const auto& atom : atoms) {
    bool found = false;
    for (auto shell = begin; shell != end; ++shell) {
      if (shell->atomic_number != atom.atomic_number) continue;
      found = true;
      add(result.shell_count, 1, maximum_index);
      add(result.primitive_count, shell->primitive_count, maximum_index);
      const auto angular = static_cast<std::size_t>(shell->angular_momentum);
      const auto count = representation == GENERATIVEQC_BASIS_SPHERICAL
                             ? 2 * angular + 1
                             : cartesian_count(shell->angular_momentum);
      if (!count) throw std::invalid_argument("bundled AO count overflows");
      add(result.ao_count, count, maximum_size);
    }
    if (!found)
      throw std::invalid_argument(std::string(basis->name) + " is not bundled for atomic number " +
                                  std::to_string(atom.atomic_number));
  }
  if (result.shell_count > maximum_size / sizeof(generativeqc_shell) ||
      result.primitive_count > maximum_size / sizeof(generativeqc_primitive))
    throw std::invalid_argument("bundled basis buffer capacity overflows");
  result.buffer_bytes = result.shell_count * sizeof(generativeqc_shell);
  add(result.buffer_bytes, result.primitive_count * sizeof(generativeqc_primitive), maximum_size);
  return result;
}

BasisData expand_bundled_basis(std::string_view name, std::span<const generativeqc_atom> atoms,
                               generativeqc_basis_representation representation) {
  const auto capacity = bundled_basis_capacity(name, atoms, representation);
  const auto* basis = find_basis(name);
  BasisData result;
  result.representation = representation;
  result.shells.reserve(capacity.shell_count);
  result.primitives.reserve(capacity.primitive_count);
  const auto begin = basis_generated::kShells.begin() + basis->shell_offset;
  const auto end = begin + basis->shell_count;
  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
    for (auto shell = begin; shell != end; ++shell) {
      if (shell->atomic_number != atoms[atom_index].atomic_number) continue;
      const auto offset = static_cast<std::uint32_t>(result.primitives.size());
      for (std::uint32_t index = 0; index < shell->primitive_count; ++index) {
        const auto& primitive = basis_generated::kPrimitives[shell->primitive_offset + index];
        result.primitives.push_back({primitive.exponent, primitive.coefficient});
      }
      result.shells.push_back({static_cast<std::uint32_t>(atom_index), shell->angular_momentum,
                               offset, shell->primitive_count});
    }
  }
  return result;
}

}  // namespace generativeqc::molecule::catalog
