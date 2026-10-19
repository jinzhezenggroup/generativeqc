#include "molecule/basis_catalog.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "molecule/basis_template.hpp"
#include "native_basis_pack.hpp"

namespace generativeqc::molecule {
namespace {

std::string normalized_basis_name(std::string_view name) {
  std::string result(name);
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
    if (value == '_') return '-';
    return static_cast<char>(std::tolower(value));
  });
  return result;
}

const basis_generated::BasisRow* find_basis(std::string_view name) {
  const std::string canonical = normalized_basis_name(name);
  for (const auto& basis : basis_generated::kBases)
    if (basis.name == canonical) return &basis;
  return nullptr;
}

generativeqc_basis_template_descriptor_v1 bundled_template(std::string_view name) {
  const auto* basis = find_basis(name);
  if (basis == nullptr) throw std::invalid_argument("unknown bundled basis: " + std::string(name));
  return {sizeof(generativeqc_basis_template_descriptor_v1),
          GENERATIVEQC_ABI_VERSION,
          basis_generated::kElements.data() + basis->element_offset,
          basis->element_count,
          basis_generated::kShells.data() + basis->shell_offset,
          basis->shell_count,
          basis_generated::kExponents.data() + basis->primitive_offset,
          basis->primitive_count,
          basis_generated::kCoefficients.data() + basis->primitive_offset,
          basis->primitive_count,
          nullptr,
          0};
}

}  // namespace

const char* bundled_basis_catalog_sha256() noexcept { return basis_generated::kSourceSha256; }

std::size_t bundled_basis_count() noexcept { return basis_generated::kBases.size(); }

std::string_view bundled_basis_name(std::size_t index) noexcept {
  return index < bundled_basis_count() ? basis_generated::kBases[index].name : std::string_view{};
}

std::string_view bundled_basis_pack_checksum() noexcept { return bundled_basis_catalog_sha256(); }

std::string_view canonical_basis_name(std::string_view name) {
  const auto* basis = find_basis(name);
  if (basis == nullptr) throw std::invalid_argument("unknown bundled basis: " + std::string(name));
  return basis->name;
}

std::vector<std::string_view> bundled_basis_names() {
  std::vector<std::string_view> result;
  result.reserve(basis_generated::kBases.size());
  for (const auto& basis : basis_generated::kBases) result.push_back(basis.name);
  return result;
}

bool bundled_basis_supports(std::string_view name, int atomic_number) {
  const auto* basis = find_basis(name);
  if (basis == nullptr) return false;
  const auto begin = basis_generated::kElements.begin() + basis->element_offset;
  const auto end = begin + basis->element_count;
  return std::any_of(begin, end, [atomic_number](const auto& shell) {
    return shell.atomic_number == atomic_number;
  });
}

ExpandedBasis expand_bundled_basis(std::string_view name, std::span<const generativeqc_atom> atoms,
                                   generativeqc_basis_representation representation) {
  return expand_basis_template(bundled_template(name), atoms, representation);
}

BasisExpansionCapacity bundled_basis_capacity(std::string_view name,
                                              std::span<const generativeqc_atom> atoms,
                                              generativeqc_basis_representation representation) {
  return basis_template_capacity(bundled_template(name), atoms, representation);
}

}  // namespace generativeqc::molecule
