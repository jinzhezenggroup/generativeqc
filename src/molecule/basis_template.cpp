#include "molecule/basis_template.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace generativeqc::molecule {
namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::invalid_argument(std::string(message));
}

std::uint32_t count(std::uint64_t value) {
  require(value <= std::numeric_limits<std::uint32_t>::max(),
          "expanded basis exceeds native uint32 dimensions");
  return static_cast<std::uint32_t>(value);
}

bool range(std::uint32_t offset, std::uint64_t length, std::uint32_t size) {
  return offset <= size && length <= size - offset;
}

std::array<const generativeqc_basis_element_template_v1*, 119> validate(
    const generativeqc_basis_template_descriptor_v1& basis) {
  require(basis.struct_size >= sizeof(basis) && basis.abi_version == GENERATIVEQC_ABI_VERSION,
          "invalid basis template ABI");
  require(basis.elements && basis.element_count > 0 && basis.element_count <= 118 && basis.shells &&
              basis.shell_count > 0 && basis.exponents && basis.exponent_count > 0 &&
              basis.coefficients && basis.coefficient_count > 0 &&
              (basis.ecp_term_count == 0 ? basis.ecp_terms == nullptr : basis.ecp_terms != nullptr),
          "invalid basis template pointers/counts");
  for (std::uint32_t index = 0; index < basis.exponent_count; ++index)
    require(std::isfinite(basis.exponents[index]) && basis.exponents[index] > 0,
            "basis exponents must be finite and positive");
  for (std::uint32_t index = 0; index < basis.coefficient_count; ++index)
    require(std::isfinite(basis.coefficients[index]), "basis coefficients must be finite");
  for (std::uint32_t index = 0; index < basis.shell_count; ++index) {
    const auto& shell = basis.shells[index];
    const auto entries = std::uint64_t(shell.primitive_count) * shell.contraction_count;
    require(shell.angular_momentum <= 32 && shell.primitive_count > 0 &&
                shell.contraction_count > 0 &&
                range(shell.exponent_offset, shell.primitive_count, basis.exponent_count) &&
                range(shell.coefficient_offset, entries, basis.coefficient_count),
            "invalid basis contraction ranges");
    for (std::uint32_t row = 0; row < shell.contraction_count; ++row) {
      const auto* begin = basis.coefficients + shell.coefficient_offset +
                          std::uint64_t(row) * shell.primitive_count;
      require(std::any_of(begin, begin + shell.primitive_count,
                          [](double value) { return value != 0; }),
              "identically zero contraction cannot be normalized");
    }
  }
  for (std::uint32_t index = 0; index < basis.ecp_term_count; ++index) {
    const auto& term = basis.ecp_terms[index];
    require(term.angular_momentum <= 32 && term.power <= 32 && std::isfinite(term.exponent) &&
                term.exponent > 0 && std::isfinite(term.coefficient),
            "invalid scalar ECP template term");
  }
  std::array<const generativeqc_basis_element_template_v1*, 119> elements{};
  for (std::uint32_t index = 0; index < basis.element_count; ++index) {
    const auto& element = basis.elements[index];
    require(element.atomic_number > 0 && element.atomic_number <= 118 &&
                elements[element.atomic_number] == nullptr,
            "basis template requires unique element labels Z=1..118");
    require(element.shell_count > 0 &&
                range(element.shell_offset, element.shell_count, basis.shell_count) &&
                range(element.ecp_term_offset, element.ecp_term_count, basis.ecp_term_count) &&
                element.ecp_core_electrons >= 0 &&
                element.ecp_core_electrons < element.atomic_number &&
                ((element.ecp_core_electrons != 0) == (element.ecp_term_count != 0)),
            "invalid basis element ranges or ECP core/term pairing");
    elements[element.atomic_number] = &element;
  }
  return elements;
}

}  // namespace

generativeqc_basis_template_descriptor_v1 BasisTemplateData::descriptor() const {
  return {sizeof(generativeqc_basis_template_descriptor_v1),
          GENERATIVEQC_ABI_VERSION,
          elements.data(),
          count(elements.size()),
          shells.data(),
          count(shells.size()),
          exponents.data(),
          count(exponents.size()),
          coefficients.data(),
          count(coefficients.size()),
          ecp_terms.empty() ? nullptr : ecp_terms.data(),
          count(ecp_terms.size())};
}

ExpandedBasis expand_basis_template(const generativeqc_basis_template_descriptor_v1& basis,
                                    std::span<const generativeqc_atom> atoms,
                                    generativeqc_basis_representation representation) {
  require(representation == GENERATIVEQC_BASIS_CARTESIAN ||
              representation == GENERATIVEQC_BASIS_SPHERICAL,
          "invalid basis representation");
  require(!atoms.empty(), "basis template requires atoms");
  count(atoms.size());
  const auto elements = validate(basis);
  // General rows multiply output storage even when their exponents are shared.
  // Check the complete expansion before allocating or normalizing any shell.
  std::uint64_t shell_count = 0, primitive_count = 0, term_count = 0;
  for (const auto& atom : atoms) {
    require(atom.atomic_number > 0 && atom.atomic_number <= 118 && elements[atom.atomic_number],
            "basis template is missing an input element");
    const auto& element = *elements[atom.atomic_number];
    for (std::uint32_t index = 0; index < element.shell_count; ++index) {
      const auto& shell = basis.shells[element.shell_offset + index];
      shell_count = count(shell_count + shell.contraction_count);
      primitive_count =
          count(primitive_count + std::uint64_t(shell.primitive_count) * shell.contraction_count);
    }
    term_count = count(term_count + element.ecp_term_count);
  }

  ExpandedBasis result;
  result.representation = representation;
  result.shells.reserve(shell_count);
  result.primitives.reserve(primitive_count);
  result.ecp_terms.reserve(term_count);
  result.core_electrons.reserve(atoms.size());
  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
    const auto& element = *elements[atoms[atom_index].atomic_number];
    result.core_electrons.push_back(element.ecp_core_electrons);
    for (std::uint32_t index = 0; index < element.shell_count; ++index) {
      const auto& shell = basis.shells[element.shell_offset + index];
      require(term_count == 0 || shell.angular_momentum <= 3,
              "ECP execution requires s/p/d/f orbital shells");
      for (std::uint32_t row = 0; row < shell.contraction_count; ++row) {
        const auto offset = count(result.primitives.size());
        for (std::uint32_t primitive = 0; primitive < shell.primitive_count; ++primitive)
          result.primitives.push_back(
              {basis.exponents[shell.exponent_offset + primitive],
               basis.coefficients[shell.coefficient_offset +
                                  std::uint64_t(row) * shell.primitive_count + primitive]});
        result.shells.push_back(
            {count(atom_index), shell.angular_momentum, offset, shell.primitive_count});
      }
    }
    // Source channel order is arbitrary; the highest channel is the local one.
    std::uint32_t local_channel = 0;
    for (std::uint32_t index = 0; index < element.ecp_term_count; ++index)
      local_channel = std::max(local_channel,
                               basis.ecp_terms[element.ecp_term_offset + index].angular_momentum);
    require(local_channel <= 4, "ECP local channel supports at most g");
    for (std::uint32_t index = 0; index < element.ecp_term_count; ++index) {
      const auto& term = basis.ecp_terms[element.ecp_term_offset + index];
      require(term.power <= 4, "ECP radial powers support 0..4");
      result.ecp_terms.push_back({count(atom_index),
                                  term.angular_momentum == local_channel
                                      ? -1
                                      : static_cast<std::int32_t>(term.angular_momentum),
                                  term.power, term.exponent, term.coefficient});
    }
  }
  return result;
}

}  // namespace generativeqc::molecule
