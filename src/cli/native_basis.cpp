#include "cli/native_basis.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "cli/json.hpp"
#include "cli/options.hpp"
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

namespace {
void basis_require(bool valid, std::string_view message) {
  if (!valid) throw UsageError(std::string(message));
}

std::uint32_t basis_integer(const Json& value, std::uint32_t maximum, std::string_view name) {
  basis_require(value.is_number_integer() && value >= 0 && value <= maximum,
                std::string(name) + " must be an integer within native bounds");
  return value.get<std::uint32_t>();
}

double decimal(const Json& value, bool positive) {
  basis_require(value.is_string(), "canonical basis decimal fields must be strings");
  const auto text = value.get<std::string>();
  basis_require(!text.empty() && text.find_first_not_of("0123456789+-.eE") == text.npos,
                "invalid canonical basis decimal");
  char* end = nullptr;
  const double result = std::strtod(text.c_str(), &end);
  basis_require(end == text.c_str() + text.size() && std::isfinite(result),
                "nonfinite or invalid basis decimal");
  const auto mantissa = text.substr(0, text.find_first_of("eE"));
  basis_require(result != 0 || mantissa.find_first_of("123456789") == mantissa.npos,
                "nonzero basis decimal underflows FP64");
  basis_require(!positive || result > 0, "basis exponents must be positive");
  return result;
}

// Validate every ECP record on load, even for elements absent from this geometry.
// The largest angular channel is the local potential in the canonical schema.
Json ecp_potentials(const Json& element) {
  const auto potentials = parse_json(element.at("ecp_data").get<std::string>());
  basis_require(potentials.is_array() && !potentials.empty(), "invalid ECP potential records");
  std::set<std::uint32_t> channels;
  for (const auto& potential : potentials) {
    basis_require(potential.is_object() && potential.size() == 5 &&
                      potential.at("ecp_type") == "scalar_ecp" &&
                      potential.at("angular_momentum").is_array() &&
                      potential.at("angular_momentum").size() == 1,
                  "unsupported scalar ECP fields or channels");
    const auto channel =
        basis_integer(potential.at("angular_momentum")[0], 4, "ECP angular channel");
    basis_require(channels.insert(channel).second, "duplicate ECP angular channel");
    const auto& exponents = potential.at("gaussian_exponents");
    const auto& powers = potential.at("r_exponents");
    const auto& coefficients = potential.at("coefficients");
    basis_require(exponents.is_array() && !exponents.empty() && powers.is_array() &&
                      powers.size() == exponents.size() && coefficients.is_array() &&
                      coefficients.size() == 1 && coefficients[0].is_array() &&
                      coefficients[0].size() == exponents.size(),
                  "invalid scalar ECP radial arrays");
    for (std::size_t index = 0; index < exponents.size(); ++index) {
      basis_integer(powers[index], 4, "ECP radial power");
      decimal(exponents[index], true);
      decimal(coefficients[0][index], false);
    }
  }
  return potentials;
}

Json local_basis(std::string_view name) {
  auto basis = read_json(std::filesystem::canonical(std::filesystem::path(name)));
  const auto checksum = basis.at("checksum");
  basis.erase("checksum");
  basis_require(checksum.is_string() && checksum == canonical_hash(basis),
                "canonical basis checksum mismatch");
  basis_require(
      basis.is_object() && basis.size() == 9 && basis.at("schema") == "generativeqc.basis" &&
          basis.at("schema_version").is_number_integer() && basis.at("schema_version") == 1,
      "unsupported canonical basis schema or fields");
  basis_require(
      basis.at("exponent_units") == "bohr^-2" &&
          basis.at("normalization") == "normalized-primitives; native-normalized-contractions" &&
          basis.at("ordering") == "CCA-cartesian; libcint-real-spherical",
      "unsupported basis units/normalization/ordering");
  basis_require(basis.at("name").is_string() && !basis.at("name").get<std::string>().empty(),
                "basis requires a name");
  basis_require(
      basis.at("representation") == "cartesian" || basis.at("representation") == "spherical",
      "invalid canonical basis representation");
  const auto& provenance = basis.at("provenance");
  basis_require(provenance.is_object() && provenance.size() == 4,
                "basis requires exact source provenance");
  for (const auto field : {"source", "version", "license"})
    basis_require(
        provenance.at(field).is_string() && !provenance.at(field).get<std::string>().empty(),
        "missing basis provenance");
  basis_require(provenance.at("checksum").is_string() &&
                    is_digest(provenance.at("checksum").get<std::string>()),
                "invalid basis source checksum");
  const auto& elements = basis.at("elements");
  basis_require(elements.is_array() && !elements.empty(), "basis requires element records");
  std::set<std::uint32_t> seen;
  for (const auto& element : elements) {
    basis_require(element.is_object() && element.size() == 5, "invalid canonical element fields");
    const auto atomic_number = basis_integer(element.at("atomic_number"), 118, "atomic number");
    basis_require(atomic_number > 0 && seen.insert(atomic_number).second,
                  "invalid or duplicate basis element");
    basis_require(
        basis_integer(element.at("nuclear_charge"), 118, "nuclear charge") == atomic_number,
        "native basis requires nuclear charge to match element identity");
    const auto core =
        basis_integer(element.at("ecp_core_electrons"), atomic_number - 1, "ECP core count");
    basis_require(core ? element.at("ecp_data").is_string() : element.at("ecp_data").is_null(),
                  "ECP cores and metadata must appear together");
    if (core) ecp_potentials(element);
    const auto& shells = element.at("shells");
    basis_require(shells.is_array() && !shells.empty(), "basis element requires shells");
    for (const auto& shell : shells) {
      basis_require(shell.is_object() && shell.size() == 4, "invalid canonical shell fields");
      basis_integer(shell.at("angular_momentum"), 32, "shell angular momentum");
      basis_integer(shell.at("source_group"), std::numeric_limits<std::uint32_t>::max(),
                    "shell source group");
      const auto& exponents = shell.at("exponents");
      const auto& rows = shell.at("coefficients");
      basis_require(exponents.is_array() && !exponents.empty() && rows.is_array() && !rows.empty(),
                    "basis requires exponents and contractions");
      for (const auto& exponent : exponents) decimal(exponent, true);
      for (const auto& row : rows) {
        basis_require(row.is_array() && row.size() == exponents.size(),
                      "coefficient columns must match primitive exponents");
        bool nonzero = false;
        for (const auto& coefficient : row) nonzero = decimal(coefficient, false) != 0 || nonzero;
        basis_require(nonzero, "identically zero contraction cannot be normalized");
      }
    }
  }
  basis["checksum"] = checksum;
  return basis;
}
}  // namespace

bool is_local_basis(std::string_view name) {
  return name.ends_with(".json") || std::filesystem::exists(std::filesystem::path(name));
}

NativeBasisData expand_basis_input(
    std::string_view name, std::span<const generativeqc_atom> atoms,
    std::optional<generativeqc_basis_representation> representation) {
  if (!is_local_basis(name))
    return expand_bundled_basis(name, atoms, representation.value_or(GENERATIVEQC_BASIS_CARTESIAN));
  const auto record = local_basis(name);
  NativeBasisData result;
  result.representation = record.at("representation") == "spherical" ? GENERATIVEQC_BASIS_SPHERICAL
                                                                     : GENERATIVEQC_BASIS_CARTESIAN;
  basis_require(!representation || *representation == result.representation,
                "explicit representation conflicts with canonical basis record");
  result.identity = record.at("checksum");
  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
    const Json* element = nullptr;
    for (const auto& candidate : record.at("elements"))
      if (candidate.at("atomic_number") == atoms[atom_index].atomic_number) element = &candidate;
    basis_require(element != nullptr, "local basis is missing an input element");
    for (const auto& shell : element->at("shells")) {
      const auto angular = shell.at("angular_momentum").get<std::uint32_t>();
      for (const auto& row : shell.at("coefficients")) {
        basis_require(
            result.primitives.size() <= std::numeric_limits<std::uint32_t>::max() - row.size() &&
                result.shells.size() < std::numeric_limits<std::uint32_t>::max(),
            "expanded basis exceeds native uint32 dimensions");
        const auto offset = static_cast<std::uint32_t>(result.primitives.size());
        for (std::size_t index = 0; index < row.size(); ++index)
          result.primitives.push_back(
              {decimal(shell.at("exponents")[index], true), decimal(row[index], false)});
        result.shells.push_back({static_cast<std::uint32_t>(atom_index), angular, offset,
                                 static_cast<std::uint32_t>(row.size())});
      }
    }
    const auto core = element->at("ecp_core_electrons").get<std::int32_t>();
    result.core_electrons.push_back(core);
    if (!core) continue;
    const auto potentials = ecp_potentials(*element);
    std::set<std::uint32_t> channels;
    for (const auto& potential : potentials) {
      channels.insert(potential.at("angular_momentum")[0].get<std::uint32_t>());
    }
    const auto local = *channels.rbegin();
    for (const auto& potential : potentials) {
      const auto& exponents = potential.at("gaussian_exponents");
      const auto& powers = potential.at("r_exponents");
      const auto& rows = potential.at("coefficients");
      const auto channel = potential.at("angular_momentum")[0].get<std::uint32_t>();
      for (std::size_t index = 0; index < exponents.size(); ++index)
        result.ecp_terms.push_back({static_cast<std::uint32_t>(atom_index),
                                    channel == local ? -1 : static_cast<std::int32_t>(channel),
                                    basis_integer(powers[index], 4, "ECP radial power"),
                                    decimal(exponents[index], true),
                                    decimal(rows[0][index], false)});
    }
  }
  if (!result.ecp_terms.empty())
    for (const auto& shell : result.shells)
      basis_require(shell.angular_momentum <= 3, "ECP execution requires s/p/d/f orbital shells");
  return result;
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
