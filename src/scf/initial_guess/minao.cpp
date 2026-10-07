#include "scf/initial_guess/minao.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/initial_guess/minao_data.hpp"

namespace generativeqc::scf::initial_guess {
namespace {

using minao_data::kElementShellOffsets;
using minao_data::kPrimitives;
using minao_data::kShells;

void validate_domain(const core::System& system) {
  if (!system.ecp_terms.empty() ||
      std::any_of(system.atoms.begin(), system.atoms.end(),
                  [](const auto& atom) { return atom.ecp_core != 0; }))
    throw std::invalid_argument("MINAO initial guess is qualified only for all-electron systems");
  for (const auto& atom : system.atoms)
    if (atom.atomic_number < 1 || atom.atomic_number > 18)
      throw std::invalid_argument("MINAO initial guess is currently qualified for H-Ar");
}

struct Source {
  core::System system;
  std::vector<double> occupations;
  std::size_t primitive_count{};
};

Source source_system(const core::System& target) {
  validate_domain(target);
  Source result;
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

double electron_trace(const Matrix& density, const Matrix& overlap, std::size_t n) {
  double value = 0.0;
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < n; ++column)
      value += density[row * n + column] * overlap[column * n + row];
  return value;
}

}  // namespace

std::size_t minao_source_ao_count(const core::System& system) {
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

std::size_t minao_source_primitive_count(const core::System& system) {
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

MinaoDensityResult minao_density(const core::System& system,
                                 const integrals::IntegralData& target,
                                 const Matrix& target_orthogonalizer) {
  const std::size_t n = target.nbf;
  if (!n || target.overlap.size() != n * n ||
      target_orthogonalizer.size() != n * n)
    throw std::invalid_argument("MINAO target overlap/orthogonalizer shape mismatch");

  auto source = source_system(system);
  const std::size_t ns = source.occupations.size();
  std::vector<double> cross(n * ns, 0.0);
  integrals::cross_overlap(system, source.system, cross);

  // PySCF project_mo_nr2nr: P = S_tt^-1 S_ts = X X^T S_ts.
  // The native symmetric orthogonalizer X is row-major and symmetric.
  std::vector<double> transformed(n * ns, 0.0);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t source_ao = 0; source_ao < ns; ++source_ao) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < n; ++inner)
        value += target_orthogonalizer[inner * n + row] * cross[inner * ns + source_ao];
      transformed[row * ns + source_ao] = value;
    }

  std::fill(cross.begin(), cross.end(), 0.0);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t source_ao = 0; source_ao < ns; ++source_ao) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < n; ++inner)
        value += target_orthogonalizer[row * n + inner] * transformed[inner * ns + source_ao];
      cross[row * ns + source_ao] = value;
    }

  Matrix density(n * n, 0.0);
  double source_electrons = 0.0;
  for (std::size_t source_ao = 0; source_ao < ns; ++source_ao) {
    const double occupation = source.occupations[source_ao];
    source_electrons += occupation;
    if (occupation == 0.0) continue;
    for (std::size_t row = 0; row < n; ++row) {
      const double left = occupation * cross[row * ns + source_ao];
      for (std::size_t column = 0; column < n; ++column)
        density[row * n + column] += left * cross[column * ns + source_ao];
    }
  }
  if (!std::all_of(density.begin(), density.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("MINAO projection produced a nonfinite density");

  return {std::move(density), ns, source.primitive_count, source_electrons,
          electron_trace(density, target.overlap, n)};
}

}  // namespace generativeqc::scf::initial_guess
