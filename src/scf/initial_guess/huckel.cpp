#include "scf/initial_guess/huckel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/types.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/huckel_capacity.hpp"
#include "scf/initial_guess/overlap.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/cpu_target_eigen.hpp"
#include "scf/solver/diis.hpp"

namespace generativeqc::scf::initial_guess {
namespace {
struct AtomicProblem {
  core::System system;
  std::vector<std::size_t> target_aos;
};
struct AtomicOrbitals {
  core::System system;
  Matrix coefficients;
  std::vector<double> energies;
};
struct Orbital {
  std::vector<std::pair<std::size_t, double>> coefficients;
  double energy{};
};

AtomicProblem atomic_problem(const core::System& system, std::size_t atom) {
  const auto configuration = huckel_atomic_configuration(system.atoms[atom].atomic_number);
  AtomicProblem result;
  result.system.atoms = {{system.atoms[atom].atomic_number, {0, 0, 0}}};
  result.system.electron_count = system.atoms[atom].atomic_number;
  result.system.multiplicity = 1 + result.system.electron_count % 2;
  result.system.basis_representation = system.basis_representation;
  std::size_t offset = 0;
  for (const auto& shell : system.shells) {
    const auto count = system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * shell.angular_momentum + 1
                           : molecule::cartesian_count(shell.angular_momentum);
    const bool occupied_channel =
        shell.angular_momentum == 0 || (shell.angular_momentum == 1 && configuration.second != 0);
    if (shell.atom_index == atom && occupied_channel) {
      result.system.shells.push_back({0, shell.angular_momentum, shell.primitives});
      for (std::size_t component = 0; component < count; ++component)
        result.target_aos.push_back(offset + component);
    }
    offset += count;
  }
  return result;
}

bool same_problem(const core::System& first, const core::System& second) {
  if (first.electron_count != second.electron_count || first.shells.size() != second.shells.size())
    return false;
  for (std::size_t shell = 0; shell < first.shells.size(); ++shell) {
    const auto& left = first.shells[shell];
    const auto& right = second.shells[shell];
    if (left.angular_momentum != right.angular_momentum ||
        left.primitives.size() != right.primitives.size())
      return false;
    for (std::size_t primitive = 0; primitive < left.primitives.size(); ++primitive)
      if (left.primitives[primitive].exponent != right.primitives[primitive].exponent ||
          left.primitives[primitive].coefficient != right.primitives[primitive].coefficient)
        return false;
  }
  return true;
}

struct AtomicFrame {
  Matrix density, coefficients;
  std::vector<double> energies;
};

AtomicFrame atomic_frame(const core::System& system, const Matrix& fock, const Matrix& overlap) {
  const auto dimension = molecule::ao_count(system);
  const auto [s_electrons, p_electrons] = huckel_atomic_configuration(system.electron_count);
  AtomicFrame frame;
  frame.density.assign(dimension * dimension, 0.0);
  std::vector<std::vector<double>> columns;
  for (unsigned angular = 0; angular < 2; ++angular) {
    const auto electrons = angular == 0 ? s_electrons : p_electrons;
    if (!electrons) continue;
    const std::size_t degeneracy = 2 * angular + 1;
    std::vector<std::size_t> indices;
    std::size_t offset = 0;
    for (const auto& shell : system.shells) {
      const auto count = 2 * shell.angular_momentum + 1;
      if (shell.angular_momentum == angular) indices.push_back(offset);
      offset += count;
    }
    const auto radial = indices.size();
    const auto occupied = (electrons + 2 * degeneracy - 1) / (2 * degeneracy);
    if (radial < occupied) throw std::invalid_argument("Huckel atomic basis lacks occupied shells");
    Matrix radial_fock(radial * radial, 0.0), radial_overlap(radial * radial, 0.0);
    for (std::size_t row = 0; row < radial; ++row)
      for (std::size_t column = 0; column < radial; ++column)
        for (std::size_t component = 0; component < degeneracy; ++component) {
          const auto index = (indices[row] + component) * dimension + indices[column] + component;
          radial_fock[row * radial + column] += fock[index] / degeneracy;
          radial_overlap[row * radial + column] += overlap[index] / degeneracy;
        }
    const auto orthogonalizer = symmetric_overlap(radial_overlap, radial, solver::cpu_target_eigen);
    const auto eigen =
        solver::cpu_target_eigen(radial_fock, &radial_overlap, &orthogonalizer, radial);
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      const auto remaining = electrons - std::min<std::size_t>(electrons, 2 * degeneracy * orbital);
      const double occupation = std::min<double>(2.0, static_cast<double>(remaining) / degeneracy);
      for (std::size_t component = 0; component < degeneracy; ++component) {
        std::vector<double> coefficients(dimension, 0.0);
        for (std::size_t row = 0; row < radial; ++row)
          coefficients[indices[row] + component] = eigen.vectors[row * radial + orbital];
        for (std::size_t row = 0; row < dimension; ++row)
          for (std::size_t column = 0; column < dimension; ++column)
            frame.density[row * dimension + column] +=
                occupation * coefficients[row] * coefficients[column];
        columns.push_back(std::move(coefficients));
        frame.energies.push_back(eigen.values[orbital]);
      }
    }
  }
  frame.coefficients.resize(dimension * columns.size());
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column < columns.size(); ++column)
      frame.coefficients[row * columns.size() + column] = columns[column][row];
  return frame;
}

AtomicOrbitals solve_atom(core::System system, const PreliminaryOptions& policy,
                          PreliminaryDiagnostic& diagnostic) {
  if (system.electron_count == 1) {
    const auto integrals = integrals::build_integrals(system, false, false);
    auto frame = atomic_frame(system, integrals.hcore, integrals.overlap);
    return {std::move(system), std::move(frame.coefficients), std::move(frame.energies)};
  }
  auto spec = make_hf_fock_spec(FockSpin::Restricted, FockApproximation::Exact);
  spec.derivative_order = 0;
  const PreparedFockPlan plan(system, nullptr, resolve_fock_build(spec, FockBackend::Cpu));
  const auto& integrals = plan.one_electron();
  auto frame = atomic_frame(system, integrals.hcore, integrals.overlap);
  // Hydrogen is the one-electron atomic Hamiltonian, not a fractional-RHF
  // self-interacting surrogate. All unused angular channels decouple exactly
  // from a spherical density and have already been removed before ERI work.
  solver::Diis diis(policy.diis_history, true);
  double previous_energy = INFINITY;
  for (unsigned iteration = 0; iteration < policy.max_iterations; ++iteration) {
    auto fock = assemble_fock(plan.strategy(), integrals.hcore, plan.build(frame.density)).alpha;
    ++diagnostic.preliminary_fock_builds;
    ++diagnostic.preliminary_iterations;
    const auto energy = reference::electronic_energy(frame.density, integrals.hcore, fock);
    const auto residual =
        reference::commutator_residual(fock, frame.density, integrals.overlap, integrals.nbf);
    auto physical = atomic_frame(system, fock, integrals.overlap);
    // Atomic eigenvalues enter a different Hamiltonian. Do not let loose
    // preliminary molecular tolerances perturb that reproducible model.
    if (std::abs(energy - previous_energy) < std::min(policy.energy_tolerance, 1e-10) &&
        reference::density_rms(frame.density, physical.density) <
            std::min(policy.density_tolerance, 1e-8) &&
        reference::residual_max_abs(residual) < 1e-7)
      return {std::move(system), std::move(physical.coefficients), std::move(physical.energies)};
    previous_energy = energy;
    frame = atomic_frame(system, diis.update(fock, residual), integrals.overlap);
  }
  throw std::runtime_error("Huckel spherical atomic RHF exhausted its iteration budget");
}
}  // namespace

std::pair<unsigned, unsigned> huckel_atomic_configuration(int atomic_number) {
  if (atomic_number < 1 || atomic_number > 18)
    throw std::invalid_argument("Huckel initial guess is currently qualified for H-Ar");
  if (atomic_number <= 4) return {static_cast<unsigned>(atomic_number), 0};
  if (atomic_number <= 10) return {4, static_cast<unsigned>(atomic_number - 4)};
  if (atomic_number <= 12) return {static_cast<unsigned>(atomic_number - 6), 6};
  return {6, static_cast<unsigned>(atomic_number - 6)};
}

std::size_t huckel_maximum_atomic_aos(const core::System& system) {
  return huckel_capacity_shape(system).atomic_aos;
}

HuckelDensityResult huckel_density(const core::System& system,
                                   const integrals::IntegralData& target,
                                   const PreliminaryOptions& policy,
                                   PreliminaryDiagnostic& diagnostic) {
  const auto dimension = target.nbf;
  if (dimension == 0 || target.overlap.size() != dimension * dimension ||
      molecule::ao_count(system) != dimension || system.electron_count <= 0 ||
      system.electron_count % 2 || system.multiplicity != 1 || !system.ecp_terms.empty() ||
      std::any_of(system.atoms.begin(), system.atoms.end(),
                  [](const auto& atom) { return atom.ecp_core != 0; }))
    throw std::invalid_argument("Huckel requires a matching all-electron restricted target");
  std::vector<AtomicOrbitals> cache;
  std::vector<Orbital> orbitals;
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
    auto problem = atomic_problem(system, atom);
    auto found = std::find_if(cache.begin(), cache.end(), [&](const auto& item) {
      return same_problem(item.system, problem.system);
    });
    if (found == cache.end()) {
      cache.push_back(solve_atom(std::move(problem.system), policy, diagnostic));
      found = cache.end() - 1;
    }
    const auto atomic_n = problem.target_aos.size(), occupied = found->energies.size();
    for (std::size_t column = 0; column < occupied; ++column) {
      Orbital orbital;
      orbital.energy = found->energies[column];
      for (std::size_t row = 0; row < atomic_n; ++row) {
        const auto value = found->coefficients[row * occupied + column];
        if (value != 0.0) orbital.coefficients.emplace_back(problem.target_aos[row], value);
      }
      orbitals.push_back(std::move(orbital));
    }
  }
  const auto minimal = orbitals.size(),
             occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (minimal < occupied)
    throw std::invalid_argument("Huckel occupied atomic subspace cannot hold target electrons");
  Matrix overlap(minimal * minimal), hamiltonian(minimal * minimal);
  for (std::size_t row = 0; row < minimal; ++row)
    for (std::size_t column = 0; column <= row; ++column) {
      double value = 0.0;
      for (const auto& [left, coefficient_left] : orbitals[row].coefficients)
        for (const auto& [right, coefficient_right] : orbitals[column].coefficients)
          value += coefficient_left * target.overlap[left * dimension + right] * coefficient_right;
      overlap[row * minimal + column] = overlap[column * minimal + row] = value;
      const auto sum = orbitals[row].energy + orbitals[column].energy;
      if (!std::isfinite(sum) || std::abs(sum) < 1e-14)
        throw std::runtime_error("Huckel atomic energies cannot define the modified GWH rule");
      const auto delta = (orbitals[row].energy - orbitals[column].energy) / sum;
      const auto square = delta * delta;
      const auto coupling = 0.5 * (1.75 + square - 0.75 * square * square) * value * sum;
      hamiltonian[row * minimal + column] = hamiltonian[column * minimal + row] =
          row == column ? orbitals[row].energy : coupling;
    }
  // Only the minimal occupied-atomic subspace is diagonalized on the native
  // CPU production provider. Sparse atomic supports avoid dense N-by-N-by-M
  // projection and do not assume atoms or shells are contiguous in the target.
  const auto orthogonalizer = symmetric_overlap(overlap, minimal, solver::cpu_preliminary_eigen);
  const auto frame = solver::cpu_preliminary_eigen(hamiltonian, &overlap, &orthogonalizer, minimal);
  Matrix coefficients(dimension * occupied, 0.0);
  for (std::size_t orbital = 0; orbital < minimal; ++orbital)
    for (const auto& [row, value] : orbitals[orbital].coefficients)
      for (std::size_t column = 0; column < occupied; ++column)
        coefficients[row * occupied + column] += value * frame.vectors[orbital * minimal + column];
  auto density = solver::cpu_preliminary_density(coefficients, dimension, occupied);
  return {std::move(density), cache.size(), minimal};
}
}  // namespace generativeqc::scf::initial_guess
