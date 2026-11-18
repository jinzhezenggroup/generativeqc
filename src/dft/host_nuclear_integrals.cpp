#include "dft/host_nuclear_integrals.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dft/host_hessian_aot.hpp"
#include "generated_one_electron_cpu.hpp"
#include "host_factorized_second.hpp"
#include "host_nuclear_integral_maps.hpp"
#include "integrals/eri_symmetry.hpp"
#include "integrals/host_factorized_primitives.hpp"
#include "integrals/host_mixed_primitives.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/host_component_trace.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_host_integral_maps;
namespace primitive = integrals::generated_one_electron_cpu;

std::size_t add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("host nuclear integral size overflow");
  return left + right;
}

std::size_t multiply(std::size_t left, std::size_t right) {
  if (right && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error("host nuclear integral size overflow");
  return left * right;
}

std::size_t integer(double value, std::size_t maximum) {
  if (!std::isfinite(value) || value < 0 || value > static_cast<double>(maximum) ||
      std::trunc(value) != value)
    throw std::invalid_argument("invalid host nuclear integral AO record");
  return static_cast<std::size_t>(value);
}

struct Ao {
  std::size_t atom{}, begin{}, count{}, terms{};
  const double* record{};
};

std::size_t offset(const AoBasis& basis) {
  if (!basis.natom || !basis.nao || !basis.nprimitive || basis.natom > INT32_MAX ||
      basis.nao > INT32_MAX || basis.nprimitive > INT32_MAX)
    throw std::invalid_argument("invalid host nuclear integral basis dimensions");
  const auto result = add(multiply(3, basis.natom), multiply(2, basis.nprimitive));
  if (basis.packed.size() != add(result, multiply(16, basis.nao)))
    throw std::invalid_argument("invalid host nuclear integral basis packing");
  if (!maps::finite(basis.packed.data(), basis.packed.size()))
    throw std::domain_error("nonfinite host nuclear integral basis");
  for (std::size_t index = 0; index < basis.nprimitive; ++index)
    if (basis.packed[3 * basis.natom + 2 * index] <= 0)
      throw std::invalid_argument("host nuclear integral exponent must be positive");
  return result;
}

Ao orbital(const AoBasis& basis, std::size_t base, std::size_t index) {
  const auto* record = basis.packed.data() + base + 16 * index;
  Ao result{integer(record[0], basis.natom - 1), integer(record[1], basis.nprimitive), 0,
            integer(record[3], 3), record};
  result.count = integer(record[2], basis.nprimitive - result.begin);
  if (!result.count || !result.terms) throw std::invalid_argument("empty host nuclear integral AO");
  for (std::size_t term = 0; term < result.terms; ++term) {
    std::size_t angular{};
    for (std::size_t axis = 0; axis < 3; ++axis) angular += integer(record[4 + 4 * term + axis], 3);
    if (angular > 3) throw std::invalid_argument("host nuclear integrals support through f");
  }
  return result;
}

struct PolynomialAo {
  std::size_t atom{}, begin{}, count{};
  unsigned degree{}, component{};
  bool operator==(const PolynomialAo&) const = default;
};

struct PolynomialLayout {
  std::vector<PolynomialAo> orbitals;
  std::vector<std::size_t> public_offsets;
  bool operator==(const PolynomialLayout&) const = default;
};

/** The public polynomial coefficients already include Cartesian normalization.
 * Keep those factors in T and use unit-coefficient monomials internally. This
 * changes only the contraction representation, not the physical AO subspace.
 * Normalized radial ranges identify adjacent shell records in the native pack.
 */
PolynomialLayout polynomial_layout(const AoBasis& basis, std::size_t base) {
  PolynomialLayout result;
  result.public_offsets.reserve(basis.nao);
  Ao previous;
  unsigned previous_degree = 4;
  std::size_t shell_offset{};
  for (std::size_t index = 0; index < basis.nao; ++index) {
    const auto ao = orbital(basis, base, index);
    unsigned degree{};
    for (std::size_t term = 0; term < ao.terms; ++term) {
      const auto* expansion = ao.record + 4 + 4 * term;
      const auto current = static_cast<unsigned>(expansion[0] + expansion[1] + expansion[2]);
      if (term && current != degree)
        throw std::invalid_argument("projected integral shell mixes polynomial degrees");
      degree = current;
    }
    if (index == 0 || ao.atom != previous.atom || ao.begin != previous.begin ||
        ao.count != previous.count || degree != previous_degree) {
      shell_offset = result.orbitals.size();
      const auto components = (degree + 1) * (degree + 2) / 2;
      for (unsigned component = 0; component < components; ++component)
        result.orbitals.push_back({ao.atom, ao.begin, ao.count, degree, component});
    }
    result.public_offsets.push_back(shell_offset);
    previous = ao;
    previous_degree = degree;
  }
  return result;
}

std::size_t polynomial_component(const double* expansion) {
  const auto degree = static_cast<unsigned>(expansion[0] + expansion[1] + expansion[2]);
  return primitive::component_index(expansion[0], expansion[1], expansion[2]) -
         degree * (degree + 1) * (degree + 2) / 6;
}

template <class Consumer>
void integral_quartets(std::size_t count, HostNuclearEriSchedule schedule, Consumer&& consume) {
  if (schedule == HostNuclearEriSchedule::SymmetryUnique) {
    integrals::for_each_unique_eri_quartet(count, consume);
  } else if (schedule == HostNuclearEriSchedule::Ordered) {
    for (std::size_t first = 0; first < count; ++first)
      for (std::size_t second = 0; second < count; ++second)
        for (std::size_t third = 0; third < count; ++third)
          for (std::size_t fourth = 0; fourth < count; ++fourth)
            consume(std::array<std::size_t, 4>{first, second, third, fourth});
  } else {
    throw std::invalid_argument("invalid host nuclear ERI schedule");
  }
}

template <class Consumer>
void quartet_incidences(const std::array<std::size_t, 4>& indices, HostNuclearEriSchedule schedule,
                        Consumer&& consume) {
  if (schedule == HostNuclearEriSchedule::SymmetryUnique)
    integrals::for_each_eri_permutation(indices, consume);
  else
    consume(indices);
}

/** Count actual radial/Cartesian primitive evaluations, without enumerating AO4.
 * Pair-prefix products give each triangular pair-of-pairs exactly once. No
 * integral screening or density-dependent work is hidden in this envelope.
 */
std::size_t eri_primitive_work(const AoBasis& basis, std::size_t base,
                               HostNuclearEriSchedule schedule) {
  if (schedule != HostNuclearEriSchedule::Ordered &&
      schedule != HostNuclearEriSchedule::SymmetryUnique)
    throw std::invalid_argument("invalid host nuclear ERI schedule");
  std::size_t prefix{}, work{};
  for (std::size_t first = 0; first < basis.nao; ++first) {
    const auto first_ao = orbital(basis, base, first);
    const auto first_terms = multiply(first_ao.count, first_ao.terms);
    if (schedule == HostNuclearEriSchedule::Ordered) {
      prefix = add(prefix, first_terms);
      continue;
    }
    for (std::size_t second = 0; second <= first; ++second) {
      const auto second_ao = orbital(basis, base, second);
      const auto pair = multiply(first_terms, multiply(second_ao.count, second_ao.terms));
      prefix = add(prefix, pair);
      work = add(work, multiply(pair, prefix));
    }
  }
  if (schedule == HostNuclearEriSchedule::Ordered) {
    const auto pairs = multiply(prefix, prefix);
    return multiply(pairs, pairs);
  }
  return work;
}

/** Price the matrix schedule's physical-center directions without an AO4 walk.
 * A canonical quartet needs one value pass on one atom, otherwise three mixed
 * passes per non-origin atom. Weighted pair prefixes count all quartets and
 * each atom's complement. Summing their differences counts distinct owners,
 * including diagonal stabilizers and radial/Cartesian multiplicities, in
 * O(natom * nao^2) rather than an AO4 or atom-pair-of-pairs walk. Equal numeric
 * coordinates never identify distinct physical atoms.
 */
std::size_t eri_matrix_primitive_work(const AoBasis& basis, std::size_t base,
                                      HostNuclearEriSchedule schedule,
                                      const PolynomialLayout* projected = nullptr) {
  if (!projected && schedule != HostNuclearEriSchedule::SymmetryUnique)
    return multiply(9, eri_primitive_work(basis, base, schedule));
  if (schedule != HostNuclearEriSchedule::SymmetryUnique &&
      schedule != HostNuclearEriSchedule::Ordered)
    throw std::invalid_argument("invalid projected nuclear ERI schedule");
  const auto count = projected ? projected->orbitals.size() : basis.nao;
  const auto get = [&](std::size_t index) {
    if (!projected) return orbital(basis, base, index);
    const auto& ao = projected->orbitals[index];
    return Ao{ao.atom, ao.begin, ao.count, 1, nullptr};
  };
  struct PairWork {
    std::size_t total{}, triangular{};
  };
  const auto accumulate = [](PairWork& work, std::size_t pair) {
    work.total = add(work.total, pair);
    work.triangular = add(work.triangular, multiply(pair, work.total));
  };
  PairWork all;
  std::vector<PairWork> absent(basis.natom), local(basis.natom);
  std::size_t ordered_terms{};
  for (std::size_t first = 0; first < count; ++first) {
    const auto first_ao = get(first);
    const auto first_terms = multiply(first_ao.count, first_ao.terms);
    if (schedule == HostNuclearEriSchedule::Ordered) {
      ordered_terms = add(ordered_terms, first_terms);
      continue;
    }
    for (std::size_t second = 0; second <= first; ++second) {
      const auto second_ao = get(second);
      const auto pair = multiply(first_terms, multiply(second_ao.count, second_ao.terms));
      accumulate(all, pair);
      if (first_ao.atom == second_ao.atom) accumulate(local[first_ao.atom], pair);
      for (std::size_t atom = 0; atom < basis.natom; ++atom)
        if (atom != first_ao.atom && atom != second_ao.atom) accumulate(absent[atom], pair);
    }
  }
  if (schedule == HostNuclearEriSchedule::Ordered) {
    const auto pairs = multiply(ordered_terms, ordered_terms);
    return multiply(9, multiply(pairs, pairs));
  }
  std::size_t incidences{}, single_center{};
  for (std::size_t atom = 0; atom < basis.natom; ++atom) {
    incidences = add(incidences, all.triangular - absent[atom].triangular);
    single_center = add(single_center, local[atom].triangular);
  }
  return add(multiply(3, incidences - all.triangular), single_center);
}

const maps::Program& map(std::size_t spins) {
  if (spins != 1 && spins != 2) throw std::invalid_argument("invalid host nuclear integral spin");
  const auto name = spins == 1 ? "quartet_unpolarized" : "quartet_polarized";
  for (const auto& entry : maps::programs())
    if (entry.name == name) return entry;
  throw std::invalid_argument("missing linked host nuclear quartet map");
}

const maps::Program& weight_map(std::string_view source, std::size_t spins, bool response = false) {
  if (spins != 1 && spins != 2) throw std::invalid_argument("invalid host integral curvature spin");
  const auto name = std::string(source) + (response ? "_response_weight_" : "_weight_") +
                    (spins == 1 ? "unpolarized" : "polarized");
  for (const auto& entry : maps::programs())
    if (entry.name == name && entry.outputs.size() == (response ? 2u : 1u)) return entry;
  throw std::invalid_argument("missing linked host integral curvature weight map");
}

const maps::Program& density_response_map(std::size_t spins) {
  if (spins != 1 && spins != 2)
    throw std::invalid_argument("invalid host integral moving-density spin");
  const auto name = spins == 1 ? "density_response_unpolarized" : "density_response_polarized";
  for (const auto& entry : maps::programs())
    if (entry.name == name && entry.outputs.size() == 1) return entry;
  throw std::invalid_argument("missing linked integral moving-density contraction");
}

bool overlap(const void* first, std::size_t first_bytes, const void* second,
             std::size_t second_bytes) {
  if (!first_bytes || !second_bytes) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first),
             right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}

bool overlaps(std::span<const double> first, std::span<const double> second) {
  return overlap(first.data(), first.size_bytes(), second.data(), second.size_bytes());
}

void finite(double value) {
  if (!std::isfinite(value)) throw std::domain_error("nonfinite host nuclear integral result");
}

void symmetric(std::span<const double> panel, std::size_t basis) {
  if (!maps::finite(panel.data(), panel.size()))
    throw std::domain_error("nonfinite host nuclear matrix");
  for (std::size_t row = 0; row < basis; ++row)
    for (std::size_t column = row + 1; column < basis; ++column) {
      const auto first = panel[row * basis + column], second = panel[column * basis + row];
      if (std::abs(first - second) > 1e-12 + 1e-10 * std::max(std::abs(first), std::abs(second)))
        throw std::domain_error("asymmetric host nuclear integral matrix");
    }
}

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("host nuclear integral reentry");
  }
  ~Busy() { flag.clear(); }
};

}  // namespace

HostNuclearIntegralPlan HostKsNuclearIntegrals::plan(const AoBasis& basis, std::size_t spins,
                                                     HostNuclearEriSchedule schedule,
                                                     HostNuclearFirstSchedule first_schedule) {
  if (first_schedule != HostNuclearFirstSchedule::Direction &&
      first_schedule != HostNuclearFirstSchedule::CoordinatePanel)
    throw std::invalid_argument("invalid host nuclear first-source schedule");
  const auto base = offset(basis);
  std::size_t primitive_terms{};
  for (std::size_t index = 0; index < basis.nao; ++index) {
    const auto ao = orbital(basis, base, index);
    primitive_terms = add(primitive_terms, multiply(ao.count, ao.terms));
  }
  const auto pairs = multiply(primitive_terms, primitive_terms);
  const auto matrix = multiply(basis.nao, basis.nao);
  const auto& entry = map(spins);
  std::size_t bytes{}, terms{};
  const maps::Extents shape{spins, 1};
  if (!entry.workspace(shape, bytes) || !entry.work(shape, terms))
    throw std::overflow_error("host nuclear integral map envelope overflow");
  const auto coordinates = multiply(3, basis.natom);
  const bool panel = first_schedule == HostNuclearFirstSchedule::CoordinatePanel;
  const auto columns = panel ? add(coordinates, 1) : 1;
  auto elements = add(multiply(multiply(spins + 1, matrix), columns), 28 + 8 * spins);
  if (panel)
    elements = add(elements, add(coordinates, add(multiply(3 * spins, matrix),
                                                  add(basis.packed.size(), basis.natom))));
  if (panel) {
    std::size_t response_bytes{};
    if (!density_response_map(spins).workspace({spins, matrix}, response_bytes))
      throw std::overflow_error("host integral moving-density workspace overflow");
    bytes = std::max(bytes, response_bytes);
  }
  bytes = add(bytes, multiply(elements, sizeof(double)));
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("host nuclear integral exceeds signed resource domain");
  return {bytes, multiply(pairs, add(basis.natom, 1)), eri_primitive_work(basis, base, schedule),
          multiply(multiply(matrix, matrix), terms)};
}

bool HostKsNuclearIntegrals::coordinate_density_supported(
    std::span<const double> density) noexcept {
  return std::all_of(density.begin(), density.end(),
                     [](double value) { return std::isfinite(value) && std::abs(value) <= 1e50; });
}

std::size_t HostKsNuclearIntegrals::density_response_terms(const AoBasis& basis,
                                                           std::size_t spins) {
  std::size_t terms{};
  if (!density_response_map(spins).work({spins, multiply(basis.nao, basis.nao)}, terms))
    throw std::overflow_error("host integral moving-density work overflow");
  return multiply(multiply(3, basis.natom), terms);
}

namespace {
class IntegralFrame {
 public:
  IntegralFrame(const core::System& system, const AoBasis& basis,
                const scf::ResolvedFockBuild& strategy, std::span<const double> density,
                std::string_view method_identity, std::function<void()> validate_current)
      : system(system),
        basis(basis),
        strategy(strategy),
        expected_strategy(strategy),
        density(density),
        validator(std::move(validate_current)),
        atoms(basis.natom),
        aos(basis.nao),
        primitives(basis.nprimitive),
        spins(strategy.spec.spin == scf::FockSpin::Restricted ? 1 : 2) {
    if (!validator) throw std::invalid_argument("host nuclear integrals require a live owner");
    const auto* method = host_hessian_aot::find_method(method_identity);
    if (!method || method->spins != spins || method_identity != maps::method_identity(spins) ||
        (strategy.spec.spin != scf::FockSpin::Restricted &&
         strategy.spec.spin != scf::FockSpin::Unrestricted) ||
        strategy.backend != scf::FockBackend::Cpu ||
        strategy.precision != scf::FockPrecision::Float64 || strategy.spec.version != 1 ||
        !strategy.spec.coulomb.present || !strategy.spec.exchange.present ||
        strategy.spec.coulomb.coefficient != 1 ||
        strategy.spec.exchange.coefficient != -method->exact_exchange / (spins == 1 ? 2 : 1) ||
        strategy.spec.coulomb.op != scf::FockOperator::FullRange ||
        strategy.spec.exchange.op != scf::FockOperator::FullRange ||
        strategy.spec.coulomb.omega != 0 || strategy.spec.exchange.omega != 0 ||
        strategy.spec.coulomb.approximation != scf::FockApproximation::Exact ||
        strategy.spec.exchange.approximation != scf::FockApproximation::Exact)
      throw std::invalid_argument(
          "host nuclear integrals require the exact linked CPU full-range method/model");
    if (!density.data() || density.size() != multiply(spins, multiply(aos, aos)))
      throw std::invalid_argument("invalid host nuclear integral density binding");
    retained = {basis.packed.data(), system.atoms.data(), system.shells.data()};
    base = offset(basis);
    validate();
    for (std::size_t index = 0; index < aos; ++index)
      orbitals.push_back(orbital(basis, base, index));
  }

  void validate() const {
    validator();
    if (basis.natom != atoms || basis.nao != aos || basis.nprimitive != primitives ||
        system.atoms.size() != atoms || strategy != expected_strategy ||
        molecule::ao_count(system) != aos || offset(basis) != base ||
        retained != std::array<const void*, 3>{basis.packed.data(), system.atoms.data(),
                                               system.shells.data()})
      throw std::invalid_argument("changed host nuclear integral frame");
    std::size_t primitive_begin{}, ao_begin{};
    for (const auto& shell : system.shells) {
      if (shell.atom_index >= atoms || shell.angular_momentum > 3 || shell.primitives.empty())
        throw std::invalid_argument("invalid host nuclear integral normalized shell");
      for (const auto& radial : shell.primitives) {
        if (primitive_begin >= primitives ||
            basis.packed[3 * atoms + 2 * primitive_begin] != radial.exponent ||
            basis.packed[3 * atoms + 2 * primitive_begin + 1] != radial.coefficient)
          throw std::invalid_argument("host nuclear integral normalized primitive mismatch");
        ++primitive_begin;
      }
      const auto count =
          molecule::shell_ao_count(shell.angular_momentum, system.basis_representation);
      for (std::size_t index = 0; index < count; ++index) {
        if (ao_begin >= aos)
          throw std::invalid_argument("host nuclear integral AO layout mismatch");
        const auto ao = orbital(basis, base, ao_begin++);
        if (ao.atom != shell.atom_index || ao.count != shell.primitives.size() ||
            ao.begin != primitive_begin - ao.count)
          throw std::invalid_argument("host nuclear integral shell map mismatch");
      }
    }
    if (primitive_begin != primitives || ao_begin != aos)
      throw std::invalid_argument("incomplete host nuclear integral basis");
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      if (system.atoms[atom].ionic_charge() < 0)
        throw std::invalid_argument("invalid ionic point charge");
      for (std::size_t axis = 0; axis < 3; ++axis)
        if (basis.packed[3 * atom + axis] != system.atoms[atom].position[axis])
          throw std::invalid_argument("host nuclear integral center mismatch");
    }
    for (std::size_t spin = 0; spin < spins; ++spin)
      symmetric(density.subspan(spin * aos * aos, aos * aos), aos);
  }

  void reject_alias(std::span<const double> output) const {
    if (overlaps(output, basis.packed) || overlaps(output, density) ||
        overlap(output.data(), output.size_bytes(), system.atoms.data(),
                atoms * sizeof(core::Atom)) ||
        overlap(output.data(), output.size_bytes(), system.shells.data(),
                system.shells.size() * sizeof(core::Shell)) ||
        overlap(output.data(), output.size_bytes(), system.ecp_terms.data(),
                system.ecp_terms.size() * sizeof(core::EcpTerm)))
      throw std::invalid_argument("host nuclear integral output aliases borrowed state");
    for (const auto& shell : system.shells)
      if (overlap(output.data(), output.size_bytes(), shell.primitives.data(),
                  shell.primitives.size() * sizeof(core::Primitive)))
        throw std::invalid_argument("host nuclear integral output aliases normalized primitives");
  }

  const core::System& system;
  const AoBasis& basis;
  const scf::ResolvedFockBuild& strategy;
  scf::ResolvedFockBuild expected_strategy;
  std::span<const double> density;
  std::function<void()> validator;
  std::size_t atoms, aos, primitives, spins, base{};
  std::array<const void*, 3> retained{};
  std::vector<Ao> orbitals;
};
}  // namespace

class HostKsNuclearIntegrals::Impl : public IntegralFrame {
 public:
  Impl(const core::System& system, const AoBasis& basis, const scf::ResolvedFockBuild& strategy,
       std::span<const double> density, std::string_view method_identity,
       HostNuclearIntegralOptions options, std::function<void()> validate_current,
       HostNuclearFirstSchedule first_schedule)
      : IntegralFrame(system, basis, strategy, density, method_identity,
                      std::move(validate_current)),
        options(options),
        resources(HostKsNuclearIntegrals::plan(basis, spins, options.eri_schedule, first_schedule)),
        quartet(map(spins)),
        coordinate_panel(first_schedule == HostNuclearFirstSchedule::CoordinatePanel) {
    if (coordinate_panel && !HostKsNuclearIntegrals::coordinate_density_supported(density))
      throw std::invalid_argument("resident nuclear first-source density range exceeded");
    if (!validator || !options.validation_quartets || options.validation_quartets > 4096)
      throw std::invalid_argument(
          "host nuclear integrals require a live owner and bounded validation cadence");
    if (resources.numeric_bytes > options.limits.numeric_bytes ||
        resources.one_electron_primitives > options.limits.one_electron_primitives ||
        resources.eri_primitives > options.limits.eri_primitives ||
        resources.map_terms > options.limits.map_terms)
      throw std::length_error("host nuclear integral numeric or complete-source work cap exceeded");
    projection = host_hessian_aot::find("integral_direction", maps::projection_identity);
    if (!projection || projection->inputs.size() != 25 || projection->outputs != 1 ||
        !projection->evaluate || quartet.inputs.size() != 6 || quartet.outputs.size() != 4)
      throw std::invalid_argument("missing host nuclear integral projection/quartet closure");
    for (std::size_t index = 0; index < 12; ++index) {
      gradient_inputs[index] = projection_input("g" + std::to_string(index));
      direction_inputs[index] = projection_input("u" + std::to_string(index));
    }
    projection_input_index = projection_input("projection");
    storage_count = resources.numeric_bytes / sizeof(double);
    storage = std::make_unique<double[]>(storage_count);
  }

  std::size_t projection_input(std::string_view name) const {
    const auto found = std::find(projection->inputs.begin(), projection->inputs.end(), name);
    if (found == projection->inputs.end())
      throw std::invalid_argument("incompatible integral projection inputs");
    return static_cast<std::size_t>(found - projection->inputs.begin());
  }

  void validate() {
    IntegralFrame::validate();
    if (!snapshots_bound) return;
    if (!std::equal(density.begin(), density.end(), density_snapshot.begin()) ||
        !std::equal(basis.packed.begin(), basis.packed.end(), basis_snapshot.begin()))
      throw std::invalid_argument("resident nuclear first-source inputs changed");
    for (std::size_t atom = 0; atom < atoms; ++atom)
      if (ionic_snapshot[atom] != system.atoms[atom].ionic_charge())
        throw std::invalid_argument("resident nuclear first-source ionic charges changed");
  }

  double project(std::span<const double, 12> gradient, const std::array<std::size_t, 4>& owners,
                 double scale, std::span<const double> direction, std::span<double> input,
                 std::span<double> output, std::size_t coordinate = SIZE_MAX) {
    finite(scale);
    input[projection_input_index] = scale;
    for (std::size_t index = 0; index < 12; ++index) {
      finite(gradient[index]);
      input[gradient_inputs[index]] = gradient[index];
      input[direction_inputs[index]] =
          owners[index / 3] == atoms ? 0
          : coordinate != SIZE_MAX   ? (coordinate == 3 * owners[index / 3] + index % 3 ? 1 : 0)
                                     : direction[3 * owners[index / 3] + index % 3];
    }
    census.projection_lanes = add(census.projection_lanes, 1);
    if (projection->evaluate(input.data(), input.size(), output.data(), output.size(), 1))
      throw std::domain_error("linked integral direction projection failed");
    return output[0];
  }

  template <class Publish>
  void project_coordinates(std::span<const double, 12> gradient,
                           const std::array<std::size_t, 4>& owners, double scale,
                           std::span<double> input, std::span<double> output, Publish publish) {
    for (std::size_t slot = 0; slot < owners.size(); ++slot) {
      const auto atom = owners[slot];
      if (atom == atoms ||
          std::find(owners.begin(), owners.begin() + slot, atom) != owners.begin() + slot)
        continue;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto coordinate = 3 * atom + axis;
        publish(coordinate, project(gradient, owners, scale, {}, input, output, coordinate));
      }
    }
  }

  void replay(std::span<const double> direction, std::span<double> output_overlap,
              std::span<double> output_fock) {
    Busy guard(busy);
    validate();
    const auto coordinates = 3 * atoms, matrix = aos * aos;
    if (!panel_ready || direction.size() != coordinates || !direction.data() ||
        !maps::finite(direction.data(), direction.size()) || output_overlap.size() != matrix ||
        output_fock.size() != density.size() || !output_overlap.data() || !output_fock.data() ||
        overlaps(output_overlap, output_fock))
      throw std::invalid_argument("invalid or unprepared resident nuclear first source");
    reject_alias(output_overlap);
    reject_alias(output_fock);
    auto* cursor = storage.get() + coordinates * (matrix + density.size());
    auto overlap_stage = std::span(cursor, matrix);
    cursor += matrix;
    auto fock_stage = std::span(cursor, density.size());
    cursor += density.size();
    auto snapshot = std::span(cursor, coordinates);
    std::copy(direction.begin(), direction.end(), snapshot.begin());
    std::fill(overlap_stage.begin(), overlap_stage.end(), 0);
    std::fill(fock_stage.begin(), fock_stage.end(), 0);
    census.resident_applications = add(census.resident_applications, 1);
    const auto* overlap_bank = storage.get();
    const auto* fock_bank = overlap_bank + coordinates * matrix;
    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      for (std::size_t element = 0; element < matrix; ++element)
        overlap_stage[element] +=
            snapshot[coordinate] * overlap_bank[coordinate * matrix + element];
      for (std::size_t element = 0; element < density.size(); ++element)
        fock_stage[element] +=
            snapshot[coordinate] * fock_bank[coordinate * density.size() + element];
    }
    if (!maps::finite(overlap_stage.data(), overlap_stage.size()) ||
        !maps::finite(fock_stage.data(), fock_stage.size()))
      throw std::domain_error("nonfinite resident nuclear source contraction");
    validate();
    if (!std::equal(snapshot.begin(), snapshot.end(), direction.begin()))
      throw std::invalid_argument("resident nuclear source direction changed before publication");
    std::copy(overlap_stage.begin(), overlap_stage.end(), output_overlap.begin());
    std::copy(fock_stage.begin(), fock_stage.end(), output_fock.begin());
  }

  void apply(std::span<const double> direction, std::span<double> output_overlap,
             std::span<double> output_fock, bool prepare = false) {
    if (coordinate_panel && !prepare) {
      replay(direction, output_overlap, output_fock);
      return;
    }
    Busy guard(busy);
    validate();
    const auto matrix = aos * aos;
    if (prepare != coordinate_panel)
      throw std::logic_error("coordinate preparation requires the resident first-source schedule");
    if (!prepare &&
        (direction.size() != 3 * atoms || !direction.data() ||
         !maps::finite(direction.data(), direction.size()) || output_overlap.size() != matrix ||
         output_fock.size() != density.size() || !output_overlap.data() || !output_fock.data() ||
         overlaps(output_overlap, output_fock)))
      throw std::invalid_argument("invalid host nuclear integral direction/output binding");
    if (prepare) {
      panel_ready = false;
      snapshots_bound = false;
    } else {
      reject_alias(output_overlap);
      reject_alias(output_fock);
    }
    census.directions_started = add(census.directions_started, 1);
    auto* cursor = storage.get();
    const auto take = [&cursor](std::size_t count) {
      auto result = std::span(cursor, count);
      cursor += count;
      return result;
    };
    const auto columns = prepare ? 3 * atoms : 1;
    auto overlap_stage = take(columns * matrix), fock_stage = take(columns * density.size());
    if (prepare) {
      take(matrix + density.size() + 3 * atoms);
      density_snapshot = take(density.size());
      basis_snapshot = take(basis.packed.size());
      ionic_snapshot = take(atoms);
      density_motion = take(density.size());
      weighted_motion = take(density.size());
      std::copy(density.begin(), density.end(), density_snapshot.begin());
      std::copy(basis.packed.begin(), basis.packed.end(), basis_snapshot.begin());
      for (std::size_t atom = 0; atom < atoms; ++atom)
        ionic_snapshot[atom] = system.atoms[atom].ionic_charge();
      snapshots_bound = true;
    }
    auto projection_input = take(25), projection_output = take(1), inputs = take(4 * spins + 2),
         outputs = take(4 * spins);
    auto arena =
        std::span(cursor, storage_count - static_cast<std::size_t>(cursor - storage.get()));
    if (prepare) response_arena = arena;
    std::fill(overlap_stage.begin(), overlap_stage.end(), 0);
    std::fill(fock_stage.begin(), fock_stage.end(), 0);
    const auto* radial = basis.packed.data() + 3 * atoms;
    for (std::size_t row = 0; row < aos; ++row)
      for (std::size_t column = 0; column < aos; ++column) {
        const auto& first = orbitals[row];
        const auto& second = orbitals[column];
        const auto* center_first = basis.packed.data() + 3 * first.atom;
        const auto* center_second = basis.packed.data() + 3 * second.atom;
        const std::array<std::size_t, 4> owners{first.atom, second.atom, atoms, atoms};
        for (std::size_t term_first = 0; term_first < first.terms; ++term_first)
          for (std::size_t term_second = 0; term_second < second.terms; ++term_second) {
            const auto* expansion_first = first.record + 4 + 4 * term_first;
            const auto* expansion_second = second.record + 4 + 4 * term_second;
            const auto component_first = primitive::component_index(
                expansion_first[0], expansion_first[1], expansion_first[2]);
            const auto component_second = primitive::component_index(
                expansion_second[0], expansion_second[1], expansion_second[2]);
            for (std::size_t index_first = first.begin; index_first < first.begin + first.count;
                 ++index_first)
              for (std::size_t index_second = second.begin;
                   index_second < second.begin + second.count; ++index_second) {
                const auto scale = expansion_first[3] * expansion_second[3] *
                                   radial[2 * index_first + 1] * radial[2 * index_second + 1];
                const auto pair =
                    primitive::make_pair(radial[2 * index_first], radial[2 * index_second],
                                         center_first[0], center_first[1], center_first[2],
                                         center_second[0], center_second[1], center_second[2]);
                census.one_electron_primitives = add(census.one_electron_primitives, 1);
                const auto st =
                    primitive::overlap_kinetic_gradient(pair, component_first, component_second);
                // S/T have only two mathematical centers; translation supplies the second slot.
                std::array<double, 12> gradient{};
                for (std::size_t axis = 0; axis < 3; ++axis) {
                  gradient[axis] = st.first[axis];
                  gradient[3 + axis] = -st.first[axis];
                }
                const auto add_fock = [&](std::size_t coordinate, double value) {
                  for (std::size_t spin = 0; spin < spins; ++spin)
                    fock_stage[(coordinate * spins + spin) * matrix + row * aos + column] += value;
                };
                if (prepare)
                  project_coordinates(gradient, owners, scale, projection_input, projection_output,
                                      [&](std::size_t coordinate, double value) {
                                        overlap_stage[coordinate * matrix + row * aos + column] +=
                                            value;
                                      });
                else
                  overlap_stage[row * aos + column] += project(gradient, owners, scale, direction,
                                                               projection_input, projection_output);
                for (std::size_t axis = 0; axis < 3; ++axis) {
                  gradient[axis] = st.second[axis];
                  gradient[3 + axis] = -st.second[axis];
                }
                double value{};
                if (prepare)
                  project_coordinates(gradient, owners, scale, projection_input, projection_output,
                                      add_fock);
                else
                  value = project(gradient, owners, scale, direction, projection_input,
                                  projection_output);
                for (std::size_t atom = 0; atom < atoms; ++atom) {
                  const auto* center = basis.packed.data() + 3 * atom;
                  census.one_electron_primitives = add(census.one_electron_primitives, 1);
                  const auto attraction = primitive::attraction_gradient(
                      pair, component_first, component_second, center[0], center[1], center[2]);
                  // Bind the attraction center independently, even when physical atom slots
                  // repeat.
                  for (std::size_t axis = 0; axis < 3; ++axis) {
                    gradient[axis] = attraction.first[axis];
                    gradient[3 + axis] = attraction.second[axis];
                    gradient[6 + axis] = -attraction.first[axis] - attraction.second[axis];
                  }
                  if (prepare)
                    project_coordinates(gradient, {first.atom, second.atom, atom, atoms},
                                        scale * system.atoms[atom].ionic_charge(), projection_input,
                                        projection_output, add_fock);
                  else
                    value += project(gradient, {first.atom, second.atom, atom, atoms},
                                     scale * system.atoms[atom].ionic_charge(), direction,
                                     projection_input, projection_output);
                }
                finite(value);
                if (!prepare)
                  for (std::size_t spin = 0; spin < spins; ++spin)
                    fock_stage[spin * matrix + row * aos + column] += value;
              }
          }
        census.ao_pairs = add(census.ao_pairs, 1);
        validate();
      }
    std::array<maps::Input, 6> input_bindings{};
    std::array<maps::Output, 4> output_bindings{};
    std::array<std::span<double>, 6> panels{};
    auto remaining = inputs;
    for (std::size_t index = 0; index < panels.size(); ++index) {
      const auto scalar =
          quartet.inputs[index] == "bar_energy" || quartet.inputs[index] == "d_integrals";
      panels[index] = remaining.first(scalar ? 1 : spins);
      remaining = remaining.subspan(panels[index].size());
      input_bindings[index] = {panels[index].data(), panels[index].size()};
    }
    for (std::size_t slot = 0; slot < 4; ++slot)
      output_bindings[slot] = {outputs.data() + slot * spins, spins};
    std::size_t map_terms{};
    const maps::Extents shape{spins, 1};
    if (!quartet.work(shape, map_terms)) throw std::overflow_error("nuclear quartet work overflow");
    constexpr std::array<std::array<std::size_t, 2>, 4> positions{
        {{{0, 1}}, {{2, 3}}, {{0, 2}}, {{1, 3}}}};
    constexpr std::array<std::string_view, 4> names{"coulomb_left", "coulomb_right",
                                                    "exchange_left", "exchange_right"};
    // The compiler supplies every density incidence and scientific coefficient before scatter.
    std::size_t since_validation{};
    integral_quartets(aos, options.eri_schedule, [&](const auto& indices) {
      std::array<const Ao*, 4> aos_in_quartet{};
      std::array<std::size_t, 4> owners{};
      std::array<double, 17> record{};
      for (std::size_t slot = 0; slot < 4; ++slot) {
        aos_in_quartet[slot] = &orbitals[indices[slot]];
        owners[slot] = aos_in_quartet[slot]->atom;
        std::copy_n(basis.packed.data() + 3 * owners[slot], 3, record.data() + 4 + 3 * slot);
      }
      record[16] = 1;
      std::array<unsigned, 4> angular{};
      double value{};
      std::array<double, 12> contracted_gradient{};
      const auto evaluate = [&](const auto& self, std::size_t slot, unsigned component,
                                double scale) -> void {
        if (slot == 4) {
          std::array<double, 12> gradient{};
          census.eri_primitives = add(census.eri_primitives, 1);
          census.eri_first_only_primitives = add(census.eri_first_only_primitives, 1);
          // This consumer never uses the scalar value. Share packaged s/p/d
          // gradients with stationary contractions, retaining the original
          // factorized first source for capability misses (including f).
          auto status = integrals::packaged_eri_primitive_gradient(angular.data(), component,
                                                                   record.data(), gradient.data());
          if (status == integrals::PackagedEriGradientStatus::Unavailable) {
            std::array<double, 13> first{};
            if (integrals::full_range_primitive_first_cpu(angular.data(), component, record.data(),
                                                          first.data())) {
              std::copy_n(first.data() + 1, gradient.size(), gradient.begin());
              status = integrals::PackagedEriGradientStatus::Success;
            }
          }
          if (status != integrals::PackagedEriGradientStatus::Success)
            throw std::domain_error("linked full-range nuclear primitive AD failed");
          if (prepare) {
            for (std::size_t index = 0; index < gradient.size(); ++index) {
              contracted_gradient[index] += scale * gradient[index];
              finite(contracted_gradient[index]);
            }
          } else
            value +=
                project(gradient, owners, scale, direction, projection_input, projection_output);
          finite(value);
          return;
        }
        const auto& ao = *aos_in_quartet[slot];
        for (std::size_t term = 0; term < ao.terms; ++term) {
          const auto* expansion = ao.record + 4 + 4 * term;
          const auto degree = static_cast<unsigned>(expansion[0] + expansion[1] + expansion[2]);
          angular[slot] = degree;
          const auto local_component =
              primitive::component_index(expansion[0], expansion[1], expansion[2]) -
              degree * (degree + 1) * (degree + 2) / 6;
          const auto count = (degree + 1) * (degree + 2) / 2;
          for (std::size_t index = ao.begin; index < ao.begin + ao.count; ++index) {
            record[slot] = radial[2 * index];
            self(self, slot + 1, component * count + local_component,
                 scale * expansion[3] * radial[2 * index + 1]);
          }
        }
      };
      const bool common_translation =
          options.eri_schedule == HostNuclearEriSchedule::SymmetryUnique &&
          std::all_of(owners.begin() + 1, owners.end(), [&](std::size_t atom) {
            return prepare ? atom == owners[0]
                           : std::equal(direction.begin() + 3 * owners[0],
                                        direction.begin() + 3 * owners[0] + 3,
                                        direction.begin() + 3 * atom);
          });
      // An ERI is unchanged when all four physical centers move together. Use
      // exact equality, never a displacement tolerance or integral screening.
      // This only zeros frozen-D Fock': curvature still needs moving-D weights.
      if (!common_translation) evaluate(evaluate, 0, 0, 1);
      std::array<std::size_t, 12> coordinates{};
      std::array<double, 12> partials{};
      std::size_t partial_count{};
      if (prepare)
        project_coordinates(contracted_gradient, owners, 1, projection_input, projection_output,
                            [&](std::size_t coordinate, double partial) {
                              coordinates[partial_count] = coordinate;
                              partials[partial_count++] = partial;
                            });
      quartet_incidences(indices, options.eri_schedule, [&](const auto& incidence) {
        for (std::size_t index = 0; index < panels.size(); ++index) {
          const auto name = quartet.inputs[index];
          if (name == "bar_energy")
            panels[index][0] = 1;
          else if (name == "d_integrals")
            panels[index][0] = prepare ? 1 : value;
          else {
            const auto found = std::find(names.begin(), names.end(), name);
            if (found == names.end())
              throw std::invalid_argument("unknown host nuclear density incidence");
            const auto pair = positions[static_cast<std::size_t>(found - names.begin())];
            for (std::size_t spin = 0; spin < spins; ++spin)
              panels[index][spin] =
                  density[spin * matrix + incidence[pair[0]] * aos + incidence[pair[1]]];
          }
        }
        census.map_terms = add(census.map_terms, map_terms);
        if (quartet.evaluate(shape, input_bindings, output_bindings, arena) !=
            maps::Status::success)
          throw std::domain_error("linked nuclear quartet Fock map failed");
        for (std::size_t slot = 0; slot < 4; ++slot)
          for (std::size_t spin = 0; spin < spins; ++spin) {
            const auto element =
                spin * matrix + incidence[positions[slot][0]] * aos + incidence[positions[slot][1]];
            if (prepare) {
              for (std::size_t partial = 0; partial < partial_count; ++partial)
                fock_stage[coordinates[partial] * density.size() + element] +=
                    outputs[slot * spins + spin] * partials[partial];
            } else
              fock_stage[element] += outputs[slot * spins + spin];
          }
        census.ao_quartets = add(census.ao_quartets, 1);
        if (++since_validation == options.validation_quartets) {
          validate();
          since_validation = 0;
        }
      });
    });
    for (std::size_t coordinate = 0; coordinate < columns; ++coordinate) {
      symmetric(overlap_stage.subspan(coordinate * matrix, matrix), aos);
      for (std::size_t spin = 0; spin < spins; ++spin)
        symmetric(fock_stage.subspan((coordinate * spins + spin) * matrix, matrix), aos);
    }
    validate();
    census.directions_completed = add(census.directions_completed, 1);
    if (prepare) {
      panel_ready = true;
      census.coordinate_panels_completed = add(census.coordinate_panels_completed, 1);
    } else {
      std::copy(overlap_stage.begin(), overlap_stage.end(), output_overlap.begin());
      std::copy(fock_stage.begin(), fock_stage.end(), output_fock.begin());
    }
  }

  void density_response(std::span<const double> density_right,
                        std::span<const double> weighted_density_right, std::span<double> output,
                        std::size_t max_map_terms) {
    Busy guard(busy);
    validate();
    if (!panel_ready || density_right.size() != density.size() ||
        weighted_density_right.size() != density.size() || !density_right.data() ||
        !weighted_density_right.data() || output.size() != 3 * atoms || !output.data())
      throw std::invalid_argument("invalid or unprepared moving-density integral contraction");
    reject_alias(output);
    const auto matrix = aos * aos;
    for (std::size_t spin = 0; spin < spins; ++spin) {
      symmetric(density_right.subspan(spin * matrix, matrix), aos);
      symmetric(weighted_density_right.subspan(spin * matrix, matrix), aos);
    }
    const auto& entry = density_response_map(spins);
    const maps::Extents shape{spins, matrix};
    std::size_t terms{};
    if (!entry.work(shape, terms))
      throw std::overflow_error("moving-density integral map work overflow");
    if (multiply(3 * atoms, terms) > max_map_terms)
      throw std::length_error("moving-density integral map work cap exceeded");
    std::copy(density_right.begin(), density_right.end(), density_motion.begin());
    std::copy(weighted_density_right.begin(), weighted_density_right.end(),
              weighted_motion.begin());
    const auto* overlap_bank = storage.get();
    const auto* fock_bank = overlap_bank + 3 * atoms * matrix;
    auto stage = std::span(storage.get() + (3 * atoms + 1) * (matrix + density.size()), 3 * atoms);
    for (std::size_t coordinate = 0; coordinate < 3 * atoms; ++coordinate) {
      std::array<maps::Input, 4> inputs{};
      if (entry.inputs.size() != inputs.size())
        throw std::invalid_argument("incompatible moving-density integral map");
      for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto name = entry.inputs[index];
        if (name == "density_right")
          inputs[index] = {density_motion.data(), density_motion.size()};
        else if (name == "weighted_density_right")
          inputs[index] = {weighted_motion.data(), weighted_motion.size()};
        else if (name == "overlap_direction")
          inputs[index] = {overlap_bank + coordinate * matrix, matrix};
        else if (name == "integral_fock_direction")
          inputs[index] = {fock_bank + coordinate * density.size(), density.size()};
        else
          throw std::invalid_argument("unknown moving-density integral map input");
      }
      const std::array<maps::Output, 1> outputs{{{stage.data() + coordinate, 1}}};
      census.density_response_map_terms = add(census.density_response_map_terms, terms);
      if (entry.evaluate(shape, inputs, outputs, response_arena) != maps::Status::success)
        throw std::domain_error("linked moving-density integral contraction failed");
    }
    validate();
    if (!std::equal(density_motion.begin(), density_motion.end(), density_right.begin()) ||
        !std::equal(weighted_motion.begin(), weighted_motion.end(), weighted_density_right.begin()))
      throw std::invalid_argument("moving-density integral inputs changed during invocation");
    std::copy(stage.begin(), stage.end(), output.begin());
  }

  HostNuclearIntegralOptions options;
  std::size_t storage_count{}, projection_input_index{};
  HostNuclearIntegralPlan resources;
  const maps::Program& quartet;
  bool coordinate_panel{}, panel_ready{}, snapshots_bound{};
  std::span<double> density_snapshot, basis_snapshot, ionic_snapshot;
  std::span<double> density_motion, weighted_motion, response_arena;
  const host_hessian_aot::Program* projection{};
  std::array<std::size_t, 12> gradient_inputs{}, direction_inputs{};
  std::unique_ptr<double[]> storage;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostNuclearIntegralWork census;
};

HostKsNuclearIntegrals::HostKsNuclearIntegrals(const core::System& system, const AoBasis& basis,
                                               const scf::ResolvedFockBuild& strategy,
                                               std::span<const double> density,
                                               std::string_view method_identity,
                                               HostNuclearIntegralOptions options,
                                               std::function<void()> validate_current,
                                               HostNuclearFirstSchedule first_schedule)
    : impl_(std::make_unique<Impl>(system, basis, strategy, density, method_identity, options,
                                   std::move(validate_current), first_schedule)) {}
HostKsNuclearIntegrals::~HostKsNuclearIntegrals() = default;
void HostKsNuclearIntegrals::apply(std::span<const double> direction,
                                   std::span<double> overlap_direction,
                                   std::span<double> integral_fock_direction) {
  impl_->apply(direction, overlap_direction, integral_fock_direction);
}
void HostKsNuclearIntegrals::prepare_coordinates() { impl_->apply({}, {}, {}, true); }
void HostKsNuclearIntegrals::apply_density_response(std::span<const double> density_right,
                                                    std::span<const double> weighted_density_right,
                                                    std::span<double> output,
                                                    std::size_t max_map_terms) {
  impl_->density_response(density_right, weighted_density_right, output, max_map_terms);
}
HostNuclearIntegralPlan HostKsNuclearIntegrals::resources() const noexcept {
  return impl_->resources;
}
HostNuclearIntegralWork HostKsNuclearIntegrals::work() const noexcept { return impl_->census; }

bool HostKsNuclearIntegralCurvature::projected_matrix_supported(
    const AoBasis& basis, std::span<const double> density) noexcept {
  try {
    const auto base = offset(basis);
    const auto matrix = multiply(basis.nao, basis.nao);
    if (!density.empty() && density.size() != matrix && density.size() != multiply(2, matrix))
      return false;
    if (!std::all_of(density.begin(), density.end(),
                     [](double value) { return std::isfinite(value) && std::abs(value) <= 1e50; }))
      return false;
    for (std::size_t index = 0; index < basis.nao; ++index) {
      const auto ao = orbital(basis, base, index);
      for (std::size_t term = 0; term < ao.terms; ++term)
        if (std::abs(ao.record[7 + 4 * term]) > 16) return false;
    }
    return !polynomial_layout(basis, base).orbitals.empty();
  } catch (const std::exception&) {
    return false;
  }
}

HostNuclearIntegralPlan HostKsNuclearIntegralCurvature::plan(
    const AoBasis& basis, std::size_t spins, HostNuclearIntegralCurvatureSchedule schedule,
    HostNuclearEriSchedule eri_schedule) {
  if (schedule != HostNuclearIntegralCurvatureSchedule::Mixed &&
      schedule != HostNuclearIntegralCurvatureSchedule::Column &&
      schedule != HostNuclearIntegralCurvatureSchedule::Matrix &&
      schedule != HostNuclearIntegralCurvatureSchedule::ProjectedMatrix)
    throw std::invalid_argument("invalid host integral curvature schedule");
  const auto column = schedule == HostNuclearIntegralCurvatureSchedule::Column;
  const auto projected = schedule == HostNuclearIntegralCurvatureSchedule::ProjectedMatrix;
  const auto full_matrix = schedule == HostNuclearIntegralCurvatureSchedule::Matrix || projected;
  const auto base = offset(basis);
  if (projected && !projected_matrix_supported(basis, {}))
    throw std::invalid_argument("projected integral matrix requires bounded polynomial scales");
  const auto polynomial = projected ? polynomial_layout(basis, base) : PolynomialLayout{};
  std::size_t primitive_terms{};
  for (std::size_t index = 0; index < basis.nao; ++index) {
    const auto ao = orbital(basis, base, index);
    primitive_terms = add(primitive_terms, multiply(ao.count, ao.terms));
  }
  const maps::Extents shape{spins, 1};
  std::size_t arena_bytes{}, pair_terms{}, quartet_terms{};
  for (const auto source : {"one_electron", "overlap_pulay", "quartet"}) {
    const auto& entry = weight_map(source, spins, column);
    std::size_t bytes{}, terms{};
    if (!entry.workspace(shape, bytes) || !entry.work(shape, terms))
      throw std::overflow_error("host integral curvature weight envelope overflow");
    arena_bytes = std::max(arena_bytes, bytes);
    if (std::string_view(source) == "quartet")
      quartet_terms = terms;
    else
      pair_terms = add(pair_terms, terms);
  }
  const auto pairs = multiply(primitive_terms, primitive_terms);
  const auto matrix = multiply(basis.nao, basis.nao);
  const auto dimension = multiply(3, basis.natom);
  const auto fields = full_matrix ? add(add(1, dimension), multiply(dimension, dimension))
                      : column    ? add(2, multiply(6, basis.natom))
                                  : 4;
  auto elements =
      add(multiply(2, fields), column ? add(3, multiply(8, spins)) : add(2, multiply(4, spins)));
  if (column)
    elements = add(elements, add(multiply(3, basis.natom), multiply(2, multiply(spins, matrix))));
  const auto eri_matrix =
      projected ? multiply(polynomial.orbitals.size(), polynomial.orbitals.size()) : matrix;
  if (projected)
    elements = add(elements,
                   add(basis.packed.size(), multiply(spins, add(matrix, multiply(2, eri_matrix)))));
  const auto bytes = add(arena_bytes, multiply(elements, sizeof(double)));
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("host integral curvature exceeds signed resource domain");
  const auto pair_visits = full_matrix ? add(9, multiply(36, basis.natom))
                           : column    ? add(3, multiply(6, basis.natom))
                                       : add(basis.natom, 1);
  return {
      bytes, multiply(pairs, pair_visits),
      full_matrix
          ? eri_matrix_primitive_work(basis, base, eri_schedule, projected ? &polynomial : nullptr)
          : eri_primitive_work(basis, base, eri_schedule),
      add(multiply(matrix, pair_terms), multiply(multiply(eri_matrix, eri_matrix), quartet_terms))};
}

class HostKsNuclearIntegralCurvature::Impl : public IntegralFrame {
 public:
  Impl(const core::System& system, const AoBasis& basis, const scf::ResolvedFockBuild& strategy,
       std::span<const double> density, std::span<const double> weighted_density,
       std::string_view method_identity, HostNuclearIntegralOptions options,
       std::function<void()> validate_current, HostNuclearIntegralCurvatureSchedule schedule)
      : IntegralFrame(system, basis, strategy, density, method_identity,
                      std::move(validate_current)),
        weighted_density(weighted_density),
        options(options),
        column_schedule(schedule == HostNuclearIntegralCurvatureSchedule::Column),
        matrix_schedule(schedule == HostNuclearIntegralCurvatureSchedule::Matrix ||
                        schedule == HostNuclearIntegralCurvatureSchedule::ProjectedMatrix),
        projected_matrix_schedule(schedule ==
                                  HostNuclearIntegralCurvatureSchedule::ProjectedMatrix),
        fields(matrix_schedule   ? add(add(1, multiply(3, atoms)), multiply(3 * atoms, 3 * atoms))
               : column_schedule ? add(2, multiply(6, atoms))
                                 : 4),
        resources(
            HostKsNuclearIntegralCurvature::plan(basis, spins, schedule, options.eri_schedule)),
        hcore(weight_map("one_electron", spins, column_schedule)),
        metric(weight_map("overlap_pulay", spins, column_schedule)),
        quartet(weight_map("quartet", spins, column_schedule)) {
    if (!options.validation_quartets || options.validation_quartets > 4096)
      throw std::invalid_argument("host integral curvature requires bounded validation cadence");
    if (resources.numeric_bytes > options.limits.numeric_bytes ||
        resources.one_electron_primitives > options.limits.one_electron_primitives ||
        resources.eri_primitives > options.limits.eri_primitives ||
        resources.map_terms > options.limits.map_terms)
      throw std::length_error(
          "host integral curvature numeric or complete-source work cap exceeded");
    validate();
    storage_count = resources.numeric_bytes / sizeof(double);
    storage = std::make_unique<double[]>(storage_count);
    if (projected_matrix_schedule) {
      if (!HostKsNuclearIntegralCurvature::projected_matrix_supported(basis, density))
        throw std::invalid_argument(
            "projected integral matrix density exceeds its safe scaling domain");
      polynomial = polynomial_layout(basis, base);
      const auto projected_elements =
          multiply(spins, multiply(polynomial.orbitals.size(), polynomial.orbitals.size()));
      const auto take = [&](std::size_t count) {
        if (count > storage_count)
          throw std::logic_error("projected integral snapshots exceed their numerical plan");
        storage_count -= count;
        return std::span(storage.get() + storage_count, count);
      };
      projected_density = take(projected_elements);
      projection_compensation = take(projected_elements);
      density_snapshot = take(density.size());
      basis_snapshot = take(basis.packed.size());
      std::copy(density.begin(), density.end(), density_snapshot.begin());
      std::copy(basis.packed.begin(), basis.packed.end(), basis_snapshot.begin());
      projection_inputs_bound = true;
      validate();
    }
  }

  void validate() const {
    IntegralFrame::validate();
    if (projection_inputs_bound &&
        (!std::equal(density.begin(), density.end(), density_snapshot.begin()) ||
         !std::equal(basis.packed.begin(), basis.packed.end(), basis_snapshot.begin())))
      throw std::invalid_argument("changed projected integral density or polynomial basis");
    if (!weighted_density.data() || weighted_density.size() != density.size())
      throw std::invalid_argument("invalid host integral curvature weighted-density binding");
    for (std::size_t spin = 0; spin < spins; ++spin)
      symmetric(weighted_density.subspan(spin * aos * aos, aos * aos), aos);
  }

  /** T is geometry independent for the fixed lab-frame real harmonics. Only
   * ERI weights use T^T D T; public S/T/V, W and moving-density sources remain
   * in their original basis. Snapshot and compensate before making reuse live.
   */
  void prepare_projected_density() {
    if (!projected_matrix_schedule || projection_ready) return;
    runtime::host_trace::Region trace("cpu_integral_density_projection", aos);
    if (polynomial != polynomial_layout(basis, base))
      throw std::invalid_argument("changed projected integral polynomial topology");
    std::fill(projected_density.begin(), projected_density.end(), 0);
    std::fill(projection_compensation.begin(), projection_compensation.end(), 0);
    const auto count = polynomial.orbitals.size();
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < aos; ++row)
        for (std::size_t column = 0; column < aos; ++column) {
          const auto& first = orbitals[row];
          const auto& second = orbitals[column];
          const auto value = density_snapshot[spin * aos * aos + row * aos + column];
          for (std::size_t first_term = 0; first_term < first.terms; ++first_term)
            for (std::size_t second_term = 0; second_term < second.terms; ++second_term) {
              const auto* left = basis_snapshot.data() + base + 16 * row + 4 + 4 * first_term;
              const auto* right = basis_snapshot.data() + base + 16 * column + 4 + 4 * second_term;
              const auto first_component =
                  polynomial.public_offsets[row] + polynomial_component(left);
              const auto second_component =
                  polynomial.public_offsets[column] + polynomial_component(right);
              const auto index = spin * count * count + first_component * count + second_component;
              const auto contribution = value * left[3] * right[3];
              finite(contribution);
              const auto adjusted = contribution - projection_compensation[index];
              const auto next = projected_density[index] + adjusted;
              projection_compensation[index] = (next - projected_density[index]) - adjusted;
              projected_density[index] = next;
              finite(next);
              census.density_projection_terms = add(census.density_projection_terms, 1);
            }
        }
    validate();
    if (!std::equal(density.begin(), density.end(), density_snapshot.begin()) ||
        !std::equal(basis.packed.begin(), basis.packed.end(), basis_snapshot.begin()))
      throw std::invalid_argument("projected integral inputs changed during congruence");
    projection_ready = true;
    runtime::df_progress::number("public_aos", aos);
    runtime::df_progress::number("polynomial_aos", count);
    runtime::df_progress::number("density_projection_terms", census.density_projection_terms);
  }

  std::array<double, 2> weight(const maps::Program& entry,
                               const std::array<std::size_t, 4>& indices, std::span<double> panels,
                               std::span<double> result, std::span<double> arena,
                               bool projected = false) {
    constexpr std::array<std::array<std::size_t, 2>, 4> positions{
        {{{0, 1}}, {{2, 3}}, {{0, 2}}, {{1, 3}}}};
    constexpr std::array<std::string_view, 4> names{"coulomb_left", "coulomb_right",
                                                    "exchange_left", "exchange_right"};
    std::array<maps::Input, 9> bindings{};
    if (entry.inputs.size() > bindings.size())
      throw std::invalid_argument("incompatible host integral curvature weight inputs");
    for (std::size_t index = 0; index < entry.inputs.size(); ++index) {
      auto name = entry.inputs[index];
      const auto moving = name.starts_with("d_");
      if (moving) name.remove_prefix(2);
      const auto count = name == "bar_energy" ? 1 : spins;
      auto panel = panels.first(count);
      panels = panels.subspan(count);
      bindings[index] = {panel.data(), count};
      if (name == "bar_energy")
        panel[0] = 1;
      else {
        auto pair = std::array<std::size_t, 2>{0, 1};
        if (name != "density_left" && name != "weighted_density") {
          const auto found = std::find(names.begin(), names.end(), name);
          if (found == names.end())
            throw std::invalid_argument("unknown curvature density incidence");
          pair = positions[static_cast<std::size_t>(found - names.begin())];
        }
        if (projected && (moving || name == "weighted_density"))
          throw std::invalid_argument(
              "polynomial projection is only a frozen ERI density contract");
        const auto count = projected ? polynomial.orbitals.size() : aos;
        const auto values = projected ? std::span<const double>(projected_density)
                            : name == "weighted_density"
                                ? (moving ? weighted_motion : weighted_density)
                                : (moving ? density_motion : density);
        for (std::size_t spin = 0; spin < spins; ++spin)
          panel[spin] = values[spin * count * count + indices[pair[0]] * count + indices[pair[1]]];
      }
    }
    const maps::Extents shape{spins, 1};
    std::size_t terms{};
    if (!entry.work(shape, terms)) throw std::overflow_error("curvature weight work overflow");
    census.map_terms = add(census.map_terms, terms);
    const std::array<maps::Output, 2> outputs{{{result.data(), 1}, {result.data() + 1, 1}}};
    if (entry.evaluate(shape, std::span(bindings).first(entry.inputs.size()),
                       std::span(outputs).first(entry.outputs.size()),
                       arena) != maps::Status::success)
      throw std::domain_error("linked integral curvature weight map failed");
    return {result[0], column_schedule ? result[1] : 0};
  }

  void apply(std::span<const double> left, std::span<const double> right,
             std::span<double> output) {
    if (column_schedule || matrix_schedule)
      throw std::logic_error("host integral curvature requires its selected matrix/column call");
    run(left, right, {}, {}, output);
  }

  void apply_column(std::span<const double> right, std::span<const double> density_right,
                    std::span<const double> weighted_density_right, std::span<double> output) {
    if (!column_schedule)
      throw std::logic_error("host integral curvature requires the column schedule");
    run({}, right, density_right, weighted_density_right, output);
  }

  void apply_matrix(std::span<double> output) {
    Busy guard(busy);
    validate();
    if (!matrix_schedule || !output.data() || output.size() != fields)
      throw std::invalid_argument("host integral matrix requires its exact output binding");
    reject_alias(output);
    if (overlaps(output, weighted_density))
      throw std::invalid_argument("host integral matrix output aliases weighted density");
    census.directions_started = add(census.directions_started, 1);
    prepare_projected_density();
    auto stage = std::span(storage.get(), fields);
    auto compensation = std::span(storage.get() + fields, fields);
    auto panels = std::span(storage.get() + 2 * fields, 4 * spins + 1);
    auto result = std::span(panels.data() + panels.size(), 1);
    auto arena = std::span(result.data() + result.size(),
                           storage_count - 2 * fields - panels.size() - result.size());
    std::fill(stage.begin(), stage.end(), 0);
    std::fill(compensation.begin(), compensation.end(), 0);
    const auto dimension = 3 * atoms;
    const auto accumulate = [&](std::size_t field, double value) {
      finite(value);
      const auto adjusted = value - compensation[field];
      const auto next = stage[field] + adjusted;
      compensation[field] = (next - stage[field]) - adjusted;
      stage[field] = next;
      finite(next);
    };
    const auto gradient = [&](std::size_t atom, std::size_t origin, std::size_t axis,
                              double value) {
      if (atom == origin) return;
      accumulate(1 + 3 * atom + axis, value);
      accumulate(1 + 3 * origin + axis, -value);
    };
    const auto mixed = [&](std::size_t left_atom, std::size_t left_origin, std::size_t left_axis,
                           std::size_t right_atom, std::size_t right_origin, std::size_t right_axis,
                           double value) {
      if (left_atom == left_origin || right_atom == right_origin) return;
      for (std::size_t left_slot = 0; left_slot < 2; ++left_slot)
        for (std::size_t right_slot = 0; right_slot < 2; ++right_slot) {
          const auto row = 3 * (left_slot ? left_origin : left_atom) + left_axis;
          const auto column = 3 * (right_slot ? right_origin : right_atom) + right_axis;
          accumulate(1 + dimension + row * dimension + column,
                     left_slot == right_slot ? value : -value);
        }
    };
    const auto* radial = basis.packed.data() + 3 * atoms;
    {
      runtime::host_trace::Region phase("cpu_hessian_integral_one_electron", aos);
      for (std::size_t row = 0; row < aos; ++row)
        for (std::size_t column = 0; column < aos; ++column) {
          const auto& first = orbitals[row];
          const auto& second = orbitals[column];
          const auto hcore_weight = weight(hcore, {row, column, 0, 0}, panels, result, arena)[0];
          const auto metric_weight = weight(metric, {row, column, 0, 0}, panels, result, arena)[0];
          std::array<double, 9> centers{}, moving_left{}, moving_right{};
          std::copy_n(basis.packed.data() + 3 * first.atom, 3, centers.data());
          std::copy_n(basis.packed.data() + 3 * second.atom, 3, centers.data() + 3);
          for (std::size_t term_first = 0; term_first < first.terms; ++term_first)
            for (std::size_t term_second = 0; term_second < second.terms; ++term_second) {
              const auto* expansion_first = first.record + 4 + 4 * term_first;
              const auto* expansion_second = second.record + 4 + 4 * term_second;
              std::array<unsigned, 6> powers{};
              for (std::size_t axis = 0; axis < 3; ++axis) {
                powers[axis] = static_cast<unsigned>(expansion_first[axis]);
                powers[3 + axis] = static_cast<unsigned>(expansion_second[axis]);
              }
              for (std::size_t index_first = first.begin; index_first < first.begin + first.count;
                   ++index_first)
                for (std::size_t index_second = second.begin;
                     index_second < second.begin + second.count; ++index_second) {
                  const double exponents[]{radial[2 * index_first], radial[2 * index_second]};
                  const auto scale = expansion_first[3] * expansion_second[3] *
                                     radial[2 * index_first + 1] * radial[2 * index_second + 1];
                  for (std::size_t right_axis = 0; right_axis < 3; ++right_axis)
                    for (std::size_t left_axis = 0; left_axis < 3; ++left_axis) {
                      moving_left.fill(0);
                      moving_right.fill(0);
                      moving_left[left_axis] = 1;
                      moving_right[right_axis] = 1;
                      std::array<double, 8> jet{};
                      census.one_electron_primitives = add(census.one_electron_primitives, 1);
                      if (!integrals::host_mixed::overlap_kinetic(exponents, centers.data(),
                                                                  powers.data(), moving_left.data(),
                                                                  moving_right.data(), jet.data()))
                        throw std::domain_error("canonical matrix S/T primitive failed");
                      for (const auto begin : {std::size_t{0}, std::size_t{4}}) {
                        const auto coefficient = scale * (begin ? hcore_weight : metric_weight);
                        if (!right_axis) {
                          if (!left_axis) accumulate(0, coefficient * jet[begin]);
                          gradient(first.atom, second.atom, left_axis,
                                   coefficient * jet[begin + 1]);
                        }
                        mixed(first.atom, second.atom, left_axis, first.atom, second.atom,
                              right_axis, coefficient * jet[begin + 3]);
                      }
                    }
                  for (std::size_t atom = 0; atom < atoms; ++atom) {
                    std::copy_n(basis.packed.data() + 3 * atom, 3, centers.data() + 6);
                    const std::array<std::size_t, 2> owners{first.atom, second.atom};
                    const auto coefficient =
                        scale * hcore_weight * system.atoms[atom].ionic_charge();
                    for (std::size_t right_lane = 0; right_lane < 6; ++right_lane)
                      for (std::size_t left_lane = 0; left_lane < 6; ++left_lane) {
                        moving_left.fill(0);
                        moving_right.fill(0);
                        moving_left[left_lane] = 1;
                        moving_right[right_lane] = 1;
                        std::array<double, 4> jet{};
                        census.one_electron_primitives = add(census.one_electron_primitives, 1);
                        if (!integrals::host_mixed::attraction(exponents, centers.data(),
                                                               powers.data(), moving_left.data(),
                                                               moving_right.data(), jet.data()))
                          throw std::domain_error("canonical matrix attraction primitive failed");
                        if (!right_lane) {
                          if (!left_lane) accumulate(0, coefficient * jet[0]);
                          gradient(owners[left_lane / 3], atom, left_lane % 3,
                                   coefficient * jet[1]);
                        }
                        mixed(owners[left_lane / 3], atom, left_lane % 3, owners[right_lane / 3],
                              atom, right_lane % 3, coefficient * jet[3]);
                      }
                  }
                }
            }
          census.ao_pairs = add(census.ao_pairs, 1);
          validate();
        }
      runtime::df_progress::number("attempted_one_electron_primitives",
                                   census.one_electron_primitives);
    }
    std::size_t since_validation{};
    const auto eri_aos = projected_matrix_schedule ? polynomial.orbitals.size() : aos;
    {
      runtime::host_trace::Region phase("cpu_hessian_integral_eri", aos);
      integral_quartets(eri_aos, options.eri_schedule, [&](const auto& indices) {
        double quartet_weight{};
        quartet_incidences(indices, options.eri_schedule, [&](const auto& incidence) {
          quartet_weight +=
              weight(quartet, incidence, panels, result, arena, projected_matrix_schedule)[0];
          finite(quartet_weight);
          census.ao_quartets = add(census.ao_quartets, 1);
          if (++since_validation == options.validation_quartets) {
            validate();
            since_validation = 0;
          }
        });
        std::array<std::size_t, 4> owners{};
        for (std::size_t slot = 0; slot < 4; ++slot)
          owners[slot] = projected_matrix_schedule ? polynomial.orbitals[indices[slot]].atom
                                                   : orbitals[indices[slot]].atom;
        const auto origin = owners[3];
        std::array<std::size_t, 3> right_owners{};
        std::array<unsigned, 4> incidence{};
        std::size_t active{};
        if (options.eri_schedule == HostNuclearEriSchedule::Ordered) {
          right_owners = {owners[0], owners[1], owners[2]};
          incidence = {0, 1, 2, 3};
          active = 3;
        } else {
          for (std::size_t slot = 0; slot < 3; ++slot) {
            if (owners[slot] == origin) continue;
            const auto end = right_owners.begin() + active;
            const auto found = std::find(right_owners.begin(), end, owners[slot]);
            incidence[slot] = static_cast<unsigned>(found - right_owners.begin());
            if (found == end) right_owners[active++] = owners[slot];
          }
          for (std::size_t slot = 0; slot < 4; ++slot)
            if (owners[slot] == origin) incidence[slot] = static_cast<unsigned>(active);
        }
        const auto value_only = active == 0;
        const auto local_dimension = 3 * active;
        std::array<double, 17> record{};
        for (std::size_t slot = 0; slot < 4; ++slot)
          for (std::size_t axis = 0; axis < 3; ++axis)
            record[4 + 3 * slot + axis] =
                basis.packed[3 * owners[slot] + axis] - basis.packed[3 * origin + axis];
        record[16] = 1;
        std::array<unsigned, 4> angular{};
        const auto evaluate = [&](const auto& self, std::size_t slot, unsigned component,
                                  double scale) -> void {
          if (slot == 4) {
            const auto coefficient = scale * quartet_weight;
            std::array<double, 91> jet{};
            census.eri_primitives = add(census.eri_primitives, 1);
            bool valid{};
            if (value_only) {
              census.eri_value_only_primitives = add(census.eri_value_only_primitives, 1);
              valid = integrals::full_range_primitive_value_cpu(angular.data(), component,
                                                                record.data(), jet.data());
            } else {
              census.eri_mixed_primitives = add(census.eri_mixed_primitives, 1);
              census.eri_matrix_primitives = add(census.eri_matrix_primitives, 1);
              census.eri_matrix_coordinates = add(census.eri_matrix_coordinates, local_dimension);
              census.eri_matrix_second_partials = add(census.eri_matrix_second_partials,
                                                      multiply(local_dimension, local_dimension));
              valid = integrals::full_range_primitive_matrix_cpu(
                  angular.data(), component, record.data(), incidence.data(),
                  static_cast<unsigned>(active), jet.data());
            }
            if (!valid) throw std::domain_error("canonical matrix ERI primitive failed");
            accumulate(0, coefficient * jet[0]);
            if (!value_only) {
              // Physical incidence is already applied before factor composition.
              // Restore the negative origin lanes, without any matrix repair.
              for (std::size_t left = 0; left < local_dimension; ++left) {
                gradient(right_owners[left / 3], origin, left % 3, coefficient * jet[1 + left]);
                for (std::size_t right = 0; right < local_dimension; ++right)
                  mixed(right_owners[left / 3], origin, left % 3, right_owners[right / 3], origin,
                        right % 3,
                        coefficient * jet[1 + local_dimension + local_dimension * left + right]);
              }
            }
            return;
          }
          if (projected_matrix_schedule) {
            const auto& ao = polynomial.orbitals[indices[slot]];
            angular[slot] = ao.degree;
            const auto count = (ao.degree + 1) * (ao.degree + 2) / 2;
            for (std::size_t index = ao.begin; index < ao.begin + ao.count; ++index) {
              record[slot] = radial[2 * index];
              self(self, slot + 1, component * count + ao.component, scale * radial[2 * index + 1]);
            }
            return;
          }
          const auto& ao = orbitals[indices[slot]];
          for (std::size_t term = 0; term < ao.terms; ++term) {
            const auto* expansion = ao.record + 4 + 4 * term;
            const auto degree = static_cast<unsigned>(expansion[0] + expansion[1] + expansion[2]);
            angular[slot] = degree;
            const auto local_component =
                primitive::component_index(expansion[0], expansion[1], expansion[2]) -
                degree * (degree + 1) * (degree + 2) / 6;
            const auto count = (degree + 1) * (degree + 2) / 2;
            for (std::size_t index = ao.begin; index < ao.begin + ao.count; ++index) {
              record[slot] = radial[2 * index];
              self(self, slot + 1, component * count + local_component,
                   scale * expansion[3] * radial[2 * index + 1]);
            }
          }
        };
        evaluate(evaluate, 0, 0, 1);
      });
      runtime::df_progress::number("attempted_eri_matrix_primitives", census.eri_matrix_primitives);
      runtime::df_progress::number("attempted_eri_matrix_second_partials",
                                   census.eri_matrix_second_partials);
    }
    validate();
    census.directions_completed = add(census.directions_completed, 1);
    std::copy(stage.begin(), stage.end(), output.begin());
  }

  void run(std::span<const double> left, std::span<const double> right,
           std::span<const double> density_right, std::span<const double> weighted_density_right,
           std::span<double> output) {
    Busy guard(busy);
    validate();
    if ((!column_schedule &&
         (left.size() != 3 * atoms || !left.data() || !maps::finite(left.data(), left.size()))) ||
        right.size() != 3 * atoms || !right.data() || !maps::finite(right.data(), right.size()) ||
        output.size() != fields || !output.data())
      throw std::invalid_argument("invalid host integral curvature directions/output");
    reject_alias(output);
    if (overlaps(output, weighted_density))
      throw std::invalid_argument("host integral curvature output aliases weighted density");
    for (const auto values : {density_right, weighted_density_right}) {
      if (values.empty()) continue;
      if (!values.data() || values.size() != density.size())
        throw std::invalid_argument("invalid integral column density response shape");
      for (std::size_t spin = 0; spin < spins; ++spin)
        symmetric(values.subspan(spin * aos * aos, aos * aos), aos);
    }
    census.directions_started = add(census.directions_started, 1);
    auto stage = std::span(storage.get(), fields);
    auto correction = std::span(storage.get() + fields, fields);
    const auto panel_count = (column_schedule ? 8 : 4) * spins + 1;
    const auto result_count = column_schedule ? 2 : 1;
    auto panels = std::span(storage.get() + 2 * fields, panel_count);
    auto result = std::span(storage.get() + 2 * fields + panel_count, result_count);
    auto arena_begin = 2 * fields + panel_count + result_count;
    const auto external_right = right;
    std::span<double> right_snapshot;
    if (column_schedule) {
      right_snapshot = std::span(storage.get() + arena_begin, right.size());
      arena_begin += right.size();
      density_motion = std::span(storage.get() + arena_begin, density.size());
      arena_begin += density.size();
      weighted_motion = std::span(storage.get() + arena_begin, density.size());
      arena_begin += density.size();
      std::copy(right.begin(), right.end(), right_snapshot.begin());
      right = right_snapshot;
      std::fill(density_motion.begin(), density_motion.end(), 0);
      std::fill(weighted_motion.begin(), weighted_motion.end(), 0);
      if (!density_right.empty())
        std::copy(density_right.begin(), density_right.end(), density_motion.begin());
      if (!weighted_density_right.empty())
        std::copy(weighted_density_right.begin(), weighted_density_right.end(),
                  weighted_motion.begin());
    }
    auto arena = std::span(storage.get() + arena_begin, storage_count - arena_begin);
    std::fill(stage.begin(), stage.end(), 0);
    std::fill(correction.begin(), correction.end(), 0);
    const auto accumulate_field = [&](std::size_t field, double value, double scale) {
      finite(scale);
      const auto term = scale * value;
      const auto adjusted = term - correction[field];
      const auto total = stage[field] + adjusted;
      finite(term);
      finite(adjusted);
      finite(total);
      correction[field] = (total - stage[field]) - adjusted;
      finite(correction[field]);
      stage[field] = total;
      census.projection_lanes = add(census.projection_lanes, 1);
    };
    const auto accumulate = [&](std::span<const double, 4> jet, double scale) {
      for (std::size_t field = 0; field < 4; ++field) accumulate_field(field, jet[field], scale);
    };
    const auto scatter = [&](std::size_t atom, std::size_t origin, std::size_t axis,
                             double gradient, double mixed, double scale, double moving_scale) {
      if (atom == origin) return;
      for (const auto field : {std::size_t{2}, 2 + 3 * atoms}) {
        const auto value = field == 2 ? gradient : mixed;
        accumulate_field(field + 3 * atom + axis, value, scale);
        accumulate_field(field + 3 * origin + axis, -value, scale);
      }
      accumulate_field(2 + 3 * atoms + 3 * atom + axis, gradient, moving_scale);
      accumulate_field(2 + 3 * atoms + 3 * origin + axis, -gradient, moving_scale);
    };
    const auto* radial = basis.packed.data() + 3 * atoms;
    for (std::size_t row = 0; row < aos; ++row)
      for (std::size_t column = 0; column < aos; ++column) {
        const auto& first = orbitals[row];
        const auto& second = orbitals[column];
        const auto hcore_weight = weight(hcore, {row, column, 0, 0}, panels, result, arena);
        const auto overlap_weight = weight(metric, {row, column, 0, 0}, panels, result, arena);
        std::array<double, 9> centers{}, moving_left{}, moving_right{};
        for (std::size_t slot = 0; slot < 2; ++slot) {
          const auto atom = slot == 0 ? first.atom : second.atom;
          std::copy_n(basis.packed.data() + 3 * atom, 3, centers.data() + 3 * slot);
          if (!column_schedule)
            std::copy_n(left.data() + 3 * atom, 3, moving_left.data() + 3 * slot);
          std::copy_n(right.data() + 3 * atom, 3, moving_right.data() + 3 * slot);
        }
        for (std::size_t term_first = 0; term_first < first.terms; ++term_first)
          for (std::size_t term_second = 0; term_second < second.terms; ++term_second) {
            const auto* expansion_first = first.record + 4 + 4 * term_first;
            const auto* expansion_second = second.record + 4 + 4 * term_second;
            std::array<unsigned, 6> powers{};
            for (std::size_t axis = 0; axis < 3; ++axis) {
              powers[axis] = static_cast<unsigned>(expansion_first[axis]);
              powers[3 + axis] = static_cast<unsigned>(expansion_second[axis]);
            }
            for (std::size_t index_first = first.begin; index_first < first.begin + first.count;
                 ++index_first)
              for (std::size_t index_second = second.begin;
                   index_second < second.begin + second.count; ++index_second) {
                const double exponents[]{radial[2 * index_first], radial[2 * index_second]};
                const auto scale = expansion_first[3] * expansion_second[3] *
                                   radial[2 * index_first + 1] * radial[2 * index_second + 1];
                for (std::size_t lane = 0; lane < (column_schedule ? 3 : 1); ++lane) {
                  if (column_schedule) {
                    moving_left.fill(0);
                    moving_left[lane] = 1;
                  }
                  std::array<double, 8> st{};
                  census.one_electron_primitives = add(census.one_electron_primitives, 1);
                  if (!integrals::host_mixed::overlap_kinetic(exponents, centers.data(),
                                                              powers.data(), moving_left.data(),
                                                              moving_right.data(), st.data()))
                    throw std::domain_error("canonical mixed S/T primitive failed");
                  for (const auto begin : {std::size_t{0}, std::size_t{4}}) {
                    const auto coefficients = begin == 0 ? overlap_weight : hcore_weight;
                    const auto weight = scale * coefficients[0],
                               moving_weight = scale * coefficients[1];
                    if (!column_schedule)
                      accumulate(std::span(st).subspan(begin, 4).first<4>(), weight);
                    else {
                      if (!lane) {
                        accumulate_field(0, st[begin], weight);
                        accumulate_field(1, st[begin + 2], weight);
                        accumulate_field(1, st[begin], moving_weight);
                      }
                      scatter(first.atom, second.atom, lane, st[begin + 1], st[begin + 3], weight,
                              moving_weight);
                    }
                  }
                }
                for (std::size_t atom = 0; atom < atoms; ++atom) {
                  std::copy_n(basis.packed.data() + 3 * atom, 3, centers.data() + 6);
                  if (!column_schedule)
                    std::copy_n(left.data() + 3 * atom, 3, moving_left.data() + 6);
                  std::copy_n(right.data() + 3 * atom, 3, moving_right.data() + 6);
                  for (std::size_t lane = 0; lane < (column_schedule ? 6 : 1); ++lane) {
                    if (column_schedule) {
                      moving_left.fill(0);
                      moving_left[lane] = 1;
                    }
                    std::array<double, 4> attraction{};
                    census.one_electron_primitives = add(census.one_electron_primitives, 1);
                    if (!integrals::host_mixed::attraction(exponents, centers.data(), powers.data(),
                                                           moving_left.data(), moving_right.data(),
                                                           attraction.data()))
                      throw std::domain_error("canonical mixed attraction primitive failed");
                    const auto ionic_scale = scale * system.atoms[atom].ionic_charge();
                    const auto weight = ionic_scale * hcore_weight[0],
                               moving_weight = ionic_scale * hcore_weight[1];
                    if (!column_schedule)
                      accumulate(attraction, weight);
                    else {
                      if (!lane) {
                        accumulate_field(0, attraction[0], weight);
                        accumulate_field(1, attraction[2], weight);
                        accumulate_field(1, attraction[0], moving_weight);
                      }
                      scatter(lane < 3 ? first.atom : second.atom, atom, lane % 3, attraction[1],
                              attraction[3], weight, moving_weight);
                    }
                  }
                }
              }
          }
        census.ao_pairs = add(census.ao_pairs, 1);
        validate();
      }
    std::size_t since_validation{};
    integral_quartets(aos, options.eri_schedule, [&](const auto& indices) {
      // Physical nuclear jets are invariant across this real-ERI orbit. Sum
      // compiler weights, including explicit D' terms, before canonical scatter.
      std::array<double, 2> quartet_weight{};
      quartet_incidences(indices, options.eri_schedule, [&](const auto& incidence) {
        const auto incidence_weight = weight(quartet, incidence, panels, result, arena);
        for (std::size_t lane = 0; lane < quartet_weight.size(); ++lane) {
          quartet_weight[lane] += incidence_weight[lane];
          finite(quartet_weight[lane]);
        }
        census.ao_quartets = add(census.ao_quartets, 1);
        if (++since_validation == options.validation_quartets) {
          validate();
          since_validation = 0;
        }
      });
      std::array<double, 17> record{};
      std::array<double, 12> moving_left{}, moving_right{};
      const auto origin = orbitals[indices[3]].atom;
      for (std::size_t slot = 0; slot < 4; ++slot) {
        const auto atom = orbitals[indices[slot]].atom;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          record[4 + 3 * slot + axis] =
              basis.packed[3 * atom + axis] - basis.packed[3 * origin + axis];
          if (!column_schedule)
            moving_left[3 * slot + axis] = left[3 * atom + axis] - left[3 * origin + axis];
          moving_right[3 * slot + axis] = right[3 * atom + axis] - right[3 * origin + axis];
        }
      }
      record[16] = 1;
      // Only the integral's mixed field vanishes under common right motion.
      // Its value and first gradients still contract nonzero moving-D/W weights.
      const bool first_only = options.eri_schedule == HostNuclearEriSchedule::SymmetryUnique &&
                              std::all_of(moving_right.begin(), moving_right.end(),
                                          [](double value) { return value == 0; });
      // A single physical owner also zeros every left nuclear scatter. Keep
      // the value for scalar and moving-D fields, not independent gradients.
      const bool value_only = options.eri_schedule == HostNuclearEriSchedule::SymmetryUnique &&
                              std::all_of(indices.begin(), indices.end(), [&](std::size_t index) {
                                return orbitals[index].atom == origin;
                              });
      std::array<unsigned, 4> angular{};
      const auto evaluate = [&](const auto& self, std::size_t slot, unsigned component,
                                double scale) -> void {
        if (slot == 4) {
          std::array<double, 25> primitive_jet{};
          census.eri_primitives = add(census.eri_primitives, 1);
          bool valid{};
          if (value_only) {
            census.eri_value_only_primitives = add(census.eri_value_only_primitives, 1);
            valid = integrals::full_range_primitive_value_cpu(angular.data(), component,
                                                              record.data(), primitive_jet.data());
          } else if (first_only) {
            census.eri_first_only_primitives = add(census.eri_first_only_primitives, 1);
            valid = integrals::full_range_primitive_first_cpu(angular.data(), component,
                                                              record.data(), primitive_jet.data());
          } else {
            census.eri_mixed_primitives = add(census.eri_mixed_primitives, 1);
            valid =
                scf::factorized_second::primitive(0, 0, angular.data(), component, record.data(),
                                                  moving_right.data(), primitive_jet.data());
          }
          if (!valid) throw std::domain_error("canonical mixed ERI primitive failed");
          const auto weight = scale * quartet_weight[0], moving_weight = scale * quartet_weight[1];
          if (column_schedule) {
            double directional{};
            for (std::size_t coordinate = 0; coordinate < 9; ++coordinate) {
              directional += moving_right[coordinate] * primitive_jet[1 + coordinate];
              scatter(orbitals[indices[coordinate / 3]].atom, origin, coordinate % 3,
                      primitive_jet[1 + coordinate], primitive_jet[13 + coordinate], weight,
                      moving_weight);
            }
            accumulate_field(0, primitive_jet[0], weight);
            accumulate_field(1, directional, weight);
            accumulate_field(1, primitive_jet[0], moving_weight);
          } else {
            std::array<double, 4> jet{primitive_jet[0], 0, 0, 0};
            for (std::size_t coordinate = 0; coordinate < 12; ++coordinate) {
              jet[1] += moving_left[coordinate] * primitive_jet[1 + coordinate];
              jet[2] += moving_right[coordinate] * primitive_jet[1 + coordinate];
              jet[3] += moving_left[coordinate] * primitive_jet[13 + coordinate];
            }
            accumulate(jet, weight);
          }
          return;
        }
        const auto& ao = orbitals[indices[slot]];
        for (std::size_t term = 0; term < ao.terms; ++term) {
          const auto* expansion = ao.record + 4 + 4 * term;
          const auto degree = static_cast<unsigned>(expansion[0] + expansion[1] + expansion[2]);
          angular[slot] = degree;
          const auto local_component =
              primitive::component_index(expansion[0], expansion[1], expansion[2]) -
              degree * (degree + 1) * (degree + 2) / 6;
          const auto size = (degree + 1) * (degree + 2) / 2;
          for (std::size_t index = ao.begin; index < ao.begin + ao.count; ++index) {
            record[slot] = radial[2 * index];
            self(self, slot + 1, component * size + local_component,
                 scale * expansion[3] * radial[2 * index + 1]);
          }
        }
      };
      evaluate(evaluate, 0, 0, 1);
    });
    validate();
    if (column_schedule &&
        (!std::equal(right.begin(), right.end(), external_right.begin()) ||
         (!density_right.empty() &&
          !std::equal(density_motion.begin(), density_motion.end(), density_right.begin())) ||
         (!weighted_density_right.empty() &&
          !std::equal(weighted_motion.begin(), weighted_motion.end(),
                      weighted_density_right.begin()))))
      throw std::invalid_argument("integral column right input changed during invocation");
    census.directions_completed = add(census.directions_completed, 1);
    std::copy(stage.begin(), stage.end(), output.begin());
  }

  std::span<const double> weighted_density;
  std::span<double> density_motion, weighted_motion;
  HostNuclearIntegralOptions options;
  const bool column_schedule;
  const bool matrix_schedule;
  const bool projected_matrix_schedule;
  const std::size_t fields;
  HostNuclearIntegralPlan resources;
  const maps::Program& hcore;
  const maps::Program& metric;
  const maps::Program& quartet;
  std::size_t storage_count{};
  std::unique_ptr<double[]> storage;
  PolynomialLayout polynomial;
  std::span<double> projected_density, projection_compensation, density_snapshot, basis_snapshot;
  bool projection_inputs_bound{}, projection_ready{};
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostNuclearIntegralWork census;
};

HostKsNuclearIntegralCurvature::HostKsNuclearIntegralCurvature(
    const core::System& system, const AoBasis& basis, const scf::ResolvedFockBuild& strategy,
    std::span<const double> density, std::span<const double> weighted_density,
    std::string_view method_identity, HostNuclearIntegralOptions options,
    std::function<void()> validate_current, HostNuclearIntegralCurvatureSchedule schedule)
    : impl_(std::make_unique<Impl>(system, basis, strategy, density, weighted_density,
                                   method_identity, options, std::move(validate_current),
                                   schedule)) {}
HostKsNuclearIntegralCurvature::~HostKsNuclearIntegralCurvature() = default;
void HostKsNuclearIntegralCurvature::apply(std::span<const double> left,
                                           std::span<const double> right,
                                           std::span<double> output) {
  impl_->apply(left, right, output);
}
void HostKsNuclearIntegralCurvature::apply_column(std::span<const double> right,
                                                  std::span<const double> density_right,
                                                  std::span<const double> weighted_density_right,
                                                  std::span<double> output) {
  impl_->apply_column(right, density_right, weighted_density_right, output);
}
void HostKsNuclearIntegralCurvature::apply_matrix(std::span<double> output) {
  impl_->apply_matrix(output);
}
HostNuclearIntegralPlan HostKsNuclearIntegralCurvature::resources() const noexcept {
  return impl_->resources;
}
HostNuclearIntegralWork HostKsNuclearIntegralCurvature::work() const noexcept {
  return impl_->census;
}

}  // namespace generativeqc::dft
