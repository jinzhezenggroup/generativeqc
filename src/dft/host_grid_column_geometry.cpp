#include "dft/host_grid_column_geometry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "host_grid_geometry_ad.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {
namespace {
namespace ad = generativeqc_grid_geometry_ad;
using runtime::size_add;
using runtime::size_mul;
using Dual = std::array<double, 2>;

std::size_t pairs(std::size_t atoms) {
  return atoms % 2 ? size_mul(atoms, (atoms - 1) / 2) : size_mul(atoms / 2, atoms - 1);
}

std::size_t capacity(std::size_t atoms) {
  return size_add(size_add(size_mul(20, pairs(atoms)), size_mul(12, atoms)), 16);
}

void finite(std::span<const double> values) {
  for (const auto value : values)
    if (!std::isfinite(value)) throw std::domain_error("nonfinite grid column geometry");
}

bool overlaps(std::span<const double> first, std::span<const double> second) {
  if (first.empty() || second.empty()) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first.data());
  const auto right = reinterpret_cast<std::uintptr_t>(second.data());
  return left <= right ? right - left < first.size_bytes() : left - right < second.size_bytes();
}

ad::Jet jet(Dual value) { return {value[0], 0, value[1], 0}; }
Dual dual(ad::Jet value) { return {value.value, value.right}; }
Dual add(Dual first, Dual second) { return dual(ad::add(jet(first), jet(second))); }
Dual multiply(Dual first, Dual second) { return dual(ad::multiply(jet(first), jet(second))); }

double binary_scale(double value, long long exponent) {
  if (!value) return value;
  if (exponent > std::numeric_limits<int>::max())
    throw std::domain_error("grid column binary scale overflow");
  if (exponent < std::numeric_limits<int>::min()) return std::copysign(0., value);
  return std::ldexp(value, static_cast<int>(exponent));
}

Dual scale(Dual value, long long exponent) {
  return {binary_scale(value[0], exponent), binary_scale(value[1], exponent)};
}

/** Reverse adjoints keep their own binary scale: a zero product can still
 * have a nonzero gradient/HVP after a very large normalization of its peers.
 * Materializing that scale as a floating coefficient would overflow first.
 */
struct Scaled {
  Dual value{};
  long long exponent{};

  void normalize() {
    finite(value);
    const auto maximum = std::max(std::abs(value[0]), std::abs(value[1]));
    if (!maximum) {
      exponent = 0;
      return;
    }
    int shift{};
    (void)std::frexp(maximum, &shift);
    value = scale(value, -shift);
    exponent += shift;
  }

  void accumulate(Scaled contribution) {
    if (contribution.value == Dual{}) return;
    if (value == Dual{}) {
      *this = contribution;
      return;
    }
    const auto common = std::max(exponent, contribution.exponent);
    value = add(scale(value, exponent - common),
                scale(contribution.value, contribution.exponent - common));
    exponent = common;
    normalize();
  }

  Dual total() const {
    auto result = scale(value, exponent);
    finite(result);
    return result;
  }
};

struct Node {
  Dual value{};
  Scaled adjoint;
  std::array<Dual, 3> partials{};
  std::array<std::size_t, 3> parents{};
  std::size_t arity{};
  bool scaling{};
  long long exponent{};
};

/** A finite-slot schedule of the incumbent scalar Graph jets, not a second
 * derivative implementation. Unit left seeds recover local Jacobian/JVP
 * coefficients; a single reverse traversal propagates them to physical leaves.
 */
class Tape {
 public:
  Tape(std::size_t bound, HostGridColumnWork& work) : bound(bound), work(work) {}

  std::size_t append(Node node) {
    if (nodes.size() == bound) throw std::length_error("grid column tape capacity exceeded");
    finite(node.value);
    nodes.push_back(node);
    work.forward_nodes = size_add(work.forward_nodes, 1);
    return nodes.size() - 1;
  }

  std::size_t leaf(double value, double right = 0) {
    Node node;
    node.value = {value, right};
    return append(node);
  }

  template <std::size_t Arity, class Evaluate>
  std::size_t action(std::array<std::size_t, Arity> parents, Evaluate evaluate) {
    Node node;
    node.arity = Arity;
    std::copy(parents.begin(), parents.end(), node.parents.begin());
    for (std::size_t seed = 0; seed < Arity; ++seed) {
      std::array<ad::Jet, Arity> inputs;
      for (std::size_t index = 0; index < Arity; ++index) {
        inputs[index] = jet(nodes[parents[index]].value);
        inputs[index].left = index == seed ? 1 : 0;
      }
      const auto result = evaluate(inputs);
      if (!seed) node.value = dual(result);
      node.partials[seed] = {result.left, result.mixed};
      finite(node.partials[seed]);
    }
    return append(node);
  }

  std::size_t binary(std::size_t first, std::size_t second, ad::Jet (*evaluate)(ad::Jet, ad::Jet)) {
    return action(std::array{first, second},
                  [evaluate](auto inputs) { return evaluate(inputs[0], inputs[1]); });
  }

  std::size_t rescale(std::size_t parent, long long exponent) {
    Node node;
    node.arity = 1;
    node.parents[0] = parent;
    node.scaling = true;
    node.exponent = exponent;
    node.value = scale(nodes[parent].value, exponent);
    return append(node);
  }

  std::size_t normalize(std::size_t parent, long long& exponent) {
    const auto value = nodes[parent].value;
    const auto maximum = std::max(std::abs(value[0]), std::abs(value[1]));
    int shift{};
    if (maximum) (void)std::frexp(maximum, &shift);
    exponent += shift;
    return rescale(parent, -shift);
  }

  std::size_t norm(const std::array<std::size_t, 3>& first,
                   const std::array<std::size_t, 3>& second) {
    std::array<std::size_t, 3> relative;
    for (std::size_t axis = 0; axis < 3; ++axis)
      relative[axis] = binary(first[axis], second[axis], ad::subtract);
    return action(relative, [](auto inputs) { return ad::norm(inputs[0], inputs[1], inputs[2]); });
  }

  void reverse(std::size_t root) {
    for (auto& node : nodes) node.adjoint = {};
    nodes[root].adjoint.value = {1, 0};
    for (std::size_t offset = nodes.size(); offset; --offset) {
      const auto& node = nodes[offset - 1];
      for (std::size_t index = 0; index < node.arity; ++index) {
        auto contribution = node.adjoint;
        if (node.scaling)
          contribution.exponent += node.exponent;
        else {
          Scaled partial{node.partials[index], 0};
          partial.normalize();
          contribution.value = multiply(contribution.value, partial.value);
          contribution.exponent += partial.exponent;
          contribution.normalize();
        }
        nodes[node.parents[index]].adjoint.accumulate(contribution);
      }
    }
    work.reverse_nodes = size_add(work.reverse_nodes, nodes.size());
  }

  std::vector<Node> nodes;
  std::size_t bound;
  HostGridColumnWork& work;
};

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("grid column overlaps or reenters");
  }
  ~Busy() { flag.clear(); }
};
}  // namespace

HostGridColumnPlan HostGridColumnGeometry::plan(const HostGridGeometry& geometry) {
  const auto atoms = geometry.basis().natom, count = geometry.tile_points();
  const auto elements =
      size_add(size_mul(20, size_mul(count, geometry.basis().nao)),
               size_add(size_mul(size_add(2, size_mul(6, atoms)), count), size_mul(3, atoms)));
  const auto metadata =
      size_add(size_mul(capacity(atoms), sizeof(Node)),
               size_add(size_mul(size_add(size_mul(2, atoms), pairs(atoms)), sizeof(std::size_t)),
                        size_mul(atoms, sizeof(long long))));
  const auto bytes =
      size_add(geometry.scratch_bytes(), size_add(size_mul(elements, sizeof(double)), metadata));
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("grid column exceeds signed resource domain");
  return {bytes, count, size_mul(count, geometry.basis().nao),
          size_mul(size_add(count, 1), pairs(atoms)),
          size_mul(size_add(size_mul(2, count), 1), capacity(atoms))};
}

class HostGridColumnGeometry::Impl {
 public:
  Impl(HostGridGeometry& reader, std::span<const double> measures, HostGridColumnPlan limits,
       std::function<void()> validate)
      : reader(reader),
        measures(measures),
        validator(std::move(validate)),
        resources(HostGridColumnGeometry::plan(reader)),
        atoms(reader.basis().natom),
        tape(capacity(atoms), census) {
    if (!validator || !measures.data() || measures.size() != reader.grid().point_count())
      throw std::invalid_argument("grid column requires authenticated atomic measures");
    if (resources.numeric_bytes > limits.numeric_bytes ||
        resources.point_visits > limits.point_visits ||
        resources.ao_point_visits > limits.ao_point_visits ||
        resources.pair_visits > limits.pair_visits || resources.tape_visits > limits.tape_visits)
      throw std::length_error("grid column capacity or work cap exceeded");
    this->validate();
    tape.nodes.reserve(tape.bound);
    right.resize(size_mul(3, atoms));
    distances.resize(pairs(atoms));
    products.resize(atoms);
    radii.resize(atoms);
    exponents.resize(atoms);
    storage = std::make_unique<double[]>(
        size_add(size_mul(20, size_mul(reader.tile_points(), reader.basis().nao)),
                 size_mul(size_add(2, size_mul(6, atoms)), reader.tile_points())));
  }

  void validate() const {
    validator();
    reader.validate_current();
  }

  std::array<std::size_t, 3> coordinates(std::size_t atom) const {
    return {3 * atom, 3 * atom + 1, 3 * atom + 2};
  }

  std::size_t clamped(std::size_t value) {
    const auto primal = tape.nodes[value].value[0];
    if (primal <= -1) return tape.leaf(-1);
    if (primal >= 1) return tape.leaf(1);
    return value;
  }

  std::size_t partition(std::size_t point, const std::array<std::size_t, 3>& position) {
    const auto& grid = reader.grid();
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      products[atom] = tape.leaf(1);
      exponents[atom] = 0;
    }
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      radii[atom] = tape.norm(position, coordinates(atom));
      if (!(tape.nodes[radii[atom]].value[0] > 0))
        throw std::domain_error("grid column point coincides with a nucleus");
    }
    for (std::size_t first = 0; first < atoms; ++first)
      for (std::size_t second = 0; second < first; ++second) {
        auto coordinate = tape.binary(tape.binary(radii[first], radii[second], ad::subtract),
                                      distances[pairs(first) + second], ad::ratio);
        coordinate = clamped(coordinate);
        if (grid.spec().version == 4) {
          const auto adjustment =
              grid_radius_adjustment(grid.spec(), grid.system().atoms[first].atomic_number,
                                     grid.system().atoms[second].atomic_number);
          coordinate = tape.action(std::array{coordinate}, [adjustment](auto inputs) {
            return ad::treutler_adjust(inputs[0], {adjustment, 0, 0, 0});
          });
          coordinate = clamped(coordinate);
        }
        auto factor = tape.action(std::array{coordinate}, [&grid](auto inputs) {
          switch (grid.spec().partition_iterations) {
            case 1:
              return ad::becke_1(inputs[0]);
            case 2:
              return ad::becke_2(inputs[0]);
            case 3:
              return ad::becke_3(inputs[0]);
            case 4:
              return ad::becke_4(inputs[0]);
            default:
              return ad::becke_5(inputs[0]);
          }
        });
        if (tape.nodes[factor].value[0] < 0) factor = tape.leaf(0);
        if (tape.nodes[factor].value[0] > 1) factor = tape.leaf(1);
        const auto complement =
            tape.action(std::array{factor}, [](auto inputs) { return ad::complement(inputs[0]); });
        for (const auto& [atom, multiplier] :
             std::array{std::pair{first, factor}, std::pair{second, complement}}) {
          long long factor_exponent{};
          const auto normalized = tape.normalize(multiplier, factor_exponent);
          exponents[atom] += factor_exponent;
          products[atom] = tape.normalize(tape.binary(products[atom], normalized, ad::multiply),
                                          exponents[atom]);
        }
        census.pair_visits = size_add(census.pair_visits, 1);
      }
    auto common = std::numeric_limits<long long>::min();
    for (std::size_t atom = 0; atom < atoms; ++atom)
      if (tape.nodes[products[atom]].value != Dual{}) common = std::max(common, exponents[atom]);
    if (common == std::numeric_limits<long long>::min())
      throw std::domain_error("grid column partition has no finite product");
    auto total = tape.leaf(0);
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      products[atom] = tape.rescale(products[atom], exponents[atom] - common);
      total = tape.binary(total, products[atom], ad::add);
    }
    if (!(tape.nodes[total].value[0] > 0))
      throw std::domain_error("invalid grid column partition normalization");
    return tape.binary(products[grid.owners()[point]], total, ad::ratio);
  }

  void read(std::size_t begin, std::size_t count, std::span<const double> direction,
            std::span<double> ao, std::span<double> weights) {
    Busy guard(busy);
    validate();
    const auto& grid = reader.grid();
    const auto lanes = size_mul(count, reader.basis().nao),
               fields = size_add(2, size_mul(6, atoms));
    if (begin > grid.point_count() || count > grid.point_count() - begin ||
        count > reader.tile_points() || direction.size() != right.size() || !direction.data() ||
        ao.size() != size_mul(20, lanes) || weights.size() != size_mul(fields, count) ||
        (!ao.empty() && !ao.data()) || (!weights.empty() && !weights.data()))
      throw std::invalid_argument("grid column geometry shape mismatch");
    reject(ao);
    reject(weights);
    if (overlaps(ao, weights)) throw std::invalid_argument("overlapping grid column outputs");
    finite(direction);
    std::copy(direction.begin(), direction.end(), right.begin());
    if (!count) return;
    census.tiles_started = size_add(census.tiles_started, 1);
    auto jets = std::span(storage.get(), size_mul(20, lanes));
    auto column = std::span(jets.data() + jets.size(), size_mul(fields, count));
    reader.basis().evaluate(grid.points().data() + 3 * begin, count, 3, 0, reader.basis().nao,
                            jets.data(), jets.size());
    census.ao_point_visits = size_add(census.ao_point_visits, lanes);
    finite(jets);
    validate();
    tape.nodes.clear();
    for (std::size_t atom = 0; atom < atoms; ++atom)
      for (std::size_t axis = 0; axis < 3; ++axis)
        (void)tape.leaf(grid.system().atoms[atom].position[axis], right[3 * atom + axis]);
    for (std::size_t first = 0; first < atoms; ++first)
      for (std::size_t second = 0; second < first; ++second) {
        const auto index = tape.norm(coordinates(first), coordinates(second));
        if (!(tape.nodes[index].value[0] > grid.spec().coincident_tolerance))
          throw std::domain_error("grid column requires separated nuclear centers");
        distances[pairs(first) + second] = index;
        census.pair_visits = size_add(census.pair_visits, 1);
      }
    const auto prefix = tape.nodes.size();
    for (std::size_t point = 0; point < count; ++point) {
      tape.nodes.resize(prefix);
      const auto owner = grid.owners()[begin + point];
      if (owner >= atoms) throw std::invalid_argument("invalid grid column point owner");
      std::array<std::size_t, 3> position;
      for (std::size_t axis = 0; axis < 3; ++axis)
        position[axis] =
            tape.leaf(grid.points()[3 * (begin + point) + axis], right[3 * owner + axis]);
      const auto root = partition(begin + point, position);
      const auto value = tape.nodes[root].value;
      const auto measure = measures[begin + point], retained = grid.weights()[begin + point];
      finite(std::array{measure, retained});
      if (measure == 0 && retained != 0)
        throw std::domain_error("inconsistent grid column zero atomic measure");
      const auto expected = measure == 0 ? 0 : retained / measure;
      if (!std::isfinite(expected) ||
          (measure != 0 && std::abs(value[0] - expected) >
                               1e-12 + 1e-10 * std::max(std::abs(value[0]), std::abs(expected))))
        throw std::domain_error("grid column retained partition identity mismatch");
      tape.reverse(root);
      column[point] = retained;
      column[count + point] = measure * value[1];
      for (std::size_t atom = 0; atom < atoms; ++atom)
        for (std::size_t axis = 0; axis < 3; ++axis) {
          auto adjoint = tape.nodes[3 * atom + axis].adjoint;
          if (atom == owner) adjoint.accumulate(tape.nodes[position[axis]].adjoint);
          const auto result = adjoint.total();
          const auto coordinate = 3 * atom + axis;
          column[(2 + coordinate) * count + point] = measure * result[0];
          column[(2 + 3 * atoms + coordinate) * count + point] = measure * result[1];
        }
      census.point_visits = size_add(census.point_visits, 1);
      validate();
    }
    finite(column);
    validate();
    if (!std::equal(right.begin(), right.end(), direction.begin()))
      throw std::invalid_argument("grid column direction changed during invocation");
    std::copy(jets.begin(), jets.end(), ao.begin());
    std::copy(column.begin(), column.end(), weights.begin());
    census.tiles_completed = size_add(census.tiles_completed, 1);
  }

  void reject(std::span<const double> values) const {
    reader.reject_borrowed_alias(values);
    if (overlaps(values, measures))
      throw std::invalid_argument("grid column output aliases measures");
  }

  HostGridGeometry& reader;
  std::span<const double> measures;
  std::function<void()> validator;
  HostGridColumnPlan resources;
  std::size_t atoms;
  HostGridColumnWork census;
  Tape tape;
  std::vector<double> right;
  std::vector<std::size_t> distances, products, radii;
  std::vector<long long> exponents;
  std::unique_ptr<double[]> storage;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

HostGridColumnGeometry::HostGridColumnGeometry(HostGridGeometry& geometry,
                                               std::span<const double> atomic_weights,
                                               HostGridColumnPlan limits,
                                               std::function<void()> validate_current)
    : impl_(std::make_unique<Impl>(geometry, atomic_weights, limits, std::move(validate_current))) {
}
HostGridColumnGeometry::~HostGridColumnGeometry() = default;
void HostGridColumnGeometry::read(std::size_t begin, std::size_t count,
                                  std::span<const double> right, std::span<double> ao_jets,
                                  std::span<double> weight_column) {
  impl_->read(begin, count, right, ao_jets, weight_column);
}
HostGridColumnPlan HostGridColumnGeometry::resources() const noexcept { return impl_->resources; }
HostGridColumnWork HostGridColumnGeometry::work() const noexcept { return impl_->census; }
const HostGridGeometry& HostGridColumnGeometry::geometry() const noexcept { return impl_->reader; }
HostGridGeometry& HostGridColumnGeometry::geometry() noexcept { return impl_->reader; }
void HostGridColumnGeometry::validate_current() const { impl_->validate(); }
void HostGridColumnGeometry::reject_borrowed_alias(std::span<const double> values) const {
  impl_->reject(values);
}
}  // namespace generativeqc::dft
