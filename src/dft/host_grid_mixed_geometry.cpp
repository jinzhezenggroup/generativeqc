#include "dft/host_grid_mixed_geometry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
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

std::size_t pairs(std::size_t atoms) {
  return atoms % 2 ? size_mul(atoms, (atoms - 1) / 2) : size_mul(atoms / 2, atoms - 1);
}

std::size_t pair_index(std::size_t first, std::size_t second) {
  return size_add(pairs(first), second);
}

bool overlaps(std::span<const double> first, std::span<const double> second) {
  if (first.empty() || second.empty()) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first.data());
  const auto right = reinterpret_cast<std::uintptr_t>(second.data());
  return left <= right ? right - left < first.size_bytes() : left - right < second.size_bytes();
}

void finite(std::span<const double> values) {
  for (const auto value : values)
    if (!std::isfinite(value)) throw std::domain_error("nonfinite mixed grid geometry");
}

double magnitude(ad::Jet value) {
  const std::array fields{value.value, value.left, value.right, value.mixed};
  finite(fields);
  return std::max(
      {std::abs(value.value), std::abs(value.left), std::abs(value.right), std::abs(value.mixed)});
}

ad::Jet scale(ad::Jet value, int exponent) {
  return {std::ldexp(value.value, exponent), std::ldexp(value.left, exponent),
          std::ldexp(value.right, exponent), std::ldexp(value.mixed, exponent)};
}

void normalize(ad::Jet& value, long long& exponent) {
  const auto maximum = magnitude(value);
  if (!maximum) {
    exponent = 0;
    return;
  }
  int shift{};
  (void)std::frexp(maximum, &shift);
  value = scale(value, -shift);
  exponent += shift;
}

void multiply(ad::Jet& product, long long& exponent, ad::Jet factor) {
  long long factor_exponent{};
  normalize(factor, factor_exponent);
  product = ad::multiply(product, factor);
  exponent += factor_exponent;
  normalize(product, exponent);
}

ad::Jet switch_pair(ad::Jet coordinate, unsigned iterations) {
  if (coordinate.value <= -1)
    coordinate = {-1, 0, 0, 0};
  else if (coordinate.value >= 1)
    coordinate = {1, 0, 0, 0};
  ad::Jet result;
  switch (iterations) {
    case 1:
      result = ad::becke_1(coordinate);
      break;
    case 2:
      result = ad::becke_2(coordinate);
      break;
    case 3:
      result = ad::becke_3(coordinate);
      break;
    case 4:
      result = ad::becke_4(coordinate);
      break;
    default:
      result = ad::becke_5(coordinate);
  }
  (void)magnitude(result);
  if (result.value < 0) return {};
  if (result.value > 1) return {1, 0, 0, 0};
  return result;
}

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("mixed grid geometry overlaps or reenters");
  }
  ~Busy() { flag.clear(); }
};
}  // namespace

HostGridMixedPlan HostGridMixedGeometry::plan(const HostGridGeometry& geometry) {
  const auto atoms = geometry.basis().natom;
  const auto count = geometry.tile_points();
  const auto lanes = size_mul(count, geometry.basis().nao);
  const auto elements = size_add(size_add(size_mul(32, lanes), size_mul(4, count)),
                                 size_mul(4, size_add(pairs(atoms), size_mul(2, atoms))));
  const auto bytes = size_add(size_mul(elements, sizeof(double)), geometry.scratch_bytes());
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("mixed grid geometry exceeds signed resource domain");
  return {bytes, count, lanes, size_mul(size_add(count, 1), pairs(atoms))};
}

class HostGridMixedGeometry::Impl {
 public:
  Impl(HostGridGeometry& geometry, std::span<const double> atomic_weights, HostGridMixedPlan limits,
       std::function<void()> validator)
      : geometry(geometry),
        atomic_weights(atomic_weights),
        validator(std::move(validator)),
        resources(HostGridMixedGeometry::plan(geometry)),
        atoms(geometry.basis().natom),
        aos(geometry.basis().nao) {
    if (!this->validator || !atomic_weights.data() ||
        atomic_weights.size() != geometry.grid().point_count())
      throw std::invalid_argument("mixed grid geometry requires authenticated atomic measures");
    if (resources.numeric_bytes > limits.numeric_bytes ||
        resources.point_visits > limits.point_visits ||
        resources.ao_point_visits > limits.ao_point_visits ||
        resources.pair_visits > limits.pair_visits)
      throw std::length_error("mixed grid geometry capacity or work cap exceeded");
    validate();
    const auto offset = size_add(size_mul(3, atoms), size_mul(2, geometry.basis().nprimitive));
    for (std::size_t orbital = 0; orbital < aos; ++orbital)
      ao_atoms.push_back(static_cast<std::size_t>(geometry.basis().packed[offset + 16 * orbital]));
    const auto count = geometry.tile_points();
    storage = std::make_unique<double[]>(
        size_add(size_mul(32, size_mul(count, aos)), size_mul(4, count)));
    scratch = std::make_unique<ad::Jet[]>(size_add(pairs(atoms), size_mul(2, atoms)));
    exponents.resize(atoms);
  }

  void validate() const {
    validator();
    geometry.validate_current();
  }

  ad::Jet partition(std::size_t point, std::span<const double> left,
                    std::span<const double> right) {
    const auto& grid = geometry.grid();
    const auto owner = grid.owners()[point];
    if (owner >= atoms) throw std::invalid_argument("mixed grid geometry owner is invalid");
    auto* distances = scratch.get() + pairs(atoms);
    auto* products = distances + atoms;
    const auto& system = grid.system();
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      std::array<ad::Jet, 3> coordinates;
      for (std::size_t axis = 0; axis < 3; ++axis)
        coordinates[axis] = {grid.points()[3 * point + axis] - system.atoms[atom].position[axis],
                             left[3 * owner + axis] - left[3 * atom + axis],
                             right[3 * owner + axis] - right[3 * atom + axis], 0};
      distances[atom] = ad::norm(coordinates[0], coordinates[1], coordinates[2]);
      (void)magnitude(distances[atom]);
      if (!(distances[atom].value > 0))
        throw std::domain_error("mixed grid geometry point coincides with a nucleus");
      products[atom] = {1, 0, 0, 0};
      exponents[atom] = 0;
    }
    for (std::size_t first = 0; first < atoms; ++first)
      for (std::size_t second = 0; second < first; ++second) {
        auto coordinate = ad::ratio(ad::subtract(distances[first], distances[second]),
                                    scratch[pair_index(first, second)]);
        (void)magnitude(coordinate);
        if (grid.spec().version == 4) {
          if (coordinate.value <= -1)
            coordinate = {-1, 0, 0, 0};
          else if (coordinate.value >= 1)
            coordinate = {1, 0, 0, 0};
          const auto adjustment = grid_radius_adjustment(
              grid.spec(), system.atoms[first].atomic_number, system.atoms[second].atomic_number);
          coordinate = ad::treutler_adjust(coordinate, {adjustment, 0, 0, 0});
        }
        const auto factor = switch_pair(coordinate, grid.spec().partition_iterations);
        multiply(products[first], exponents[first], factor);
        multiply(products[second], exponents[second], ad::complement(factor));
        census.pair_visits = size_add(census.pair_visits, 1);
      }
    long long common = LLONG_MIN;
    for (std::size_t atom = 0; atom < atoms; ++atom)
      if (magnitude(products[atom])) common = std::max(common, exponents[atom]);
    if (common == LLONG_MIN) throw std::domain_error("mixed grid partition has no finite product");
    ad::Jet total{}, numerator{};
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      const auto difference = exponents[atom] - common;
      const auto value = magnitude(products[atom]) && difference >= -1074
                             ? scale(products[atom], static_cast<int>(difference))
                             : ad::Jet{};
      total = ad::add(total, value);
      if (atom == owner) numerator = value;
    }
    (void)magnitude(total);
    if (!(total.value > 0)) throw std::domain_error("invalid mixed grid partition normalization");
    return ad::ratio(numerator, total);
  }

  void read(std::size_t begin, std::size_t count, std::span<const double> left,
            std::span<const double> right, std::span<double> ao, std::span<double> moving_ao,
            std::span<double> weights, std::span<double> moving_weights) {
    Busy guard(busy);
    validate();
    const auto& grid = geometry.grid();
    const auto lanes = size_mul(count, aos);
    if (begin > grid.point_count() || count > grid.point_count() - begin ||
        count > geometry.tile_points() || left.size() != size_mul(3, atoms) ||
        right.size() != left.size() || !left.data() || !right.data() ||
        ao.size() != size_mul(4, lanes) || moving_ao.size() != size_mul(12, lanes) ||
        weights.size() != count || moving_weights.size() != size_mul(3, count))
      throw std::invalid_argument("mixed grid geometry shape mismatch");
    const std::array outputs{ao, moving_ao, weights, moving_weights};
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto output = outputs[index];
      if (!output.empty() && !output.data()) throw std::invalid_argument("null mixed grid output");
      geometry.reject_borrowed_alias(output);
      if (overlaps(output, atomic_weights))
        throw std::invalid_argument("mixed grid output aliases atomic measures");
      for (std::size_t other = 0; other < index; ++other)
        if (overlaps(output, outputs[other]))
          throw std::invalid_argument("overlapping mixed grid outputs");
    }
    finite(left);
    finite(right);
    if (!count) return;
    census.tiles_started = size_add(census.tiles_started, 1);
    census.ao_point_visits = size_add(census.ao_point_visits, lanes);
    auto jets = std::span(storage.get(), size_mul(20, lanes));
    auto tangent = std::span(jets.data() + jets.size(), size_mul(12, lanes));
    auto weight = std::span(tangent.data() + tangent.size(), count);
    auto weight_tangent = std::span(weight.data() + count, size_mul(3, count));
    geometry.basis().evaluate(grid.points().data() + 3 * begin, count, 3, 0, aos, jets.data(),
                              jets.size());
    finite(jets);
    validate();
    for (std::size_t first = 0; first < atoms; ++first)
      for (std::size_t second = 0; second < first; ++second) {
        std::array<ad::Jet, 3> coordinates;
        for (std::size_t axis = 0; axis < 3; ++axis)
          coordinates[axis] = {grid.system().atoms[first].position[axis] -
                                   grid.system().atoms[second].position[axis],
                               left[3 * first + axis] - left[3 * second + axis],
                               right[3 * first + axis] - right[3 * second + axis], 0};
        const auto distance = ad::norm(coordinates[0], coordinates[1], coordinates[2]);
        (void)magnitude(distance);
        if (!(distance.value > grid.spec().coincident_tolerance))
          throw std::domain_error("mixed grid geometry requires separated nuclear centers");
        scratch[pair_index(first, second)] = distance;
        census.pair_visits = size_add(census.pair_visits, 1);
      }
    for (std::size_t point = 0; point < count; ++point) {
      const auto owner = grid.owners()[begin + point];
      if (owner >= atoms) throw std::invalid_argument("mixed grid point owner is invalid");
      for (std::size_t orbital = 0; orbital < aos; ++orbital) {
        std::array<double, 3> relative_left, relative_right;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          relative_left[axis] = left[3 * owner + axis] - left[3 * ao_atoms[orbital] + axis];
          relative_right[axis] = right[3 * owner + axis] - right[3 * ao_atoms[orbital] + axis];
        }
        std::array<double, 12> values;
        ad::ao_motion(jets.data() + point * aos + orbital, lanes, relative_left.data(),
                      relative_right.data(), values.data());
        for (std::size_t component = 0; component < 4; ++component)
          for (std::size_t moving = 0; moving < 3; ++moving)
            tangent[(moving * 4 + component) * lanes + point * aos + orbital] =
                values[3 * component + moving];
      }
      const auto measure = atomic_weights[begin + point];
      const auto retained = grid.weights()[begin + point];
      finite(std::array{measure, retained});
      // Some Lebedev rules have signed measures; only zero consistency and
      // the dimensionless retained partition, not the measure's sign, are gates.
      if (measure == 0 && retained != 0)
        throw std::domain_error("invalid mixed grid atomic measure");
      auto value = partition(begin + point, left, right);
      (void)magnitude(value);
      const auto expected = measure == 0 ? 0 : retained / measure;
      if (!std::isfinite(expected) ||
          (measure != 0 && std::abs(value.value - expected) >
                               1e-12 + 1e-10 * std::max(std::abs(value.value), std::abs(expected))))
        throw std::domain_error("mixed grid partition does not match the retained grid");
      value = ad::multiply(value, {measure, 0, 0, 0});
      (void)magnitude(value);
      weight[point] = retained;
      weight_tangent[point] = value.left;
      weight_tangent[count + point] = value.right;
      weight_tangent[2 * count + point] = value.mixed;
      census.point_visits = size_add(census.point_visits, 1);
    }
    finite(tangent);
    finite(weight_tangent);
    validate();
    census.tiles_completed = size_add(census.tiles_completed, 1);
    std::copy_n(jets.begin(), ao.size(), ao.begin());
    std::copy(tangent.begin(), tangent.end(), moving_ao.begin());
    std::copy(weight.begin(), weight.end(), weights.begin());
    std::copy(weight_tangent.begin(), weight_tangent.end(), moving_weights.begin());
  }

  HostGridGeometry& geometry;
  std::span<const double> atomic_weights;
  std::function<void()> validator;
  HostGridMixedPlan resources;
  std::size_t atoms, aos;
  std::vector<std::size_t> ao_atoms;
  std::unique_ptr<double[]> storage;
  std::unique_ptr<ad::Jet[]> scratch;
  std::vector<long long> exponents;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostGridMixedWork census;
};

HostGridMixedGeometry::HostGridMixedGeometry(HostGridGeometry& geometry,
                                             std::span<const double> atomic_weights,
                                             HostGridMixedPlan limits,
                                             std::function<void()> validate_current)
    : impl_(std::make_unique<Impl>(geometry, atomic_weights, limits, std::move(validate_current))) {
}
HostGridMixedGeometry::~HostGridMixedGeometry() = default;
void HostGridMixedGeometry::read(std::size_t begin, std::size_t count, std::span<const double> left,
                                 std::span<const double> right, std::span<double> ao,
                                 std::span<double> moving_ao, std::span<double> weights,
                                 std::span<double> moving_weights) {
  impl_->read(begin, count, left, right, ao, moving_ao, weights, moving_weights);
}
HostGridMixedPlan HostGridMixedGeometry::resources() const noexcept { return impl_->resources; }
HostGridMixedWork HostGridMixedGeometry::work() const noexcept { return impl_->census; }
const HostGridGeometry& HostGridMixedGeometry::geometry() const noexcept { return impl_->geometry; }
void HostGridMixedGeometry::validate_current() const { impl_->validate(); }
void HostGridMixedGeometry::reject_borrowed_alias(std::span<const double> values) const {
  impl_->geometry.reject_borrowed_alias(values);
  if (overlaps(values, impl_->atomic_weights))
    throw std::invalid_argument("mixed grid output aliases atomic measures");
}
}  // namespace generativeqc::dft
