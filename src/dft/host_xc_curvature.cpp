#include "dft/host_xc_curvature.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "dft/host_gga_fields.hpp"
#include "host_xc_curvature_maps.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_host_curvature_maps;
using runtime::size_add;
using runtime::size_mul;

const maps::Program& program(std::string_view name) {
  for (const auto& entry : maps::programs())
    if (entry.name == name) return entry;
  throw std::invalid_argument("missing linked frozen GGA curvature map");
}

std::size_t workspace(maps::Extents shape) {
  std::size_t bytes{};
  for (const auto name : {"features", "first", "mixed"}) {
    std::size_t needed{};
    if (!program(name).workspace(shape, needed))
      throw std::overflow_error("GGA curvature workspace overflow");
    bytes = std::max(bytes, needed);
  }
  return bytes;
}

std::size_t terms(maps::Extents shape) {
  std::size_t result{};
  for (const auto name : {"features", "first", "first", "mixed"}) {
    std::size_t work{};
    if (!program(name).work(shape, work))
      throw std::overflow_error("GGA curvature map work overflow");
    result = size_add(result, work);
  }
  return result;
}

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("GGA curvature overlaps or reenters");
  }
  ~Busy() { flag.clear(); }
};

struct Sum {
  double value{}, correction{};
  void add(double contribution) {
    const auto next = value + contribution;
    correction += std::abs(value) >= std::abs(contribution) ? (value - next) + contribution
                                                            : (contribution - next) + value;
    value = next;
  }
  double total() const { return value + correction; }
};
}  // namespace

HostGgaCurvaturePlan HostGgaNuclearCurvature::plan(const HostGridMixedGeometry& geometry,
                                                   std::size_t spins) {
  if (spins != 1 && spins != 2) throw std::invalid_argument("invalid frozen GGA spin count");
  const auto& reader = geometry.geometry();
  const auto basis = reader.basis().nao, tile = reader.tile_points(),
             points = reader.grid().point_count();
  const auto elements =
      size_mul(tile, size_add(size_mul(16, basis), size_add(size_mul(16, spins), 12)));
  const auto bytes =
      size_add(size_add(size_mul(elements, sizeof(double)), workspace({basis, tile})),
               geometry.resources().numeric_bytes);
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("GGA curvature exceeds signed resource domain");
  auto work = size_mul(points / tile, terms({basis, tile}));
  if (points % tile) work = size_add(work, terms({basis, points % tile}));
  const auto pairs = size_mul(reader.basis().natom, reader.basis().natom - 1) / 2;
  const auto tiles = size_add(points / tile, points % tile != 0);
  return {bytes, size_mul(spins, work), size_mul(points, basis),
          size_mul(size_add(points, tiles), pairs)};
}

class HostGgaNuclearCurvature::Impl {
 public:
  Impl(HostGridMixedGeometry& geometry, std::size_t spins, std::span<const double> density,
       std::string_view method, HostGgaCurvaturePlan limits, std::function<void()> validator)
      : geometry(geometry),
        spins(spins),
        density(density),
        validator(std::move(validator)),
        resources(HostGgaNuclearCurvature::plan(geometry, spins)),
        basis(geometry.geometry().basis().nao) {
    if (!this->validator || !density.data() ||
        density.size() != size_mul(spins, size_mul(basis, basis)))
      throw std::invalid_argument("frozen GGA curvature requires authenticated spin densities");
    if (resources.numeric_bytes > limits.numeric_bytes || resources.map_terms > limits.map_terms ||
        resources.ao_point_visits > limits.ao_point_visits ||
        resources.grid_pair_visits > limits.grid_pair_visits)
      throw std::length_error("frozen GGA curvature numeric or complete-source cap exceeded");
    const auto* binding = host_hessian_aot::find_method(method);
    if (binding && binding->spins == spins)
      field = host_hessian_aot::find(
          spins == 1 ? "gga_energy_mixed_unpolarized" : "gga_energy_mixed_polarized",
          binding->energy_mixed_identity);
    if (!field || !field->evaluate || field->outputs != 4 || field->inputs.size() != 16 * spins + 4)
      throw std::invalid_argument("missing exact method/spin frozen GGA mixed closure");
    for (std::size_t moving = 0; moving < 4; ++moving) {
      const auto prefix = moving == 0 ? "" : moving == 1 ? "l" : moving == 2 ? "r" : "lr";
      weights[moving] = input(std::string(prefix) + "weight");
      for (std::size_t index = 0; index < 4 * spins; ++index)
        fields[moving][index] = input(std::string(prefix) + "f" + std::to_string(index));
    }
    validate();
    const auto tile = geometry.geometry().tile_points();
    owned_elements = size_add(size_mul(tile, size_add(size_mul(16, basis), 16 * spins + 12)),
                              workspace({basis, tile}) / sizeof(double));
    storage = std::make_unique<double[]>(owned_elements);
  }

  std::size_t input(std::string_view name) const {
    const auto found = std::find(field->inputs.begin(), field->inputs.end(), name);
    if (found == field->inputs.end() ||
        std::find(found + 1, field->inputs.end(), name) != field->inputs.end())
      throw std::invalid_argument("incompatible frozen GGA mixed input ABI");
    return static_cast<std::size_t>(found - field->inputs.begin());
  }

  void validate() const {
    validator();
    geometry.validate_current();
    if (!maps::finite(density.data(), density.size()))
      throw std::domain_error("nonfinite frozen GGA density");
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = row + 1; column < basis; ++column) {
          const auto first = density[(spin * basis + row) * basis + column];
          const auto second = density[(spin * basis + column) * basis + row];
          if (std::abs(first - second) >
              1e-12 + 1e-10 * std::max(std::abs(first), std::abs(second)))
            throw std::invalid_argument("frozen GGA curvature requires symmetric density");
        }
  }

  void invoke(const maps::Program& entry, maps::Extents shape, std::size_t spin, std::size_t moving,
              std::span<const double> ao, std::span<const double> motion, std::span<double> scalar,
              std::span<double> arena) {
    std::array<maps::Input, 17> input_bindings{};
    std::array<maps::Output, 4> output_bindings{};
    const auto count = shape.points, lanes = size_mul(count, basis);
    if (entry.inputs.size() > input_bindings.size() || entry.outputs.size() != 4)
      throw std::invalid_argument("incompatible frozen GGA feature ABI");
    for (std::size_t index = 0; index < entry.inputs.size(); ++index) {
      const auto name = entry.inputs[index];
      if (name == "density") {
        input_bindings[index] = {density.data() + spin * basis * basis, basis * basis};
        continue;
      }
      const auto component = name.back() - '0';
      if (component < 0 || component > 3)
        throw std::invalid_argument("invalid frozen GGA AO component");
      std::span<const double> values;
      const auto offset = size_mul(static_cast<std::size_t>(component), lanes);
      if (name.starts_with("ao"))
        values = ao.subspan(offset, lanes);
      else if (name.starts_with("d_left_ao"))
        values = motion.subspan(8 * lanes + offset, lanes);
      else if (name.starts_with("left_ao"))
        values = motion.subspan(offset, lanes);
      else if (name.starts_with("d_ao"))
        values = motion.subspan((moving == 1 ? 0 : 4 * lanes) + offset, lanes);
      else
        throw std::invalid_argument("unknown frozen GGA feature input");
      input_bindings[index] = {values.data(), values.size()};
    }
    const std::array<std::string_view, 4> names{"rho", "g0", "g1", "g2"};
    for (std::size_t index = 0; index < entry.outputs.size(); ++index) {
      const auto found = std::find(names.begin(), names.end(), entry.outputs[index]);
      if (found == names.end()) throw std::invalid_argument("unknown frozen GGA feature output");
      const auto component = static_cast<std::size_t>(found - names.begin());
      output_bindings[index] = {scalar.data() + fields[moving][4 * spin + component] * count,
                                count};
    }
    std::size_t work{};
    if (!entry.work(shape, work)) throw std::overflow_error("frozen GGA feature work overflow");
    census.map_terms = size_add(census.map_terms, work);
    if (entry.evaluate(shape, std::span(input_bindings).first(entry.inputs.size()), output_bindings,
                       arena) != maps::Status::success)
      throw std::domain_error("linked frozen GGA feature action failed");
  }

  void apply(std::span<const double> left, std::span<const double> right,
             std::span<double> output) {
    Busy guard(busy);
    validate();
    const auto& reader = geometry.geometry();
    if (left.size() != 3 * reader.basis().natom || right.size() != left.size() || !left.data() ||
        !right.data() || output.size() != 4 || !output.data() ||
        !maps::finite(left.data(), left.size()) || !maps::finite(right.data(), right.size()))
      throw std::invalid_argument("invalid frozen GGA curvature directions/output");
    geometry.reject_borrowed_alias(output);
    if (maps::overlap(output.data(), output.size(), density.data(), density.size()))
      throw std::invalid_argument("frozen GGA curvature output aliases density");
    census.pairs_started = size_add(census.pairs_started, 1);
    std::array<Sum, 4> sums;
    const auto tile = reader.tile_points(), points = reader.grid().point_count();
    for (std::size_t begin = 0; begin < points; begin += tile) {
      const auto count = std::min(tile, points - begin), lanes = size_mul(count, basis);
      auto* cursor = storage.get();
      const auto take = [&cursor](std::size_t elements) {
        auto result = std::span(cursor, elements);
        cursor += elements;
        return result;
      };
      auto ao = take(4 * lanes), motion = take(12 * lanes), weight = take(count),
           weight_motion = take(3 * count);
      auto scalar = take(field->inputs.size() * count), values = take(4 * count);
      auto arena =
          std::span(cursor, owned_elements - static_cast<std::size_t>(cursor - storage.get()));
      geometry.read(begin, count, left, right, ao, motion, weight, weight_motion);
      for (std::size_t moving = 0; moving < 4; ++moving) {
        const auto input_weight =
            moving == 0 ? weight : weight_motion.subspan((moving - 1) * count, count);
        std::copy(input_weight.begin(), input_weight.end(),
                  scalar.begin() + weights[moving] * count);
      }
      const maps::Extents shape{basis, count};
      for (std::size_t spin = 0; spin < spins; ++spin) {
        invoke(program("features"), shape, spin, 0, ao, motion, scalar, arena);
        invoke(program("first"), shape, spin, 1, ao, motion, scalar, arena);
        invoke(program("first"), shape, spin, 2, ao, motion, scalar, arena);
        invoke(program("mixed"), shape, spin, 3, ao, motion, scalar, arena);
        for (std::size_t point = 0; point < count; ++point) {
          const auto rho = scalar[fields[0][4 * spin] * count + point];
          if (!host_gga_spin_fields_admitted(rho,
                                             {scalar[fields[0][4 * spin + 1] * count + point],
                                              scalar[fields[0][4 * spin + 2] * count + point],
                                              scalar[fields[0][4 * spin + 3] * count + point]}))
            throw std::domain_error("frozen GGA curvature requires physical spin fields");
        }
      }
      if (field->evaluate(scalar.data(), scalar.size(), values.data(), values.size(), count))
        throw std::domain_error("linked frozen GGA energy mixed action failed");
      for (std::size_t component = 0; component < 4; ++component)
        for (std::size_t point = 0; point < count; ++point)
          sums[component].add(values[component * count + point]);
      census.tiles_completed = size_add(census.tiles_completed, 1);
      census.point_visits = size_add(census.point_visits, count);
      validate();
    }
    std::array<double, 4> result;
    for (std::size_t index = 0; index < result.size(); ++index) result[index] = sums[index].total();
    if (!maps::finite(result.data(), result.size()))
      throw std::domain_error("nonfinite frozen GGA curvature sum");
    validate();
    census.pairs_completed = size_add(census.pairs_completed, 1);
    std::copy(result.begin(), result.end(), output.begin());
  }

  HostGridMixedGeometry& geometry;
  std::size_t spins;
  std::span<const double> density;
  std::function<void()> validator;
  HostGgaCurvaturePlan resources;
  std::size_t basis, owned_elements{};
  const host_hessian_aot::Program* field{};
  std::array<std::array<std::size_t, 8>, 4> fields{};
  std::array<std::size_t, 4> weights{};
  std::unique_ptr<double[]> storage;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostGgaCurvatureWork census;
};

HostGgaNuclearCurvature::HostGgaNuclearCurvature(HostGridMixedGeometry& geometry, std::size_t spins,
                                                 std::span<const double> density,
                                                 std::string_view method,
                                                 HostGgaCurvaturePlan limits,
                                                 std::function<void()> validate_current)
    : impl_(std::make_unique<Impl>(geometry, spins, density, method, limits,
                                   std::move(validate_current))) {}
HostGgaNuclearCurvature::~HostGgaNuclearCurvature() = default;
void HostGgaNuclearCurvature::apply(std::span<const double> left, std::span<const double> right,
                                    std::span<double> output) {
  impl_->apply(left, right, output);
}
HostGgaCurvaturePlan HostGgaNuclearCurvature::resources() const noexcept {
  return impl_->resources;
}
HostGgaCurvatureWork HostGgaNuclearCurvature::work() const noexcept { return impl_->census; }
}  // namespace generativeqc::dft
