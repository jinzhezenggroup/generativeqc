#include "dft/host_xc_nuclear.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "dft/host_gga_fields.hpp"
#include "host_grid_maps.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_host_grid_maps;

std::size_t add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("host nuclear XC size overflow");
  return left + right;
}

std::size_t multiply(std::size_t left, std::size_t right) {
  if (right && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error("host nuclear XC size overflow");
  return left * right;
}

const maps::Program& program(std::string_view name) {
  for (const auto& entry : maps::programs())
    if (entry.name == name) return entry;
  throw std::invalid_argument("missing linked host grid contraction map");
}

std::size_t workspace(maps::Extents shape) {
  std::size_t result{};
  for (const auto& entry : maps::programs()) {
    std::size_t bytes{};
    if (!entry.workspace(shape, bytes))
      throw std::overflow_error("host nuclear XC map workspace overflow");
    result = std::max(result, bytes);
  }
  return result;
}

std::size_t terms(maps::Extents shape) {
  std::size_t result{};
  for (const auto& entry : maps::programs()) {
    std::size_t count{};
    if (!entry.work(shape, count)) throw std::overflow_error("host nuclear XC map work overflow");
    result = add(result, count);
  }
  return result;
}

struct Input {
  std::string_view name;
  std::span<const double> values;
};
struct Output {
  std::string_view name;
  std::span<double> values;
};

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("host nuclear XC reentry");
  }
  ~Busy() { flag.clear(); }
};

}  // namespace

HostGgaNuclearPlan HostGgaNuclearXc::plan(const HostGridGeometry& geometry, std::size_t spins) {
  if (spins != 1 && spins != 2) throw std::invalid_argument("invalid host nuclear XC spin count");
  const auto basis = geometry.basis().nao, points = geometry.grid().point_count();
  const auto tile = geometry.tile_points();
  const maps::Extents shape{basis, tile};
  auto elements = multiply(spins + 2, multiply(basis, basis));
  elements = add(elements, multiply(tile, add(multiply(8, basis), 15 + 16 * spins)));
  auto bytes = add(multiply(elements, sizeof(double)), workspace(shape));
  bytes = add(bytes, geometry.scratch_bytes());
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("host nuclear XC exceeds signed resource domain");
  auto work = multiply(points / tile, terms(shape));
  if (points % tile) work = add(work, terms({basis, points % tile}));
  return {bytes, multiply(spins, work), multiply(points, basis),
          geometry.grid().weight_direction_work(points).pair_visits};
}

class HostGgaNuclearXc::Impl {
 public:
  Impl(HostGridGeometry& geometry, std::size_t spins, std::span<const double> density,
       std::string_view method_identity, HostGgaNuclearPlan limits,
       std::function<void()> validate_current)
      : geometry(geometry),
        spins(spins),
        basis(geometry.basis().nao),
        density(density),
        resources(HostGgaNuclearXc::plan(geometry, spins)),
        validator(std::move(validate_current)) {
    if (resources.numeric_bytes > limits.numeric_bytes || resources.map_terms > limits.map_terms ||
        resources.ao_point_visits > limits.ao_point_visits ||
        resources.grid_pair_visits > limits.grid_pair_visits)
      throw std::length_error("host nuclear XC numeric or complete-source work cap exceeded");
    const auto* method = host_hessian_aot::find_method(method_identity);
    if (!method || method->spins != spins || method->field_response_identity.empty())
      throw std::invalid_argument("missing exact host nuclear XC method/spin closure");
    field = host_hessian_aot::find(
        spins == 1 ? "gga_field_response_unpolarized" : "gga_field_response_polarized",
        method->field_response_identity);
    if (!field || !field->evaluate || field->outputs != 8 * spins ||
        field->inputs.size() != 8 * spins + 2)
      throw std::invalid_argument("missing exact linked host GGA field response");
    if (!validator || !density.data() || density.size() != multiply(spins, multiply(basis, basis)))
      throw std::invalid_argument("host nuclear XC requires immutable density and live owner");
    for (std::size_t index = 0; index < 4 * spins; ++index) {
      values[index] = input("f" + std::to_string(index));
      tangents[index] = input("d" + std::to_string(index));
    }
    weight = input("weight");
    moving_weight = input("dweight");
    validate();
    storage_count = (resources.numeric_bytes - geometry.scratch_bytes()) / sizeof(double);
    storage = std::make_unique<double[]>(storage_count);
  }

  std::size_t input(std::string_view name) const {
    const auto found = std::find(field->inputs.begin(), field->inputs.end(), name);
    if (found == field->inputs.end() ||
        std::find(found + 1, field->inputs.end(), name) != field->inputs.end())
      throw std::invalid_argument("incompatible linked host GGA field inputs");
    return static_cast<std::size_t>(found - field->inputs.begin());
  }

  void validate() const {
    validator();
    geometry.validate_current();
    if (!maps::finite(density.data(), density.size()))
      throw std::domain_error("nonfinite host nuclear XC density");
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = row + 1; column < basis; ++column) {
          const auto first = density[(spin * basis + row) * basis + column];
          const auto second = density[(spin * basis + column) * basis + row];
          if (std::abs(first - second) >
              1e-12 + 1e-10 * std::max(std::abs(first), std::abs(second)))
            throw std::invalid_argument("host nuclear XC requires symmetric spin density");
        }
  }

  void invoke(std::string_view name, maps::Extents shape, std::initializer_list<Input> inputs,
              std::initializer_list<Output> outputs, std::span<double> arena) {
    const auto& entry = program(name);
    std::array<maps::Input, 20> input_bindings{};
    std::array<maps::Output, 5> output_bindings{};
    if (entry.inputs.size() != inputs.size() || entry.outputs.size() != outputs.size() ||
        entry.inputs.size() > input_bindings.size() ||
        entry.outputs.size() > output_bindings.size())
      throw std::invalid_argument("incompatible linked host grid map ABI");
    for (std::size_t index = 0; index < entry.inputs.size(); ++index) {
      const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const Input& binding) {
        return binding.name == entry.inputs[index];
      });
      if (found == inputs.end()) throw std::invalid_argument("missing host grid map input");
      input_bindings[index] = {found->values.data(), found->values.size()};
    }
    for (std::size_t index = 0; index < entry.outputs.size(); ++index) {
      const auto found = std::find_if(outputs.begin(), outputs.end(), [&](const Output& binding) {
        return binding.name == entry.outputs[index];
      });
      if (found == outputs.end()) throw std::invalid_argument("missing host grid map output");
      output_bindings[index] = {found->values.data(), found->values.size()};
    }
    std::size_t work{};
    if (!entry.work(shape, work)) throw std::overflow_error("host nuclear XC map work overflow");
    census.map_evaluations = add(census.map_evaluations, 1);
    census.scheduled_map_terms = add(census.scheduled_map_terms, work);
    if (entry.evaluate(shape, std::span(input_bindings).first(inputs.size()),
                       std::span(output_bindings).first(outputs.size()),
                       arena) != maps::Status::success)
      throw std::domain_error("host nuclear XC contraction map failed");
    validate();
  }

  void apply(std::span<const double> direction, std::span<double> output) {
    Busy guard(busy);
    validate();
    if (output.size() != density.size() || !output.data() ||
        direction.size() != 3 * geometry.basis().natom || !direction.data() ||
        maps::overlap(output.data(), output.size(), density.data(), density.size()))
      throw std::invalid_argument("invalid or borrowed-density-aliasing host nuclear XC binding");
    geometry.reject_borrowed_alias(output);
    if (!maps::finite(direction.data(), direction.size()))
      throw std::domain_error("nonfinite host nuclear XC direction");
    census.directions_started = add(census.directions_started, 1);
    auto result = std::span(storage.get(), density.size());
    std::fill(result.begin(), result.end(), 0);
    const auto matrix = basis * basis;
    const auto zero_density = std::span(storage.get() + density.size(), matrix);
    std::fill(zero_density.begin(), zero_density.end(), 0);
    for (std::size_t begin = 0; begin < geometry.grid().point_count();) {
      const auto count = std::min(geometry.tile_points(), geometry.grid().point_count() - begin);
      auto* cursor = zero_density.data() + matrix;
      const auto take = [&cursor](std::size_t size) {
        auto panel = std::span(cursor, size);
        cursor += size;
        return panel;
      };
      auto potential = take(matrix);
      auto ao = take(4 * count * basis), moving_ao = take(4 * count * basis);
      auto weights = take(count), moving_weights = take(count), fields = take(10 * count);
      auto scalar_input = take((8 * spins + 2) * count), scalar_output = take(8 * spins * count);
      auto zero = take(count);
      std::fill(zero.begin(), zero.end(), 0);
      auto arena =
          std::span(cursor, storage_count - static_cast<std::size_t>(cursor - storage.get()));
      geometry.read(begin, count, direction, ao, moving_ao, weights, moving_weights);
      validate();
      std::copy(weights.begin(), weights.end(), scalar_input.begin() + weight * count);
      std::copy(moving_weights.begin(), moving_weights.end(),
                scalar_input.begin() + moving_weight * count);
      const maps::Extents shape{basis, count};
      const auto component = [&](std::span<double> panel, std::size_t index) {
        return panel.subspan(index * count * basis, count * basis);
      };
      const auto panel = [&](std::size_t index) { return fields.subspan(index * count, count); };
      for (std::size_t spin = 0; spin < spins; ++spin) {
        const auto spin_density = density.subspan(spin * matrix, matrix);
        invoke("features", shape,
               {{"ao0", component(ao, 0)},
                {"ao1", component(ao, 1)},
                {"ao2", component(ao, 2)},
                {"ao3", component(ao, 3)},
                {"density", spin_density}},
               {{"rho", panel(0)},
                {"g0", panel(1)},
                {"g1", panel(2)},
                {"g2", panel(3)},
                {"tau", panel(4)}},
               arena);
        invoke("feature_direction", shape,
               {{"ao0", component(ao, 0)},
                {"ao1", component(ao, 1)},
                {"ao2", component(ao, 2)},
                {"ao3", component(ao, 3)},
                {"d_ao0", component(moving_ao, 0)},
                {"d_ao1", component(moving_ao, 1)},
                {"d_ao2", component(moving_ao, 2)},
                {"d_ao3", component(moving_ao, 3)},
                {"density", spin_density},
                {"d_density", zero_density}},
               {{"rho", panel(5)},
                {"g0", panel(6)},
                {"g1", panel(7)},
                {"g2", panel(8)},
                {"tau", panel(9)}},
               arena);
        for (std::size_t point = 0; point < count; ++point) {
          if (!host_gga_spin_fields_admitted(
                  fields[point],
                  {fields[count + point], fields[2 * count + point], fields[3 * count + point]})) {
            std::ostringstream message;
            message.precision(17);
            message << "host nuclear GGA requires nonnegative physical spin fields; spin=" << spin
                    << "; point=" << begin + point << "; rho=" << fields[point]
                    << "; gradient=" << fields[count + point] << ',' << fields[2 * count + point]
                    << ',' << fields[3 * count + point];
            throw std::domain_error(message.str());
          }
          for (std::size_t axis = 0; axis < 4; ++axis) {
            scalar_input[values[4 * spin + axis] * count + point] = fields[axis * count + point];
            scalar_input[tangents[4 * spin + axis] * count + point] =
                fields[(5 + axis) * count + point];
          }
        }
      }
      if (field->evaluate(scalar_input.data(), scalar_input.size(), scalar_output.data(),
                          scalar_output.size(), count))
        throw std::domain_error("linked host nuclear GGA field response failed");
      validate();
      const auto coefficient = [&](std::size_t index) {
        return scalar_output.subspan(index * count, count);
      };
      for (std::size_t spin = 0; spin < spins; ++spin) {
        invoke("potential_direction", shape,
               {{"ao0", component(ao, 0)},
                {"ao1", component(ao, 1)},
                {"ao2", component(ao, 2)},
                {"ao3", component(ao, 3)},
                {"d_ao0", component(moving_ao, 0)},
                {"d_ao1", component(moving_ao, 1)},
                {"d_ao2", component(moving_ao, 2)},
                {"d_ao3", component(moving_ao, 3)},
                {"bar_rho", coefficient(4 * spin)},
                {"bar_g0", coefficient(4 * spin + 1)},
                {"bar_g1", coefficient(4 * spin + 2)},
                {"bar_g2", coefficient(4 * spin + 3)},
                {"bar_tau", zero},
                {"d_bar_rho", coefficient(4 * spins + 4 * spin)},
                {"d_bar_g0", coefficient(4 * spins + 4 * spin + 1)},
                {"d_bar_g1", coefficient(4 * spins + 4 * spin + 2)},
                {"d_bar_g2", coefficient(4 * spins + 4 * spin + 3)},
                {"d_bar_tau", zero}},
               {{"potential", potential}}, arena);
        for (std::size_t index = 0; index < matrix; ++index)
          result[spin * matrix + index] += potential[index];
      }
      if (!maps::finite(result.data(), result.size()))
        throw std::domain_error("nonfinite host nuclear XC accumulation");
      census.tiles_completed = add(census.tiles_completed, 1);
      census.point_visits = add(census.point_visits, count);
      begin += count;
    }
    validate();
    census.directions_completed = add(census.directions_completed, 1);
    std::copy(result.begin(), result.end(), output.begin());
  }

  HostGridGeometry& geometry;
  std::size_t spins, basis;
  std::span<const double> density;
  HostGgaNuclearPlan resources;
  std::function<void()> validator;
  const host_hessian_aot::Program* field{};
  std::array<std::size_t, 8> values{}, tangents{};
  std::size_t weight{}, moving_weight{}, storage_count{};
  std::unique_ptr<double[]> storage;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
  HostGgaNuclearWork census;
};

HostGgaNuclearXc::HostGgaNuclearXc(HostGridGeometry& geometry, std::size_t spins,
                                   std::span<const double> density,
                                   std::string_view method_identity, HostGgaNuclearPlan limits,
                                   std::function<void()> validate_current)
    : impl_(std::make_unique<Impl>(geometry, spins, density, method_identity, limits,
                                   std::move(validate_current))) {}
HostGgaNuclearXc::~HostGgaNuclearXc() = default;
void HostGgaNuclearXc::apply(std::span<const double> direction, std::span<double> output) {
  impl_->apply(direction, output);
}
HostGgaNuclearWork HostGgaNuclearXc::work() const noexcept { return impl_->census; }
HostGgaNuclearPlan HostGgaNuclearXc::resources() const noexcept { return impl_->resources; }

}  // namespace generativeqc::dft
