#include "dft/host_xc_column.hpp"

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

#include "dft/host_gga_fields.hpp"
#include "host_grid_geometry_ad.hpp"
#include "host_xc_curvature_maps.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {
namespace {
namespace ad = generativeqc_grid_geometry_ad;
namespace maps = generativeqc_host_curvature_maps;
using runtime::size_add;
using runtime::size_mul;

const maps::Program& program(std::string_view name) {
  for (const auto& entry : maps::programs())
    if (entry.name == name) return entry;
  throw std::invalid_argument("missing linked GGA column map");
}

std::size_t workspace(maps::Extents shape) {
  std::size_t result{};
  for (const auto name : {"features", "first_relaxed", "pullback", "pullback_direction_relaxed"}) {
    std::size_t bytes{};
    if (!program(name).workspace(shape, bytes))
      throw std::overflow_error("GGA column workspace overflow");
    result = std::max(result, bytes);
  }
  return result;
}

std::size_t terms(maps::Extents shape) {
  std::size_t result{};
  for (const auto name :
       {"features", "first_relaxed", "pullback", "pullback_direction_relaxed", "pullback"}) {
    std::size_t count{};
    if (!program(name).work(shape, count)) throw std::overflow_error("GGA column work overflow");
    result = size_add(result, count);
  }
  return result;
}

std::size_t elements(const HostGridGeometry& reader, std::size_t spins) {
  const auto fields = size_add(2, size_mul(6, reader.basis().natom));
  return size_mul(reader.tile_points(), size_add(size_mul(36, reader.basis().nao),
                                                 size_add(fields, size_add(24 * spins, 10))));
}

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

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("GGA column overlaps or reenters");
  }
  ~Busy() { flag.clear(); }
};
}  // namespace

HostGgaColumnPlan HostGgaNuclearColumn::plan(const HostGridColumnGeometry& geometry,
                                             std::size_t spins) {
  if (spins != 1 && spins != 2) throw std::invalid_argument("invalid GGA column spin count");
  const auto& reader = geometry.geometry();
  const auto atoms = reader.basis().natom, basis = reader.basis().nao,
             points = reader.grid().point_count(), tile = reader.tile_points();
  const auto fields = size_add(2, size_mul(6, atoms));
  const auto state_bytes = size_add(
      size_mul(fields, sizeof(Sum)),
      size_add(size_mul(size_mul(3, atoms), sizeof(double)), size_mul(basis, sizeof(std::size_t))));
  const auto density_bytes = size_mul(size_mul(spins, size_mul(basis, basis)), sizeof(double));
  const auto own_bytes =
      size_add(size_mul(elements(reader, spins), sizeof(double)),
               size_add(workspace({basis, tile}), size_add(state_bytes, density_bytes)));
  const auto bytes = size_add(geometry.resources().numeric_bytes, own_bytes);
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("GGA column exceeds signed resource domain");
  auto work = size_mul(points / tile, terms({basis, tile}));
  if (points % tile) work = size_add(work, terms({basis, points % tile}));
  const auto pairs = size_mul(atoms, atoms - 1) / 2;
  const auto tiles = size_add(points / tile, points % tile != 0);
  return {bytes,
          size_mul(spins, work),
          size_mul(4 * spins + 2, points),
          size_mul(points, basis),
          size_mul(1 + 3 * spins, size_mul(points, basis)),
          size_mul(size_add(points, tiles), pairs),
          size_mul(tiles, geometry.resources().tape_visits)};
}

class HostGgaNuclearColumn::Impl {
 public:
  Impl(HostGridColumnGeometry& geometry, std::size_t spins, std::span<const double> density,
       std::string_view method, HostGgaColumnPlan limits, std::function<void()> validate)
      : geometry(geometry),
        spins(spins),
        density(density),
        validator(std::move(validate)),
        resources(HostGgaNuclearColumn::plan(geometry, spins)),
        atoms(geometry.geometry().basis().natom),
        basis(geometry.geometry().basis().nao) {
    if (!validator || !density.data() || density.size() != size_mul(spins, size_mul(basis, basis)))
      throw std::invalid_argument("GGA column requires authenticated spin densities");
    if (resources.numeric_bytes > limits.numeric_bytes || resources.map_terms > limits.map_terms ||
        resources.scalar_point_visits > limits.scalar_point_visits ||
        resources.ao_point_visits > limits.ao_point_visits ||
        resources.ao_motion_visits > limits.ao_motion_visits ||
        resources.grid_pair_visits > limits.grid_pair_visits ||
        resources.tape_visits > limits.tape_visits)
      throw std::length_error("GGA column capacity or complete-source work cap exceeded");
    const auto* binding = host_hessian_aot::find_method(method);
    if (binding && binding->spins == spins)
      field = host_hessian_aot::find(
          spins == 1 ? "gga_energy_mixed_unpolarized" : "gga_energy_mixed_polarized",
          binding->energy_mixed_identity);
    if (!field || !field->evaluate || field->outputs != 4 || field->inputs.size() != 16 * spins + 4)
      throw std::invalid_argument("missing exact method/spin GGA column mixed closure");
    for (std::size_t moving = 0; moving < 4; ++moving) {
      const auto prefix = moving == 0 ? "" : moving == 1 ? "l" : moving == 2 ? "r" : "lr";
      weights[moving] = input(std::string(prefix) + "weight");
      for (std::size_t index = 0; index < 4 * spins; ++index)
        fields[moving][index] = input(std::string(prefix) + "f" + std::to_string(index));
    }
    this->validate();
    const auto& reader = geometry.geometry();
    owned_elements = size_add(elements(reader, spins),
                              workspace({basis, reader.tile_points()}) / sizeof(double));
    storage = std::make_unique<double[]>(owned_elements);
    right.resize(size_mul(3, atoms));
    moving_density.resize(density.size());
    sums.resize(size_add(2, size_mul(6, atoms)));
    ao_atoms.reserve(basis);
    const auto offset = size_add(size_mul(3, atoms), size_mul(2, reader.basis().nprimitive));
    for (std::size_t orbital = 0; orbital < basis; ++orbital)
      ao_atoms.push_back(static_cast<std::size_t>(reader.basis().packed[offset + 16 * orbital]));
  }

  std::size_t input(std::string_view name) const {
    const auto found = std::find(field->inputs.begin(), field->inputs.end(), name);
    if (found == field->inputs.end() ||
        std::find(found + 1, field->inputs.end(), name) != field->inputs.end())
      throw std::invalid_argument("incompatible linked GGA column closure inputs");
    return static_cast<std::size_t>(found - field->inputs.begin());
  }

  void validate() const {
    validator();
    geometry.validate_current();
    if (!maps::finite(density.data(), density.size()))
      throw std::domain_error("nonfinite GGA column density");
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = row + 1; column < basis; ++column) {
          const auto first = density[(spin * basis + row) * basis + column];
          const auto second = density[(spin * basis + column) * basis + row];
          if (std::abs(first - second) >
              1e-12 + 1e-10 * std::max(std::abs(first), std::abs(second)))
            throw std::invalid_argument("GGA column requires symmetric density");
        }
  }

  void map(std::string_view name, std::size_t spin, std::size_t count, std::span<const double> jets,
           std::span<const double> motion, std::span<double> scalar,
           std::span<const double> coefficients, std::span<double> output,
           std::span<double> arena) {
    const auto& entry = program(name);
    const auto lanes = size_mul(count, basis);
    std::array<maps::Input, 17> inputs{};
    std::array<maps::Output, 4> outputs{};
    const std::array<std::string_view, 4> labels{"rho", "g0", "g1", "g2"};
    for (std::size_t index = 0; index < entry.inputs.size(); ++index) {
      const auto label = entry.inputs[index];
      std::span<const double> values;
      if (label == "density")
        values = density.subspan(spin * basis * basis, basis * basis);
      else if (label == "d_density")
        values = std::span(moving_density).subspan(spin * basis * basis, basis * basis);
      else if (label.starts_with("bar_")) {
        const auto found = std::find(labels.begin(), labels.end(), label.substr(4));
        if (found == labels.end()) throw std::invalid_argument("unknown GGA column field adjoint");
        values =
            coefficients.subspan(static_cast<std::size_t>(found - labels.begin()) * count, count);
      } else {
        const auto component = static_cast<std::size_t>(label.back() - '0');
        if (component > 3) throw std::invalid_argument("unknown GGA column AO input");
        if (label.starts_with("ao"))
          values = jets.subspan(component * lanes, lanes);
        else if (label.starts_with("d_ao"))
          values = motion.subspan(component * lanes, lanes);
        else
          throw std::invalid_argument("unknown GGA column map input");
      }
      inputs[index] = {values.data(), values.size()};
    }
    for (std::size_t index = 0; index < entry.outputs.size(); ++index) {
      if (name == "features" || name == "first_relaxed") {
        const auto found = std::find(labels.begin(), labels.end(), entry.outputs[index]);
        if (found == labels.end()) throw std::invalid_argument("unknown GGA column feature output");
        const auto component = static_cast<std::size_t>(found - labels.begin());
        const auto moving = name == "features" ? 0 : 2;
        outputs[index] = {scalar.data() + fields[moving][4 * spin + component] * count, count};
      } else {
        const auto component = static_cast<std::size_t>(entry.outputs[index].back() - '0');
        if (component > 3) throw std::invalid_argument("unknown GGA column pullback output");
        outputs[index] = {output.data() + component * lanes, lanes};
      }
    }
    std::size_t work{};
    const maps::Extents shape{basis, count};
    if (!entry.work(shape, work)) throw std::overflow_error("GGA column map work overflow");
    census.map_terms = size_add(census.map_terms, work);
    if (entry.evaluate(shape, std::span(inputs).first(entry.inputs.size()), outputs, arena) !=
        maps::Status::success)
      throw std::domain_error("linked GGA column map failed");
  }

  void closure(std::span<const double> scalar, std::span<double> values, std::size_t count) {
    census.scalar_point_visits = size_add(census.scalar_point_visits, count);
    if (field->evaluate(scalar.data(), scalar.size(), values.data(), values.size(), count))
      throw std::domain_error("linked GGA column mixed-energy action failed");
  }

  void apply(std::span<const double> direction, std::span<const double> density_right,
             std::span<double> output) {
    Busy guard(busy);
    validate();
    if (direction.size() != 3 * atoms || !direction.data() || output.size() != sums.size() ||
        !output.data() || !maps::finite(direction.data(), direction.size()))
      throw std::invalid_argument("invalid GGA column direction/output");
    geometry.reject_borrowed_alias(output);
    if (maps::overlap(output.data(), output.size(), density.data(), density.size()))
      throw std::invalid_argument("GGA column output aliases density");
    if (!density_right.empty() &&
        (density_right.size() != density.size() || !density_right.data() ||
         !maps::finite(density_right.data(), density_right.size())))
      throw std::invalid_argument("invalid GGA column density direction");
    for (std::size_t spin = 0; spin < spins && !density_right.empty(); ++spin)
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = row + 1; column < basis; ++column) {
          const auto first = density_right[(spin * basis + row) * basis + column];
          const auto second = density_right[(spin * basis + column) * basis + row];
          if (std::abs(first - second) >
              1e-12 + 1e-10 * std::max(std::abs(first), std::abs(second)))
            throw std::invalid_argument("GGA column density direction must be symmetric");
        }
    std::copy(direction.begin(), direction.end(), right.begin());
    std::fill(moving_density.begin(), moving_density.end(), 0);
    if (!density_right.empty())
      std::copy(density_right.begin(), density_right.end(), moving_density.begin());
    std::fill(sums.begin(), sums.end(), Sum{});
    census.columns_started = size_add(census.columns_started, 1);
    const auto& reader = geometry.geometry();
    const auto tile = reader.tile_points(), points = reader.grid().point_count(),
               ingredients = 4 * spins + 1;
    for (std::size_t begin = 0; begin < points; begin += tile) {
      const auto count = std::min(tile, points - begin), lanes = size_mul(count, basis);
      auto* cursor = storage.get();
      const auto take = [&cursor](std::size_t size) {
        auto values = std::span(cursor, size);
        cursor += size;
        return values;
      };
      auto jets = take(20 * lanes), motion = take(4 * lanes), bar = take(4 * lanes),
           geometric = take(4 * lanes), induced = take(4 * lanes),
           weight = take(sums.size() * count), scalar = take(field->inputs.size() * count),
           values = take(4 * count), coefficients = take(2 * ingredients * count);
      auto arena =
          std::span(cursor, owned_elements - static_cast<std::size_t>(cursor - storage.get()));
      geometry.read(begin, count, right, jets, weight);
      validate();
      const std::array<double, 3> zero{};
      for (std::size_t point = 0; point < count; ++point)
        for (std::size_t orbital = 0; orbital < basis; ++orbital) {
          const auto owner = static_cast<std::size_t>(reader.grid().owners()[begin + point]),
                     center = ao_atoms[orbital];
          std::array<double, 3> relative;
          for (std::size_t axis = 0; axis < 3; ++axis)
            relative[axis] = right[3 * owner + axis] - right[3 * center + axis];
          std::array<double, 12> moving;
          ad::ao_motion(jets.data() + point * basis + orbital, lanes, zero.data(), relative.data(),
                        moving.data());
          census.ao_motion_visits = size_add(census.ao_motion_visits, 1);
          for (std::size_t component = 0; component < 4; ++component)
            motion[component * lanes + point * basis + orbital] = moving[3 * component + 1];
        }
      std::fill(scalar.begin(), scalar.end(), 0);
      std::copy_n(weight.begin(), count, scalar.begin() + weights[0] * count);
      std::copy_n(weight.begin() + count, count, scalar.begin() + weights[2] * count);
      for (std::size_t spin = 0; spin < spins; ++spin) {
        map("features", spin, count, jets, motion, scalar, {}, {}, arena);
        map("first_relaxed", spin, count, jets, motion, scalar, {}, {}, arena);
        for (std::size_t point = 0; point < count; ++point) {
          const auto rho = scalar[fields[0][4 * spin] * count + point];
          if (!host_gga_spin_fields_admitted(rho,
                                             {scalar[fields[0][4 * spin + 1] * count + point],
                                              scalar[fields[0][4 * spin + 2] * count + point],
                                              scalar[fields[0][4 * spin + 3] * count + point]}))
            throw std::domain_error("GGA column requires physical spin fields");
        }
      }
      closure(scalar, values, count);
      for (std::size_t point = 0; point < count; ++point) {
        sums[0].add(values[point]);
        sums[1].add(values[2 * count + point]);
      }
      /** Fixed field/weight seeds extract the exact linked adjoints. The
       * number of scalar passes is 4*spins+2, independent of atom count. */
      for (std::size_t ingredient = 0; ingredient < ingredients; ++ingredient) {
        const auto slot = ingredient == 4 * spins ? weights[1] : fields[1][ingredient];
        std::fill_n(scalar.begin() + slot * count, count, 1);
        closure(scalar, values, count);
        std::copy_n(values.begin() + count, count, coefficients.begin() + ingredient * count);
        std::copy_n(values.begin() + 3 * count, count,
                    coefficients.begin() + (ingredients + ingredient) * count);
        std::fill_n(scalar.begin() + slot * count, count, 0);
      }
      for (std::size_t coordinate = 0; coordinate < 3 * atoms; ++coordinate)
        for (std::size_t point = 0; point < count; ++point) {
          const auto first = weight[(2 + coordinate) * count + point];
          sums[2 + coordinate].add(coefficients[(ingredients - 1) * count + point] * first);
          sums[2 + 3 * atoms + coordinate].add(
              coefficients[(2 * ingredients - 1) * count + point] * first +
              coefficients[(ingredients - 1) * count + point] *
                  weight[(2 + 3 * atoms + coordinate) * count + point]);
        }
      for (std::size_t spin = 0; spin < spins; ++spin) {
        const auto base = coefficients.subspan(4 * spin * count, 4 * count);
        const auto delta = coefficients.subspan((ingredients + 4 * spin) * count, 4 * count);
        map("pullback", spin, count, jets, motion, scalar, base, bar, arena);
        map("pullback_direction_relaxed", spin, count, jets, motion, scalar, base, geometric,
            arena);
        map("pullback", spin, count, jets, motion, scalar, delta, induced, arena);
        for (std::size_t point = 0; point < count; ++point)
          for (std::size_t orbital = 0; orbital < basis; ++orbital) {
            const auto owner = static_cast<std::size_t>(reader.grid().owners()[begin + point]),
                       center = ao_atoms[orbital];
            if (owner == center) continue;
            std::array<double, 3> relative;
            for (std::size_t axis = 0; axis < 3; ++axis)
              relative[axis] = right[3 * owner + axis] - right[3 * center + axis];
            for (std::size_t axis = 0; axis < 3; ++axis) {
              std::array<double, 3> left{};
              left[axis] = 1;
              std::array<double, 12> moving;
              ad::ao_motion(jets.data() + point * basis + orbital, lanes, left.data(),
                            relative.data(), moving.data());
              census.ao_motion_visits = size_add(census.ao_motion_visits, 1);
              Sum gradient, mixed;
              for (std::size_t component = 0; component < 4; ++component) {
                const auto lane = component * lanes + point * basis + orbital;
                gradient.add(bar[lane] * moving[3 * component]);
                mixed.add((geometric[lane] + induced[lane]) * moving[3 * component] +
                          bar[lane] * moving[3 * component + 2]);
              }
              sums[2 + 3 * owner + axis].add(gradient.total());
              sums[2 + 3 * center + axis].add(-gradient.total());
              sums[2 + 3 * atoms + 3 * owner + axis].add(mixed.total());
              sums[2 + 3 * atoms + 3 * center + axis].add(-mixed.total());
            }
          }
      }
      census.point_visits = size_add(census.point_visits, count);
      census.tiles_completed = size_add(census.tiles_completed, 1);
      validate();
    }
    for (const auto& sum : sums)
      if (!std::isfinite(sum.total())) throw std::domain_error("nonfinite GGA column sum");
    validate();
    if (!std::equal(right.begin(), right.end(), direction.begin()))
      throw std::invalid_argument("GGA column direction changed during invocation");
    if (!density_right.empty() &&
        !std::equal(moving_density.begin(), moving_density.end(), density_right.begin()))
      throw std::invalid_argument("GGA column density direction changed during invocation");
    census.columns_completed = size_add(census.columns_completed, 1);
    for (std::size_t index = 0; index < sums.size(); ++index) output[index] = sums[index].total();
  }

  HostGridColumnGeometry& geometry;
  std::size_t spins;
  std::span<const double> density;
  std::function<void()> validator;
  HostGgaColumnPlan resources;
  std::size_t atoms, basis, owned_elements{};
  const host_hessian_aot::Program* field{};
  std::array<std::array<std::size_t, 8>, 4> fields{};
  std::array<std::size_t, 4> weights{};
  std::unique_ptr<double[]> storage;
  std::vector<double> right, moving_density;
  std::vector<Sum> sums;
  std::vector<std::size_t> ao_atoms;
  HostGgaColumnWork census;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

HostGgaNuclearColumn::HostGgaNuclearColumn(HostGridColumnGeometry& geometry, std::size_t spins,
                                           std::span<const double> density,
                                           std::string_view method_identity,
                                           HostGgaColumnPlan limits,
                                           std::function<void()> validate_current)
    : impl_(std::make_unique<Impl>(geometry, spins, density, method_identity, limits,
                                   std::move(validate_current))) {}
HostGgaNuclearColumn::~HostGgaNuclearColumn() = default;
void HostGgaNuclearColumn::apply(std::span<const double> right,
                                 std::span<const double> density_right, std::span<double> output) {
  impl_->apply(right, density_right, output);
}
HostGgaColumnPlan HostGgaNuclearColumn::resources() const noexcept { return impl_->resources; }
HostGgaColumnWork HostGgaNuclearColumn::work() const noexcept { return impl_->census; }
}  // namespace generativeqc::dft
