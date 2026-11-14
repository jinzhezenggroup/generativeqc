#include "dft/dispersion/host_d3_column.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "d3_data.hpp"
#include "dft/host_hessian_aot.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft::dispersion {
namespace {
using runtime::size_add;
using runtime::size_mul;

constexpr std::size_t slots = 7, maximum_inputs = 128;

std::size_t pairs(std::size_t atoms) {
  if (!atoms || atoms > kD3MaximumAtomsPerSystem)
    throw std::invalid_argument("D3 column atom count outside 1..4096");
  return atoms % 2 ? size_mul(atoms, (atoms - 1) / 2) : size_mul(atoms / 2, atoms - 1);
}

void finite(std::span<const double> values) {
  if (!values.empty() && !values.data()) throw std::invalid_argument("null D3 column data");
  for (const auto value : values)
    if (!std::isfinite(value)) throw std::domain_error("nonfinite D3 column input or accumulation");
}

bool overlaps(std::span<const double> first, std::span<const double> second) {
  if (first.empty() || second.empty()) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first.data());
  const auto right = reinterpret_cast<std::uintptr_t>(second.data());
  return left <= right ? right - left < first.size_bytes() : left - right < second.size_bytes();
}

std::size_t axis(std::string_view name, std::string_view prefix, std::size_t base) {
  if (name.size() == prefix.size() + 1 && name.starts_with(prefix) && name.back() >= '0' &&
      name.back() <= '2')
    return base + static_cast<std::size_t>(name.back() - '0');
  return maximum_inputs;
}

/** The registry owns input spelling/identity; bind its permutation once rather
 * than searching names inside pair or reference work. Unknown closures refuse.
 */
struct Leaf {
  const host_hessian_aot::Program* program{};
  std::array<std::size_t, maximum_inputs> permutation{};
  std::array<double, maximum_inputs> scratch{};

  template <class Decode>
  void bind(std::string_view name, std::size_t inputs, std::size_t outputs, Decode decode) {
    for (const auto& entry : host_hessian_aot::programs())
      if (entry.name == name) program = &entry;
    if (!program || program->identity.empty() || !program->evaluate ||
        program->inputs.size() != inputs || program->outputs != outputs || inputs > maximum_inputs)
      throw std::invalid_argument("missing or incompatible linked D3 column leaf");
    std::array<bool, maximum_inputs> seen{};
    for (std::size_t index = 0; index < inputs; ++index) {
      const auto decoded = decode(program->inputs[index]);
      if (decoded >= inputs || seen[decoded])
        throw std::invalid_argument("incompatible linked D3 column input contract");
      seen[decoded] = true;
      permutation[index] = decoded;
    }
  }

  void evaluate(std::span<const double> values, std::span<double> output) {
    for (std::size_t index = 0; index < values.size(); ++index)
      scratch[index] = values[permutation[index]];
    if (program->evaluate(scratch.data(), values.size(), output.data(), output.size(), 1))
      throw std::domain_error("D3 column nonfinite leaf: " + std::string(program->name));
  }
};

struct Pair {
  std::size_t first{}, second{};
  std::array<double, 8> coordination{};
};

std::size_t pair_index(std::int32_t first, std::int32_t second) {
  const auto low = std::min(first, second), high = std::max(first, second);
  return static_cast<std::size_t>(low - 1 + high * (high - 1) / 2);
}

double coefficient(std::int32_t first, std::int32_t second, std::size_t left, std::size_t right) {
  const auto& entry = d3_data::kPairs[pair_index(first, second)];
  const auto first_count = d3_data::kElements[static_cast<std::size_t>(first - 1)].reference_count;
  const auto second_count =
      d3_data::kElements[static_cast<std::size_t>(second - 1)].reference_count;
  if (left >= first_count || right >= second_count) return 0;
  const auto offset = first <= second ? right * entry.first_reference_count + left
                                      : left * entry.first_reference_count + right;
  return d3_data::kReferenceC6[entry.c6_offset + offset];
}
}  // namespace

HostD3ColumnPlan HostD3Column::plan(std::size_t atoms) {
  const auto count = pairs(atoms);
  auto bytes = size_add(size_mul(size_mul(44, atoms), sizeof(double)),
                        size_mul(atoms, sizeof(std::int32_t)));
  bytes = size_add(bytes, size_mul(count, sizeof(Pair)));
  bytes = size_add(bytes, size_mul(4 * maximum_inputs + 256, sizeof(double)));
  bytes = size_add(bytes, size_mul(4 * maximum_inputs, sizeof(std::size_t)));
  return {bytes,
          sizeof(d3_data::kElements) + sizeof(d3_data::kPairs) + sizeof(d3_data::kReferenceCn) +
              sizeof(d3_data::kReferenceC6),
          size_mul(3, count), size_add(size_mul(49, count), size_mul(slots, atoms))};
}

class HostD3Column::Impl {
 public:
  std::vector<std::int32_t> numbers;
  std::vector<double> coordinates;
  D3Parameters parameters;
  ValidateCurrent validate;
  Leaf coordination, reference, pair, scatter;
  HostD3ColumnWork last{};
  std::atomic_flag active = ATOMIC_FLAG_INIT;

  Impl(std::span<const std::int32_t> elements, std::span<const double> positions,
       D3Parameters model, HostD3ColumnLimits limits, ValidateCurrent validator)
      : parameters(model), validate(std::move(validator)) {
    const auto bound = HostD3Column::plan(elements.size());
    if (limits.max_numeric_bytes < bound.numeric_bytes ||
        limits.max_pair_visits < bound.pair_visits ||
        limits.max_reference_terms < bound.reference_terms)
      throw std::length_error("D3 column numeric or semantic work ceiling exceeded");
    if (!elements.data() || positions.size() != size_mul(3, elements.size()))
      throw std::invalid_argument("D3 column input shape mismatch");
    finite(positions);
    const std::array values{model.s6, model.s8,        model.a1,          model.a2,
                            model.s9, model.cn_cutoff, model.pair_cutoff, model.pair_switch_width};
    finite(values);
    if (model.s6 < 0 || model.s8 < 0 || model.a1 < 0 || model.a2 < 0 ||
        !(model.a1 > 0 || model.a2 > 0) || model.s9 != 0 || model.cn_cutoff != 0 ||
        model.pair_cutoff != 0 || model.pair_switch_width != 0)
      throw std::invalid_argument("D3 columns require uncut two-body BJ with ATM off");
    for (const auto number : elements)
      if (number < 1 || static_cast<std::size_t>(number) > d3_data::kElements.size() ||
          !d3_data::kElements[static_cast<std::size_t>(number - 1)].reference_count ||
          d3_data::kElements[static_cast<std::size_t>(number - 1)].reference_count > slots)
        throw std::invalid_argument("unsupported D3 column element");
    coordination.bind("d3_coordination_column", 7, 8, [](std::string_view name) {
      if (name == "radius") return std::size_t{6};
      return name.starts_with("dr") ? axis(name, "dr", 3) : axis(name, "r", 0);
    });
    reference.bind("d3_reference_column", 16, 29, [](std::string_view name) {
      if (name == "cn") return std::size_t{0};
      if (name == "dcn") return std::size_t{1};
      if (name.back() < '0' || name.back() > '6') return maximum_inputs;
      const auto slot = static_cast<std::size_t>(name.back() - '0');
      if (name.size() == 5 && name.starts_with("mask")) return 2 + slot;
      if (name.size() == 4 && name.starts_with("ref")) return 9 + slot;
      return maximum_inputs;
    });
    pair.bind("d3_pair_column", 115, 12, [](std::string_view name) {
      if (name == "rr") return std::size_t{6};
      if (name == "radius") return std::size_t{7};
      if (name == "s6") return std::size_t{8};
      if (name == "s8") return std::size_t{9};
      const auto coordinate = name.starts_with("dr") ? axis(name, "dr", 3) : axis(name, "r", 0);
      if (coordinate != maximum_inputs) return coordinate;
      if (name.size() == 3 && name[0] == 'c' && name[1] >= '0' && name[1] <= '6' &&
          name[2] >= '0' && name[2] <= '6')
        return 66 + static_cast<std::size_t>(7 * (name[1] - '0') + name[2] - '0');
      const auto moving = name.starts_with("d");
      if (moving) name.remove_prefix(1);
      if (name.size() != 3 || (name[0] != 'w' && name[0] != 'p') || name[1] < '0' ||
          name[1] > '1' || name[2] < '0' || name[2] > '6')
        return maximum_inputs;
      const auto group = (name[0] == 'p' ? 2 : 0) + (moving ? 1 : 0);
      return 10 + static_cast<std::size_t>(14 * group + 7 * (name[1] - '0') + name[2] - '0');
    });
    scatter.bind("d3_scatter_column", 8, 6, [](std::string_view name) {
      if (name == "bar") return std::size_t{0};
      if (name == "dbar") return std::size_t{1};
      return name.starts_with("dg") ? axis(name, "dg", 5) : axis(name, "g", 2);
    });
    numbers.assign(elements.begin(), elements.end());
    coordinates.assign(positions.begin(), positions.end());
  }

  HostD3ColumnValue apply(std::span<const double> direction, std::span<double> gradient,
                          std::span<double> action) {
    if (active.test_and_set()) throw std::logic_error("D3 column calls may not overlap or reenter");
    struct Guard {
      std::atomic_flag& flag;
      ~Guard() { flag.clear(); }
    } guard{active};
    last = {};
    const auto atoms = numbers.size(), dimension = coordinates.size();
    if (direction.size() != dimension || gradient.size() != dimension ||
        action.size() != dimension || !gradient.data() || !action.data() ||
        overlaps(gradient, action))
      throw std::invalid_argument("D3 column output shape or alias mismatch");
    finite(direction);
    std::vector<double> right(direction.begin(), direction.end());
    if (validate) validate();
    std::vector<double> cn(2 * atoms), weights(28 * atoms), bar(2 * atoms);
    std::vector<double> staged_gradient(dimension), staged_action(dimension);
    std::vector<Pair> jets;
    jets.reserve(pairs(atoms));
    for (std::size_t first = 0; first < atoms; ++first)
      for (std::size_t second = 0; second < first; ++second) {
        std::array<double, 7> input{};
        double squared{};
        for (std::size_t component = 0; component < 3; ++component) {
          input[component] =
              coordinates[3 * first + component] - coordinates[3 * second + component];
          input[3 + component] = right[3 * first + component] - right[3 * second + component];
          squared += input[component] * input[component];
        }
        if (!(squared > 1e-12) || !std::isfinite(squared))
          throw std::domain_error("D3 column pair distance outside physical domain");
        input[6] =
            d3_data::kElements[static_cast<std::size_t>(numbers[first] - 1)].covalent_radius +
            d3_data::kElements[static_cast<std::size_t>(numbers[second] - 1)].covalent_radius;
        jets.push_back({first, second, {}});
        coordination.evaluate(input, jets.back().coordination);
        ++last.coordination_pairs;
        for (const auto atom : {first, second})
          for (std::size_t leg = 0; leg < 2; ++leg)
            cn[leg * atoms + atom] += jets.back().coordination[leg];
      }
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      const auto& element = d3_data::kElements[static_cast<std::size_t>(numbers[atom] - 1)];
      std::array<double, 16> input{};
      input[0] = cn[atom];
      input[1] = cn[atoms + atom];
      for (std::size_t slot = 0; slot < slots; ++slot) {
        input[2 + slot] = slot < element.reference_count ? 1 : 0;
        input[9 + slot] = slot < element.reference_count
                              ? d3_data::kReferenceCn[element.reference_offset + slot]
                              : 0;
      }
      std::array<double, 29> output{};
      reference.evaluate(input, output);
      if (!(output[28] > 0))
        throw std::domain_error(
            "D3 reference normalization requires an unqualified underflow fallback");
      std::copy_n(output.begin(), 28, weights.begin() + 28 * atom);
      ++last.reference_atoms;
      last.reference_terms += slots;
    }
    HostD3ColumnValue value;
    for (const auto& jet : jets) {
      const auto first = jet.first, second = jet.second;
      const auto zi = numbers[first], zj = numbers[second];
      std::array<double, 115> input{};
      for (std::size_t component = 0; component < 3; ++component) {
        input[component] = coordinates[3 * first + component] - coordinates[3 * second + component];
        input[3 + component] = right[3 * first + component] - right[3 * second + component];
      }
      input[6] = 3 * d3_data::kElements[static_cast<std::size_t>(zi - 1)].r4r2 *
                 d3_data::kElements[static_cast<std::size_t>(zj - 1)].r4r2;
      input[7] = parameters.a1 * std::sqrt(input[6]) + parameters.a2;
      if (!(input[6] > 0) || !std::isfinite(input[6]) || !(input[7] > 0) ||
          !std::isfinite(input[7]))
        throw std::domain_error("D3 column damping radius outside physical domain");
      input[8] = parameters.s6;
      input[9] = parameters.s8;
      for (std::size_t group = 0; group < 4; ++group)
        for (std::size_t slot = 0; slot < slots; ++slot) {
          input[10 + 14 * group + slot] = weights[28 * first + 7 * group + slot];
          input[10 + 14 * group + 7 + slot] = weights[28 * second + 7 * group + slot];
        }
      for (std::size_t left = 0; left < slots; ++left)
        for (std::size_t right_slot = 0; right_slot < slots; ++right_slot)
          input[66 + 7 * left + right_slot] = coefficient(zi, zj, left, right_slot);
      std::array<double, 12> output{};
      pair.evaluate(input, output);
      ++last.energy_pairs;
      last.reference_terms += 49;
      value.energy += output[0];
      value.right += output[1];
      bar[first] += output[8];
      bar[second] += output[9];
      bar[atoms + first] += output[10];
      bar[atoms + second] += output[11];
      for (std::size_t component = 0; component < 3; ++component) {
        staged_gradient[3 * first + component] += output[2 + component];
        staged_gradient[3 * second + component] -= output[2 + component];
        staged_action[3 * first + component] += output[5 + component];
        staged_action[3 * second + component] -= output[5 + component];
      }
    }
    for (const auto& jet : jets) {
      std::array<double, 8> input{};
      input[0] = bar[jet.first] + bar[jet.second];
      input[1] = bar[atoms + jet.first] + bar[atoms + jet.second];
      std::copy(jet.coordination.begin() + 2, jet.coordination.end(), input.begin() + 2);
      std::array<double, 6> output{};
      scatter.evaluate(input, output);
      ++last.reverse_pairs;
      for (std::size_t component = 0; component < 3; ++component) {
        staged_gradient[3 * jet.first + component] += output[component];
        staged_gradient[3 * jet.second + component] -= output[component];
        staged_action[3 * jet.first + component] += output[3 + component];
        staged_action[3 * jet.second + component] -= output[3 + component];
      }
    }
    finite(std::array{value.energy, value.right});
    finite(staged_gradient);
    finite(staged_action);
    if (validate) validate();
    if (!std::equal(right.begin(), right.end(), direction.begin()))
      throw std::runtime_error("D3 column direction changed before publication");
    std::copy(staged_gradient.begin(), staged_gradient.end(), gradient.begin());
    std::copy(staged_action.begin(), staged_action.end(), action.begin());
    return value;
  }
};

HostD3Column::HostD3Column(std::span<const std::int32_t> elements,
                           std::span<const double> coordinates, D3Parameters parameters,
                           HostD3ColumnLimits limits, ValidateCurrent validate_current)
    : impl_(std::make_unique<Impl>(elements, coordinates, parameters, limits,
                                   std::move(validate_current))) {}
HostD3Column::~HostD3Column() = default;
HostD3ColumnValue HostD3Column::apply(std::span<const double> direction, std::span<double> gradient,
                                      std::span<double> action) {
  return impl_->apply(direction, gradient, action);
}
HostD3ColumnWork HostD3Column::work() const noexcept { return impl_->last; }
std::string_view HostD3Column::table_identity() noexcept { return d3_data::kTableSha256; }
std::string_view HostD3Column::radii_identity() noexcept { return d3_data::kRadiiSha256; }

}  // namespace generativeqc::dft::dispersion
