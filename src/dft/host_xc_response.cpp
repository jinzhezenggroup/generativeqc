#include "dft/host_xc_response.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "dft/host_gga_fields.hpp"
#include "dft/host_gga_ordered.hpp"

namespace generativeqc::dft {
namespace {

std::size_t add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("host XC response size overflow");
  return left + right;
}

std::size_t multiply(std::size_t left, std::size_t right) {
  if (right && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error("host XC response size overflow");
  return left * right;
}

std::size_t features(std::size_t spins) { return spins == 1 ? 2 : 5; }
std::size_t xc_roots(std::size_t spins) { return spins == 1 ? 10 : 36; }
std::size_t coefficient_inputs(std::size_t spins) { return spins == 1 ? 9 : 20; }

std::size_t matrix_elements(HostGgaResponseShape shape) {
  if (!shape.basis_functions || !shape.points || !shape.tile_points ||
      (shape.spins != 1 && shape.spins != 2))
    throw std::invalid_argument("invalid host GGA response shape");
  return multiply(shape.spins, multiply(shape.basis_functions, shape.basis_functions));
}

void finite(std::span<const double> values) {
  if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
    throw std::domain_error("nonfinite host XC response input or result");
}

std::size_t input_index(const host_hessian_aot::Program& program, std::string_view name) {
  const auto found = std::find(program.inputs.begin(), program.inputs.end(), name);
  if (found == program.inputs.end() ||
      std::find(found + 1, program.inputs.end(), name) != program.inputs.end())
    throw std::invalid_argument("incompatible host GGA response program inputs");
  return static_cast<std::size_t>(found - program.inputs.begin());
}

void evaluate(const host_hessian_aot::Program& program, std::span<const double> input,
              std::span<double> output, std::size_t count) {
  if (!program.evaluate) throw std::invalid_argument("linked host GGA scalar evaluator is absent");
  const auto status =
      program.evaluate(input.data(), input.size(), output.data(), output.size(), count);
  if (status < 0) throw std::invalid_argument("linked host GGA scalar ABI admission failed");
  if (status > 0) throw std::domain_error("nonfinite linked host GGA response derivative");
}

bool overlaps(std::span<const double> left, std::span<const double> right) {
  if (left.empty() || right.empty()) return false;
  const auto left_start = reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_start = reinterpret_cast<std::uintptr_t>(right.data());
  return left_start <= right_start ? right_start - left_start < left.size_bytes()
                                   : left_start - right_start < right.size_bytes();
}

}  // namespace

std::size_t HostGgaXcResponse::workspace_bytes(HostGgaResponseShape shape) {
  const auto matrices = matrix_elements(shape);
  const auto spins = shape.spins;
  auto per_point = add(multiply(4, shape.basis_functions), 1);
  per_point = add(per_point, multiply(12, spins));
  per_point = add(per_point, features(spins));
  per_point = add(per_point, xc_roots(spins));
  per_point = add(per_point, coefficient_inputs(spins));
  const auto count = add(matrices, multiply(std::min(shape.points, shape.tile_points), per_point));
  const auto bytes = multiply(count, sizeof(double));
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("host XC response byte count exceeds signed resource domain");
  return bytes;
}

std::size_t HostGgaXcResponse::primal_cache_bytes(HostGgaResponseShape shape,
                                                  std::size_t maximum_bytes) {
  const auto base = workspace_bytes(shape);
  const auto available = std::min(
      maximum_bytes, static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) - base);
  const auto per_point =
      multiply(add(features(shape.spins), xc_roots(shape.spins)), sizeof(double));
  auto points = std::min(shape.points, available / per_point);
  if (points < shape.points) points -= points % std::min(shape.points, shape.tile_points);
  return multiply(points, per_point);
}

std::size_t HostGgaXcResponse::batch_workspace_bytes(HostGgaResponseShape shape,
                                                     std::size_t directions) {
  const auto base = workspace_bytes(shape);
  if (!directions || directions > 4)
    throw std::invalid_argument("host XC batch requires one to four directions");
  if (directions == 1) return 0;
  const auto matrices = matrix_elements(shape);
  const auto count = std::min(shape.points, shape.tile_points);
  const auto copies = add(multiply(2, directions), 1);
  const auto panels = add(matrices, multiply(multiply(4, shape.spins), count));
  const auto bytes =
      multiply(add(multiply(copies, panels), multiply(4, shape.basis_functions)), sizeof(double));
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - base)
    throw std::overflow_error("host XC batch exceeds signed resource domain");
  return bytes;
}

HostGgaXcResponse::HostGgaXcResponse(HostGgaResponseShape shape, std::size_t budget_bytes,
                                     std::span<const double> density, std::string_view xc_identity,
                                     std::string_view coefficient_response_identity,
                                     HostGgaTileReader reader,
                                     std::function<void()> validate_current)
    : shape_(shape),
      density_(density),
      reader_(std::move(reader)),
      validate_current_(std::move(validate_current)) {
  const auto bytes = workspace_bytes(shape);
  if (bytes > budget_bytes) throw std::length_error("host XC response scratch budget exceeded");
  if (density.size() != matrix_elements(shape) || !density.data() || !reader_ || !validate_current_)
    throw std::invalid_argument("host XC response requires density and live grid/state owners");
  const auto suffix = shape.spins == 1 ? "unpolarized" : "polarized";
  xc_ = host_hessian_aot::find(std::string("xc_") + suffix, xc_identity);
  coefficients_ = host_hessian_aot::find(std::string("coefficient_response_") + suffix,
                                         coefficient_response_identity);
  const auto count = features(shape.spins);
  if (!xc_ || !coefficients_ || xc_->inputs.size() != count ||
      xc_->outputs != xc_roots(shape.spins) ||
      coefficients_->inputs.size() != coefficient_inputs(shape.spins) ||
      coefficients_->outputs != 4 * shape.spins)
    throw std::invalid_argument("linked host GGA response identity or layout is unavailable");
  constexpr std::array<std::string_view, 5> restricted{"rho", "sigma"};
  constexpr std::array<std::string_view, 5> unrestricted{"rho_a", "rho_b", "sigma_aa", "sigma_ab",
                                                         "sigma_bb"};
  for (std::size_t feature = 0; feature < count; ++feature) {
    feature_inputs_[feature] =
        input_index(*xc_, shape.spins == 1 ? restricted[feature] : unrestricted[feature]);
    if (feature >= shape.spins)
      values_[feature] = input_index(*coefficients_, "v" + std::to_string(feature));
    tangents_[feature] = input_index(*coefficients_, "dv" + std::to_string(feature));
  }
  for (std::size_t spin = 0; spin < shape.spins; ++spin)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto label = std::to_string(spin) + "_" + std::to_string(axis);
      gradients_[3 * spin + axis] = input_index(*coefficients_, "g" + label);
      gradient_tangents_[3 * spin + axis] = input_index(*coefficients_, "dg" + label);
    }
  validate_current_();
  finite(density_);
  storage_count_ = bytes / sizeof(double);
  storage_ = std::make_unique<double[]>(storage_count_);
}

bool HostGgaXcResponse::admit_primal_cache(std::size_t maximum_bytes) {
  if (busy_.test_and_set()) throw std::logic_error("host XC response calls overlap or reenter");
  struct Release {
    std::atomic_flag& flag;
    ~Release() { flag.clear(); }
  } release{busy_};
  validate_current_();
  if (work_.actions_started)
    throw std::logic_error("host XC primal cache must be admitted before actions");
  if (cache_) return cache_count_ * sizeof(double) <= maximum_bytes;
  const auto bytes = primal_cache_bytes(shape_, maximum_bytes);
  if (!bytes) return false;
  std::unique_ptr<double[]> candidate;
  try {
    candidate = std::make_unique<double[]>(bytes / sizeof(double));
  } catch (const std::bad_alloc&) {
    return false;
  }
  cache_ = std::move(candidate);
  cache_count_ = bytes / sizeof(double);
  cache_points_ = cache_count_ / add(xc_->inputs.size(), xc_->outputs);
  return true;
}

void HostGgaXcResponse::apply(std::span<const double> direction, std::span<double> output) {
  if (direction.size() != matrix_elements(shape_))
    throw std::invalid_argument("host XC response matrix shape or reference alias mismatch");
  apply_batch(direction, output, {});
}

void HostGgaXcResponse::apply_batch(std::span<const double> directions, std::span<double> output,
                                    std::span<double> workspace) {
  if (busy_.test_and_set()) throw std::logic_error("host XC response calls overlap or reenter");
  struct Release {
    std::atomic_flag& flag;
    bool& ready;
    bool previously_ready, completed{};
    ~Release() {
      if (!completed) ready = previously_ready;
      flag.clear();
    }
  } release{busy_, cache_ready_, cache_ready_};
  const auto matrix_count = matrix_elements(shape_);
  const auto direction_count = directions.size() / matrix_count;
  if (directions.size() % matrix_count || !direction_count || direction_count > 4 ||
      output.size() != directions.size() || !directions.data() || !output.data() ||
      overlaps(density_, output))
    throw std::invalid_argument("host XC response matrix shape or reference alias mismatch");
  const auto extra = batch_workspace_bytes(shape_, direction_count) / sizeof(double);
  if (workspace.size() < extra)
    throw std::length_error("host XC batch workspace capacity exceeded");
  workspace = workspace.first(extra);
  if (overlaps(workspace, directions) || overlaps(workspace, output) ||
      overlaps(workspace, density_) ||
      overlaps(workspace, std::span(storage_.get(), storage_count_)) ||
      overlaps(workspace, std::span(cache_.get(), cache_count_)))
    throw std::invalid_argument(
        "host XC batch workspace aliases a live input, output or owner panel");
  work_.actions_started = add(work_.actions_started, direction_count);
  validate_current_();
  finite(density_);
  finite(directions);
  const auto basis = shape_.basis_functions;
  const auto spins = shape_.spins;
  const auto feature_count = features(spins);
  const std::size_t full_features = spins == 1 ? 3 : 7;
  const auto tile_capacity = std::min(shape_.points, shape_.tile_points);
  std::span<double> storage(storage_.get(), storage_count_);
  const auto batched = direction_count > 1;
  const auto take_batch = [&workspace](std::size_t count) {
    auto panel = workspace.first(count);
    workspace = workspace.subspan(count);
    return panel;
  };
  const auto packed_inputs = take_batch(batched ? (direction_count + 1) * matrix_count : 0);
  auto result = batched ? take_batch(direction_count * matrix_count) : storage.first(matrix_count);
  const auto field_bank =
      take_batch(batched ? (direction_count + 1) * 4 * spins * tile_capacity : 0);
  const auto coefficient_bank =
      take_batch(batched ? direction_count * 4 * spins * tile_capacity : 0);
  const auto packing = take_batch(batched ? 4 * basis : 0);
  std::fill(result.begin(), result.end(), 0.0);
  for (std::size_t begin = 0; begin < shape_.points;) {
    validate_current_();
    const auto count = std::min(tile_capacity, shape_.points - begin);
    auto cursor = storage.subspan(matrix_count);
    const auto take = [&cursor, count](std::size_t components) {
      const auto panel = cursor.first(components * count);
      cursor = cursor.subspan(panel.size());
      return panel;
    };
    const auto ao = take(4 * basis), weights = take(1);
    const auto single_primal = take(4 * spins), single_tangent = take(4 * spins);
    const auto primal = batched ? field_bank.first(4 * spins * count) : single_primal;
    const auto xc_input = take(xc_->inputs.size()), xc_output = take(xc_->outputs);
    const auto coefficient_input = take(coefficients_->inputs.size());
    const auto coefficient_output = take(coefficients_->outputs);
    std::fill(ao.begin(), ao.end(), std::numeric_limits<double>::quiet_NaN());
    std::fill(weights.begin(), weights.end(), std::numeric_limits<double>::quiet_NaN());
    work_.tile_reads = add(work_.tile_reads, 1);
    reader_(begin, count, ao, weights);
    validate_current_();
    finite(ao);
    finite(weights);
    // Coefficient inputs are not live until scalar pullback preparation. Reuse
    // that admitted panel for AO packing; small tiles keep the scalar schedule.
    if (batched) {
      for (std::size_t element = 0; element < matrix_count; ++element) {
        packed_inputs[(direction_count + 1) * element] = density_[element];
        for (std::size_t direction = 0; direction < direction_count; ++direction)
          packed_inputs[(direction_count + 1) * element + direction + 1] =
              directions[direction * matrix_count + element];
      }
      host_gga_ordered::batch_fields(basis, count, spins, direction_count + 1, ao.data(),
                                     packed_inputs.data(), field_bank.data(), packing);
      finite(field_bank.first((direction_count + 1) * 4 * spins * count));
    } else {
      host_gga_ordered::fields(basis, count, spins, ao.data(), density_.data(), directions.data(),
                               primal.data(), single_tangent.data(), coefficient_input);
      finite(single_tangent);
    }
    finite(primal);
    for (std::size_t point = 0; point < count; ++point) {
      std::array<double, 5> feature{};
      for (std::size_t spin = 0; spin < spins; ++spin) {
        feature[spin] = primal[4 * spin * count + point];
        if (feature[spin] < 0.0)
          throw std::domain_error("host GGA response requires nonnegative spin point densities");
        if (!host_gga_spin_fields_admitted(feature[spin], {primal[(4 * spin + 1) * count + point],
                                                           primal[(4 * spin + 2) * count + point],
                                                           primal[(4 * spin + 3) * count + point]}))
          throw std::domain_error(
              "host GGA zero spin density requires a zero gradient (subnormal roundoff is "
              "permitted)");
        for (std::size_t axis = 0; axis < 3; ++axis) {
          const auto value = primal[(4 * spin + axis + 1) * count + point];
          feature[spins == 1 ? 1 : 2 + 2 * spin] += value * value;
        }
      }
      if (spins == 2)
        for (std::size_t axis = 0; axis < 3; ++axis)
          feature[3] += primal[(axis + 1) * count + point] * primal[(axis + 5) * count + point];
      for (std::size_t index = 0; index < feature_count; ++index)
        xc_input[feature_inputs_[index] * count + point] = feature[index];
    }
    finite(xc_input);
    work_.point_visits = add(work_.point_visits, multiply(direction_count, count));
    // The linked production graph owns its lazy vacuum branch and threshold.
    // Do not floor densities, evaluate an inactive inverse power, or replace
    // nonfinite derivatives at a non-vacuum spin endpoint with zero.
    if (cache_ready_ && begin < cache_points_) {
      for (std::size_t component = 0; component < xc_->inputs.size(); ++component)
        if (std::memcmp(xc_input.data() + component * count,
                        cache_.get() + component * cache_points_ + begin,
                        count * sizeof(double)) != 0)
          throw std::invalid_argument("host XC response cached point inputs changed");
      for (std::size_t component = 0; component < xc_->outputs; ++component)
        std::copy_n(cache_.get() + (xc_->inputs.size() + component) * cache_points_ + begin, count,
                    xc_output.data() + component * count);
      work_.cached_xc_points = add(work_.cached_xc_points, count);
    } else {
      work_.xc_evaluated_points = add(work_.xc_evaluated_points, count);
      evaluate(*xc_, xc_input, xc_output, count);
      if (cache_ && !cache_ready_ && begin < cache_points_) {
        for (std::size_t component = 0; component < xc_->inputs.size(); ++component)
          std::copy_n(xc_input.data() + component * count, count,
                      cache_.get() + component * cache_points_ + begin);
        for (std::size_t component = 0; component < xc_->outputs; ++component)
          std::copy_n(xc_output.data() + component * count, count,
                      cache_.get() + (xc_->inputs.size() + component) * cache_points_ + begin);
      }
    }
    for (std::size_t direction = 0; direction < direction_count; ++direction) {
      const auto tangent =
          batched ? field_bank.subspan((direction + 1) * 4 * spins * count, 4 * spins * count)
                  : single_tangent;
      for (std::size_t point = 0; point < count; ++point) {
        std::array<double, 7> feature_tangent{}, potential_tangent{};
        for (std::size_t spin = 0; spin < spins; ++spin) {
          feature_tangent[spin] = tangent[4 * spin * count + point];
          for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto index = (4 * spin + axis + 1) * count + point;
            feature_tangent[spins == 1 ? 1 : 2 + 2 * spin] += 2 * primal[index] * tangent[index];
            coefficient_input[gradients_[3 * spin + axis] * count + point] = primal[index];
            coefficient_input[gradient_tangents_[3 * spin + axis] * count + point] = tangent[index];
          }
        }
        if (spins == 2)
          for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto alpha = (axis + 1) * count + point, beta = (axis + 5) * count + point;
            feature_tangent[3] += tangent[alpha] * primal[beta] + primal[alpha] * tangent[beta];
          }
        std::size_t root = 1 + full_features;
        for (std::size_t left = 0; left < full_features; ++left)
          for (std::size_t right = left; right < full_features; ++right, ++root) {
            const auto value = xc_output[root * count + point];
            potential_tangent[left] += value * feature_tangent[right];
            if (left != right) potential_tangent[right] += value * feature_tangent[left];
          }
        for (std::size_t index = 0; index < feature_count; ++index) {
          if (index >= spins)
            coefficient_input[values_[index] * count + point] =
                xc_output[(index + 1) * count + point];
          coefficient_input[tangents_[index] * count + point] = potential_tangent[index];
        }
      }
      finite(coefficient_input);
      evaluate(*coefficients_, coefficient_input, coefficient_output, count);
      if (batched) {
        for (std::size_t component = 0; component < 4 * spins; ++component)
          for (std::size_t point = 0; point < count; ++point)
            coefficient_bank[(component * count + point) * direction_count + direction] =
                coefficient_output[component * count + point];
      } else {
        host_gga_ordered::pullback(basis, count, spins, ao.data(), coefficient_output.data(),
                                   weights.data(), result.data());
      }
    }
    if (batched)
      host_gga_ordered::batch_pullback(basis, count, spins, direction_count, ao.data(),
                                       coefficient_bank.data(), weights.data(), result.data());
    finite(result);
    begin += count;
  }
  validate_current_();
  work_.actions_completed = add(work_.actions_completed, direction_count);
  if (batched) {
    for (std::size_t element = 0; element < matrix_count; ++element)
      for (std::size_t direction = 0; direction < direction_count; ++direction)
        output[direction * matrix_count + element] = result[element * direction_count + direction];
  } else {
    std::copy(result.begin(), result.end(), output.begin());
  }
  cache_ready_ = cache_ != nullptr;
  release.completed = true;
}

}  // namespace generativeqc::dft
