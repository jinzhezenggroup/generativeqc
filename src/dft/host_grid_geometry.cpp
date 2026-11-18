#include "dft/host_grid_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace generativeqc::dft {
namespace {

std::size_t add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("host grid geometry size overflow");
  return left + right;
}

std::size_t multiply(std::size_t left, std::size_t right) {
  if (right && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error("host grid geometry size overflow");
  return left * right;
}

bool overlap(const void* left, std::size_t left_bytes, const void* right, std::size_t right_bytes) {
  if (!left_bytes || !right_bytes) return false;
  const auto first = reinterpret_cast<std::uintptr_t>(left);
  const auto second = reinterpret_cast<std::uintptr_t>(right);
  return first <= second ? second - first < left_bytes : first - second < right_bytes;
}

void finite(std::span<const double> values) {
  if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
    throw std::domain_error("nonfinite host grid geometry input or result");
}

std::size_t integer(double value, std::size_t maximum) {
  if (!std::isfinite(value) || value < 0 || value > static_cast<double>(maximum) ||
      std::trunc(value) != value)
    throw std::invalid_argument("invalid packed AO record in host grid geometry");
  return static_cast<std::size_t>(value);
}

struct Busy {
  std::atomic_flag& flag;
  explicit Busy(std::atomic_flag& value) : flag(value) {
    if (flag.test_and_set()) throw std::logic_error("host grid geometry reentry");
  }
  ~Busy() { flag.clear(); }
};

}  // namespace

HostGridGeometryPlan HostGridGeometry::plan(const AoBasis& basis, const MolecularGrid& grid,
                                            std::size_t tile_points) {
  if (!tile_points || !basis.natom || !basis.nao || !basis.nprimitive || !grid.point_count() ||
      basis.natom != grid.system().atoms.size())
    throw std::invalid_argument("invalid host grid geometry dimensions");
  const auto count = std::min(tile_points, grid.point_count());
  const auto work = grid.weight_direction_work(count);
  auto elements = add(multiply(26, multiply(count, basis.nao)), multiply(2, count));
  auto bytes =
      add(multiply(elements, sizeof(double)), grid.weight_direction_workspace_bytes(count));
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("host grid geometry exceeds signed resource domain");
  return {bytes, count, multiply(count, basis.nao), work.pair_visits};
}

HostGridGeometry::HostGridGeometry(const AoBasis& basis, const MolecularGrid& grid,
                                   std::size_t tile_points, HostGridGeometryPlan limits,
                                   std::string_view ao_motion_identity,
                                   std::function<void()> validate_current)
    : basis_(basis),
      grid_(grid),
      tile_points_(std::min(tile_points, grid.point_count())),
      atoms_(basis.natom),
      primitives_(basis.nprimitive),
      aos_(basis.nao),
      points_(grid.point_count()),
      plan_(plan(basis, grid, tile_points)),
      validate_current_(std::move(validate_current)) {
  if (plan_.numeric_bytes > limits.numeric_bytes || plan_.point_visits > limits.point_visits ||
      plan_.ao_point_visits > limits.ao_point_visits ||
      plan_.grid_pair_visits > limits.grid_pair_visits)
    throw std::length_error("host grid geometry capacity or work cap exceeded");
  if (!validate_current_) throw std::invalid_argument("host grid geometry requires a live owner");
  motion_ = host_hessian_aot::find("ao_first_motion", ao_motion_identity);
  if (!motion_ || !motion_->evaluate || motion_->outputs != 4 || motion_->inputs.size() != 12)
    throw std::invalid_argument("missing exact linked AO motion program");
  for (std::size_t index = 0; index < inputs_.size(); ++index) {
    const auto name =
        index < 9 ? "jet" + std::to_string(index + 1) : "left" + std::to_string(index - 9);
    const auto found = std::find(motion_->inputs.begin(), motion_->inputs.end(), name);
    if (found == motion_->inputs.end() ||
        std::find(found + 1, motion_->inputs.end(), name) != motion_->inputs.end())
      throw std::invalid_argument("incompatible linked AO motion inputs");
    inputs_[index] = static_cast<std::size_t>(found - motion_->inputs.begin());
  }
  retained_ = {basis.packed.data(), grid.points().data(), grid.weights().data(),
               grid.owners().data(), grid.system().atoms.data()};
  this->validate_current();
  ao_atoms_.reserve(aos_);
  const auto offset = add(multiply(3, atoms_), multiply(2, primitives_));
  for (std::size_t orbital = 0; orbital < aos_; ++orbital)
    ao_atoms_.push_back(integer(basis.packed[offset + 16 * orbital], atoms_ - 1));
  storage_ = std::make_unique<double[]>(plan_.numeric_bytes / sizeof(double));
}

void HostGridGeometry::validate_current() const {
  validate_current_();
  const auto offset = add(multiply(3, atoms_), multiply(2, primitives_));
  if (basis_.natom != atoms_ || basis_.nprimitive != primitives_ || basis_.nao != aos_ ||
      basis_.packed.size() != add(offset, multiply(16, aos_)) || grid_.point_count() != points_ ||
      grid_.points().size() != multiply(3, points_) || grid_.owners().size() != points_ ||
      grid_.system().atoms.size() != atoms_ ||
      retained_ != std::array<const void*, 5>{basis_.packed.data(), grid_.points().data(),
                                              grid_.weights().data(), grid_.owners().data(),
                                              grid_.system().atoms.data()})
    throw std::invalid_argument("replaced or reshaped host grid geometry frame");
  finite(basis_.packed);
  for (std::size_t primitive = 0; primitive < primitives_; ++primitive)
    if (basis_.packed[3 * atoms_ + 2 * primitive] <= 0)
      throw std::invalid_argument("host AO primitive exponents must be positive");
  for (std::size_t atom = 0; atom < atoms_; ++atom)
    for (std::size_t axis = 0; axis < 3; ++axis)
      if (basis_.packed[3 * atom + axis] != grid_.system().atoms[atom].position[axis])
        throw std::invalid_argument("host AO and grid geometry mismatch");
  for (std::size_t orbital = 0; orbital < aos_; ++orbital) {
    const auto* record = basis_.packed.data() + offset + 16 * orbital;
    const auto atom = integer(record[0], atoms_ - 1);
    if (!ao_atoms_.empty() && ao_atoms_[orbital] != atom)
      throw std::invalid_argument("changed host AO center map");
    const auto begin = integer(record[1], primitives_);
    const auto count = integer(record[2], primitives_ - begin);
    const auto terms = integer(record[3], 3);
    if (!count || !terms) throw std::invalid_argument("empty host AO contraction or expansion");
    for (std::size_t term = 0; term < terms; ++term) {
      std::size_t degree = 0;
      for (std::size_t axis = 0; axis < 3; ++axis)
        degree += integer(record[4 + 4 * term + axis], 3);
      if (degree > 3) throw std::invalid_argument("host AO geometry supports through f");
    }
  }
}

void HostGridGeometry::reject_borrowed_alias(std::span<const double> values) const {
  if (overlap(values.data(), values.size_bytes(), basis_.packed.data(),
              basis_.packed.size() * sizeof(double)) ||
      overlap(values.data(), values.size_bytes(), grid_.points().data(),
              grid_.points().size() * sizeof(double)) ||
      overlap(values.data(), values.size_bytes(), grid_.weights().data(),
              grid_.weights().size() * sizeof(double)) ||
      overlap(values.data(), values.size_bytes(), grid_.owners().data(),
              grid_.owners().size() * sizeof(std::uint32_t)) ||
      overlap(values.data(), values.size_bytes(), grid_.system().atoms.data(),
              atoms_ * sizeof(core::Atom)))
    throw std::invalid_argument("host grid geometry output aliases borrowed frame");
}

void HostGridGeometry::read(std::size_t begin, std::size_t count, std::span<const double> direction,
                            std::span<double> ao, std::span<double> moving_ao,
                            std::span<double> weights, std::span<double> moving_weights) {
  Busy guard(busy_);
  validate_current();
  const auto lanes = multiply(count, aos_);
  if (begin > points_ || count > points_ - begin || count > tile_points_ ||
      direction.size() != multiply(3, atoms_) || !direction.data() ||
      ao.size() != multiply(4, lanes) || moving_ao.size() != ao.size() || weights.size() != count ||
      moving_weights.size() != count)
    throw std::invalid_argument("invalid host grid geometry tile binding");
  const std::array<std::span<double>, 4> outputs{ao, moving_ao, weights, moving_weights};
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto output = outputs[index];
    if (!output.empty() && !output.data())
      throw std::invalid_argument("null host grid geometry output");
    reject_borrowed_alias(output);
    for (std::size_t other = 0; other < index; ++other)
      if (overlap(output.data(), output.size_bytes(), outputs[other].data(),
                  outputs[other].size_bytes()))
        throw std::invalid_argument("overlapping host grid geometry outputs");
  }
  finite(direction);
  if (!count) return;
  auto* cursor = storage_.get();
  const auto take = [&cursor](std::size_t size) {
    auto result = std::span<double>(cursor, size);
    cursor += size;
    return result;
  };
  auto jets = take(10 * lanes), input = take(12 * lanes), tangent = take(4 * lanes);
  auto weight = take(count), delta_weight = take(count);
  auto workspace = take(grid_.weight_direction_workspace_bytes(count) / sizeof(double));
  const auto scheduled = grid_.weight_direction_work(count);
  work_.tiles_started = add(work_.tiles_started, 1);
  work_.point_visits = add(work_.point_visits, count);
  work_.ao_point_visits = add(work_.ao_point_visits, lanes);
  basis_.evaluate(grid_.points().data() + 3 * begin, count, 2, 0, aos_, jets.data(), jets.size());
  finite(jets);
  validate_current();
  for (std::size_t index = 0; index < 9; ++index)
    std::copy_n(jets.data() + (index + 1) * lanes, lanes, input.data() + inputs_[index] * lanes);
  for (std::size_t point = 0; point < count; ++point) {
    const auto owner = grid_.owners()[begin + point];
    if (owner >= atoms_) throw std::invalid_argument("invalid host grid point owner");
    for (std::size_t orbital = 0; orbital < aos_; ++orbital)
      for (std::size_t axis = 0; axis < 3; ++axis)
        input[inputs_[9 + axis] * lanes + point * aos_ + orbital] =
            direction[3 * owner + axis] - direction[3 * ao_atoms_[orbital] + axis];
  }
  finite(input);
  const auto status =
      motion_->evaluate(input.data(), input.size(), tangent.data(), tangent.size(), lanes);
  if (status) throw std::domain_error("linked host AO motion evaluation failed");
  work_.grid_pair_visits = add(work_.grid_pair_visits, scheduled.pair_visits);
  grid_.weight_direction(begin, count, direction, delta_weight, workspace);
  std::copy_n(grid_.weights().data() + begin, count, weight.data());
  finite(weight);
  finite(delta_weight);
  finite(tangent);
  validate_current();
  work_.tiles_completed = add(work_.tiles_completed, 1);
  std::copy_n(jets.data(), ao.size(), ao.data());
  std::copy(tangent.begin(), tangent.end(), moving_ao.begin());
  std::copy(weight.begin(), weight.end(), weights.begin());
  std::copy(delta_weight.begin(), delta_weight.end(), moving_weights.begin());
}

}  // namespace generativeqc::dft
