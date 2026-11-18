#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "geometry/harmonic_modes.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void refuses(Function operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "invalid harmonic input was admitted");
}

void diatomic() {
  const std::vector<double> positions{0, 0, 0, 0, 0, 1.4}, masses{1, 2};
  std::vector<double> matrix(36);
  const double spring = 0.4;
  matrix[2 * 6 + 2] = matrix[5 * 6 + 5] = spring;
  matrix[2 * 6 + 5] = matrix[5 * 6 + 2] = -spring;
  const auto original = matrix;
  const auto result = generativeqc::geometry::harmonic_modes(matrix, positions, masses, 1 << 20);
  require(matrix == original, "harmonic analysis changed the raw Hessian");
  require(result.rigid_modes == 5 && result.active_modes == 1 && !result.negative_modes,
          "diatomic rigid-motion or negative-mode count is wrong");
  require(std::abs(result.eigenvalues[0] - spring * (1 / masses[0] + 1 / masses[1])) < 1e-13,
          "diatomic reduced-mass frequency is wrong");
  require(std::abs(result.frequencies_cm1[0] - std::sqrt(0.6) * 5140.487143715828) < 1e-7,
          "frequency unit conversion is wrong");
  double norm = 0, translation = 0;
  for (std::size_t coordinate = 0; coordinate < 6; ++coordinate) {
    const double component = result.cartesian_modes[coordinate];
    norm += masses[coordinate / 3] * component * component;
    translation += masses[coordinate / 3] * component;
  }
  require(std::abs(norm - 1) < 1e-13 && std::abs(translation) < 1e-13,
          "Cartesian modes are not mass-normalized or translation-free");
  for (double& value : matrix) value = -value;
  const auto unstable = generativeqc::geometry::harmonic_modes(matrix, positions, masses, 1 << 20);
  require(unstable.negative_modes == 1 && unstable.frequencies_cm1[0] < 0,
          "imaginary vibration lost its signed frequency");
  const std::vector<double> heavier{4, 8};
  const auto isotope = generativeqc::geometry::harmonic_modes(matrix, positions, heavier, 1 << 20);
  require(std::abs(isotope.frequencies_cm1[0] - 0.5 * unstable.frequencies_cm1[0]) < 1e-9,
          "uniform isotope scaling is wrong");
  for (double& value : matrix) value *= 1e-14;
  const auto soft = generativeqc::geometry::harmonic_modes(matrix, positions, masses, 1 << 20);
  require(soft.active_modes == 1 && soft.frequencies_cm1.size() == 1 && soft.frequencies_cm1[0] < 0,
          "a small physical vibration was deleted");
}

void linear_and_nonlinear() {
  const std::vector<double> positions{-1, 0, 0, 0, 0, 0, 1, 0, 0}, masses{1, 1, 1};
  std::vector<double> matrix(81);
  const double bend[]{1, -2, 1};
  const double chain[3][3]{{1, -1, 0}, {-1, 2, -1}, {0, -1, 1}};
  for (std::size_t first = 0; first < 3; ++first)
    for (std::size_t second = 0; second < 3; ++second)
      for (std::size_t axis = 0; axis < 3; ++axis)
        matrix[(3 * first + axis) * 9 + 3 * second + axis] =
            axis == 0 ? chain[first][second] : (axis == 1 ? -1 : 1) * bend[first] * bend[second];
  const auto result = generativeqc::geometry::harmonic_modes(matrix, positions, masses, 1 << 20);
  const std::vector<double> expected{-6, 1, 3, 6};
  require(result.rigid_modes == 5 && result.active_modes == 4, "linear molecular rank is wrong");
  for (std::size_t mode = 0; mode < expected.size(); ++mode)
    require(std::abs(result.eigenvalues[mode] - expected[mode]) < 1e-12,
            "independent linear-chain spectrum is wrong");
  const double cosine = std::cos(0.713), sine = std::sin(0.713);
  const double rotation[3][3]{{cosine, -sine, 0}, {sine, cosine, 0}, {0, 0, 1}};
  std::vector<double> rotated(9), transformed(81);
  for (std::size_t atom = 0; atom < 3; ++atom)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      rotated[3 * atom + axis] = 7 + static_cast<double>(axis);
      for (std::size_t inner = 0; inner < 3; ++inner)
        rotated[3 * atom + axis] += rotation[axis][inner] * positions[3 * atom + inner];
    }
  for (std::size_t first = 0; first < 3; ++first)
    for (std::size_t second = 0; second < 3; ++second)
      for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
          for (std::size_t inner = 0; inner < 3; ++inner)
            for (std::size_t other = 0; other < 3; ++other)
              transformed[(3 * first + row) * 9 + 3 * second + column] +=
                  rotation[row][inner] * matrix[(3 * first + inner) * 9 + 3 * second + other] *
                  rotation[column][other];
  const auto moved = generativeqc::geometry::harmonic_modes(transformed, rotated, masses, 1 << 20);
  for (std::size_t mode = 0; mode < expected.size(); ++mode)
    require(std::abs(moved.eigenvalues[mode] - expected[mode]) < 1e-12,
            "harmonics depend on the Cartesian frame");
  const std::vector<double> bent{0, 0, 0, 1, 0, 0, 0, 1, 0};
  const auto nonlinear = generativeqc::geometry::harmonic_modes(matrix, bent, masses, 1 << 20);
  require(nonlinear.rigid_modes == 6 && nonlinear.active_modes == 3, "nonlinear rank is wrong");
  const std::vector<double> atom{0, 0, 0}, mass{2}, zero(9);
  const auto monatomic = generativeqc::geometry::harmonic_modes(zero, atom, mass, 1 << 20);
  require(monatomic.rigid_modes == 3 && monatomic.frequencies_cm1.empty(),
          "monatomic geometry acquired a vibration");
}

void invalid_inputs() {
  const std::vector<double> positions{0, 0, 0, 0, 0, 1}, masses{1, 1};
  std::vector<double> matrix(36);
  const auto analyze = [&](const auto& geometry, const auto& weights, std::size_t bytes = 1 << 20) {
    return generativeqc::geometry::harmonic_modes(matrix, geometry, weights, bytes);
  };
  refuses([&] { analyze(positions, masses, 1); });
  refuses([&] { analyze(positions, std::vector<double>{1}); });
  refuses([&] { analyze(positions, std::vector<double>{1, -1}); });
  refuses(
      [&] { analyze(positions, std::vector<double>{1, std::numeric_limits<double>::infinity()}); });
  refuses([&] { analyze(positions, std::vector<double>{1e300, 1e-300}); });
  refuses([&] { analyze(std::vector<double>(6), masses); });
  auto bad_positions = positions;
  bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
  refuses([&] { analyze(bad_positions, masses); });
  matrix[1] = 1e-4;
  refuses([&] { analyze(positions, masses); });
  matrix[1] = 1e-12;
  const auto admitted = analyze(positions, masses);
  require(admitted.raw_symmetry_error == 1e-12 && matrix[1] == 1e-12 && matrix[6] == 0,
          "roundoff averaging hid or modified raw asymmetry");
  matrix[1] = std::numeric_limits<double>::quiet_NaN();
  refuses([&] { analyze(positions, masses); });
}
}  // namespace

int main() {
  try {
    diatomic();
    linear_and_nonlinear();
    invalid_inputs();
    std::cout << "native supplied-Hessian harmonic controls passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
