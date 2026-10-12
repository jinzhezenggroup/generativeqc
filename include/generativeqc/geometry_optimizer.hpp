#ifndef GENERATIVEQC_GEOMETRY_OPTIMIZER_HPP
#define GENERATIVEQC_GEOMETRY_OPTIMIZER_HPP

// Native Cartesian minimum search driven by fully converged analytic forces.
// The callback is the only owner of the electronic method and its provenance.
// Positions, steps: Bohr; energies: Hartree; forces: Hartree/Bohr.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace generativeqc::geometry {

struct EnergyForces {
  double energy{};
  std::vector<double> forces;
};
struct Options {
  std::size_t max_steps{80};
  double force_tolerance{3e-4};
  double max_atom_step{0.35};
  std::size_t history{8};
  std::size_t max_backtracks{20};
};
struct Result {
  std::vector<double> coordinates;
  double energy{}, max_force{};
  std::size_t steps{}, evaluations{};
};
using Evaluator = std::function<EnergyForces(std::span<const double>)>;

inline double dot(std::span<const double> a, std::span<const double> b) {
  if (a.size() != b.size()) throw std::invalid_argument("geometry vector sizes differ");
  double value = 0;
  for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
  return value;
}

inline double maximum_atomic_norm(std::span<const double> v) {
  if (v.empty() || v.size() % 3 != 0)
    throw std::invalid_argument("geometry vectors require three components per atom");
  double maximum = 0;
  for (std::size_t i = 0; i < v.size(); i += 3) {
    const double norm = std::hypot(v[i], v[i + 1], v[i + 2]);
    if (!std::isfinite(norm)) throw std::runtime_error("nonfinite geometry or force norm");
    maximum = std::max(maximum, norm);
  }
  return maximum;
}

// Standalone L-BFGS; accepts only energy-decreasing moves under Armijo and
// atom-wise step bounds. A fresh call resets all Hessian history at stage changes.
// Exhaustion and native electronic failures never masquerade as convergence.
inline Result minimize(std::span<const double> initial, const Options& options,
                       const Evaluator& evaluate) {
  if (initial.empty() || initial.size() % 3 != 0 ||
      !std::all_of(initial.begin(), initial.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("initial geometry must be finite atom-major Bohr coordinates");
  if (options.max_steps == 0 || options.history == 0 || options.max_backtracks == 0 ||
      !std::isfinite(options.force_tolerance) || options.force_tolerance <= 0 ||
      !std::isfinite(options.max_atom_step) || options.max_atom_step <= 0)
    throw std::invalid_argument("invalid geometry optimization limits");
  if (!evaluate) throw std::invalid_argument("missing geometry energy/force evaluator");

  const std::size_t n = initial.size();
  std::vector<double> x(initial.begin(), initial.end());
  std::size_t evaluations = 0;
  const auto checked = [&](std::span<const double> position) {
    EnergyForces result = evaluate(position);
    ++evaluations;
    if (!std::isfinite(result.energy) || result.forces.size() != n ||
        !std::all_of(result.forces.begin(), result.forces.end(),
                     [](double f) { return std::isfinite(f); }))
      throw std::runtime_error("geometry evaluator returned nonfinite energy or invalid forces");
    return result;
  };

  EnergyForces current = checked(x);
  struct Pair {
    std::vector<double> s, y;
    double inv_sy{};
  };
  std::vector<Pair> history;
  history.reserve(options.history);

  for (std::size_t step = 0; step <= options.max_steps; ++step) {
    const double fmax = maximum_atomic_norm(current.forces);
    if (fmax <= options.force_tolerance)
      return {std::move(x), current.energy, fmax, step, evaluations};
    if (step == options.max_steps)
      throw std::runtime_error("geometry optimization exhausted its step budget without convergence");

    std::vector<double> g(n), direction(n);
    for (std::size_t i = 0; i < n; ++i) g[i] = -current.forces[i];
    direction = g;
    std::vector<double> alpha(history.size(), 0);
    for (std::size_t j = history.size(); j-- > 0;) {
      alpha[j] = history[j].inv_sy * dot(history[j].s, direction);
      for (std::size_t i = 0; i < n; ++i) direction[i] -= alpha[j] * history[j].y[i];
    }
    if (!history.empty()) {
      const auto& last = history.back();
      const double yy = dot(last.y, last.y);
      if (yy > 0) {
        const double gamma = (1.0 / last.inv_sy) / yy;
        for (double& v : direction) v *= std::clamp(gamma, 1e-5, 1e4);
      }
    }
    for (std::size_t j = 0; j < history.size(); ++j) {
      const double beta = history[j].inv_sy * dot(history[j].y, direction);
      for (std::size_t i = 0; i < n; ++i)
        direction[i] += history[j].s[i] * (alpha[j] - beta);
    }
    for (double& v : direction) v = -v;
    double slope = dot(g, direction);
    if (!std::isfinite(slope) || slope >= -1e-14 * dot(g, g)) {
      history.clear();
      for (std::size_t i = 0; i < n; ++i) direction[i] = -g[i];
      slope = -dot(g, g);
    }
    const double norm = maximum_atomic_norm(direction);
    if (!(norm > 0)) throw std::runtime_error("zero geometry descent direction");
    if (norm > options.max_atom_step) {
      const double scale = options.max_atom_step / norm;
      for (double& v : direction) v *= scale;
      slope *= scale;
    }

    EnergyForces accepted;
    std::vector<double> trial(n);
    double scale = 1;
    bool found = false;
    for (std::size_t attempt = 0; attempt < options.max_backtracks; ++attempt) {
      for (std::size_t i = 0; i < n; ++i) trial[i] = x[i] + scale * direction[i];
      // Do not suppress electronic convergence or resource failures by retry.
      EnergyForces candidate = checked(trial);
      if (candidate.energy <= current.energy + 1e-4 * scale * slope) {
        accepted = std::move(candidate);
        found = true;
        break;
      }
      scale *= 0.5;
    }
    if (!found) throw std::runtime_error("geometry Armijo line search failed");

    Pair update;
    update.s.resize(n);
    update.y.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      update.s[i] = trial[i] - x[i];
      update.y[i] = current.forces[i] - accepted.forces[i];
    }
    const double sy = dot(update.s, update.y);
    const double ss = dot(update.s, update.s);
    const double yy = dot(update.y, update.y);
    if (std::isfinite(sy) && sy > 1e-10 * std::sqrt(ss * yy) && yy > 0) {
      update.inv_sy = 1.0 / sy;
      if (history.size() == options.history) history.erase(history.begin());
      history.push_back(std::move(update));
    } else {
      history.clear();
    }
    x = std::move(trial);
    current = std::move(accepted);
  }
  throw std::logic_error("unreachable geometry convergence branch");
}
}  // namespace generativeqc::geometry
#endif
