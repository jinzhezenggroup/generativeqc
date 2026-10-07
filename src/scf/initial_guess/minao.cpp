#include "scf/initial_guess/minao.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/minao_basis.hpp"
#include "integrals/s_integrals.hpp"

namespace generativeqc::scf::initial_guess {
namespace {

double electron_trace(const Matrix& density, const Matrix& overlap, std::size_t n) {
  double value = 0.0;
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < n; ++column)
      value += density[row * n + column] * overlap[column * n + row];
  return value;
}

}  // namespace

std::size_t minao_source_ao_count(const core::System& system) {
  return integrals::minao_basis_ao_count(system);
}

std::size_t minao_source_primitive_count(const core::System& system) {
  return integrals::minao_basis_primitive_count(system);
}

MinaoDensityResult minao_density(const core::System& system, const integrals::IntegralData& target,
                                 const Matrix& target_orthogonalizer) {
  const std::size_t n = target.nbf;
  if (!n || n > std::numeric_limits<std::size_t>::max() / n)
    throw std::invalid_argument("MINAO target dimensions overflow");
  const auto n2 = n * n;
  if (!n || target.overlap.size() != n2 || target_orthogonalizer.size() != n2)
    throw std::invalid_argument("MINAO target overlap/orthogonalizer shape mismatch");

  auto source = integrals::make_minao_basis_source(system);
  const std::size_t ns = source.occupations.size();
  if (ns > std::numeric_limits<std::size_t>::max() / n)
    throw std::invalid_argument("MINAO cross-overlap dimensions overflow");
  const auto cross_size = n * ns;
  std::vector<double> cross(cross_size, 0.0);
  integrals::cross_overlap(system, source.system, cross);

  // PySCF project_mo_nr2nr: P = S_tt^-1 S_ts = X X^T S_ts.
  // The native symmetric orthogonalizer X is row-major and symmetric.
  std::vector<double> transformed(cross_size, 0.0);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t source_ao = 0; source_ao < ns; ++source_ao) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < n; ++inner)
        value += target_orthogonalizer[inner * n + row] * cross[inner * ns + source_ao];
      transformed[row * ns + source_ao] = value;
    }

  std::fill(cross.begin(), cross.end(), 0.0);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t source_ao = 0; source_ao < ns; ++source_ao) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < n; ++inner)
        value += target_orthogonalizer[row * n + inner] * transformed[inner * ns + source_ao];
      cross[row * ns + source_ao] = value;
    }

  Matrix density(n * n, 0.0);
  double source_electrons = 0.0;
  for (std::size_t source_ao = 0; source_ao < ns; ++source_ao) {
    const double occupation = source.occupations[source_ao];
    source_electrons += occupation;
    if (occupation == 0.0) continue;
    for (std::size_t row = 0; row < n; ++row) {
      const double left = occupation * cross[row * ns + source_ao];
      for (std::size_t column = 0; column < n; ++column)
        density[row * n + column] += left * cross[column * ns + source_ao];
    }
  }
  if (!std::all_of(density.begin(), density.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("MINAO projection produced a nonfinite density");

  const double projected_electrons = electron_trace(density, target.overlap, n);
  return {std::move(density), ns, source.primitive_count, source_electrons, projected_electrons};
}

Matrix admissible_minao_density(const core::System& system, const integrals::IntegralData& target,
                                const Matrix& target_orthogonalizer, const Matrix& raw_density,
                                const EigenOperation& eigen) {
  const auto n = target.nbf;
  if (!n || n > std::numeric_limits<std::size_t>::max() / n)
    throw std::invalid_argument("MINAO target dimensions overflow");
  const auto n2 = n * n;
  if (system.electron_count <= 0 || static_cast<std::size_t>(system.electron_count) > 2 * n ||
      target.overlap.size() != n2 || target_orthogonalizer.size() != n2 || raw_density.size() != n2)
    throw std::invalid_argument("MINAO admission has invalid dimensions or electron capacity");
  const auto finite = [](const Matrix& matrix) {
    return std::all_of(matrix.begin(), matrix.end(), [](double v) { return std::isfinite(v); });
  };
  if (!finite(target.overlap) || !finite(target_orthogonalizer))
    throw std::invalid_argument("MINAO admission has a nonfinite target metric");

  // This is a MINAO-specific construction, not a repair in the shared seed
  // validator. Overlapping atomic orbitals may have occupations above two.
  // Preserve the raw PySCF projection separately, then find the closest
  // trace-normalized orthogonal density in the restricted ensemble set.
  const auto normalized = normalized_warm_density(system, target, raw_density);
  const auto root = reference::multiply(target.overlap, target_orthogonalizer, n);
  auto metric_density = reference::multiply(root, reference::multiply(normalized, root, n), n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      const auto value = 0.5 * (metric_density[i * n + j] + metric_density[j * n + i]);
      metric_density[i * n + j] = metric_density[j * n + i] = value;
    }
  if (!finite(metric_density)) throw std::runtime_error("nonfinite MINAO metric density");
  auto spectrum = eigen ? eigen(metric_density, nullptr, nullptr, n)
                        : reference::symmetric_eigen(std::move(metric_density), n);
  if (spectrum.values.size() != n || spectrum.vectors.size() != n2)
    throw std::runtime_error("MINAO provider returned an invalid eigenframe shape");
  if (!finite(spectrum.values) || !finite(spectrum.vectors))
    throw std::runtime_error("nonfinite MINAO metric spectrum");

  // KKT solution of min ||f-lambda||^2, 0<=f<=2, sum(f)=N:
  // f_i = clamp(lambda_i + shift, 0, 2). The fixed iteration bound is
  // independent of chemistry and no SCF, Fock, J/K or XC work is involved.
  const double electrons = system.electron_count;
  double lower = -spectrum.values.back();
  double upper = 2.0 - spectrum.values.front();
  for (unsigned step = 0; step < 128; ++step) {
    const double shift = lower / 2.0 + upper / 2.0;
    double total = 0.0;
    for (double value : spectrum.values) total += std::clamp(value + shift, 0.0, 2.0);
    if (total < electrons)
      lower = shift;
    else
      upper = shift;
  }
  const double shift = lower / 2.0 + upper / 2.0;
  for (double& value : spectrum.values) value = std::clamp(value + shift, 0.0, 2.0);
  // Remove only summation roundoff without rescaling a saturated occupation.
  double residual =
      electrons - std::accumulate(spectrum.values.begin(), spectrum.values.end(), 0.0);
  for (double& value : spectrum.values) {
    const double change = std::clamp(residual, -value, 2.0 - value);
    value += change;
    residual -= change;
  }
  if (std::abs(residual) > 1e-12)
    throw std::runtime_error("MINAO occupation projection failed electron conservation");

  const auto coefficients = reference::multiply(target_orthogonalizer, spectrum.vectors, n);
  Matrix density(n2, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      double value = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        value += coefficients[i * n + k] * spectrum.values[k] * coefficients[j * n + k];
      density[i * n + j] = density[j * n + i] = value;
    }
  if (!finite(density)) throw std::runtime_error("MINAO admission produced a nonfinite density");
  return density;
}

}  // namespace generativeqc::scf::initial_guess
