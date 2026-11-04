#include "posthf/df_interaction_source.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "generated_df_ao_source.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/df_ao_source.hpp"

namespace generativeqc::posthf {
namespace {
namespace graph = df_ao_generated;
std::size_t payload(const std::vector<double>& values) {
  return checked_mul(values.capacity(), sizeof(double));
}
bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}
}  // namespace

DensityFittedInteractionSource::DensityFittedInteractionSource(
    const integrals::ElectronInteractionSource& reference, core::System auxiliary, double threshold,
    std::size_t budget, bool cuda, int device, bool forces,
    integrals::CorrelationIntegralBackend backend)
    : reference_(reference),
      auxiliary_(std::move(auxiliary)),
      threshold_(threshold),
      budget_(budget),
      cuda_(cuda),
      forces_(forces),
      device_(device),
      backend_(backend) {
  if (cuda_ && !backend_.density_fitting)
    throw std::invalid_argument("CUDA fitted interaction requires an explicit integral backend");
  const auto orbital_count = reference.nbf(), auxiliary_count = molecule::ao_count(auxiliary_);
  const auto cartesian_orbitals = molecule::cartesian_ao_count(reference.orbital());
  const auto cartesian_auxiliary = molecule::cartesian_ao_count(auxiliary_);
  if (!orbital_count || !auxiliary_count || !std::isfinite(threshold) || threshold <= 0 ||
      threshold >= 1 || auxiliary_.atoms.size() != reference.orbital().atoms.size())
    throw std::invalid_argument(
        "invalid unrestricted fitted interaction topology/metric threshold");
  const auto raw_count = graph::checked_product({orbital_count, orbital_count, auxiliary_count});
  const auto metric_count = checked_mul(auxiliary_count, auxiliary_count);
  const auto eri_count =
      graph::checked_product({orbital_count, orbital_count, orbital_count, orbital_count});
  const auto raw_cartesian =
      graph::checked_product({cartesian_orbitals, cartesian_orbitals, cartesian_auxiliary});
  const auto metric_cartesian = checked_mul(cartesian_auxiliary, cartesian_auxiliary);
  const auto arena =
      std::max(graph::required_value(orbital_count, auxiliary_count),
               forces ? graph::required_pullback(orbital_count, auxiliary_count) : 0);
  auto count =
      checked_add(checked_mul(12, checked_add(raw_count, metric_count)),
                  checked_add(checked_mul(8, eri_count), checked_mul(cuda ? 4 : 2, arena)));
  auto raw_generation = checked_mul(4, checked_add(raw_cartesian, metric_cartesian));
  if (forces && cuda)
    raw_generation = checked_mul(raw_generation,
                                 checked_add(1, checked_mul(3, reference.orbital().atoms.size())));
  count = checked_add(count, raw_generation);
  capacity_ = checked_add(checked_mul(count, sizeof(double)), checked_mul(2, source_scratch_bytes));
  capacity_ = checked_add(
      capacity_, checked_add(source_capacity(reference.orbital()), source_capacity(auxiliary_)));
  if (capacity_ > budget)
    throw std::length_error("unrestricted DF source/response exceeds numeric memory budget");
  // Complete Gram, reverse Gram and whitening work are bounded before source generation.
  auto work = graph::checked_product({eri_count, auxiliary_count, forces ? 8U : 2U});
  work = checked_add(work, graph::checked_product({raw_count, auxiliary_count, forces ? 8U : 2U}));
  if (work > 100000000000ULL)
    throw std::length_error("unrestricted DF source exceeds scalar-work bound");
  if (cuda) {
#if GENERATIVEQC_HAS_CUDA
    raw_ = backend_.density_fitting(device_, reference.orbital(), auxiliary_, false);
#else
    throw std::invalid_argument("CUDA fitted interaction requires a CUDA build");
#endif
  } else
    raw_ = integrals::build_density_fitting_integrals(reference.orbital(), auxiliary_, false);
  if (raw_.nbf != orbital_count || raw_.naux != auxiliary_count ||
      raw_.three_center.size() != raw_count || raw_.metric.size() != metric_count)
    throw std::runtime_error("unrestricted DF source dimension mismatch");
  metric_ = integrals::factor_density_fitting_metric(raw_.metric, auxiliary_count, threshold);
  if (cuda) {
#if GENERATIVEQC_HAS_CUDA
    eri_ = evaluate_df_ao_cuda(orbital_count, auxiliary_count, raw_.three_center,
                               metric_.inverse_square_root, {}, device, budget)
               .eri;
#endif
  } else {
    std::vector<double> arena(graph::required_value(orbital_count, auxiliary_count));
    const auto values = graph::evaluate_value(
        orbital_count, auxiliary_count,
        {raw_.three_center.data(), metric_.inverse_square_root.data(), nullptr}, arena.data(),
        arena.size());
    eri_.assign(values.eri, values.eri + eri_count);
  }
  if (!finite(eri_)) throw std::runtime_error("nonfinite unrestricted fitted interaction");
}

std::size_t DensityFittedInteractionSource::retained_numeric_bytes() const {
  auto result =
      checked_add(payload(eri_), checked_add(payload(raw_.three_center), payload(raw_.metric)));
  result = checked_add(result, payload(metric_.inverse_square_root));
  return checked_add(result, reference_.retained_numeric_bytes());
}

void DensityFittedInteractionSource::read(Operator op, const std::array<std::size_t, 4>& begin,
                                          const std::array<std::size_t, 4>& count, double* output,
                                          std::size_t elements) const {
  if (op == Operator::overlap || op == Operator::hcore) {
    reference_.read(op, begin, count, output, elements);
    return;
  }
  const std::vector<double>* values = nullptr;
  std::array<std::size_t, 4> extent{1, 1, 1, 1};
  unsigned rank = 0;
  if (op == Operator::eri) {
    values = &eri_;
    extent = {nbf(), nbf(), nbf(), nbf()};
    rank = 4;
  }
  if (op == Operator::metric) {
    values = &raw_.metric;
    extent = {naux(), naux(), 1, 1};
    rank = 2;
  }
  if (op == Operator::three_center) {
    values = &raw_.three_center;
    extent = {nbf(), nbf(), naux(), 1};
    rank = 3;
  }
  if (!values) throw std::invalid_argument("unknown unrestricted DF source operator");
  std::size_t required = 1;
  for (unsigned axis = 0; axis < 4; ++axis) {
    if (begin[axis] > extent[axis] || count[axis] > extent[axis] - begin[axis] ||
        (axis >= rank && (begin[axis] || count[axis] != 1)))
      throw std::invalid_argument("unrestricted DF source tile out of bounds");
    required = checked_mul(required, count[axis]);
  }
  if (required != elements || (elements && !output))
    throw std::invalid_argument("invalid unrestricted DF source destination");
  for (std::size_t flat = 0; flat < elements; ++flat) {
    auto remaining = flat;
    std::array<std::size_t, 4> indices{};
    for (unsigned reverse = rank; reverse > 0; --reverse) {
      const auto axis = reverse - 1;
      indices[axis] = begin[axis] + remaining % count[axis];
      remaining /= count[axis];
    }
    std::size_t source_offset = 0;
    for (unsigned axis = 0; axis < rank; ++axis)
      source_offset = source_offset * extent[axis] + indices[axis];
    output[flat] = (*values)[source_offset];
  }
}

std::vector<double> DensityFittedInteractionSource::nuclear_derivative(
    std::span<const double> eri_cotangent) const {
  if (!forces_ || eri_cotangent.size() != eri_.size() || !finite(eri_cotangent))
    throw std::invalid_argument("unadmitted or invalid unrestricted DF nuclear response");
  DFAOSourceResult bars;
  if (cuda_) {
#if GENERATIVEQC_HAS_CUDA
    bars = evaluate_df_ao_cuda(nbf(), naux(), raw_.three_center, metric_.inverse_square_root,
                               eri_cotangent, device_, budget_);
#endif
  } else {
    std::vector<double> arena(graph::required_pullback(nbf(), naux()));
    const auto result = graph::evaluate_pullback(
        nbf(), naux(),
        {raw_.three_center.data(), metric_.inverse_square_root.data(), eri_cotangent.data()},
        arena.data(), arena.size());
    bars.three_center.assign(result.bar_three_center,
                             result.bar_three_center + raw_.three_center.size());
    bars.inverse_root.assign(result.bar_inverse_root, result.bar_inverse_root + raw_.metric.size());
  }
  const auto metric_bar = integrals::density_fitting_metric_response(
      raw_.metric, metric_.inverse_square_root, bars.inverse_root, naux(), threshold_,
      tensor::SymmetricMatrixFunction::inverse_sqrt);
  if (!cuda_)
    return integrals::contract_weighted_density_fitting_derivative(
        orbital(), auxiliary_, bars.three_center, metric_bar, budget_);
#if GENERATIVEQC_HAS_CUDA
  // Detached coordinate tensors are an explicit bounded hybrid DF consumer;
  // unlike the conventional shell path this must not claim streaming flags.
  const auto derivative = backend_.density_fitting(device_, orbital(), auxiliary_, true);
  const auto coordinates = checked_mul(3, orbital().atoms.size());
  if (derivative.metric_derivative.size() != checked_mul(coordinates, raw_.metric.size()) ||
      derivative.three_center_derivative.size() !=
          checked_mul(coordinates, raw_.three_center.size()))
    throw std::runtime_error("unrestricted DF derivative source shape mismatch");
  std::vector<double> gradient(coordinates);
  for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
    gradient[coordinate] = std::inner_product(
        metric_bar.begin(), metric_bar.end(),
        derivative.metric_derivative.begin() + coordinate * metric_bar.size(), 0.0);
    gradient[coordinate] = std::inner_product(
        bars.three_center.begin(), bars.three_center.end(),
        derivative.three_center_derivative.begin() + coordinate * bars.three_center.size(),
        gradient[coordinate]);
  }
  if (!finite(gradient)) throw std::runtime_error("nonfinite unrestricted DF nuclear response");
  return gradient;
#else
  throw std::invalid_argument("CUDA DF response requires a CUDA build");
#endif
}
}  // namespace generativeqc::posthf
