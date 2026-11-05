#include "posthf/df_interaction_source.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "core/work_budget.hpp"
#include "generated_df_ao_source.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/df_ao_source.hpp"
#include "posthf/df_factor_blocks.hpp"
#include "posthf/native_provider.hpp"
#include "runtime/df_progress_trace.hpp"

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
    const integrals::ElectronInteractionSource& reference,
    const hf::UnrestrictedPhysicalReference& frame, core::System auxiliary, double threshold,
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
  runtime::df_progress::Scope trace("unrestricted_df_factors", cuda ? "cuda" : "host");
  core::validate_electronic_reference_shape(frame.electronic_reference());
  if (frame.nbf != reference.nbf() || !frame.source_identity ||
      !frame.source_identity->matches(reference.orbital()))
    throw std::invalid_argument("unrestricted DF factors require their physical reference");
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
  const auto raw_cartesian =
      graph::checked_product({cartesian_orbitals, cartesian_orbitals, cartesian_auxiliary});
  const auto metric_cartesian = checked_mul(cartesian_auxiliary, cartesian_auxiliary);
  const auto arena =
      std::max(graph::required_factors(orbital_count, auxiliary_count),
               forces ? graph::required_factor_pullback(orbital_count, auxiliary_count) : 0);
  auto count = checked_add(checked_mul(24, checked_add(raw_count, metric_count)),
                           checked_mul(cuda ? 4 : 2, arena));
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
  auto work = checked_mul(3, graph::work_factors(orbital_count, auxiliary_count));
  if (forces)
    work = checked_add(work,
                       checked_mul(4, graph::work_factor_pullback(orbital_count, auxiliary_count)));
  const auto alpha = static_cast<std::size_t>(
      (reference.orbital().electron_count + reference.orbital().multiplicity - 1) / 2);
  const auto beta = static_cast<std::size_t>(reference.orbital().electron_count) - alpha;
  core::admit_correlation_work("df_source", work, capacity_, orbital_count, alpha, beta);
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
  coefficients_ = frame.coefficients;
  occupied_ = frame.nocc;
  std::vector<double> identity(checked_mul(orbital_count, orbital_count));
  for (std::size_t orbital = 0; orbital < orbital_count; ++orbital)
    identity[orbital * orbital_count + orbital] = 1;
  for (unsigned spin = 0; spin < 3; ++spin) {
    const auto& coefficients = spin < 2 ? coefficients_[spin] : identity;
    auto& destination = spin < 2 ? factors_[spin] : ao_factors_;
    if (cuda) {
#if GENERATIVEQC_HAS_CUDA
      destination =
          evaluate_df_factors_cuda(orbital_count, auxiliary_count, raw_.three_center,
                                   metric_.inverse_square_root, coefficients, {}, device, budget)
              .factors;
#endif
    } else {
      std::vector<double> scratch(graph::required_factors(orbital_count, auxiliary_count));
      const auto values = graph::evaluate_factors(
          orbital_count, auxiliary_count,
          {raw_.three_center.data(), metric_.inverse_square_root.data(), coefficients.data()},
          scratch.data(), scratch.size());
      destination.assign(values.factors, values.factors + raw_count);
    }
    if (!finite(destination)) throw std::runtime_error("nonfinite unrestricted DF factors");
  }
#if GENERATIVEQC_HAS_CUDA
  if (cuda) {
    const auto retained = retained_numeric_bytes();
    cuda_blocks_ =
        make_df_factor_cuda_blocks(orbital_count, auxiliary_count, {factors_[0], factors_[1]},
                                   device, retained < capacity_ ? capacity_ - retained : 0);
  }
#endif
  runtime::df_progress::number("ao_values_retained", raw_count);
  runtime::df_progress::number("factor_transform_summands", work);
}

DensityFittedInteractionSource::~DensityFittedInteractionSource() = default;

std::size_t DensityFittedInteractionSource::retained_numeric_bytes() const {
  auto result = checked_add(payload(ao_factors_),
                            checked_add(payload(raw_.three_center), payload(raw_.metric)));
  result = checked_add(result, payload(metric_.inverse_square_root));
  for (unsigned spin = 0; spin < 2; ++spin)
    result =
        checked_add(result, checked_add(payload(factors_[spin]), payload(coefficients_[spin])));
  if (cuda_blocks_) result = checked_add(result, cuda_blocks_->retained_numeric_bytes());
  return checked_add(result, reference_.retained_numeric_bytes());
}

void DensityFittedInteractionSource::validate_frame(
    const core::ElectronicReferenceView& frame) const {
  core::validate_electronic_reference_shape(frame);
  if (frame.basis_functions != nbf() || frame.spin_channels != 2)
    throw std::invalid_argument("DF factor cache has a different orbital frame");
  for (unsigned spin = 0; spin < 2; ++spin)
    if (frame.channels[spin].occupied != occupied_[spin] ||
        !std::equal(coefficients_[spin].begin(), coefficients_[spin].end(),
                    frame.channels[spin].coefficients.begin()))
      throw std::invalid_argument("stale DF factor orbital frame");
}

const std::vector<double>& DensityFittedInteractionSource::spin_factors(unsigned spin) const {
  return factors_.at(spin);
}

std::size_t DensityFittedInteractionSource::block_capacity(const std::array<std::size_t, 4>& shape,
                                                           bool cuda) const {
  const auto left = checked_mul(shape[0], shape[1]), right = checked_mul(shape[2], shape[3]);
  const auto panels = checked_mul(checked_add(left, right), naux());
  const auto output = checked_mul(left, right);
  auto values = checked_add(panels, checked_mul(2, output));
  if (cuda) values = checked_mul(2, values);
  const auto indices =
      checked_add(checked_add(shape[0], shape[1]), checked_add(shape[2], shape[3]));
  return checked_add(
      checked_mul(values, sizeof(double)),
      checked_add(checked_mul(cuda ? 2 : 0, checked_mul(indices, sizeof(std::size_t))),
                  cuda ? sizeof(int) : 0));
}

void DensityFittedInteractionSource::add_pair_cotangent(std::array<std::vector<double>, 2>& seeds,
                                                        unsigned left_spin, unsigned right_spin,
                                                        const std::array<std::size_t, 4>& orbitals,
                                                        double weight) const {
  if (left_spin > 1 || right_spin > 1 || !std::isfinite(weight) ||
      seeds[0].size() != factors_[0].size() || seeds[1].size() != factors_[1].size() ||
      std::any_of(orbitals.begin(), orbitals.end(), [&](auto orbital) { return orbital >= nbf(); }))
    throw std::invalid_argument("invalid DF Gram pair cotangent");
  const auto left = (orbitals[0] * nbf() + orbitals[1]) * naux();
  const auto right = (orbitals[2] * nbf() + orbitals[3]) * naux();
  for (std::size_t auxiliary = 0; auxiliary < naux(); ++auxiliary) {
    seeds[left_spin][left + auxiliary] += weight * factors_[right_spin][right + auxiliary];
    seeds[right_spin][right + auxiliary] += weight * factors_[left_spin][left + auxiliary];
  }
}

std::vector<double> DensityFittedInteractionSource::factor_block(const SpinMOSlots& request,
                                                                 bool cuda, int device,
                                                                 std::size_t budget,
                                                                 ProviderWork* work) const {
  if (cuda != cuda_ || (cuda && device != device_))
    throw std::invalid_argument("DF factor execution backend differs from its source");
  std::array<std::size_t, 4> shape{};
  for (unsigned axis = 0; axis < 4; ++axis) {
    shape[axis] = request.slots[axis].size();
    if (!shape[axis] || request.spins[axis] > 1 || request.spins[axis] != request.spins[axis ^ 1])
      throw std::invalid_argument("DF factor block requires nonempty same-spin pairs");
    for (std::size_t item = 0; item < shape[axis]; ++item) {
      const auto orbital = request.slots[axis][item];
      if (orbital != padded_mo &&
          (orbital >= nbf() ||
           std::find(request.slots[axis].begin(), request.slots[axis].begin() + item, orbital) !=
               request.slots[axis].begin() + item))
        throw std::invalid_argument("invalid or repeated DF factor orbital slot");
    }
  }
  if (block_capacity(shape, cuda) > budget)
    throw std::length_error("DF factor block exceeds numeric memory budget");
  if (work) {
    ++work->mo_blocks;
    work->transform_fmas = checked_add(
        work->transform_fmas, graph::work_factor_block(checked_mul(shape[0], shape[1]),
                                                       checked_mul(shape[2], shape[3]), naux()));
  }
#if GENERATIVEQC_HAS_CUDA
  if (cuda) return cuda_blocks_->get(request, work);
#endif
  std::array<std::vector<double>, 2> panels;
  for (unsigned pair = 0; pair < 2; ++pair) {
    auto& panel = panels[pair];
    const auto first = 2 * pair, second = first + 1;
    panel.resize(checked_mul(checked_mul(shape[first], shape[second]), naux()));
    std::size_t row = 0;
    for (const auto left : request.slots[first])
      for (const auto right : request.slots[second]) {
        if (left != padded_mo && right != padded_mo)
          std::copy_n(factors_[request.spins[first]].data() + (left * nbf() + right) * naux(),
                      naux(), panel.data() + row * naux());
        ++row;
      }
  }
  const auto left = checked_mul(shape[0], shape[1]), right = checked_mul(shape[2], shape[3]);
  std::vector<double> arena(graph::required_factor_block(left, right, naux()));
  graph::Inputs inputs;
  inputs.left_factors = panels[0].data();
  inputs.right_factors = panels[1].data();
  const auto output =
      graph::evaluate_factor_block(left, right, naux(), inputs, arena.data(), arena.size());
  std::vector<double> values(output.block, output.block + checked_mul(left, right));
  if (!finite(values)) throw std::runtime_error("nonfinite DF factor block");
  return values;
}

DFFactorSourceResult DensityFittedInteractionSource::pullback_factors(
    unsigned spin, std::span<const double> seed) const {
  if (!forces_ || spin > 1 || seed.size() != raw_.three_center.size() || !finite(seed))
    throw std::invalid_argument("unadmitted or invalid unrestricted DF factor response");
  runtime::df_progress::Scope trace("unrestricted_df_factor_pullback", cuda_ ? "cuda" : "host");
#if GENERATIVEQC_HAS_CUDA
  if (cuda_)
    return evaluate_df_factors_cuda(nbf(), naux(), raw_.three_center, metric_.inverse_square_root,
                                    coefficients_[spin], seed, device_, budget_);
#endif
  std::vector<double> arena(graph::required_factor_pullback(nbf(), naux()));
  const auto output =
      graph::evaluate_factor_pullback(nbf(), naux(),
                                      {raw_.three_center.data(), metric_.inverse_square_root.data(),
                                       coefficients_[spin].data(), seed.data()},
                                      arena.data(), arena.size());
  DFFactorSourceResult result;
  result.three_center.assign(output.bar_three_center, output.bar_three_center + seed.size());
  result.inverse_root.assign(output.bar_inverse_root, output.bar_inverse_root + raw_.metric.size());
  result.coefficients.assign(output.bar_coefficients,
                             output.bar_coefficients + checked_mul(nbf(), nbf()));
  return result;
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
  if (!values && op != Operator::eri)
    throw std::invalid_argument("unknown unrestricted DF source operator");
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
    if (op == Operator::eri) {
      const auto left = (indices[0] * nbf() + indices[1]) * naux();
      const auto right = (indices[2] * nbf() + indices[3]) * naux();
      double value = 0;
      for (std::size_t auxiliary = 0; auxiliary < naux(); ++auxiliary)
        value += ao_factors_[left + auxiliary] * ao_factors_[right + auxiliary];
      output[flat] = value;
    } else {
      output[flat] = (*values)[source_offset];
    }
  }
}

std::vector<double> DensityFittedInteractionSource::nuclear_derivative(
    const std::vector<double>& raw_cotangent, const std::vector<double>& root_cotangent) const {
  if (!forces_ || raw_cotangent.size() != raw_.three_center.size() ||
      root_cotangent.size() != raw_.metric.size() || !finite(raw_cotangent) ||
      !finite(root_cotangent))
    throw std::invalid_argument("unadmitted or invalid unrestricted DF nuclear response");
  const auto metric_bar = integrals::density_fitting_metric_response(
      raw_.metric, metric_.inverse_square_root, root_cotangent, naux(), threshold_,
      tensor::SymmetricMatrixFunction::inverse_sqrt);
  if (!cuda_)
    return integrals::contract_weighted_density_fitting_derivative(
        orbital(), auxiliary_, raw_cotangent, metric_bar, budget_);
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
        raw_cotangent.begin(), raw_cotangent.end(),
        derivative.three_center_derivative.begin() + coordinate * raw_cotangent.size(),
        gradient[coordinate]);
  }
  if (!finite(gradient)) throw std::runtime_error("nonfinite unrestricted DF nuclear response");
  return gradient;
#else
  throw std::invalid_argument("CUDA DF response requires a CUDA build");
#endif
}
}  // namespace generativeqc::posthf
