#ifndef GENERATIVEQC_SCF_INTERACTION_SOURCE_VIEW_HPP
#define GENERATIVEQC_SCF_INTERACTION_SOURCE_VIEW_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/electron_interaction_source.hpp"
#include "scf/fock_prepared.hpp"
#if GENERATIVEQC_HAS_CUDA
#include "scf/cuda_direct_jk_device.hpp"
#endif

namespace generativeqc::scf {

/** Read-only AO interaction view over one prepared Fock owner.
 *
 * This adapter does not change the prepared plan's scientific identity or
 * allocate another integral tensor. Exact CPU owners expose their resident
 * four-center ERI. Exact CUDA owners expose the same unscreened full-range ERI
 * semantics through bounded device tiles generated from their retained Direct
 * basis/geometry metadata; they do not materialize or publish a host N^4 tensor.
 * Density-fitted CPU owners retain the existing metric/three-center view.
 *
 * The PreparedFockPlan must outlive this view and all device work borrowed
 * through it. The device consumer is responsible for completing its stream
 * before releasing the view/plan.
 */
class PreparedFockInteractionSourceView final : public integrals::ElectronInteractionSource {
 public:
  explicit PreparedFockInteractionSourceView(const PreparedFockPlan& plan) : plan_(plan) {}

  const core::System& orbital() const override { return plan_.system(); }
  std::size_t nbf() const override { return plan_.one_electron().nbf; }
  std::size_t naux() const override {
    const auto* fitted = plan_.cpu_fitted_data();
    return fitted ? fitted->raw.naux : 0;
  }
  std::size_t retained_numeric_bytes() const override {
    auto bytes = plan_.cpu_observation_capacity();
    if (plan_.cuda_direct_source())
      bytes = checked_add(bytes, plan_.diagnostic().direct.device_bytes);
    return bytes;
  }

  bool supports(Operator op) const noexcept override {
    const auto n = nbf();
    const auto& one = plan_.one_electron();
    switch (op) {
      case Operator::overlap:
        return matches_size(one.overlap, n, n);
      case Operator::hcore:
        return matches_size(one.hcore, n, n);
      case Operator::eri:
        return matches_size(one.eri, n, n, n, n) || plan_.cuda_direct_source() != nullptr;
      case Operator::metric: {
        const auto* fitted = plan_.cpu_fitted_data();
        return fitted && matches_size(fitted->raw.metric, fitted->raw.naux, fitted->raw.naux);
      }
      case Operator::three_center: {
        const auto* fitted = plan_.cpu_fitted_data();
        return fitted && matches_size(fitted->raw.three_center, fitted->raw.nbf, fitted->raw.nbf,
                                      fitted->raw.naux);
      }
    }
    return false;
  }

  bool supports_host_read(Operator op) const noexcept override {
    if (op != Operator::eri) return supports(op);
    const auto n = nbf();
    return matches_size(plan_.one_electron().eri, n, n, n, n);
  }

  bool supports_device_read(Operator op, int device) const noexcept override {
#if GENERATIVEQC_HAS_CUDA
    auto* source = plan_.cuda_direct_source();
    return op == Operator::eri && source != nullptr && device >= 0 &&
           cuda_direct_jk_device(source) == device;
#else
    (void)op;
    (void)device;
    return false;
#endif
  }

  void read(Operator op, const std::array<std::size_t, 4>& begin,
            const std::array<std::size_t, 4>& count, double* out,
            std::size_t elements) const override {
    std::array<std::size_t, 4> extent{1, 1, 1, 1};
    std::size_t rank = 0;
    const auto& values = values_for(op, extent, rank);

    std::size_t requested = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (begin[axis] > extent[axis] || count[axis] > extent[axis] - begin[axis])
        throw std::invalid_argument("prepared interaction tile out of bounds");
      requested = checked_mul(requested, count[axis]);
    }
    for (std::size_t axis = rank; axis < 4; ++axis)
      if (begin[axis] != 0 || count[axis] != 1)
        throw std::invalid_argument("prepared interaction tile rank mismatch");
    if (requested != elements)
      throw std::invalid_argument("prepared interaction tile element count mismatch");
    if (elements && !out) throw std::invalid_argument("null prepared interaction tile output");

    // All validation precedes publication. Decode the caller's row-major tile
    // and gather from the resident row-major tensor without temporary storage.
    for (std::size_t local = 0; local < elements; ++local) {
      auto remainder = local;
      std::array<std::size_t, 4> index{};
      for (std::size_t reverse = 0; reverse < rank; ++reverse) {
        const auto axis = rank - reverse - 1;
        index[axis] = begin[axis] + remainder % count[axis];
        remainder /= count[axis];
      }
      std::size_t flat = index[0];
      for (std::size_t axis = 1; axis < rank; ++axis)
        flat = checked_add(checked_mul(flat, extent[axis]), index[axis]);
      out[local] = values[flat];
    }
  }

  void read_device(Operator op, const std::array<std::size_t, 4>& begin,
                   const std::array<std::size_t, 4>& count,
                   integrals::DeviceInteractionTarget target, std::size_t elements) const override {
#if GENERATIVEQC_HAS_CUDA
    if (!supports_device_read(op, target.device))
      throw std::invalid_argument("prepared Fock owner has no requested device interaction");
    if (!target.values || !target.stream || target.capacity < elements)
      throw std::invalid_argument("prepared device interaction target is invalid");
    const auto n = nbf();
    std::size_t requested = 1;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      if (!count[axis] || begin[axis] > n || count[axis] > n - begin[axis])
        throw std::invalid_argument("prepared device interaction tile out of bounds");
      requested = checked_mul(requested, count[axis]);
    }
    if (requested != elements)
      throw std::invalid_argument("prepared device interaction element count mismatch");

    std::string detail;
    const auto status = enqueue_cuda_direct_eri_tile(
        plan_.cuda_direct_source(), 0, begin, count, target.values, elements,
        reinterpret_cast<cudaStream_t>(target.stream), detail);
    if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT)
      throw std::invalid_argument(detail.empty() ? "prepared device ERI tile rejected" : detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "prepared device ERI tile failed" : detail);
#else
    (void)op;
    (void)begin;
    (void)count;
    (void)target;
    (void)elements;
    throw std::invalid_argument("prepared device interactions require CUDA support");
#endif
  }

 private:
  static std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::overflow_error("prepared interaction index overflow");
    return a + b;
  }

  static std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
      throw std::overflow_error("prepared interaction extent overflow");
    return a * b;
  }

  template <class... Extents>
  static bool matches_size(const std::vector<double>& values, Extents... extents) noexcept {
    std::size_t total = 1;
    const std::size_t shape[]{static_cast<std::size_t>(extents)...};
    for (const auto extent : shape) {
      if (total && extent > std::numeric_limits<std::size_t>::max() / total) return false;
      total *= extent;
    }
    return values.size() == total;
  }

  const std::vector<double>& values_for(Operator op, std::array<std::size_t, 4>& extent,
                                        std::size_t& rank) const {
    if (!supports_host_read(op))
      throw std::invalid_argument("prepared Fock owner does not retain requested host interaction");
    const auto n = nbf();
    const auto& one = plan_.one_electron();
    switch (op) {
      case Operator::overlap:
        extent = {n, n, 1, 1};
        rank = 2;
        return one.overlap;
      case Operator::hcore:
        extent = {n, n, 1, 1};
        rank = 2;
        return one.hcore;
      case Operator::eri:
        extent = {n, n, n, n};
        rank = 4;
        return one.eri;
      case Operator::metric: {
        const auto& raw = plan_.cpu_fitted_data()->raw;
        extent = {raw.naux, raw.naux, 1, 1};
        rank = 2;
        return raw.metric;
      }
      case Operator::three_center: {
        const auto& raw = plan_.cpu_fitted_data()->raw;
        extent = {raw.nbf, raw.nbf, raw.naux, 1};
        rank = 3;
        return raw.three_center;
      }
    }
    throw std::invalid_argument("unknown prepared interaction operator");
  }

  const PreparedFockPlan& plan_;
};

}  // namespace generativeqc::scf
#endif
