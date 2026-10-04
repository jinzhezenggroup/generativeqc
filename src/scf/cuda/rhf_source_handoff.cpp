#include "scf/cuda/rhf_source_handoff.hpp"

#include <chrono>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "integrals/electron_interaction_source.hpp"
#include "posthf/capacity.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "tensor/cuda_error.hpp"

namespace generativeqc::scf {
namespace {
using posthf::checked_add;
using posthf::checked_mul;

/** Only the existing public-AO ERI producer's immutable inputs are copied.
 * Quartet queues, density-aware bounds, Cartesian J/K projections and all
 * solver arrays are intentionally absent from this source-only owner. */
template <class Visit>
void source_fields(const CudaRhfBucketPlan& owner, cuda_execution::DeviceBatch& target,
                   Visit visit) {
  const auto& h = owner.topology;
  const auto& l = owner.layout;
  visit(target.atom_offsets, l.atom_offsets, h.atom_offsets.size());
  visit(target.atom_systems, l.atom_systems, h.atom_systems.size());
  visit(target.atomic_numbers, l.atomic_numbers, h.atomic_numbers.size());
  // Dynamic positions are removed from the host topology after packing.
  visit(target.positions, l.positions, checked_mul(owner.total_atoms, 3));
  visit(target.system_shell_offsets, l.system_shell_offsets, h.system_shell_offsets.size());
  visit(target.shell_atoms, l.shell_atoms, h.shell_atoms.size());
  visit(target.shell_angular, l.shell_angular, h.shell_angular.size());
  visit(target.shell_ao_offsets, l.shell_ao_offsets, h.shell_ao_offsets.size());
  visit(target.shell_primitive_offsets, l.shell_primitive_offsets,
        h.shell_primitive_offsets.size());
  visit(target.ao_shells, l.ao_shells, h.ao_shells.size());
  visit(target.ao_term_counts, l.ao_term_counts, h.ao_term_counts.size());
  visit(target.ao_term_angular, l.ao_term_angular, h.ao_term_angular.size());
  visit(target.ao_term_coefficients, l.ao_term_coefficients, h.ao_term_coefficients.size());
  visit(target.primitive_exponents, l.primitive_exponents, h.primitive_exponents.size());
  visit(target.primitive_coefficients, l.primitive_coefficients, h.primitive_coefficients.size());
}

class RhfInteractionSource final : public integrals::ElectronInteractionSource {
 public:
  RhfInteractionSource(core::System&& system, std::unique_ptr<CudaDirectJkPlan> plan,
                       std::size_t retained)
      : system_(std::move(system)), plan_(std::move(plan)), retained_(retained) {}
  const core::System& orbital() const override { return system_; }
  std::size_t nbf() const override { return plan_->diagnostic.nbf; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return retained_; }
  bool supports(Operator op) const noexcept override { return op == Operator::eri; }
  bool supports_host_read(Operator) const noexcept override { return false; }
  bool supports_device_read(Operator op, int device) const noexcept override {
    return op == Operator::eri && device == plan_->device_id;
  }
  void read(Operator, const std::array<std::size_t, 4>&, const std::array<std::size_t, 4>&, double*,
            std::size_t) const override {
    throw std::invalid_argument("detached RHF source only publishes device ERI tiles");
  }
  void read_device(Operator op, const std::array<std::size_t, 4>& begin,
                   const std::array<std::size_t, 4>& count,
                   integrals::DeviceInteractionTarget target, std::size_t elements) const override {
    if (!supports_device_read(op, target.device) || target.capacity < elements)
      throw std::invalid_argument("detached RHF interaction target is incompatible");
    std::string detail;
    const auto status =
        enqueue_cuda_direct_eri_tile(plan_.get(), 0, begin, count, target.values, elements,
                                     reinterpret_cast<cudaStream_t>(target.stream), detail);
    if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
  }

 private:
  core::System system_;
  std::unique_ptr<CudaDirectJkPlan> plan_;
  std::size_t retained_;
};
}  // namespace

CudaRhfSourceHandoff detach_rhf_cuda_source(const CudaRhfBucketPlan& owner, core::System&& orbital,
                                            std::size_t reference_peak_bytes,
                                            std::size_t numeric_budget) {
  using generativeqc_tensor::cuda_check;
  const auto started = std::chrono::steady_clock::now();
  if (!owner.initialized || owner.batch_size != 1 || owner.unrestricted ||
      !owner.options.export_physical_reference || !owner.resources.arena_ ||
      !owner.resources.stream_ || !reference_peak_bytes || !numeric_budget)
    throw std::invalid_argument("exact source handoff requires one published CUDA RHF reference");

  CudaRhfSourceHandoff result;
  result.numeric_peak_bytes = reference_peak_bytes;
  cuda_execution::DeviceBatch batch{};
  batch.batch_size = 1;
  batch.nbf = static_cast<std::int32_t>(owner.nbf);
  batch.direct_nbf = batch.nbf;
  batch.total_atoms = static_cast<std::int64_t>(owner.total_atoms);
  batch.total_shells = static_cast<std::int64_t>(owner.total_shells);
  std::size_t compact_bytes = 0, copy_bytes = 0;
  // Both passes use exactly the same aligned layout and checked extents.
  const auto reserve = [&](auto*& pointer, std::size_t source_offset, std::size_t count) {
    using T =
        std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>>;
    const auto bytes = checked_mul(count, sizeof(T));
    if (source_offset > owner.layout.bytes || bytes > owner.layout.bytes - source_offset)
      throw std::logic_error("RHF metadata view exceeds its admitted arena");
    compact_bytes =
        checked_add(compact_bytes, (alignof(T) - compact_bytes % alignof(T)) % alignof(T));
    const auto offset = compact_bytes;
    compact_bytes = checked_add(compact_bytes, bytes);
    copy_bytes = checked_add(copy_bytes, bytes);
    return std::pair{offset, bytes};
  };
  source_fields(owner, batch, [&](auto*& pointer, auto offset, auto count) {
    (void)reserve(pointer, offset, count);
  });
  const auto allocation_bytes = compact_bytes;
  // The moved system and provider control headers are included conservatively;
  // no public one-electron matrix, J/K scratch or AO^4 tensor is allocated.
  const auto retained = checked_add(
      allocation_bytes, checked_add(posthf::source_capacity(orbital),
                                    sizeof(RhfInteractionSource) + sizeof(CudaDirectJkPlan) + 128));
  result.required_peak_bytes = checked_add(reference_peak_bytes, retained);
  if (result.required_peak_bytes > numeric_budget) {
    result.resource_fallback = true;
  } else {
    result.numeric_peak_bytes = result.required_peak_bytes;
    try {
      auto plan = std::make_unique<CudaDirectJkPlan>();
      plan->device_id = owner.resources.device_id_;
      plan->batch = batch;
      plan->diagnostic.batch_size = 1;
      plan->diagnostic.nbf = owner.nbf;
      plan->device_bytes = allocation_bytes;
      plan->allocations.resize(1, nullptr);
      cuda_check(cudaSetDevice(plan->device_id));
      cuda_check(cudaStreamCreateWithFlags(&plan->stream, cudaStreamNonBlocking));
      cuda_check(runtime::resource_cuda_malloc(&plan->allocations[0], allocation_bytes));
      // Physical-reference publication already fenced the producing RHF stream.
      // Fence this compaction stream before the adapter releases the old arena.
      compact_bytes = copy_bytes = 0;
      source_fields(owner, plan->batch, [&](auto*& pointer, auto source_offset, auto count) {
        const auto [offset, bytes] = reserve(pointer, source_offset, count);
        auto* destination = static_cast<unsigned char*>(plan->allocations[0]) + offset;
        pointer = reinterpret_cast<std::remove_reference_t<decltype(pointer)>>(destination);
        if (bytes)
          cuda_check(cudaMemcpyAsync(
              destination,
              static_cast<const unsigned char*>(owner.resources.arena_) + source_offset, bytes,
              cudaMemcpyDeviceToDevice, plan->stream));
      });
      cuda_check(cudaStreamSynchronize(plan->stream));
      result.device_copy_bytes = copy_bytes;
      result.source =
          std::make_shared<RhfInteractionSource>(std::move(orbital), std::move(plan), retained);
      result.retained_numeric_bytes = retained;
      result.numeric_peak_bytes = result.required_peak_bytes;
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      const auto error = cudaGetLastError();
      if (error != cudaSuccess && error != cudaErrorMemoryAllocation) cuda_check(error);
      result.resource_fallback = true;
    } catch (const std::bad_alloc&) {
      result.resource_fallback = true;
    }
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}
}  // namespace generativeqc::scf
