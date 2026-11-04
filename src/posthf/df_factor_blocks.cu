#include <algorithm>
#include <mutex>

#include "posthf/capacity.hpp"
#include "posthf/df_ao_source.hpp"
#include "posthf/df_factor_blocks.hpp"
#include "runtime/cuda_resources.cuh"
#include "runtime/df_progress_trace.hpp"

namespace generativeqc::posthf {
namespace {
__global__ void gather_pairs(const double* factors, const std::size_t* first,
                             const std::size_t* second, std::size_t rows, std::size_t columns,
                             std::size_t nbf, std::size_t naux, double* output) {
  const auto count = rows * columns * naux;
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto pair = index / naux, auxiliary = index % naux;
    const auto left = first[pair / columns], right = second[pair % columns];
    output[index] = left == padded_mo || right == padded_mo
                        ? 0.0
                        : factors[(left * nbf + right) * naux + auxiliary];
  }
}

/** The immutable host factors outlive this owner. A refused optional resident
 * allocation retires only that allocation; panel-upload execution remains CUDA. */
class FactorBlocks final : public DFFactorCudaBlocks {
 public:
  FactorBlocks(std::size_t nbf, std::size_t naux, std::array<std::span<const double>, 2> factors,
               int device, std::size_t resident_budget)
      : nbf_(nbf), naux_(naux), factors_(factors), device_(device), stream_(device) {
    const auto values = checked_mul(2, checked_mul(checked_mul(nbf, nbf), naux));
    if (checked_mul(values, sizeof(double)) <= resident_budget) {
      try {
        resident_.allocate(device, values, stream_.get());
      } catch (const std::bad_alloc&) {
        resident_.reset();
      }
      if (resident_) {
        runtime::CudaDeviceScope scope(device_);
        for (unsigned spin = 0; spin < 2; ++spin)
          runtime::cuda_resource_check(
              cudaMemcpyAsync(resident_.get() + spin * factors_[spin].size(), factors_[spin].data(),
                              factors_[spin].size_bytes(), cudaMemcpyHostToDevice, stream_.get()));
        stream_.synchronize();
      }
    }
    runtime::df_progress::number("device_factor_cache_bytes", retained_numeric_bytes());
  }

  std::size_t retained_numeric_bytes() const noexcept override {
    return resident_ ? 2 * factors_[0].size_bytes() : 0;
  }

  std::vector<double> get(const SpinMOSlots& request, ProviderWork* work) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    runtime::CudaDeviceScope scope(device_);
    std::array<std::size_t, 4> shape{}, offsets{};
    std::size_t selection_count = 0;
    for (unsigned axis = 0; axis < 4; ++axis) {
      offsets[axis] = selection_count;
      shape[axis] = request.slots[axis].size();
      selection_count = checked_add(selection_count, shape[axis]);
    }
    // Allocate the admitted selection payload once, without geometric growth.
    std::vector<std::size_t> indices(selection_count);
    for (unsigned axis = 0; axis < 4; ++axis)
      std::copy(request.slots[axis].begin(), request.slots[axis].end(),
                indices.begin() + offsets[axis]);
    const auto left = checked_mul(shape[0], shape[1]), right = checked_mul(shape[2], shape[3]);
    const auto panel_values = checked_mul(checked_add(left, right), naux_);
    const auto output_values = checked_mul(left, right);
    runtime::OwnedCudaBuffer<double> storage(device_, checked_add(panel_values, output_values),
                                             stream_.get());
    runtime::OwnedCudaBuffer<int> error(device_, 1, stream_.get());
    runtime::OwnedCudaBuffer<std::size_t> selected;
    std::array<std::vector<double>, 2> panels;
    if (resident_) {
      selected.allocate(device_, indices.size(), stream_.get());
      runtime::cuda_resource_check(cudaMemcpyAsync(selected.get(), indices.data(),
                                                   indices.size() * sizeof(std::size_t),
                                                   cudaMemcpyHostToDevice, stream_.get()));
      if (work)
        work->h2d_bytes = checked_add(work->h2d_bytes, indices.size() * sizeof(std::size_t));
    }
    std::vector<double> output(output_values);
    int failed = 0;
    try {
      std::size_t start = 0;
      for (unsigned pair = 0; pair < 2; ++pair) {
        const auto first = 2 * pair, second = first + 1;
        const auto values = checked_mul(checked_mul(shape[first], shape[second]), naux_);
        if (resident_) {
          const auto blocks =
              static_cast<unsigned>(std::min<std::size_t>((values + 255) / 256, 65535));
          gather_pairs<<<blocks, 256, 0, stream_.get()>>>(
              resident_.get() + request.spins[first] * factors_[0].size(),
              selected.get() + offsets[first], selected.get() + offsets[second], shape[first],
              shape[second], nbf_, naux_, storage.get() + start);
          runtime::cuda_resource_check(cudaGetLastError());
        } else {
          auto& panel = panels[pair];
          panel.resize(values);
          std::size_t row = 0;
          for (const auto left : request.slots[first])
            for (const auto right : request.slots[second]) {
              if (left != padded_mo && right != padded_mo)
                std::copy_n(factors_[request.spins[first]].data() + (left * nbf_ + right) * naux_,
                            naux_, panel.data() + row * naux_);
              ++row;
            }
          runtime::cuda_resource_check(cudaMemcpyAsync(storage.get() + start, panel.data(),
                                                       values * sizeof(double),
                                                       cudaMemcpyHostToDevice, stream_.get()));
          if (work) work->h2d_bytes = checked_add(work->h2d_bytes, values * sizeof(double));
        }
        start += values;
      }
      evaluate_df_factor_block_device(left, right, naux_, storage.get(),
                                      storage.get() + left * naux_, storage.get() + panel_values,
                                      error.get(), stream_.get());
      runtime::cuda_resource_check(cudaMemcpyAsync(output.data(), storage.get() + panel_values,
                                                   output.size() * sizeof(double),
                                                   cudaMemcpyDeviceToHost, stream_.get()));
      runtime::cuda_resource_check(cudaMemcpyAsync(&failed, error.get(), sizeof(int),
                                                   cudaMemcpyDeviceToHost, stream_.get()));
      stream_.synchronize();
    } catch (...) {
      stream_.synchronize();
      throw;
    }
    if (failed) throw std::runtime_error("nonfinite CUDA DF factor block");
    if (work) {
      ++work->cuda_transform_calls;
      work->d2h_bytes = checked_add(work->d2h_bytes, output.size() * sizeof(double) + sizeof(int));
    }
    return output;
  }

 private:
  std::size_t nbf_, naux_;
  std::array<std::span<const double>, 2> factors_;
  int device_;
  runtime::OwnedCudaStream stream_;
  runtime::OwnedCudaBuffer<double> resident_;
  mutable std::mutex mutex_;
};
}  // namespace

std::unique_ptr<DFFactorCudaBlocks> make_df_factor_cuda_blocks(
    std::size_t nbf, std::size_t naux, std::array<std::span<const double>, 2> factors, int device,
    std::size_t resident_budget) {
  const auto count = checked_mul(checked_mul(nbf, nbf), naux);
  if (!nbf || !naux || device < 0 || factors[0].size() != count || factors[1].size() != count)
    throw std::invalid_argument("invalid CUDA DF factor cache dimensions");
  return std::make_unique<FactorBlocks>(nbf, naux, factors, device, resident_budget);
}
}  // namespace generativeqc::posthf
