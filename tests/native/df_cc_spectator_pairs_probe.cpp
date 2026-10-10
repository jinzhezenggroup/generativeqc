// Qualification only: supplied pair coordinates are admitted by the Python
// test oracle, not by a production solver or an implicit symmetry projection.
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "generated_df_ccsd_spectator_pairs_cpu.hpp"
#if defined(DF_PROBE_CUDA)
#include "generated_df_ccsd_spectator_pairs_cuda.cuh"
#include "tensor/cuda_error.hpp"
#endif

namespace folded = generativeqc::cc::generated::dfpairs;

std::size_t ladder_offset(std::size_t flat, std::size_t nocc, std::size_t nvir) {
  auto second_virtual = flat % nvir;
  flat /= nvir;
  auto first_virtual = flat % nvir;
  flat /= nvir;
  auto second_occupied = flat % nocc;
  auto first_occupied = flat / nocc;
  if (first_occupied > second_occupied) {
    std::swap(first_occupied, second_occupied);
    std::swap(first_virtual, second_virtual);
  }
  const auto pair =
      first_occupied * (2 * nocc - first_occupied + 1) / 2 + second_occupied - first_occupied;
  return (pair * nvir + first_virtual) * nvir + second_virtual;
}

int main() {
  try {
    std::size_t nocc{}, nvir{}, naux{}, batch_limit{};
    int seed{};
    if (!(std::cin >> nocc >> nvir >> naux >> batch_limit >> seed) || !nocc || !nvir || !naux ||
        !batch_limit || batch_limit > naux)
      throw std::invalid_argument("invalid spectator-pair probe dimensions");
    const auto singles = folded::checked_mul(nocc, nvir);
    const auto doubles = folded::checked_mul(singles, singles);
    const auto virtual_matrix = folded::checked_mul(nvir, nvir);
    const auto pairs = folded::checked_mul(nocc, folded::checked_add(nocc, 1)) / 2;
    const auto paired_doubles = folded::checked_mul(pairs, virtual_matrix);
    const std::array<std::size_t, 6> input_sizes{singles,
                                                 doubles,
                                                 doubles,
                                                 paired_doubles,
                                                 folded::checked_mul(naux, singles),
                                                 folded::checked_mul(naux, virtual_matrix)};
    std::array<std::vector<double>, 6> inputs;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      inputs[index].resize(input_sizes[index]);
      for (auto& value : inputs[index])
        if (!(std::cin >> value)) throw std::invalid_argument("truncated pair probe input");
    }
    const std::array<std::size_t, 6> output_sizes{virtual_matrix, doubles, doubles,
                                                  doubles,        doubles, singles};
    std::array<std::vector<double>, 6> outputs;
    for (std::size_t index = 0; index < outputs.size(); ++index)
      outputs[index].resize(output_sizes[index]);
    std::size_t work = 0;
    int observed = 0;
    constexpr double sentinel = 913.25;
#if defined(DF_PROBE_CUDA)
    using generativeqc_tensor::cuda_check;
    struct DeviceArrays {
      std::vector<void*> allocations;
      ~DeviceArrays() {
        for (auto* pointer : allocations) cudaFree(pointer);
      }
      void* allocate(std::size_t bytes) {
        void* pointer{};
        generativeqc_tensor::cuda_check(cudaMalloc(&pointer, bytes));
        allocations.push_back(pointer);
        return pointer;
      }
    } device;
    generativeqc::tensor::CudaContractionContext context;
    folded::CudaState state;
    state.o = nocc;
    state.v = nvir;
    std::array<double*, 6> sources{}, targets{};
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      sources[index] = static_cast<double*>(device.allocate(inputs[index].size() * sizeof(double)));
      cuda_check(cudaMemcpy(sources[index], inputs[index].data(),
                            inputs[index].size() * sizeof(double), cudaMemcpyHostToDevice));
      targets[index] =
          static_cast<double*>(device.allocate(outputs[index].size() * sizeof(double)));
      cuda_check(cudaMemset(targets[index], 0, outputs[index].size() * sizeof(double)));
    }
    state.t1 = sources[0];
    state.t2 = sources[1];
    state.df_tau = sources[2];
    state.df_tau_occupied_pairs = sources[3];
    state.error = static_cast<int*>(device.allocate(sizeof(int)));
    cuda_check(cudaMemcpy(state.error, &seed, sizeof(int), cudaMemcpyHostToDevice));
    const auto required =
        std::max(folded::auxiliary_packed_arena_elements(nocc, nvir),
                 folded::auxiliary_batched_arena_elements(nocc, nvir, batch_limit));
    std::vector<double> canary(required + 2, sentinel);
    auto* allocation = static_cast<double*>(device.allocate(canary.size() * sizeof(double)));
    cuda_check(cudaMemcpy(allocation, canary.data(), canary.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    state.auxiliary_arena = allocation + 1;
    if (!context.prepare(nullptr)) throw std::runtime_error("matrix provider was not admitted");
    std::size_t calls = 0, summands = 0;
    folded::prepare_contractions(state, context, batch_limit, naux % batch_limit, calls, summands);
    for (std::size_t auxiliary = 0; auxiliary < naux;) {
      const auto batch = std::min(batch_limit, naux - auxiliary);
      state.q = batch;
      state.bov = sources[4] + auxiliary * singles;
      state.bvv = sources[5] + auxiliary * virtual_matrix;
      const auto row = folded::run_auxiliary_cuda(state);
      folded::accumulate_occupied_auxiliary_cuda(state, row, targets[0], targets[1], targets[2],
                                                 targets[3], targets[4], targets[5]);
      work = folded::checked_add(
          work, batch > 1 ? folded::auxiliary_batched_contraction_terms(nocc, nvir, batch)
                          : folded::auxiliary_packed_contraction_terms(nocc, nvir));
      auxiliary += batch;
    }
    for (std::size_t index = 0; index < outputs.size(); ++index)
      cuda_check(cudaMemcpy(outputs[index].data(), targets[index],
                            outputs[index].size() * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&observed, state.error, sizeof(int), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(canary.data(), allocation, canary.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    if (canary.front() != sentinel || canary.back() != sentinel)
      throw std::runtime_error("folded CUDA arena crossed a canary");
    if (observed != seed) throw std::runtime_error("folded action changed sticky arithmetic state");
#else
    folded::Inputs state;
    state.t1 = inputs[0].data();
    state.t2 = inputs[1].data();
    state.df_tau = inputs[2].data();
    state.df_tau_occupied_pairs = inputs[3].data();
    const auto required =
        std::max(folded::auxiliary_arena_elements(nocc, nvir),
                 folded::auxiliary_batched_arena_elements(nocc, nvir, batch_limit));
    std::vector<double> arena(required + 2, sentinel);
    for (std::size_t auxiliary = 0; auxiliary < naux;) {
      const auto batch = std::min(batch_limit, naux - auxiliary);
      state.bov = inputs[4].data() + auxiliary * singles;
      state.bvv = inputs[5].data() + auxiliary * virtual_matrix;
      const auto row =
          batch > 1 ? folded::run_auxiliary_batched_cpu(nocc, nvir, batch, state, arena.data() + 1,
                                                        required)
                    : folded::run_auxiliary_cpu(nocc, nvir, state, arena.data() + 1, required);
      const std::array<const double*, 6> values{row.lvv, row.wvoov,  row.wvovo,
                                                row.xv,  row.ladder, row.singles};
      for (std::size_t field = 0; field < values.size(); ++field) {
        const auto stride = field == 4 ? paired_doubles : output_sizes[field];
        for (std::size_t flat = 0; flat < output_sizes[field]; ++flat) {
          const auto source = field == 4 ? ladder_offset(flat, nocc, nvir) : flat;
          for (std::size_t row_index = 0; row_index < batch; ++row_index)
            outputs[field][flat] += values[field][row_index * stride + source];
        }
      }
      work = folded::checked_add(
          work, batch > 1 ? folded::auxiliary_batched_contraction_terms(nocc, nvir, batch)
                          : folded::auxiliary_contraction_terms(nocc, nvir));
      auxiliary += batch;
    }
    if (arena.front() != sentinel || arena.back() != sentinel)
      throw std::runtime_error("folded CPU arena crossed a canary");
#endif
    std::cout << "work " << work << '\n' << "status " << observed << '\n' << std::setprecision(17);
    for (const auto& field : outputs) {
      for (double value : field) std::cout << value << ' ';
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
