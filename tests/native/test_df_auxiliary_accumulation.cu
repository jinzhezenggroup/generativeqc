// Adversarial test of the generated primal Q consumer, independent of CC
// equations and providers. Tile subtotals would fail the cancellation case.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_df_ccsd_hoisted_cuda.cuh"

int main() {
  using generativeqc_tensor::cuda_check;
  namespace generated = generativeqc::cc::generated::dfhoist;
  try {
    constexpr std::size_t o = 2, v = 3, q = 5, stride = 36;
    const std::array<std::size_t, 6> sizes{v * v, stride, stride, stride, stride, o * v};
    double *source = nullptr, *target = nullptr;
    generated::CudaState state;
    state.o = o;
    state.v = v;
    cuda_check(cudaStreamCreate(&state.stream));
    generativeqc::tensor::CudaContractionContext context;
    if (!context.prepare(state.stream)) throw std::runtime_error("provider preparation failed");
    std::size_t calls = 0, summands = 0;
    // Mark the admitted matrix path without providing an executable request:
    // this ordered-consumer test must never invoke a contraction.
    state.auxiliary_contractions.add(o, v, 1, {}, context, calls, summands);
    cuda_check(cudaMalloc(&source, 6 * q * stride * sizeof(double)));
    cuda_check(cudaMalloc(&target, 6 * stride * sizeof(double)));
    cuda_check(cudaMalloc(&state.error, sizeof(int)));
    const std::array<double, q> contributions{-1e16, 1, 2, 3, 4};
    for (bool overflow : {false, true}) {
      for (std::size_t tile : {1, 2, 3, 5}) {
        std::vector<double> input(6 * q * stride, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> output(6 * stride, -17.25);
        const double initial = overflow ? std::numeric_limits<double>::max() : 1e16;
        volatile double expected = initial;
        for (const auto value : contributions) expected = expected + value;
        for (std::size_t field = 0; field < sizes.size(); ++field) {
          std::fill_n(output.data() + field * stride, sizes[field], initial);
          for (std::size_t Q = 0; Q < q; ++Q) {
            const auto value = overflow ? (Q ? -std::numeric_limits<double>::max()
                                             : std::numeric_limits<double>::max())
                                        : contributions[Q];
            std::fill_n(input.data() + field * q * stride + Q * sizes[field], sizes[field], value);
          }
        }
        cuda_check(cudaMemcpyAsync(source, input.data(), input.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, state.stream));
        cuda_check(cudaMemcpyAsync(target, output.data(), output.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, state.stream));
        cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), state.stream));
        for (std::size_t Q = 0; Q < q; Q += tile) {
          state.q = std::min(tile, q - Q);
          const auto at = [&](std::size_t field) {
            return source + field * q * stride + Q * sizes[field];
          };
          const generated::AuxiliaryOutputs values{at(0), at(1), at(2), at(3), at(4), at(5)};
          generated::accumulate_auxiliary_cuda(state, values, target, target + stride,
                                               target + 2 * stride, target + 3 * stride,
                                               target + 4 * stride, target + 5 * stride);
        }
        int error = 0;
        cuda_check(cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost,
                                   state.stream));
        cuda_check(cudaMemcpyAsync(output.data(), target, output.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, state.stream));
        cuda_check(cudaStreamSynchronize(state.stream));
        if (bool(error) != overflow) throw std::runtime_error("lost/stale Q accumulation error");
        for (std::size_t field = 0; field < sizes.size(); ++field)
          for (std::size_t x = 0; x < stride; ++x) {
            const auto value = output[field * stride + x];
            if (x >= sizes[field] && value != -17.25)
              throw std::runtime_error("Q consumer overwrote another output span");
            if (!overflow && x < sizes[field] && value != expected)
              throw std::runtime_error("Q consumer changed serial FP64 accumulation order");
          }
      }
    }
    cuda_check(cudaFree(state.error));
    cuda_check(cudaFree(target));
    cuda_check(cudaFree(source));
    if (calls || summands) throw std::runtime_error("accumulation executed a contraction");
    context.reset();
    cuda_check(cudaStreamDestroy(state.stream));
    std::cout << "ordered Q accumulation: 8 cancellation/overflow/tail cases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
