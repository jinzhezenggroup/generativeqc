// Compare actual native schedules without decimal conversion of IEEE values.
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "generated_df_ccsd_cpu.hpp"
#include "tensor/cuda_runtime.cuh"

using generativeqc::cc::generated::df::checked_add;
using generativeqc::cc::generated::df::checked_product;

struct State {
  std::size_t o{1}, v{16};
  const double* source{};
  const double* ones{};
  double* arena{};
  int* error{};
  cudaStream_t stream{};
};

struct Outputs {
  const double* result;
};

#include "generated_native_copy_roundtrips.cuh"

int main() {
  try {
    constexpr std::size_t count = 16, arena_count = 2 * count + 2;
    constexpr std::uint64_t sentinel = 0x4031000000000000ULL;
    const std::array<std::uint64_t, count> finite{0,
                                                  0x8000000000000000ULL,
                                                  1,
                                                  0x8000000000000001ULL,
                                                  0x0010000000000000ULL,
                                                  0x8010000000000000ULL,
                                                  0x7fefffffffffffffULL,
                                                  0xffefffffffffffffULL,
                                                  0x3ff0000000000000ULL,
                                                  0xbff0000000000000ULL,
                                                  0x000fffffffffffffULL,
                                                  0x800fffffffffffffULL,
                                                  0x4000000000000000ULL,
                                                  0xc000000000000000ULL,
                                                  0x3fd5555555555555ULL,
                                                  0xbfd5555555555555ULL};
    std::array<double, count> ones;
    ones.fill(1.0);
    double *source{}, *unit{}, *arena{};
    int* error{};
    using generativeqc_tensor::cuda_check;
    cuda_check(cudaMalloc(&source, sizeof(finite)));
    cuda_check(cudaMalloc(&unit, sizeof(ones)));
    cuda_check(cudaMalloc(&arena, arena_count * sizeof(double)));
    cuda_check(cudaMalloc(&error, sizeof(int)));
    cuda_check(cudaMemcpy(unit, ones.data(), sizeof(ones), cudaMemcpyHostToDevice));
    State state{1, count, source, unit, arena + 1, error, nullptr};
    const std::array runners{run_original_2, run_selected_2, run_original_5, run_selected_5};
    for (bool nonfinite : {false, true}) {
      auto inputs = finite;
      if (nonfinite) {
        inputs[12] = 0x7ff0000000000000ULL;
        inputs[13] = 0xfff0000000000000ULL;
        inputs[14] = 0x7ff8000000000017ULL;
        inputs[15] = 0x7ff0000000000017ULL;
      }
      cuda_check(cudaMemcpy(source, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice));
      for (int seed : {0, 173}) {
        std::array<std::uint64_t, count> reference{};
        for (std::size_t runner = 0; runner < runners.size(); ++runner) {
          std::array<std::uint64_t, arena_count> storage;
          storage.fill(sentinel);
          cuda_check(cudaMemcpy(arena, storage.data(), sizeof(storage), cudaMemcpyHostToDevice));
          cuda_check(cudaMemcpy(error, &seed, sizeof(seed), cudaMemcpyHostToDevice));
          const auto output = runners[runner](state);
          std::array<std::uint64_t, count> actual;
          int observed{};
          cuda_check(
              cudaMemcpy(actual.data(), output.result, sizeof(actual), cudaMemcpyDeviceToHost));
          cuda_check(cudaMemcpy(storage.data(), arena, sizeof(storage), cudaMemcpyDeviceToHost));
          cuda_check(cudaMemcpy(&observed, error, sizeof(observed), cudaMemcpyDeviceToHost));
          if (storage.front() != sentinel || storage.back() != sentinel)
            throw std::runtime_error("copy elision crossed an arena canary");
          // The scalar producer has index two and must retain the first fault.
          if (observed != (seed ? seed : nonfinite ? 3 : 0))
            throw std::runtime_error("copy elision changed the sticky first error");
          if (runner == 0) reference = actual;
          if (actual != reference) throw std::runtime_error("copy elision changed output bits");
          if (!nonfinite && actual != finite)
            throw std::runtime_error("native unit copies changed finite input bits");
        }
      }
    }
    cuda_check(cudaFree(error));
    cuda_check(cudaFree(arena));
    cuda_check(cudaFree(unit));
    cuda_check(cudaFree(source));
    std::cout << "bitwise, canaries, and first-error gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
