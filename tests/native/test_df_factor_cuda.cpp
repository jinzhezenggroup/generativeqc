#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "posthf/df_factor_blocks.hpp"

int main() {
  const auto* enabled = std::getenv("GENERATIVEQC_OPEN_SHELL_CUDA_TEST");
  if (!enabled || std::string(enabled) != "1") return 77;
  try {
    if (!std::getenv("SLURM_JOB_ID"))
      throw std::runtime_error("real GPU factor contracts require srun");
    constexpr std::size_t nbf = 4, naux = 3;
    std::array<std::vector<double>, 2> factors;
    for (unsigned spin = 0; spin < 2; ++spin) {
      factors[spin].resize(nbf * nbf * naux);
      for (std::size_t index = 0; index < factors[spin].size(); ++index)
        factors[spin][index] = std::sin(double(index + 5 * spin)) * 0.2;
    }
    for (const bool zero_beta : {false, true}) {
      if (zero_beta) std::fill(factors[1].begin(), factors[1].end(), 0);
      auto resident = generativeqc::posthf::make_df_factor_cuda_blocks(
          nbf, naux, {factors[0], factors[1]}, 0, 1 << 20);
      auto bounded = generativeqc::posthf::make_df_factor_cuda_blocks(
          nbf, naux, {factors[0], factors[1]}, 0, 0);
      if (!resident->retained_numeric_bytes() || bounded->retained_numeric_bytes())
        throw std::runtime_error("optional factor residency did not honor its capacity");
      for (const auto spins :
           {std::array<unsigned, 4>{0, 0, 0, 0}, std::array<unsigned, 4>{1, 1, 1, 1},
            std::array<unsigned, 4>{0, 0, 1, 1}}) {
        const generativeqc::posthf::SpinMOSlots request{
            {{{2, generativeqc::posthf::padded_mo, 0}, {1}, {0, 3}, {2}}}, spins};
        const auto actual = resident->get(request, nullptr);
        const auto fallback = bounded->get(request, nullptr);
        std::size_t position = 0;
        for (const auto left : request.slots[0])
          for (const auto right : request.slots[2]) {
            double expected = 0;
            if (left != generativeqc::posthf::padded_mo)
              for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary)
                expected += factors[spins[0]][(left * nbf + 1) * naux + auxiliary] *
                            factors[spins[2]][(right * nbf + 2) * naux + auxiliary];
            if (std::abs(actual[position] - expected) > 2e-14 ||
                std::abs(fallback[position] - expected) > 2e-14)
              throw std::runtime_error("resident/bounded CUDA DF Gram disagrees with oracle");
            ++position;
          }
      }
    }
    std::cout << "CUDA three-index factor contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
