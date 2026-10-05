#include <iostream>
#include <stdexcept>

#include "hf/rhf_frame_response.hpp"

/** Resource selection must remain CPU-testable without initializing CUDA or
 * evaluating the physical response. Explicit ceilings override only the
 * crossover; they never override the caller's remaining complete budget. */
int main() {
  try {
    const auto require = [](bool condition, const char* reason) {
      if (!condition) throw std::runtime_error(reason);
    };
    generativeqc::hf::RHFFrameResponseOptions options;
    constexpr std::size_t ceiling = 8ULL << 30;
    require(!options.resident_jk_maximum_bytes, "default must select the conservative policy");
    for (const auto dimension : {7U, 58U, 63U})
      require(options.resident_jk_allowance(dimension, ceiling) == 0,
              "small frames must retain ordinary recomputation");
    for (const auto dimension : {64U, 230U, 400U}) {
      require(options.resident_jk_allowance(dimension, ceiling * 2) == ceiling,
              "automatic allowance exceeded the ceiling");
      require(options.resident_jk_allowance(dimension, ceiling - 1) == ceiling - 1,
              "automatic allowance exceeded the endpoint budget");
      require(options.resident_jk_allowance(dimension, 0) == 0,
              "empty remaining budget must not admit a lease");
    }
    options.relax_orbitals = false;
    require(options.resident_jk_allowance(230, ceiling) == 0,
            "fixed-frame requests must retain recomputation by default");
    options.relax_orbitals = true;
    options.orbital_screening_tolerance = 1e-12;
    require(options.resident_jk_allowance(230, ceiling) == 0,
            "fixed-mask experiments must retain recomputation by default");
    options.resident_jk_maximum_bytes = ceiling;
    require(options.resident_jk_allowance(7, ceiling / 2) == ceiling / 2,
            "explicit opt-in must respect the remaining endpoint budget");
    options.resident_jk_maximum_bytes = 0;
    require(options.resident_jk_allowance(230, ceiling) == 0,
            "explicit zero must disable resident preparation");
    std::cout << "RHF resident resource policy passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
