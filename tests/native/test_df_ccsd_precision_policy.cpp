#include <iostream>
#include <limits>
#include <stdexcept>

#include "cc/solver.hpp"

int main() {
  try {
    const auto require = [](bool condition, const char* message) {
      if (!condition) throw std::runtime_error(message);
    };
    using generativeqc::cc::default_df_ccsd_mixed_precision;
    require(!default_df_ccsd_mixed_precision(5, 19), "small water must retain FP64");
    require(!default_df_ccsd_mixed_precision(9, 49), "small ethane must retain FP64");
    require(!default_df_ccsd_mixed_precision(5, 87), "medium water must retain FP64");
    require(!default_df_ccsd_mixed_precision(1, 221), "low occupied work must retain FP64");
    require(!default_df_ccsd_mixed_precision(9, 127), "auto threshold lost its lower boundary");
    require(default_df_ccsd_mixed_precision(9, 128), "auto threshold lost its upper boundary");
    require(default_df_ccsd_mixed_precision(9, 135), "medium ethane must request warmup");
    require(default_df_ccsd_mixed_precision(9, 221), "large ethane must request warmup");
    require(default_df_ccsd_mixed_precision(std::numeric_limits<std::size_t>::max(),
                                            std::numeric_limits<std::size_t>::max()),
            "shape selection must not overflow before bounded CUDA admission");
    std::cout << "validated native DF-CCSD default precision selection\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
