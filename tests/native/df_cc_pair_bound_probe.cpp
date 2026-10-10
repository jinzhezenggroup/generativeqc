#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "generated_df_ccsd_spectator_pairs_cpu.hpp"

namespace bound = generativeqc::cc::pair_bound;
namespace folded = generativeqc::cc::generated::dfpairs;

void print(bound::Power value) {
  std::cout << static_cast<unsigned>(value.state) << ' ' << value.exponent << '\n';
}

int main() {
  std::string operation;
  while (std::cin >> operation) {
    if (operation == "magnitude") {
      std::uint64_t bits{};
      std::cin >> bits;
      print(bound::magnitude_upper(bits));
    } else if (operation == "norm") {
      std::size_t rows{}, columns{}, inspected{};
      unsigned axes{};
      std::cin >> rows >> columns >> axes;
      std::vector<double> values(rows * columns);
      for (auto& value : values) {
        std::uint64_t bits{};
        std::cin >> bits;
        value = std::bit_cast<double>(bits);
      }
      print(bound::row_l1_upper(values.data(), rows, columns, axes, inspected));
      std::cout << inspected << '\n';
    } else if (operation == "unsafe") {
      std::size_t inspected = 0;
      print(bound::row_l1_upper(nullptr, std::numeric_limits<std::size_t>::max(), 2, 1, inspected));
      print(bound::product(
          {{bound::Power::State::bounded, std::numeric_limits<int>::max()}, bound::Power::of(1)}));
      print(bound::product({bound::Power::zero(), bound::Power::refused()}));
      print(bound::sum({bound::Power::zero(), bound::Power::refused()}));
    } else if (operation == "margin") {
      int exponent{};
      std::uint64_t bits{};
      std::cin >> exponent >> bits;
      std::cout << bound::within_residual_margin(bound::Power::of(exponent),
                                                 std::bit_cast<double>(bits))
                << '\n';
    } else if (operation == "factorization-range") {
      int bov{}, bvv{};
      std::size_t occupied{}, virtuals{};
      bound::ProjectionMaxima maxima;
      std::cin >> bov >> bvv >> occupied >> virtuals >> maxima.t1_magnitude_bits >>
          maxima.tau_magnitude_bits >> maxima.refused;
      std::cout << bound::within_factorization_range(maxima, bound::Power::of(bov),
                                                     bound::Power::of(bvv), occupied, virtuals)
                << '\n';
    } else if (operation == "coefficient") {
      std::array<bound::Power, folded::geometry_norms.size()> norms;
      for (auto& value : norms) {
        int exponent{};
        std::cin >> exponent;
        value = bound::Power::of(exponent);
      }
      for (auto value : folded::ladder_coefficient_bounds(norms)) print(value);
      print(folded::ladder_residual_gain);
    } else if (operation == "aggregate") {
      int first{}, second{}, delta{}, amplitude{};
      std::size_t auxiliary{};
      std::cin >> first >> second >> delta >> amplitude >> auxiliary;
      print(bound::product(
          {bound::Power::of(delta), bound::extent_upper(auxiliary),
           bound::sum({bound::Power::of(first),
                       bound::product({bound::Power::of(second), bound::Power::of(amplitude)})})}));
    } else {
      return 2;
    }
    if (!std::cin) return 3;
  }
}
