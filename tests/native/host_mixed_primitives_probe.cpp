#include <array>
#include <cmath>

#include "integrals/host_mixed_primitives.hpp"

extern "C" bool host_mixed_st(const double* exponents, const double* centers,
                              const unsigned* powers, const double* left, const double* right,
                              double* output) {
  return generativeqc::integrals::host_mixed::overlap_kinetic(exponents, centers, powers, left,
                                                              right, output);
}

extern "C" bool host_mixed_v(const double* exponents, const double* centers, const unsigned* powers,
                             const double* left, const double* right, double* output) {
  return generativeqc::integrals::host_mixed::attraction(exponents, centers, powers, left, right,
                                                         output);
}

#ifdef GENERATIVEQC_HOST_MIXED_STANDALONE
int main() {
  const double exponents[]{.6, .8};
  const double centers[]{.125, -.25, .375, -.5, .25, .625, .75, -.125, -.25};
  const unsigned powers[]{2, 0, 1, 1, 1, 1};
  const double left[]{.25, -.125, .375, -.25, .5, .125, .125, .25, -.5};
  const double right[]{-.375, .25, .125, .5, -.25, .375, -.125, .5, .25};
  const double translation[]{.25, -.5, .125, .25, -.5, .125, .25, -.5, .125};
  std::array<double, 8> st{}, reversed{};
  std::array<double, 4> attraction{};
  if (!host_mixed_st(exponents, centers, powers, left, right, st.data()) ||
      !host_mixed_st(exponents, centers, powers, right, left, reversed.data()) ||
      !host_mixed_v(exponents, centers, powers, left, right, attraction.data()))
    return 1;
  if (std::abs(st[3]) < 1e-8 || std::abs(st[7]) < 1e-8 || std::abs(attraction[3]) < 1e-8 ||
      std::abs(st[3] - reversed[3]) > 1e-12 || std::abs(st[7] - reversed[7]) > 1e-12)
    return 2;
  if (!host_mixed_st(exponents, centers, powers, translation, right, st.data()) ||
      !host_mixed_v(exponents, centers, powers, translation, right, attraction.data()))
    return 3;
  if (st[1] != 0 || st[3] != 0 || st[5] != 0 || st[7] != 0 || attraction[1] != 0 ||
      attraction[3] != 0)
    return 4;
  return 0;
}
#endif
