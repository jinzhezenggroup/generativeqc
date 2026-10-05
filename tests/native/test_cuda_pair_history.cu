// Independent full-layout oracle for the actual compressed history consumers.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "generated_rccsd_cpu.hpp"
#include "tensor/cuda_history.cuh"

namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
template <class T>
struct Device {
  T* p{};
  explicit Device(std::size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Device() { cudaFree(p); }
  void put(const std::vector<T>& v) {
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get(std::size_t n) {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
void run(std::size_t o, std::size_t v) {
  using namespace generativeqc_tensor;
  const auto n = o * v, n2 = n * n, full = n + n2, packed = n + n * (n + 1) / 2;
  const generativeqc::cc::generated::RestrictedPairCoordinates map{o, v};
  std::vector<double> x(2 * full), r(2 * full);
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t i = 0; i < n; ++i) {
      x[row * full + i] = std::sin(double(i + row + 1));
      r[row * full + i] = std::cos(double(i + 2 * row + 1));
    }
    // Fill from independent ijab loops, without using the generated map to
    // construct the full-layout numerical oracle.
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const auto p = i * v + a, q = j * v + b, k = ((i * o + j) * v + a) * v + b;
            const double label = double(std::min(p, q) + 3 * std::max(p, q));
            x[row * full + n + k] = std::sin(label + row + 1);
            r[row * full + n + k] = std::cos(label + 2 * row + 1);
          }
  }
  // Singleton subnormal values must survive bitwise (no half+half underflow).
  x[n] = std::numeric_limits<double>::denorm_min();
  Device<double> dx(x.size()), dr(r.size()), vectors(2 * packed), errors(2 * packed), gram(4),
      asymmetry(1), out(n2), coefficients(2);
  Device<unsigned char> weights(packed);
  Device<int> refused(1), arithmetic(1), status(1);
  dx.put(x);
  dr.put(r);
  asymmetry.put({0.});
  refused.put({0});
  arithmetic.put({0});
  status.put({0});
  coefficients.put({.25, -.5});
  for (std::size_t row = 0; row < 2; ++row) {
    history_insert_orbits<<<blocks(full, 256), 256>>>(
        dx.p + row * full, dx.p + row * full + n, dr.p + row * full, dr.p + row * full + n, I(n),
        I(n2), vectors.p + row * packed, errors.p + row * packed, weights.p, map,
        generativeqc::cc::generated::pair_rounding_tolerance, refused.p, asymmetry.p, arithmetic.p);
    history_gram_row<<<row + 1, 256>>>(errors.p, I(packed), 2, 0, int(row + 1), int(row), gram.p,
                                       weights.p);
  }
  check(cudaDeviceSynchronize());
  require(refused.get(1)[0] == 0 && arithmetic.get(1)[0] == 0, "symmetric storage refused");
  const auto saved = vectors.get(2 * packed), metric = gram.get(4);
  const auto metric_weights = weights.get(packed);
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t k = 0; k < n2; ++k) {
      const auto orbit = map(k);
      require(saved[row * packed + n + orbit.slot] == x[row * full + n + k],
              "packed/full roundtrip mismatch");
      require(metric_weights[n + orbit.slot] == orbit.weight, "orbit weight mismatch");
    }
  for (std::size_t a = 0; a < 2; ++a)
    for (std::size_t b = 0; b < 2; ++b) {
      long double dot = 0.;
      for (std::size_t k = 0; k < full; ++k)
        dot += static_cast<long double>(r[a * full + k]) * r[b * full + k];
      require(std::abs(metric[a * 2 + b] - dot) < 2e-13 * (1 + std::abs(dot)),
              "weighted full-metric oracle mismatch");
    }
  diis_combine_orbits<<<blocks(n2, 256), 256>>>(vectors.p, coefficients.p, I(packed), I(n), I(n2),
                                                2, status.p, out.p, arithmetic.p, 2, 0, map);
  const auto actual = out.get(n2);
  for (std::size_t k = 0; k < n2; ++k)
    require(std::abs(actual[k] - (.25 * x[n + k] - .5 * x[full + n + k])) < 2e-15,
            "full combined output mismatch");
  if (n > 1) {
    // Test both a small rounding orbit and an explicit large refusal.
    const std::size_t k = 1, mate = map(k).partner;
    x[n + k] = std::nextafter(x[n + mate], 2.);
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (attempt) x[n + k] = x[n + mate] + 1e-4;
      dx.put(x);
      refused.put({0});
      history_insert_orbits<<<blocks(full, 256), 256>>>(
          dx.p, dx.p + n, dr.p, dr.p + n, I(n), I(n2), vectors.p, errors.p, weights.p, map,
          generativeqc::cc::generated::pair_rounding_tolerance, refused.p, asymmetry.p,
          arithmetic.p);
      require(refused.get(1)[0] == attempt, "rounding/asymmetric orbit admission mismatch");
    }
  }
}
}  // namespace
int main() {
  try {
    for (const auto shape :
         {std::pair{1, 1}, std::pair{1, 7}, std::pair{2, 3}, std::pair{4, 1}, std::pair{3, 11}})
      run(shape.first, shape.second);
    std::cout << "Packed history full metric, roundtrip, tails and refusal gates passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
