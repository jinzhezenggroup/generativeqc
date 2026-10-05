#pragma once

#include "cosx_contraction_cases.cuh"

namespace cosx_checked_pair_test {
using namespace cosx_contraction_test;
using namespace generativeqc::dft;

/** Original simultaneous traversal protects update/publication rounding;
 * independent long-double loops separately check both mathematical directions. */
__global__ void incumbent(const double* matrix, const double* first_vector,
                          const double* second_vector, std::size_t points, std::size_t n,
                          double* first_output, double* second_output) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= points * n) return;
  const auto p = index / n, row = index % n;
  double first = 0, second = 0;
  bool valid = true;
  for (std::size_t k = 0; k < n; ++k)
    if (!generated_cosx_derivative::accumulate_bidirectional(
            matrix[(p * n + row) * n + k], matrix[(p * n + k) * n + row], first_vector[p * n + k],
            second_vector[p * n + k], first, second)) {
      valid = false;
      break;
    }
  first_output[index] = valid ? first : 0;
  second_output[index] = valid ? second : 0;
}

struct WrongStep : cosx_derivative_lowering::PairedStep {
  static constexpr std::string_view update_identity =
      "0000000000000000000000000000000000000000000000000000000000000000";
};
struct PartialStep : cosx_derivative_lowering::ScalarStep<false> {
  static constexpr auto update_identity = cosx_derivative_lowering::PairedStep::update_identity;
};

void cases() {
  constexpr std::size_t n = 17, full = 13, tail = 5;
  for (unsigned qualification : {0U, 255U}) {
    Stream stream;
    Buffer<double> matrix(full * n * n), first(full * n), second(full * n), right(full * n),
        left(full * n), old_right(full * n), old_left(full * n);
    Buffer<int> error(1);
    generativeqc::tensor::contraction_sites_qualification_for_test = qualification;
    auto plan = cosx_derivative_lowering::prepare_molecular(n, full, tail, stream.value);
    generativeqc::tensor::contraction_sites_qualification_for_test = 0;
    require(!plan->provider_bytes(), "coupled checked region reserved an optional provider");
    std::vector<double> a(full * n * n), b(full * n), c(full * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::cos(0.31 * (i + 7));
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = std::sin(0.23 * (i + 1));
      c[i] = std::cos(0.17 * (i + 3));
    }
    const auto upload = [&] {
      check(cudaMemcpyAsync(matrix.data, a.data(), a.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(first.data, b.data(), b.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(second.data, c.data(), c.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
    };
    upload();
    for (bool short_tile : {false, true}) {
      const auto points = short_tile ? tail : full;
      check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
      cosx_derivative_lowering::apply_bidirectional(*plan, short_tile, stream.value, matrix.data,
                                                    first.data, second.data, right.data, left.data,
                                                    error.data);
      incumbent<<<(points * n + 127) / 128, 128, 0, stream.value>>>(
          matrix.data, first.data, second.data, points, n, old_right.data, old_left.data);
      check(cudaGetLastError());
      std::vector<double> r(points * n), l(points * n), old_r(points * n), old_l(points * n);
      int failure{};
      for (const auto pair : {std::pair{r.data(), right.data},
                              {l.data(), left.data},
                              {old_r.data(), old_right.data},
                              {old_l.data(), old_left.data}})
        check(cudaMemcpyAsync(pair.first, pair.second, r.size() * 8, cudaMemcpyDeviceToHost,
                              stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(!failure, "finite coupled region flagged an error");
      for (std::size_t p = 0; p < points; ++p)
        for (std::size_t row = 0; row < n; ++row) {
          long double expected_r = 0, expected_l = 0;
          for (std::size_t k = 0; k < n; ++k) {
            expected_r += static_cast<long double>(a[(p * n + row) * n + k]) * b[p * n + k];
            expected_l += static_cast<long double>(a[(p * n + k) * n + row]) * c[p * n + k];
          }
          const auto i = p * n + row;
          require(
              std::abs(r[i] - expected_r) < 3e-12 && std::abs(l[i] - expected_l) < 3e-12 &&
                  std::bit_cast<std::uint64_t>(r[i]) == std::bit_cast<std::uint64_t>(old_r[i]) &&
                  std::bit_cast<std::uint64_t>(l[i]) == std::bit_cast<std::uint64_t>(old_l[i]),
              "coupled region changed an independent oracle or incumbent rounding");
        }
      for (std::size_t slot : {short_tile ? 5U : 4U, short_tile ? 7U : 6U}) {
        const auto& info = plan->diagnostics()[slot];
        require(info.calls == 1 && info.summands == points * n * n &&
                    info.candidate.provider == "generated.cuda",
                "coupled work counters mismatch");
        for (std::size_t offer = 0; offer < info.offer_count; ++offer)
          if (info.offers[offer].provider != "generated.cuda")
            require(!info.offers[offer].rejection.empty(),
                    "coupled region admitted a split provider");
      }
    }
    bool rejected = false;
    try {
      plan->execute_checked<PartialStep>(4, stream.value, matrix.data, first.data, right.data,
                                         error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "coupled region admitted partial execution with a matching scalar hash");
    rejected = false;
    try {
      plan->execute_checked_transpose_pair<WrongStep>(4, 6, stream.value, matrix.data, first.data,
                                                      second.data, right.data, left.data,
                                                      error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "coupled region admitted a different scalar helper");
    rejected = false;
    try {
      cosx_derivative_lowering::apply_bidirectional(*plan, false, stream.value, matrix.data,
                                                    first.data, second.data, right.data, right.data,
                                                    error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "coupled region admitted overlapping publications");
    // Only one update fails. Its finite counterpart must also become zero;
    // sticky errors from the enclosing endpoint must survive unchanged.
    for (unsigned scenario : {0U, 1U, 2U}) {
      std::fill(a.begin(), a.end(), scenario == 2 ? std::numeric_limits<double>::max() : 0);
      std::fill(b.begin(), b.end(), scenario == 0 ? std::numeric_limits<double>::quiet_NaN() : 0);
      std::fill(c.begin(), c.end(), scenario == 1 ? std::numeric_limits<double>::quiet_NaN() : 2);
      upload();
      for (int sticky : {0, 2}) {
        check(cudaMemcpyAsync(error.data, &sticky, sizeof(int), cudaMemcpyHostToDevice,
                              stream.value));
        cosx_derivative_lowering::apply_bidirectional(*plan, false, stream.value, matrix.data,
                                                      first.data, second.data, right.data,
                                                      left.data, error.data);
        std::vector<double> r(b.size()), l(c.size());
        int failure{};
        check(cudaMemcpyAsync(r.data(), right.data, r.size() * 8, cudaMemcpyDeviceToHost,
                              stream.value));
        check(cudaMemcpyAsync(l.data(), left.data, l.size() * 8, cudaMemcpyDeviceToHost,
                              stream.value));
        check(cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost,
                              stream.value));
        check(cudaStreamSynchronize(stream.value));
        require(std::all_of(r.begin(), r.end(), [](double x) { return x == 0; }) &&
                    std::all_of(l.begin(), l.end(), [](double x) { return x == 0; }) &&
                    failure == (sticky ? sticky : 1),
                "coupled zero/sticky publication changed");
      }
    }
  }
}
}  // namespace cosx_checked_pair_test
