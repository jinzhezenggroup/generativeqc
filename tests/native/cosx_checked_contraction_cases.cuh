#pragma once

#include "cosx_contraction_cases.cuh"

namespace cosx_checked_test {
using namespace cosx_contraction_test;
using namespace generativeqc::dft;

/** Retained incumbent traversal for a bitwise rounding comparison. Independent
 * long-double loops below separately qualify the finite mathematical results. */
__global__ void incumbent(const double* left, const double* right, const double* weights,
                          std::size_t points, std::size_t n, double* output) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= points * n) return;
  const auto point = index / n, column = index % n;
  double value = 0;
  bool valid = true;
  for (std::size_t k = 0; k < n; ++k) {
    const auto ai = weights ? point * n * n + column * n + k : point * n + k;
    const auto bi = weights ? point * n + k : k * n + column;
    if (!generated_cosx_derivative::accumulate_projection(left[ai], right[bi], value)) {
      valid = false;
      break;
    }
  }
  double publication = value;
  if (valid && weights)
    valid = generated_cosx_derivative::scale(weights[point], value, publication);
  output[index] = valid ? publication : 0;
}

struct WrongStep : cosx_derivative_lowering::ScalarStep<false> {
  static constexpr std::string_view update_identity =
      "0000000000000000000000000000000000000000000000000000000000000000";
};

__global__ void incumbent_symmetric(const double* ao, const double* density, std::size_t points,
                                    std::size_t n, double* output) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= points * n) return;
  const auto point = index / n, column = index % n;
  double value = 0;
  for (std::size_t k = 0; k < n; ++k)
    if (!generated_cosx_derivative::accumulate_symmetric_projection(
            ao[point * n + k], density[k * n + column], density[column * n + k], value)) {
      value = 0;
      break;
    }
  output[index] = value;
}

struct MissingSymmetricViews : cosx_derivative_lowering::ScalarStep<false> {
  static constexpr auto update_identity = cosx_derivative_lowering::SymmetricStep::update_identity;
};

void symmetric_cases() {
  constexpr std::size_t n = 17, full = 13, tail = 5;
  for (unsigned qualification : {0U, 15U}) {
    Stream stream;
    Buffer<double> left(full * n), right(n * n), output(full * n), reference(full * n);
    Buffer<int> error(1);
    generativeqc::tensor::contraction_sites_qualification_for_test = qualification;
    auto plan = cosx_derivative_lowering::prepare_molecular(n, full, tail, stream.value);
    generativeqc::tensor::contraction_sites_qualification_for_test = 0;
    require(!plan->provider_bytes(), "symmetric checked recipe reserved a library");
    std::vector<double> a(full * n), b(n * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::sin(0.23 * (i + 1));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.17 * (i + 3));
    check(cudaMemcpyAsync(left.data, a.data(), a.size() * 8, cudaMemcpyHostToDevice, stream.value));
    check(
        cudaMemcpyAsync(right.data, b.data(), b.size() * 8, cudaMemcpyHostToDevice, stream.value));
    for (bool short_tile : {false, true}) {
      const auto points = short_tile ? tail : full;
      check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
      cosx_derivative_lowering::project_symmetric(*plan, short_tile, stream.value, left.data,
                                                  right.data, output.data, error.data);
      incumbent_symmetric<<<(points * n + 127) / 128, 128, 0, stream.value>>>(
          left.data, right.data, points, n, reference.data);
      check(cudaGetLastError());
      std::vector<double> actual(points * n), old(points * n);
      int failure{};
      check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(cudaMemcpyAsync(old.data(), reference.data, old.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(!failure, "finite symmetric recipe flagged an error");
      for (std::size_t p = 0; p < points; ++p)
        for (std::size_t c = 0; c < n; ++c) {
          long double expected = 0;
          for (std::size_t k = 0; k < n; ++k)
            expected += static_cast<long double>(a[p * n + k]) / 2 *
                        (static_cast<long double>(b[k * n + c]) + b[c * n + k]);
          require(std::abs(actual[p * n + c] - expected) < 3e-12 &&
                      std::bit_cast<std::uint64_t>(actual[p * n + c]) ==
                          std::bit_cast<std::uint64_t>(old[p * n + c]),
                  "symmetric recipe changed the oracle or incumbent rounding");
        }
      const auto& info = plan->diagnostics()[short_tile ? 3 : 2];
      require(info.calls == 1 && info.summands == points * n * n &&
                  info.candidate.provider == "generated.cuda",
              "symmetric full/tail work diagnostics mismatch");
      for (std::size_t offer = 0; offer < info.offer_count; ++offer)
        if (info.offers[offer].provider != "generated.cuda")
          require(!info.offers[offer].rejection.empty(),
                  "symmetric recipe admitted an unchecked provider");
    }
    bool rejected{};
    try {
      plan->execute_checked<MissingSymmetricViews>(2, stream.value, left.data, right.data,
                                                   output.data, error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "symmetric recipe accepted a helper with missing transpose views");
    rejected = false;
    try {
      plan->execute(2, stream.value, left.data, right.data, output.data, error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "symmetric recipe accepted ordinary execution");
    // A zero AO must not hide overflow in the original density-pair sum.
    std::fill(a.begin(), a.end(), 0);
    std::fill(b.begin(), b.end(), std::numeric_limits<double>::max());
    check(cudaMemcpyAsync(left.data, a.data(), a.size() * 8, cudaMemcpyHostToDevice, stream.value));
    check(
        cudaMemcpyAsync(right.data, b.data(), b.size() * 8, cudaMemcpyHostToDevice, stream.value));
    for (int sticky : {0, 2}) {
      check(
          cudaMemcpyAsync(error.data, &sticky, sizeof(int), cudaMemcpyHostToDevice, stream.value));
      cosx_derivative_lowering::project_symmetric(*plan, false, stream.value, left.data, right.data,
                                                  output.data, error.data);
      std::vector<double> actual(a.size());
      int failure{};
      check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(std::all_of(actual.begin(), actual.end(), [](double x) { return x == 0; }) &&
                  failure == (sticky ? sticky : 1),
              "symmetric recipe lost invalid zero/sticky publication");
    }
  }
}

void cases() {
  symmetric_cases();
  constexpr std::size_t n = 17, full = 13, tail = 5;
  for (unsigned qualification : {0U, 63U}) {
    Stream stream;
    Buffer<double> left(full * n * n), right(full * n * n), weights(full), output(3 * full * n),
        reference(3 * full * n);
    Buffer<int> error(1);
    generativeqc::tensor::contraction_sites_qualification_for_test = qualification;
    auto plan = cosx_derivative_lowering::prepare_point(n, full, tail, stream.value);
    generativeqc::tensor::contraction_sites_qualification_for_test = 0;
    require(!plan->provider_bytes(), "checked derivative reserved a library provider");
    for (const auto& site : plan->diagnostics()) {
      require(site.candidate.provider == "generated.cuda",
              "ordered scalar checks selected a library");
      for (std::size_t i = 0; i < site.offer_count; ++i)
        if (site.offers[i].provider != "generated.cuda")
          require(!site.offers[i].rejection.empty(), "unchecked optional recipe was admitted");
    }
    for (std::size_t slot = 0; slot < 6; ++slot) {
      const auto points = slot % 2 ? tail : full;
      const bool weighted = slot >= 4, jet = slot >= 2 && slot < 4;
      const auto rows = jet ? 3 * points : points;
      std::vector<double> a(weighted ? points * n * n : rows * n), b(weighted ? points * n : n * n),
          w(points);
      for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::sin(0.17 * (i + 1));
      for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.29 * (i + 1));
      for (std::size_t i = 0; i < w.size(); ++i) w[i] = i % 3 ? 0.03 * (double(i) - 4) : 0;
      check(
          cudaMemcpyAsync(left.data, a.data(), a.size() * 8, cudaMemcpyHostToDevice, stream.value));
      check(cudaMemcpyAsync(right.data, b.data(), b.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(weights.data, w.data(), w.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
      if (weighted)
        cosx_derivative_lowering::apply_esp(*plan, slot % 2, stream.value, left.data, right.data,
                                            weights.data, output.data, error.data);
      else if (jet)
        cosx_derivative_lowering::project_jets(*plan, slot % 2, stream.value, left.data, right.data,
                                               output.data, error.data);
      else
        cosx_derivative_lowering::project(*plan, slot % 2, stream.value, left.data, right.data,
                                          output.data, error.data);
      incumbent<<<(rows * n + 127) / 128, 128, 0, stream.value>>>(
          left.data, right.data, weighted ? weights.data : nullptr, rows, n, reference.data);
      check(cudaGetLastError());
      std::vector<double> actual(rows * n), old(rows * n);
      int failure{};
      check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(cudaMemcpyAsync(old.data(), reference.data, old.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(!failure, "checked finite derivative failed");
      for (std::size_t point = 0; point < rows; ++point)
        for (std::size_t column = 0; column < n; ++column) {
          long double expected = 0;
          for (std::size_t k = 0; k < n; ++k)
            expected += static_cast<long double>(
                            a[weighted ? point * n * n + column * n + k : point * n + k]) *
                        b[weighted ? point * n + k : k * n + column];
          if (weighted) expected *= w[point];
          const auto index = point * n + column;
          require(std::abs(actual[index] - expected) < 3e-12,
                  "checked derivative differs from independent long double");
          require(std::bit_cast<std::uint64_t>(actual[index]) ==
                      std::bit_cast<std::uint64_t>(old[index]),
                  "checked derivative changed incumbent rounding");
        }
      const auto& info = plan->diagnostics()[slot];
      require(info.calls == (jet ? 3 : 1) && info.summands == rows * n * n &&
                  info.scaled_elements == (weighted ? points * n : 0) && !info.publication_passes,
              "checked derivative semantic work mismatch");
    }
    const auto calls = plan->diagnostics()[0].calls;
    for (unsigned wrong = 0; wrong < 2; ++wrong) {
      bool rejected{};
      try {
        if (wrong)
          plan->execute_checked<WrongStep>(0, stream.value, left.data, right.data, output.data,
                                           error.data);
        else
          plan->execute(0, stream.value, left.data, right.data, output.data, error.data);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected && plan->diagnostics()[0].calls == calls, "checked contract was bypassed");
    }
    // Mathematically cancelling finite terms overflow an ordered partial sum.
    // A final-only audit or a zero publication weight must not suppress it.
    std::vector<double> a(full * n * n, 0), b(full * n, 1), w(full, 0);
    a[0] = a[1] = 1e308;
    a[2] = a[3] = -1e308;
    for (int sticky : {0, 2}) {
      check(
          cudaMemcpyAsync(left.data, a.data(), a.size() * 8, cudaMemcpyHostToDevice, stream.value));
      check(cudaMemcpyAsync(right.data, b.data(), b.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(weights.data, w.data(), w.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(
          cudaMemcpyAsync(error.data, &sticky, sizeof(int), cudaMemcpyHostToDevice, stream.value));
      cosx_derivative_lowering::apply_esp(*plan, false, stream.value, left.data, right.data,
                                          weights.data, output.data, error.data);
      double published{};
      int failure{};
      check(cudaMemcpyAsync(&published, output.data, 8, cudaMemcpyDeviceToHost, stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(published == 0 && failure == (sticky ? sticky : 1),
              "ordered invalid reduction lost zero/sticky publication");
    }
    a[0] = std::numeric_limits<double>::quiet_NaN();
    check(cudaMemcpyAsync(left.data, a.data(), a.size() * 8, cudaMemcpyHostToDevice, stream.value));
    check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
    cosx_derivative_lowering::apply_esp(*plan, false, stream.value, left.data, right.data,
                                        weights.data, output.data, error.data);
    double published{};
    int failure{};
    check(cudaMemcpyAsync(&published, output.data, 8, cudaMemcpyDeviceToHost, stream.value));
    check(cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
    check(cudaStreamSynchronize(stream.value));
    require(published == 0 && failure == 1, "zero weighting masked a nonfinite checked input");
  }
}
}  // namespace cosx_checked_test
