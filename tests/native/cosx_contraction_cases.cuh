#pragma once

namespace cosx_contraction_test {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
template <class T>
struct Buffer {
  T* data{};
  explicit Buffer(std::size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
  ~Buffer() { cudaFree(data); }
};

/** Independent long-double full/tail products, including an asymmetric seed.
 * Direct device nonfinites must set the sticky error before another consumer
 * can mask them. Capture is explicitly unsupported by this physical owner. */
void scalar_cases() {
  using namespace generativeqc::dft;
  constexpr std::size_t n = 17, points = 13, tail = 5;
  for (unsigned mask : {0U, 3U}) {
    Stream stream;
    Buffer<double> a(points * n), b(n * n), output(n * n);
    Buffer<int> error(1);
    cosx_contraction_qualification_for_test(mask, false);
    auto binding = cosx_lowering::prepare(n, points, tail, stream.value, 96ULL << 20);
    cosx_contraction_qualification_for_test(0, false);
    for (unsigned slot = 0; slot < 4; ++slot) {
      const bool update = slot % 2;
      const auto p = slot < 2 ? points : tail;
      std::vector<double> left(p * n), right(update ? p * n : n * n);
      std::vector<double> expected(update ? n * n : p * n), actual(expected.size());
      for (std::size_t i = 0; i < left.size(); ++i) left[i] = std::sin(0.13 * (i + 1));
      for (std::size_t i = 0; i < right.size(); ++i) right[i] = std::cos(0.27 * (i + 1));
      for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = update ? 0.001 * i : std::numeric_limits<double>::quiet_NaN();
      check(cudaMemcpyAsync(a.data, left.data(), left.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(b.data, right.data(), right.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(output.data, expected.data(), expected.size() * 8,
                            cudaMemcpyHostToDevice, stream.value));
      check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
      binding->execute(slot, stream.value, a.data, b.data, output.data, error.data);
      int failure{};
      check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(failure == 0, "COSX scalar product flagged a finite output");
      for (std::size_t i = 0; i < (update ? n : p); ++i)
        for (std::size_t j = 0; j < n; ++j) {
          long double value = update ? expected[i * n + j] : 0;
          for (std::size_t k = 0; k < (update ? p : n); ++k)
            value += static_cast<long double>(update ? left[k * n + i] : left[i * n + k]) *
                     right[k * n + j];
          require(std::isfinite(actual[i * n + j]) && std::abs(actual[i * n + j] - value) < 3e-12,
                  "COSX prepared product differs from long-double oracle");
        }
    }
    bool rejected{};
    try {
      binding->execute(0, stream.value, a.data, b.data, a.data, error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "COSX contraction accepted aliased output");
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    rejected = false;
    try {
      binding->execute(0, stream.value, a.data, b.data, output.data, error.data);
    } catch (const std::logic_error&) {
      rejected = true;
    }
    cudaGraph_t graph{};
    check(cudaStreamEndCapture(stream.value, &graph));
    check(cudaGraphDestroy(graph));
    require(rejected, "COSX contraction capture bypassed physical work accounting");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(cudaMemcpyAsync(a.data, &nan, sizeof(double), cudaMemcpyHostToDevice, stream.value));
    for (int prior : {0, 9}) {
      check(cudaMemcpyAsync(error.data, &prior, sizeof(int), cudaMemcpyHostToDevice, stream.value));
      binding->execute(2, stream.value, a.data, b.data, output.data, error.data);
      int failure{};
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(failure == (prior ? prior : 1), "COSX product lost nonfinite/sticky error state");
    }
  }
}

/** Independent asymmetric point-batched arithmetic and publication failures.
 * Generated execution must retain the incumbent fused FMA chain; both routes
 * also meet the independent long-double gate, including zero/signed weights. */
void weighted_cases() {
  using namespace generativeqc::dft;
  constexpr std::size_t columns = 17, points = 13, tail = 5;
  for (unsigned mask : {0U, 4U}) {
    Stream stream;
    Buffer<double> esp(points * columns * columns), projected(points * columns), weight(points),
        output(points * columns);
    Buffer<int> error(1);
    cosx_contraction_qualification_for_test(mask, false);
    auto binding = cosx_lowering::prepare(columns, points, tail, stream.value, 96ULL << 20);
    cosx_contraction_qualification_for_test(0, false);
    std::vector<double> left(points * columns * columns), right(points * columns), scales(points);
    for (std::size_t index = 0; index < left.size(); ++index)
      left[index] = std::sin(0.13 * (index + 1));
    for (std::size_t index = 0; index < right.size(); ++index)
      right[index] = std::cos(0.27 * (index + 1));
    for (std::size_t point = 0; point < points; ++point)
      scales[point] = point % 3 == 0 ? 0 : (point % 2 ? -0.3 : 0.7);
    check(cudaMemcpyAsync(esp.data, left.data(), left.size() * 8, cudaMemcpyHostToDevice,
                          stream.value));
    check(cudaMemcpyAsync(projected.data, right.data(), right.size() * 8, cudaMemcpyHostToDevice,
                          stream.value));
    check(cudaMemcpyAsync(weight.data, scales.data(), scales.size() * 8, cudaMemcpyHostToDevice,
                          stream.value));
    for (unsigned slot : {4U, 5U}) {
      const auto extent = slot == 4 ? points : tail;
      std::vector<double> actual(extent * columns, std::numeric_limits<double>::quiet_NaN());
      check(cudaMemcpyAsync(output.data, actual.data(), actual.size() * 8, cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
      binding->execute(slot, stream.value, esp.data, projected.data, output.data, error.data,
                       weight.data);
      int failure{};
      check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * 8, cudaMemcpyDeviceToHost,
                            stream.value));
      check(
          cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(!failure, "finite weighted contraction flagged an error");
      for (std::size_t point = 0; point < extent; ++point)
        for (std::size_t row = 0; row < columns; ++row) {
          long double oracle = 0;
          double fused = 0;
          for (std::size_t column = 0; column < columns; ++column) {
            const auto matrix = left[(point * columns + row) * columns + column];
            const auto vector = right[point * columns + column];
            oracle += static_cast<long double>(matrix) * vector;
            fused = std::fma(matrix, vector, fused);
          }
          const auto value = actual[point * columns + row];
          require(std::isfinite(value) && std::abs(value - oracle * scales[point]) < 3e-12,
                  "weighted contraction differs from independent long-double oracle");
          require(mask || value == scales[point] * fused,
                  "generated route changed the fused FMA chain");
        }
    }
    bool rejected{};
    try {
      binding->execute(4, stream.value, esp.data, projected.data, output.data, error.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "weighted contraction accepted a missing scale input");
    rejected = false;
    try {
      binding->execute(0, stream.value, esp.data, projected.data, output.data, error.data,
                       weight.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "ordinary contraction accepted an undeclared scale input");
    rejected = false;
    try {
      binding->execute(4, stream.value, esp.data, projected.data, output.data, error.data,
                       output.data);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "weighted contraction accepted scale/output aliasing");
    auto malformed = cosx_lowering::esp_application(points, columns);
    malformed.batch_scale.modes[0] = 99;
    rejected = false;
    try {
      generativeqc::tensor::PreparedContractionSites<1> invalid(std::array{malformed}, stream.value,
                                                                96ULL << 20);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "weighted contraction accepted a different semantic batch axis");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (bool bad_weight : {false, true}) {
      check(cudaMemcpyAsync(esp.data, &left[0], sizeof(double), cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(weight.data, &scales[0], sizeof(double), cudaMemcpyHostToDevice,
                            stream.value));
      check(cudaMemcpyAsync(bad_weight ? weight.data : esp.data, &nan, sizeof(double),
                            cudaMemcpyHostToDevice, stream.value));
      for (int prior : {0, 9}) {
        check(
            cudaMemcpyAsync(error.data, &prior, sizeof(int), cudaMemcpyHostToDevice, stream.value));
        binding->execute(5, stream.value, esp.data, projected.data, output.data, error.data,
                         weight.data);
        int failure{};
        double published{};
        check(cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost,
                              stream.value));
        check(cudaMemcpyAsync(&published, output.data, sizeof(double), cudaMemcpyDeviceToHost,
                              stream.value));
        check(cudaStreamSynchronize(stream.value));
        require(failure == (prior ? prior : 1) && published == 0,
                "weighted publication lost nonfinite/sticky-error semantics");
      }
    }
  }
}

void endpoint_cases() {
  using namespace generativeqc::dft;
  const auto molecule = spherical_sdf();
  const auto n = AoBasis(molecule).nao;
  std::vector<double> xyz(3 * 17), weights(17);
  for (std::size_t p = 0; p < weights.size(); ++p) {
    xyz[3 * p] = -0.6 + 0.07 * p;
    xyz[3 * p + 1] = 0.13 * std::sin(p + 0.3);
    xyz[3 * p + 2] = 0.4 + 0.03 * p;
    weights[p] = p % 3 == 0 ? -0.02 : 0.01 * (p + 1);
  }
  for (std::size_t tile : {1U, 7U, 17U})
    for (unsigned route = 0; route < 10; ++route) {
      const unsigned mask = route < 8 ? route : 7;
      const auto base = cuda_cosx_staging_diagnostic(molecule, weights.size(), tile).device_bytes;
      cosx_contraction_qualification_for_test(mask, route == 8);
      CudaCosxStagingPlan plan(molecule, xyz, weights, tile, 0,
                               base + (route == 9 ? 0 : 96ULL << 20));
      cosx_contraction_qualification_for_test(0, false);
      for (unsigned changed = 0; changed < 2; ++changed) {
        auto density = symmetric_density(n);
        // Deliberately asymmetric off-diagonals make a hidden transpose visible.
        density[1] += 0.03 + 0.01 * changed;
        const auto convention =
            changed ? CosxDensityConvention::spin_resolved : CosxDensityConvention::rhf_spin_summed;
        const auto cpu = build_cosx_reference(molecule, xyz, weights, density, convention);
        const auto gpu = plan.build(density, convention);
        require(max_error(gpu.raw_exchange, cpu.raw_exchange) < 3e-11 &&
                    max_error(gpu.exchange, cpu.exchange) < 3e-11 &&
                    std::abs(gpu.exchange_energy - cpu.exchange_energy) < 3e-11,
                "COSX prepared endpoint differs from independent CPU E/K oracle");
        const auto& info = plan.diagnostic();
        require(info.provider_allowance == ((mask && route < 8) ? 96ULL << 20 : 0) &&
                    info.device_bytes == base + info.provider_allowance &&
                    info.contraction_host_bytes,
                "COSX shared provider reservation was omitted or charged per site");
        for (unsigned slot = 0; slot < 6; ++slot) {
          const auto& site = info.contractions[slot];
          const bool library = route < 8 && (mask & (1U << (slot < 4 ? slot % 2 : 2)));
          require(site.candidate.provider == (library ? "cublas" : "generated.cuda"),
                  "COSX did not independently bind projection/accumulation/weighted ESP");
          const auto& candidates = slot >= 4  ? cosx_lowering::esp_application_candidates
                                   : slot % 2 ? cosx_lowering::accumulation_candidates
                                              : cosx_lowering::projection_candidates;
          require(site.offer_count == candidates.size() && site.selected < site.offer_count &&
                      site.selected == (library ? 0U : 1U) &&
                      site.offers[site.selected].identity == site.candidate.identity &&
                      site.offers[site.selected].provider == site.candidate.provider &&
                      site.offers[site.selected].rejection.empty() &&
                      site.offers[1].rejection.empty() &&
                      site.offers[0].rejection.empty() == library && info.contraction_device == 0 &&
                      info.compute_major > 0 && info.runtime_version > 0,
                  "COSX discarded negative candidates or runtime provenance");
          for (std::size_t i = 0; i < candidates.size(); ++i)
            require(site.offers[i].identity == candidates[i].identity &&
                        site.offers[i].provider == candidates[i].provider &&
                        (i < 2 || !site.offers[i].rejection.empty()),
                    "COSX changed the generated catalog or admitted an optional provider");
          const bool full = slot < 2 || slot == 4;
          const auto calls = full ? 17 / tile : std::size_t(17 % tile != 0);
          const auto points = full ? tile : 17 % tile;
          require(site.calls == (changed + 1) * calls &&
                      site.summands == (changed + 1) * calls * points * n * n,
                  "COSX full/tail semantic work accounting changed");
          require(site.scaled_elements == (slot >= 4 ? (changed + 1) * calls * points * n : 0) &&
                      site.publication_passes == (slot >= 4 && library ? (changed + 1) * calls : 0),
                  "weighted COSX omitted publication work");
        }
      }
    }
}
}  // namespace cosx_contraction_test

void cosx_contraction_cases() {
  cosx_contraction_test::scalar_cases();
  cosx_contraction_test::weighted_cases();
  cosx_contraction_test::endpoint_cases();
  std::cout
      << "COSX prepared full/tail scalar, CPU E/K, independent-site and resource gates passed\n";
}
