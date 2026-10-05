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
    for (unsigned route = 0; route < 6; ++route) {
      const unsigned mask = route < 4 ? route : 3;
      const auto base = cuda_cosx_staging_diagnostic(molecule, weights.size(), tile).device_bytes;
      cosx_contraction_qualification_for_test(mask, route == 4);
      CudaCosxStagingPlan plan(molecule, xyz, weights, tile, 0,
                               base + (route == 5 ? 0 : 96ULL << 20));
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
        require(info.provider_allowance == ((mask && route < 4) ? 96ULL << 20 : 0) &&
                    info.device_bytes == base + info.provider_allowance &&
                    info.contraction_host_bytes,
                "COSX shared provider reservation was omitted or charged per site");
        for (unsigned slot = 0; slot < 4; ++slot) {
          const auto& site = info.contractions[slot];
          const bool library = route < 4 && (mask & (1U << (slot % 2)));
          require(site.candidate.provider == (library ? "cublas" : "generated.cuda"),
                  "COSX did not independently bind projection/accumulation");
          require(site.offer_count == 3 && !site.offers[2].rejection.empty() &&
                      site.offers[site.selected].identity == site.candidate.identity &&
                      info.contraction_device == 0 && info.compute_major > 0 &&
                      info.runtime_version > 0,
                  "COSX discarded negative candidates or runtime provenance");
          const auto calls = slot < 2 ? 17 / tile : std::size_t(17 % tile != 0);
          const auto points = slot < 2 ? tile : 17 % tile;
          require(site.calls == (changed + 1) * calls &&
                      site.summands == (changed + 1) * calls * points * n * n,
                  "COSX full/tail semantic work accounting changed");
        }
      }
    }
}
}  // namespace cosx_contraction_test

void cosx_contraction_cases() {
  cosx_contraction_test::scalar_cases();
  cosx_contraction_test::endpoint_cases();
  std::cout
      << "COSX prepared full/tail scalar, CPU E/K, independent-site and resource gates passed\n";
}
