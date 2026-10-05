// Included in the DFT test namespace. Exercise the actual compiler materializer
// and shared provider independently of the endpoint's symmetric-input guard.
namespace density_provider_test {
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Buffer {
  double* value{};
  explicit Buffer(std::size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(double)));
  }
  ~Buffer() { cudaFree(value); }
};
}  // namespace density_provider_test

void density_provider_scalar_cases() {
  using namespace density_provider_test;
  for (std::size_t n : {1U, 17U})
    for (bool uks : {false, true}) {
      Stream stream;
      const auto layout = cuda_xc_layout_shape(1, 1, n, 17, 2, uks, 7);
      const auto spins = layout.spins, jets = layout.work_jets;
      xc_density_provider_for_test(true, false);
      auto product = cuda_xc_detail::prepare_density_provider(layout, stream.value, 128ULL << 20);
      xc_density_provider_for_test(false, false);
      require(product->enabled(), "scalar gate did not bind GEMM");
      const auto& diagnostic = product->diagnostic();
      require(diagnostic.matrix_bytes == spins * n * n * sizeof(double) &&
                  diagnostic.provider_allowance == 96ULL << 20 &&
                  diagnostic.host_bytes == 16ULL << 10 && diagnostic.provider_version > 0,
              "density provider resource/provenance mismatch");
      Buffer d(spins * n * n), a(jets * 7 * n), out(spins * jets * 7 * n), error(1);
      bool alias_rejected{};
      try {
        product->execute(stream.value, jets, a.value, a.value, reinterpret_cast<int*>(error.value));
      } catch (const std::invalid_argument&) {
        alias_rejected = true;
      }
      require(alias_rejected, "panel product alias was admitted");
      for (std::size_t points : {1U, 7U}) {
        const auto rows = jets * points;
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        for (unsigned replay = 0; replay < 3; ++replay) {
          std::vector<double> density_values(spins * n * n), ao(rows * n), actual(spins * rows * n);
          for (std::size_t i = 0; i != density_values.size(); ++i)
            density_values[i] = std::sin(0.37 * (i + 1) + replay);
          for (std::size_t i = 0; i != ao.size(); ++i) ao[i] = std::cos(0.19 * (i + 1) - replay);
          check(cudaMemcpyAsync(d.value, density_values.data(),
                                density_values.size() * sizeof(double), cudaMemcpyHostToDevice,
                                stream.value));
          check(cudaMemcpyAsync(a.value, ao.data(), ao.size() * sizeof(double),
                                cudaMemcpyHostToDevice, stream.value));
          check(cudaMemsetAsync(error.value, 0, sizeof(int), stream.value));
          check(cudaStreamSynchronize(stream.value));
          if (!replay) {
            check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
            xc_density_materialize_for_test(stream.value, d.value, n, spins,
                                            product->materialized_matrices(),
                                            reinterpret_cast<int*>(error.value));
            product->execute(stream.value, rows, a.value, out.value,
                             reinterpret_cast<int*>(error.value));
            check(cudaStreamEndCapture(stream.value, &graph));
            check(cudaGraphInstantiate(&executable, graph, 0));
          }
          check(cudaGraphLaunch(executable, stream.value));
          check(cudaMemcpyAsync(actual.data(), out.value, actual.size() * sizeof(double),
                                cudaMemcpyDeviceToHost, stream.value));
          int failure{};
          check(cudaMemcpyAsync(&failure, error.value, sizeof(int), cudaMemcpyDeviceToHost,
                                stream.value));
          check(cudaStreamSynchronize(stream.value));
          require(failure == 0, "materialized density finite gate");
          for (std::size_t spin = 0; spin != spins; ++spin)
            for (std::size_t row = 0; row != rows; ++row)
              for (std::size_t mu = 0; mu != n; ++mu) {
                long double expected{};
                for (std::size_t nu = 0; nu != n; ++nu)
                  expected += (0.5L * density_values[(spin * n + mu) * n + nu] +
                               0.5L * density_values[(spin * n + nu) * n + mu]) *
                              ao[row * n + nu];
                close(actual[(spin * rows + row) * n + mu], static_cast<double>(expected),
                      "nonsymmetric density scalar oracle", 2e-12);
              }
        }
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
      }
      // Publication replaces nonfinite values and preserves an earlier sticky
      // error. Neither a successful GEMM nor a later finite audit may clear it.
      std::vector<double> bad(spins * n * n, std::numeric_limits<double>::quiet_NaN());
      check(cudaMemcpyAsync(d.value, bad.data(), bad.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream.value));
      int sticky = 7;
      check(
          cudaMemcpyAsync(error.value, &sticky, sizeof(int), cudaMemcpyHostToDevice, stream.value));
      xc_density_materialize_for_test(stream.value, d.value, n, spins,
                                      product->materialized_matrices(),
                                      reinterpret_cast<int*>(error.value));
      product->execute(stream.value, jets, a.value, out.value, reinterpret_cast<int*>(error.value));
      check(
          cudaMemcpyAsync(&sticky, error.value, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
      check(cudaStreamSynchronize(stream.value));
      require(sticky == 7, "panel publication cleared the earlier error");
    }
}

void density_provider_cases() {
  density_provider_scalar_cases();
  auto molecule = system(3);
  molecule.shells.push_back({0, 1, {{0.4, 1.0}}});
  molecule.shells.push_back({1, 0, {{0.3, 1.0}}});
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
  for (unsigned route : {0U, 1U, 2U, 3U})
    for (unsigned functional : {0U, 1U, 2U})
      for (bool uks : {false, true}) {
        Fixture fixture(basis, grid, functional, uks, 17);
        xc_density_provider_for_test(route != 0, route == 3);
        fixture.plan->prepare_density(generativeqc::runtime::strict_fp64_precision(), 10,
                                      route == 2 ? 1 : 128ULL << 20);
        xc_density_provider_for_test(false, false);
        const auto& binding =
            fixture.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit);
        require((binding.candidate.provider == "cublas") == (route == 1),
                "density admission route");
        auto d = density(basis.nao, uks ? 2 : 1);
        compare(fixture, basis, grid, d);
        for (auto& value : d) value *= 0.8;
        compare(fixture, basis, grid, d);
        fixture.canary();
        std::cout << "density route=" << route << " functional=" << functional
                  << " spins=" << fixture.layout.spins << " passed\n";
      }
  // A library-capable strict audit must not replace the distinct mixed rule.
  Fixture mixed(basis, grid, 1, false, 17);
  xc_density_provider_for_test(true, false);
  mixed.plan->prepare_density(
      generativeqc::runtime::fp32_compute_fp64_accumulation("dft.cuda.auto/density-contraction-v1"),
      10, 128ULL << 20);
  xc_density_provider_for_test(false, false);
  require(mixed.plan->density_binding(generativeqc::runtime::PrecisionPhase::Admitted)
                      .candidate.provider == "generated.cuda" &&
              mixed.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit)
                      .candidate.provider == "cublas",
          "mixed arithmetic was silently implemented by GEMM");
  compare(mixed, basis, grid, density(basis.nao, 1));
  Fixture control(basis, grid, 1, false, 17);
  control.plan->prepare_density(generativeqc::runtime::fp32_compute_fp64_accumulation(
      "dft.cuda.auto/density-contraction-v1"));
  const auto d = density(basis.nao, 1);
  mixed.submit(d, generativeqc::runtime::PrecisionPhase::Admitted);
  control.submit(d, generativeqc::runtime::PrecisionPhase::Admitted);
  require(mixed.scalars().energy == control.scalars().energy &&
              mixed.potential() == control.potential(),
          "optional provider changed mixed execution");
  xc_density_provider_for_test(true, false);
  density_provider_qualification_budget = 128ULL << 20;
  local_ao_cases();
  for (unsigned functional : {0U, 1U}) matrix_response_case(basis, grid, functional, true, 19);
  density_provider_qualification_budget = 0;
  xc_density_provider_for_test(false, false);
}
