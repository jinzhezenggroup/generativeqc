/** Public Schwarz preparation: tight diagonal values, batch layout and fallback.
 * Included by the provider fixture to use its independent CPU integral owner. */
void public_schwarz_preparation() {
  constexpr const char* selector = "GENERATIVEQC_DIRECT_SPECIALIZED_SCHWARZ";
  const auto* previous = std::getenv(selector);
  const bool had_previous = previous != nullptr;
  const std::string saved = previous ? previous : "";
  struct Restore {
    const char* name;
    bool present;
    const std::string& value;
    ~Restore() {
      if (present)
        setenv(name, value.c_str(), 1);
      else
        unsetenv(name);
    }
  } restore{selector, had_previous, saved};
  constexpr const char* ordering = "GENERATIVEQC_DIRECT_ORDERED_SCHWARZ";
  const auto* previous_order = std::getenv(ordering);
  const bool had_order = previous_order != nullptr;
  const std::string saved_order = previous_order ? previous_order : "";
  Restore restore_order{ordering, had_order, saved_order};

  for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
    generativeqc::core::System first;
    first.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
    // Signed contractions exercise cancellation before sqrt(abs(diagonal)).
    // f shells retain the generic recurrence in both preparation schedules.
    first.shells = {{0, 0, {{0.8, 0.7}, {0.2, -0.15}}},
                    {1, 1, {{0.6, 1.0}}},
                    {0, 2, {{0.7, 1.0}}},
                    {1, 3, {{0.9, 1.0}}}};
    first.electron_count = 2;
    first.basis_representation = representation;
    auto second = first;
    second.atoms[1].position[0] += 0.17;
    second.shells[1].primitives[0].exponent = 0.4;
    std::string detail;
    for (auto* system : {&first, &second})
      require(generativeqc::molecule::validate_and_normalize(*system, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
    const auto a = generativeqc::integrals::build_integrals(first, false);
    const auto b = generativeqc::integrals::build_integrals(second, false);
    const auto n = a.nbf, matrix = n * n;
    std::vector<double> reference(2 * matrix), ordinary;
    for (std::size_t item = 0; item < 2; ++item) {
      const auto& eri = item ? b.eri : a.eri;
      for (std::size_t ij = 0; ij < matrix; ++ij)
        reference[item * matrix + ij] = std::sqrt(std::abs(eri[ij * matrix + ij]));
    }
    // Exact mandatory capacity tests that specialization adds no resident
    // metadata and does not silently depend on the optional generated owner.
    const auto capacity = cuda_direct_jk_device_bytes(2, n, 4, 8, 10, 0);
    for (const char* order_choice : {"0", "1"})
      for (const char* choice : {"0", "1"}) {
        require(setenv(ordering, order_choice, 1) == 0, "cannot select Schwarz lane ordering");
        require(setenv(selector, choice, 1) == 0, "cannot select Schwarz schedule");
        CudaDirectJkPlan* raw{};
        CudaDirectJkDiagnostic diagnostic;
        require(create_cuda_direct_jk_plan(0, {first, second}, 0, 1e-12, capacity, &raw, diagnostic,
                                           detail) == GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());
        std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
            raw, &destroy_cuda_direct_jk_plan);
        require(diagnostic.device_bytes == capacity, "Schwarz capacity changed");
        std::vector<double> bounds(2 * matrix);
        check(cudaMemcpy(bounds.data(), plan->bounds, bounds.size() * sizeof(double),
                         cudaMemcpyDeviceToHost));
        for (std::size_t ij = 0; ij < bounds.size(); ++ij) {
          require(std::isfinite(bounds[ij]) &&
                      std::abs(bounds[ij] - reference[ij]) <= 2e-11 * (1 + reference[ij]),
                  "public Schwarz diagonal differs from independent CPU integrals");
          if (!ordinary.empty())
            require(std::abs(bounds[ij] - ordinary[ij]) <= 2e-12 * (1 + ordinary[ij]),
                    "specialized Schwarz differs from ordinary preparation");
        }
        ordinary = bounds;
        // Poison one primitive only after normal admission to exercise the
        // device failure flag, rather than merely host input validation.
        const double invalid = std::numeric_limits<double>::quiet_NaN();
        check(cudaMemcpy(const_cast<double*>(plan->batch.primitive_coefficients), &invalid,
                         sizeof(double), cudaMemcpyHostToDevice));
        launch_independent_jk_bounds_kernel(static_cast<unsigned>((bounds.size() + 127) / 128), 128,
                                            0, plan->stream, plan->batch, plan->bounds,
                                            plan->numerical_failure);
        check(cudaGetLastError());
        int failed = 0;
        check(cudaMemcpyAsync(&failed, plan->numerical_failure, sizeof(failed),
                              cudaMemcpyDeviceToHost, plan->stream));
        check(cudaStreamSynchronize(plan->stream));
        require(failed == 1, "nonfinite Schwarz diagonal did not fail closed");
      }
  }
  std::cout << "CUDA public Schwarz s/p/d/f, spherical/Cartesian, batch, capacity and NaN PASS\n";
}
