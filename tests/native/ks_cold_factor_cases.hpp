// Included inside test_ks_cuda.cpp's anonymous namespace to reuse the real
// provider fixtures and independent CPU physical-state oracle.
void run_density_fitted_cold_factor_case() {
  using namespace scf::initial_guess;
  struct Environment {
    const char* name;
    bool present;
    std::string value;
    explicit Environment(const char* key)
        : name(key), present(std::getenv(key) != nullptr), value(present ? std::getenv(key) : "") {}
    ~Environment() {
      if (present)
        ::setenv(name, value.c_str(), 1);
      else
        ::unsetenv(name);
    }
  } control("GENERATIVEQC_CUDA_KS_COLD_FACTOR"), audit("GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT");
  // Audit the actual first occupied K at its admitted D, before SCF can hide
  // a representation error by converging to the same final determinant.
  ::setenv(audit.name, "1", 1);
  const auto system = hydrogens(2, true);
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 24, 12, 24, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  const scf::PreparedFockPlan cpu(system, &system,
                                  fitted_exchange_strategy(true, scf::FockBackend::Cpu));
  const scf::PreparedFockPlan gpu(system, &system,
                                  fitted_exchange_strategy(true, scf::FockBackend::Cuda), 0);
  scf::ScfOptions options;
  options.semilocal_exchange_scale = 0.75;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 200;
  dft::CudaKsPlan plan(gpu, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
  PreliminaryOptions policy;
  policy.kind = PreliminaryKind::Huckel;
  PreliminaryDiagnostic diagnostic;
  NativeColdDensityFactor factor;
  const auto seed =
      prepare_preliminary_density(gpu, policy, diagnostic, plan.seed_eigen_operation(), &factor);
  require(seed && diagnostic.outcome == PreliminaryOutcome::Used && factor.target == &gpu &&
              factor.nbf == basis.nao && factor.occupied == 1 &&
              factor.coefficients.size() == basis.nao,
          "native Huckel preparation did not publish its density-generating factor");
  const auto reference = scf::run_pbe_rks(cpu, basis, grid, options);
  require(reference.converged, "independent CPU DF cold-factor oracle failed");
  const auto retained = plan.resources().state_device_bytes;
  const auto run = [&](const char* selection, const NativeColdDensityFactor* candidate,
                       const std::vector<double>* density, dft::CudaKsSeedOrigin origin,
                       bool expected, bool ledger_bound = false) {
    if (selection)
      ::setenv(control.name, selection, 1);
    else
      ::unsetenv(control.name);
    const auto before = plan.transfers();
    // Public bounded owners reserve no optional cold packing. Exercise the
    // same thread-local admission boundary without allocating provider memory.
    const auto saved_ledger = runtime::active_device_resource_ledger;
    if (ledger_bound) {
      auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
      ledger->limit = std::numeric_limits<std::size_t>::max();
      runtime::active_device_resource_ledger = std::move(ledger);
      require(!plan.accepts_native_cold_factor(), "bounded owner admitted optional cold work");
    }
    try {
      plan.begin(density, false, origin, candidate);
    } catch (...) {
      runtime::active_device_resource_ledger = saved_ledger;
      throw;
    }
    runtime::active_device_resource_ledger = saved_ledger;
    plan.enqueue_iteration();
    plan.finish_iteration();
    const auto first = plan.transfers();
    require(first.fitted_cold_factor_restores - before.fitted_cold_factor_restores == expected &&
                first.fitted_dense_exchange_builds - before.fitted_dense_exchange_builds ==
                    (expected ? 0U : 1U) &&
                first.fitted_occupied_exchange_builds - before.fitted_occupied_exchange_builds ==
                    (expected ? 1U : 0U),
            "first cold K did not follow the accepted factor/dense fallback");
    while (plan.active()) {
      plan.enqueue_iteration();
      plan.finish_iteration();
    }
    const auto result = plan.result();
    require(result.converged && !plan.failed() &&
                std::abs(result.energy - reference.energy) < 1e-10 &&
                plan.resources().state_device_bytes == retained,
            "cold-factor solve changed the endpoint or retained allocation");
    physical_check(cpu, basis, grid, 1U, result, options.semilocal_exchange_scale);
  };
  const auto native = dft::CudaKsSeedOrigin::NativeColdGuess;
  for (const char* selection : {static_cast<const char*>(nullptr), "1", "0", "invalid"})
    run(selection, &factor, &*seed, native, !selection || std::string(selection) == "1");
  // Reusing this owner deliberately leaves a last-good warm frame present.
  // Neither rejected factors nor explicit seeds may borrow that old frame.
  run(nullptr, nullptr, &*seed, native, false);
  run(nullptr, &factor, &*seed, dft::CudaKsSeedOrigin::External, false);
  run(nullptr, &factor, nullptr, native, false);
  run(nullptr, &factor, &*seed, native, false, true);
  for (unsigned fault = 0; fault < 7; ++fault) {
    auto bad = factor;
    if (fault == 0) bad.target = &cpu;
    if (fault == 1) ++bad.nbf;
    if (fault == 2) ++bad.occupied;
    if (fault == 3) bad.coefficients.pop_back();
    if (fault == 4) bad.coefficients[0] += 0.02;
    if (fault == 5) bad.coefficients[0] = std::numeric_limits<double>::quiet_NaN();
    if (fault == 6) bad.coefficients[0] = std::numeric_limits<double>::max();
    run(nullptr, &bad, &*seed, native, false);
  }
  // Scale both D and its producer C. Normalization must restore the target
  // electron count while preserving the exact occupied representation.
  auto scaled_density = *seed;
  auto scaled_factor = factor;
  for (auto& value : scaled_density) value *= 1.02;
  for (auto& value : scaled_factor.coefficients) value *= std::sqrt(1.02);
  run(nullptr, &scaled_factor, &scaled_density, native, true);
  plan.clear_warm_start();
  run(nullptr, &factor, &*seed, native, true);

  const auto moved_system = hydrogens(2, true, 0.05);
  const dft::AoBasis moved_basis(moved_system);
  const dft::MolecularGrid moved_grid(moved_system, grid_spec);
  const scf::PreparedFockPlan moved_gpu(moved_system, &moved_system,
                                        fitted_exchange_strategy(true, scf::FockBackend::Cuda), 0);
  dft::CudaKsPlan moved(moved_gpu, moved_basis, moved_grid, options, dft::SemilocalFamily::Pbe,
                        257);
  moved.begin(&*seed, false, native, &factor);
  moved.enqueue_iteration();
  moved.finish_iteration();
  require(moved.transfers().fitted_cold_factor_restores == 0 &&
              moved.transfers().fitted_dense_exchange_builds == 1,
          "moved geometry borrowed a cold factor from another target");
  while (moved.active()) {
    moved.enqueue_iteration();
    moved.finish_iteration();
  }
  const scf::PreparedFockPlan moved_cpu(moved_system, &moved_system,
                                        fitted_exchange_strategy(true, scf::FockBackend::Cpu));
  const auto moved_reference = scf::run_pbe_rks(moved_cpu, moved_basis, moved_grid, options);
  require(moved.result().converged && moved_reference.converged &&
              std::abs(moved.result().energy - moved_reference.energy) < 1e-10,
          "moved cold-factor fallback differs from CPU DF");
  std::cout << "Native CUDA DF cold-factor oracle, matrix and fallback gates passed\n";
}
