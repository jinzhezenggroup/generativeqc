/** Cold fitted KS uses an algebraic density factor without publishing an
 * orbital lease. The independent CPU provider owns the scientific reference;
 * the optional same-density audit additionally compares GPU occupied/dense K.
 */
void ks_cold_factor_cases() {
  struct Environment {
    const char* name;
    bool present;
    std::string previous;
    Environment(const char* key, const char* value)
        : name(key),
          present(std::getenv(key) != nullptr),
          previous(present ? std::getenv(key) : "") {
      require(::setenv(key, value, 1) == 0, "cannot configure cold-factor test");
    }
    ~Environment() {
      if (present)
        ::setenv(name, previous.c_str(), 1);
      else
        ::unsetenv(name);
    }
  } factor("GENERATIVEQC_CUDA_KS_COLD_FACTOR", "1"),
      audit("GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT", "1");
  for (double displacement : {0.0, 0.001}) {
    auto system = water();
    system.atoms[1].position[2] += displacement;
    // The library eigensolver begins at 17 AOs. Distinct polarization/diffuse
    // shells exercise a nonorthogonal metric without an ill-conditioned copy.
    system.shells.push_back({0, 2, {{0.8, 1.0}}});
    system.shells.push_back({0, 1, {{0.2, 1.0}}});
    for (unsigned atom : {1U, 2U}) system.shells.push_back({atom, 0, {{0.12, 1.0}}});
    std::string detail;
    require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const dft::AoBasis basis(system);
    const dft::MolecularGrid grid(system, {1, 24, 12, 24, 3, 1e-12});
    const scf::PreparedFockPlan cpu(system, &system,
                                    fitted_exchange_strategy(true, scf::FockBackend::Cpu));
    const scf::PreparedFockPlan gpu(system, &system,
                                    fitted_exchange_strategy(true, scf::FockBackend::Cuda), 0);
    scf::ScfOptions options;
    options.compute_forces = false;
    options.semilocal_exchange_scale = 0.75;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-10;
    options.max_iterations = 200;
    options.strict_initial_density = true;
    const auto reference = scf::run_pbe_rks(cpu, basis, grid, options);
    require(reference.converged, "cold-factor CPU oracle did not converge");
    const auto x = scf::reference::symmetric_orthogonalizer(cpu.one_electron().overlap, basis.nao);
    const auto ensemble = scf::reference::density_from_orbitals(
        x, basis.nao, basis.nao, double(system.electron_count) / basis.nao);
    dft::CudaKsPlan plan(gpu, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
    const auto bytes = plan.resources().state_device_bytes;
    // Native determinant, explicit opt-out, rank-full native proposal, and
    // caller-supplied determinant: only the first may recover a cold factor.
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
      require(::setenv("GENERATIVEQC_CUDA_KS_COLD_FACTOR", scenario == 1 ? "0" : "1", 1) == 0,
              "cannot select cold factor rollback");
      const Matrix* input = scenario == 2   ? &ensemble
                            : scenario == 3 ? &reference.density
                                            : nullptr;
      const auto origin =
          scenario == 3 ? dft::CudaKsSeedOrigin::External : dft::CudaKsSeedOrigin::NativeColdGuess;
      const auto before = plan.transfers();
      plan.begin(input, false, origin);
      dft::CudaKsFinalStateToken token;
      require(plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "algebraic seed published a final-state token");
      plan.enqueue_iteration();
      plan.finish_iteration();
      const auto first = plan.transfers();
      require(first.fitted_occupied_exchange_builds - before.fitted_occupied_exchange_builds ==
                      (scenario == 0 ? 1U : 0U) &&
                  first.fitted_dense_exchange_builds - before.fitted_dense_exchange_builds ==
                      (scenario == 0 ? 0U : 1U),
              "cold density proof or external/ensemble fallback selected the wrong first K");
      while (plan.active()) {
        plan.enqueue_iteration();
        plan.finish_iteration();
      }
      const auto result = plan.result();
      require(result.converged && std::abs(result.energy - reference.energy) < 1e-10 &&
                  plan.resources().state_device_bytes == bytes,
              "cold-factor recovery changed the endpoint or retained device storage");
      physical_check(cpu, basis, grid, 1U, result, options.semilocal_exchange_scale);
    }
    // A failed begin revokes the preceding final state; its independent warm
    // snapshot remains available, and a fresh cold retry must prove its own D.
    const auto warm = plan.warm_density();
    ks_cuda_fail_next_runtime_for_test_v1();
    bool failed = false;
    try {
      plan.begin(nullptr, false);
    } catch (const std::exception&) {
      failed = true;
    }
    dft::CudaKsFinalStateToken token;
    require(failed && plan.failed() && plan.warm_density() == warm &&
                plan.final_state_token(token, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "failed cold begin retained a token or damaged last-good warm storage");
    require(plan.run(nullptr, false).converged, "cold-factor retry failed after injected error");
  }
}
