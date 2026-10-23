/** Exercise the raw-only pure-J representation against independent CPU
 * integrals, including replay after force scratch use and rank-truncated
 * fallback. The environment guard restores the caller's execution policy.
 */
void raw_coulomb_cases() {
  struct Environment {
    const char* name;
    bool present;
    std::string previous;
    Environment(const char* key, const char* value)
        : name(key),
          present(std::getenv(key) != nullptr),
          previous(present ? std::getenv(key) : "") {
      require(::setenv(key, value, 1) == 0, "cannot configure raw Coulomb fixture");
    }
    ~Environment() {
      if (present)
        ::setenv(name, previous.c_str(), 1);
      else
        ::unsetenv(name);
    }
  } environment("GENERATIVEQC_DF_RAW_COULOMB", "0"),
      storage("GENERATIVEQC_DF_VALUE_STORAGE", "packed-single");
  for (bool redundant : {false, true}) {
    for (double displacement : {0.0, 0.017}) {
      auto orbital = fixture(true);
      orbital.atoms[1].position[2] += displacement;
      auto auxiliary = orbital;
      if (redundant) auxiliary.shells.push_back(auxiliary.shells[0]);
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(auxiliary, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      auto spec = make_hf_fock_spec(FockSpin::Restricted);
      spec.coulomb.approximation = FockApproximation::DensityFitted;
      spec.exchange.present = false;
      spec.derivative_order = 1;
      constexpr double cutoff = 1e-8;
      const auto cpu_strategy = resolve_fock_build(spec, FockBackend::Cpu, 0.0, cutoff);
      const auto gpu_strategy = resolve_fock_build(spec, FockBackend::Cuda, 0.0, cutoff);
      const PreparedFockPlan cpu(orbital, &auxiliary, cpu_strategy);
      const auto n = cpu.one_electron().nbf;
      std::vector<double> density(n * n);
      for (const char* control : {"0", "1"}) {
        require(::setenv("GENERATIVEQC_DF_RAW_COULOMB", control, 1) == 0,
                "cannot configure raw Coulomb test");
        const PreparedFockPlan gpu(orbital, &auxiliary, gpu_strategy, 0, 64U << 20);
        const auto& diagnostic = gpu.diagnostic().fitted.at(0);
        const bool raw = control[0] == '1' && !redundant;
        require((diagnostic.raw_factor_bytes != 0) == raw &&
                    (diagnostic.stored_factor_bytes == 0) == raw,
                "raw Coulomb admission or rank fallback differs from its storage contract");
        const auto bytes = gpu.diagnostic().device_bytes;
        for (unsigned replay = 0; replay < 3; ++replay) {
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
              density[i * n + j] =
                  (i == j ? 0.4 : 0.0) + 0.03 * std::cos((i + j + 1) * (replay + 1));
          matrix(gpu.build(density).coulomb, cpu.build(density).coulomb,
                 "raw Coulomb independent J or changed-density replay");
          matrix(gpu.energy_derivative(density), cpu.energy_derivative(density),
                 "raw Coulomb independent metric/three-center response");
          matrix(gpu.build(density).coulomb, cpu.build(density).coulomb,
                 "force response damaged retained raw Coulomb values");
          if (raw) {
            // A low-level caller can still request K from a pure-J owner.
            // Reject both dense and factor dispatch before A is mistaken for B,
            // then prove the failed call did not damage the supported J path.
            std::vector<double> rejected_j, rejected_k;
            std::vector<std::uint8_t> selected;
            require(execute_cuda_density_fitting_rhf_jk(
                        gpu.cuda_fitted_source(), density, rejected_j, rejected_k, detail,
                        {false, true}) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                    "raw Coulomb owner accepted dense fitted exchange");
            require(execute_cuda_density_fitting_occupied_exchange(
                        gpu.cuda_fitted_source(), density, DensityFactorSpin::Restricted, {},
                        rejected_k, selected, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                    "raw Coulomb owner accepted occupied fitted exchange");
            matrix(gpu.build(density).coulomb, cpu.build(density).coulomb,
                   "rejected exchange damaged the raw Coulomb owner");
          }
          require(gpu.diagnostic().device_bytes == bytes,
                  "raw Coulomb replay changed retained storage");
        }
      }
    }
  }
  // Invalid explicit controls are errors, not a silent policy fallback.
  require(::setenv("GENERATIVEQC_DF_RAW_COULOMB", "invalid", 1) == 0,
          "cannot configure raw Coulomb rejection");
  const auto system = fixture();
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  spec.exchange.present = false;
  rejected(
      [&] {
        PreparedFockPlan plan(system, &system, resolve_fock_build(spec, FockBackend::Cuda), 0);
      },
      "invalid raw Coulomb policy was accepted");
}
