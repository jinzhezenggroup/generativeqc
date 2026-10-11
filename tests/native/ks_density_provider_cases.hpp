// Included in the KS test namespace. Exercise the ordinary native owner with
// explicit caller reservations, including the shared numeric allocation ledger.
void ks_density_provider_cases() {
  const char* previous = std::getenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
  const bool had_previous = previous != nullptr;
  const std::string saved = previous ? previous : "";
  struct Restore {
    const char* variable;
    bool present;
    std::string value;
    ~Restore() {
      if (present)
        ::setenv(variable, value.c_str(), 1);
      else
        ::unsetenv(variable);
      xc_density_provider_for_test(false, false);
    }
  } restore{"GENERATIVEQC_CUDA_KS_ACTIVE_AO", had_previous, saved};
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 200;
  options.semilocal_exchange_scale = 0.75;
  options.semilocal_correlation_scale = 1.0;
  for (bool restricted : {true, false}) {
    auto system = restricted ? water() : hydrogens(3, false);
    std::vector<double> moved_seed;
    for (unsigned geometry = 0; geometry != 2; ++geometry) {
      if (geometry) system.atoms[1].position[2] += 0.02;
      const dft::AoBasis basis(system);
      const dft::MolecularGrid grid(system, {1, 12, 8, 16, 3, 1e-12});
      const scf::PreparedFockPlan cpu(system, nullptr,
                                      exact_exchange_strategy(restricted, scf::FockBackend::Cpu));
      const auto reference = restricted ? scf::run_pbe_rks(cpu, basis, grid, options)
                                        : scf::run_uks(cpu, basis, grid, options, true);
      require(reference.converged, "density-provider PBE0 CPU oracle did not converge");
      const scf::PreparedFockPlan gpu(
          system, nullptr, exact_exchange_strategy(restricted, scf::FockBackend::Cuda), 0);
      require(::setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "0", 1) == 0, "set dense XC");
      std::size_t baseline_bytes{};
      {
        // Exhaustion must reserve incumbent storage only: optional point panels
        // would otherwise create headroom for the density cache under test.
        const auto* previous_budget = std::getenv("GENERATIVEQC_CUDA_XC_BATCH_BYTES");
        Restore budget_restore{"GENERATIVEQC_CUDA_XC_BATCH_BYTES", previous_budget != nullptr,
                               previous_budget ? previous_budget : ""};
        require(::setenv("GENERATIVEQC_CUDA_XC_BATCH_BYTES", "0", 1) == 0,
                "disable optional residency in the ledger baseline");
        dft::CudaKsPlan baseline(gpu, basis, grid, options, dft::SemilocalFamily::Pbe, 257);
        baseline_bytes =
            baseline.resources().state_device_bytes + baseline.resources().xc_device_bytes;
        require(!baseline.density_provider_diagnostic(), "default owner reserved a provider");
      }
      const auto geometry_seed = moved_seed;
      for (unsigned route = 0; route != 7; ++route) {
        // 0=unqualified, 1=qualified, 2=device budget, 3=host budget,
        // 4=unavailable, 5=local-map request under the public ledger,
        // 6=actual numeric-ledger exhaustion.
        require(::setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", route == 5 ? "1" : "0", 1) == 0,
                "select density-provider AO domain");
        dft::CudaXcPreparationBudget budget{128ULL << 20, 16U << 10};
        if (route == 2) budget.device_bytes = 1;
        if (route == 3) budget.host_bytes = 1;
        auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
        ledger->device = 0;
        ledger->limit = route == 6 ? baseline_bytes : 512ULL << 20;
        const auto previous_ledger = runtime::active_device_resource_ledger;
        runtime::active_device_resource_ledger = ledger;
        try {
          {
            xc_density_provider_for_test(route != 0, route == 4);
            const auto preparing = std::chrono::steady_clock::now();
            dft::CudaKsPlan plan(gpu, basis, grid, options, dft::SemilocalFamily::Pbe, 257, nullptr,
                                 nullptr, dft::nlc::Vv10DensityDomain::StrictPositive, budget);
            const auto prepared = std::chrono::steady_clock::now();
            xc_density_provider_for_test(false, false);
            const auto* selected = plan.density_provider_diagnostic();
            require((selected != nullptr) == (route != 3), "host reservation was not enforced");
            // The public ledger does not reserve optional AO maps. Route 5
            // remains dense despite its explicit request, so the qualified
            // dense provider has the same admission and charge as route 1.
            const bool expect_library = route == 1 || route == 5;
            require((selected && selected->candidate.provider == "cublas") == expect_library,
                    "PBE0 density provider admission route");
            if (selected) {
              require(
                  selected->host_bytes <= budget.host_bytes &&
                      selected->matrix_bytes + selected->provider_allowance <= budget.device_bytes,
                  "prepared resources exceeded caller reservation");
              const auto expected =
                  expect_library ? (restricted ? 1U : 2U) * basis.nao * basis.nao * sizeof(double)
                                 : 0;
              require(selected->matrix_bytes == expected, "density matrix cache charge");
            }
            require(ledger->live ==
                        plan.resources().state_device_bytes + plan.resources().xc_device_bytes,
                    "PBE0 density cache escaped the numeric resource ledger");
            const auto starting = std::chrono::steady_clock::now();
            const auto cold = plan.run(geometry ? &geometry_seed : nullptr, false);
            const auto cold_done = std::chrono::steady_clock::now();
            const auto warm = plan.run();
            const auto warm_done = std::chrono::steady_clock::now();
            if (route == 5)
              require(!cold.dft_diagnostic.cuda_ao_selection.selected &&
                          !warm.dft_diagnostic.cuda_ao_selection.selected,
                      "public ledger unexpectedly admitted optional AO maps");
            require(cold.converged && warm.converged &&
                        std::abs(cold.energy - reference.energy) < 1e-9 &&
                        std::abs(warm.energy - reference.energy) < 1e-9 &&
                        cold.physical_residual_rms < 1e-9 && warm.physical_residual_rms < 1e-9,
                    "prepared density changed the complete PBE0 endpoint");
            require(cold.density.size() == reference.density.size(), "PBE0 density extent");
            for (std::size_t i = 0; i != cold.density.size(); ++i)
              require(std::abs(cold.density[i] - reference.density[i]) < 2e-8,
                      "PBE0 provider density disagrees with CPU");
            if (route == 1) moved_seed = cold.density;
            std::cout << "PBE0 density route=" << route << " restricted=" << restricted
                      << " geometry=" << geometry << " cold_iterations=" << cold.iterations
                      << " warm_iterations=" << warm.iterations
                      << " energy_error=" << std::abs(cold.energy - reference.energy)
                      << " nao=" << basis.nao << " points=" << grid.point_count() << " prepare_s="
                      << std::chrono::duration<double>(prepared - preparing).count()
                      << " cold_s=" << std::chrono::duration<double>(cold_done - starting).count()
                      << " warm_s=" << std::chrono::duration<double>(warm_done - cold_done).count()
                      << " provider="
                      << (selected ? selected->candidate.provider : "generated.cuda")
                      << " matrix_bytes=" << (selected ? selected->matrix_bytes : 0)
                      << " allowance=" << (selected ? selected->provider_allowance : 0) << '\n';
          }
          require(ledger->live == 0, "PBE0 density resources survived owner destruction");
          require((ledger->rejected > 0) == (route == 6), "numeric ledger fallback not exercised");
        } catch (...) {
          runtime::active_device_resource_ledger = previous_ledger;
          throw;
        }
        runtime::active_device_resource_ledger = previous_ledger;
      }
    }
  }
}
