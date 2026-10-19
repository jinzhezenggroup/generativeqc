/** Opt-in actual-domain diagnostic, never an automatic provider promotion.
 * The caller supplies geometry/basis plus a binary public-AO FP64 density.
 * Compare complete fixed-density r2SCAN XC, including map gathers, finite
 * publication and E/V downloads, with identical scientific input in all arms.
 */
void r2scan_provider_benchmark(const char* molecule_path, const char* density_path,
                               bool spatial_order = false) {
  using namespace mapped_density_test;
  const auto molecule = read_system(molecule_path);
  const AoBasis basis(molecule);
  const auto grid = MolecularGrid::from_cuda(molecule, {1, 48, 16, 32, 3, 1e-12}, 0, true);
  std::vector<double> d(basis.nao * basis.nao);
  std::ifstream input(density_path, std::ios::binary);
  require(bool(input.read(reinterpret_cast<char*>(d.data()), d.size() * sizeof(double))),
          "missing binary public-AO density");
  require(input.peek() == std::char_traits<char>::eof(), "density extent mismatch");
  std::vector<double> reference;
  double reference_energy{};
  for (std::size_t tile : {256U, 1024U}) {
    for (unsigned route = 0; route < (spatial_order ? 1U : 4U); ++route) {
      const auto started = std::chrono::steady_clock::now();
      Fixture fixture(basis, grid, 2U, false, tile, CudaXcAoPrecision::Fp64, false, 1.0, 1.0,
                      nullptr, true, spatial_order);
      if (spatial_order) {
        const auto bytes = grid.point_count() * 4 * sizeof(double);
        const auto host_bytes =
            bytes + (2 * grid.point_count() + 4 * basis.natom) * sizeof(std::size_t);
        require(fixture.plan->prepare_spatial_grid(grid, host_bytes, bytes),
                "diagnostic spatial grid resource admission");
      }
      auto& plan = *fixture.plan;
      const auto admission = cuda_xc_ao_selection_resources(plan.layout());
      require(plan.select_local_ao(1e-16, admission.host_peak_bytes), "r2SCAN map admission");
      const bool density_library = route & 1U, potential_library = route & 2U;
      xc_density_indexed_provider_for_test(density_library);
      plan.prepare_density(generativeqc::runtime::strict_fp64_precision(), 6, 128ULL << 20);
      xc_density_indexed_provider_for_test(false);
      xc_potential_indexed_qualification_for_test(potential_library);
      plan.prepare_potential(128ULL << 20);
      xc_potential_indexed_qualification_for_test(false);
      require((plan.density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit)
                   .candidate.provider == "cublas") == density_library,
              "density diagnostic provider unavailable");
      require(bool(plan.potential_lowering().provider_allowance) == potential_library,
              "potential diagnostic provider unavailable");
      // The production point budget scales with four meta-GGA work jets.
      // Existing providers intentionally retain the serial mapped contraction
      // path; count their whole cost instead of timing GEMM in isolation.
      plan.prepare_point_batches(32, 128ULL << 20, true, true);
      const auto preparation = elapsed(started);
      std::array<double, 6> samples{};
      double energy_error = 0, potential_error = 0;
      for (auto& seconds : samples) {
        const auto begin = std::chrono::steady_clock::now();
        fixture.submit(d);
        const auto scalar = fixture.scalars();
        const auto potential = fixture.potential();
        seconds = elapsed(begin);
        require(scalar.error == 0, "r2SCAN finite publication failed");
        if (reference.empty()) {
          reference = potential;
          reference_energy = scalar.energy;
          std::ofstream output(std::string(density_path) + ".native-vxc.bin", std::ios::binary);
          require(bool(output.write(reinterpret_cast<const char*>(reference.data()),
                                    reference.size() * sizeof(double))),
                  "cannot retain native XC comparison matrix");
        }
        energy_error = std::max(energy_error, std::abs(scalar.energy - reference_energy));
        close(scalar.energy, reference_energy, "r2SCAN provider energy parity", 1e-8);
        for (std::size_t i = 0; i < potential.size(); ++i) {
          potential_error = std::max(potential_error, std::abs(potential[i] - reference[i]));
          close(potential[i], reference[i], "r2SCAN provider potential parity", 1e-8);
        }
      }
      const auto& work = plan.ao_selection_work();
      auto ordered = samples;
      std::sort(ordered.begin() + 1, ordered.end());
      std::cout << std::setprecision(12) << "r2scan_provider spatial_order=" << spatial_order
                << " nao=" << basis.nao << " points=" << grid.point_count() << " tile=" << tile
                << " route=" << route << " density_library=" << density_library
                << " potential_library=" << potential_library << " prepare_s=" << preparation
                << " median_s=" << ordered[3] << " active_sum=" << work.active_sum
                << " max_active=" << work.max_active << " empty_tiles=" << work.empty_tiles
                << " ao_visits=" << work.point_ao_visits
                << " point_ao_square=" << work.point_ao_square_sum
                << " batch_bytes=" << plan.point_batch_plan().device_bytes
                << " reference_energy=" << reference_energy << " max_energy_error=" << energy_error
                << " max_potential_error=" << potential_error;
      for (std::size_t i = 0; i < samples.size(); ++i)
        std::cout << " sample" << i << "_s=" << samples[i];
      std::cout << std::endl;
      fixture.canary();
    }
  }
}
