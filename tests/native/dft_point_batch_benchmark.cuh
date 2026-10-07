/** Scheduling-only fixed-density XC endpoints, not SCF/force promotion evidence.
 * Keep 256-point maps, FP64 canonical points and generated contractions in both
 * arms. Discovery, optional preparation, publication and work are explicit. */
void point_batch_benchmark(const char* original, const char* moved) {
  using namespace mapped_density_test;
  constexpr std::size_t tile = 256;
  for (unsigned geometry = 0; geometry < 2; ++geometry) {
    const auto molecule = read_system(geometry ? moved : original);
    const auto geometry_begin = std::chrono::steady_clock::now();
    const AoBasis basis(molecule);
    const auto grid = MolecularGrid::from_cuda(molecule, {1, 48, 16, 32, 3, 1e-12}, 0, true);
    require(grid.point_count() % tile == 0, "point batch work census requires full tiles");
    const auto geometry_seconds = elapsed(geometry_begin);
    const auto d = density(basis.nao, 1);
    std::array<std::unique_ptr<Fixture>, 2> fixtures;
    std::array<double, 2> preparation{}, batch_preparation{};
    for (unsigned route = 0; route < 2; ++route) {
      const auto started = std::chrono::steady_clock::now();
      fixtures[route] = std::make_unique<Fixture>(
          basis, grid, 1U, false, tile, CudaXcAoPrecision::Fp64, false, 0.75, 1.0, nullptr, true);
      auto& plan = *fixtures[route]->plan;
      const auto admission = cuda_xc_ao_selection_resources(plan.layout());
      require(plan.select_local_ao(1e-16, admission.host_peak_bytes), "point batch map discovery");
      const auto batch_started = std::chrono::steady_clock::now();
      plan.prepare_point_batches(route ? 32 : 1, 32ULL << 20);
      batch_preparation[route] = elapsed(batch_started);
      preparation[route] = elapsed(started);
    }
    const auto& baseline = fixtures[0]->plan->ao_selection_work();
    const auto& candidate = fixtures[1]->plan->ao_selection_work();
    require(baseline.point_ao_square_sum == candidate.point_ao_square_sum &&
                baseline.point_ao_visits == candidate.point_ao_visits &&
                baseline.empty_tiles == candidate.empty_tiles,
            "point batching changed selected work");
    std::array<std::array<double, 6>, 2> samples{};
    double energy_error = 0, potential_error = 0;
    for (unsigned sample = 0; sample < 6; ++sample) {
      std::array<std::vector<double>, 2> potentials;
      std::array<double, 2> energies{};
      for (unsigned step = 0; step < 2; ++step) {
        const auto route = (step + sample) % 2;
        const auto started = std::chrono::steady_clock::now();
        auto& fixture = *fixtures[route];
        fixture.submit(d);
        const auto scalars = fixture.scalars();
        require(scalars.error == 0, "invalid point batch endpoint");
        potentials[route] = fixture.potential();
        energies[route] = scalars.energy;
        samples[route][sample] = elapsed(started);
      }
      energy_error = std::max(energy_error, std::abs(energies[0] - energies[1]));
      close(energies[0], energies[1], "point batch energy parity", 1e-8);
      for (std::size_t entry = 0; entry < potentials[0].size(); ++entry) {
        potential_error =
            std::max(potential_error, std::abs(potentials[0][entry] - potentials[1][entry]));
        close(potentials[0][entry], potentials[1][entry], "point batch potential parity", 1e-8);
      }
    }
    for (unsigned route = 0; route < 2; ++route) {
      const auto& fixture = *fixtures[route];
      const auto& plan = *fixture.plan;
      const auto& batch = plan.point_batch_plan();
      const auto& work = plan.ao_selection_work();
      const auto& movement = plan.transfers();
      const auto submissions = 1 + (grid.point_count() - 1) / (batch.tiles * tile);
      auto ordered = samples[route];
      std::sort(ordered.begin() + 1, ordered.end());
      std::cout << std::setprecision(12) << "point_batch_endpoint atoms=" << molecule.atoms.size()
                << " nao=" << basis.nao << " geometry=" << geometry << " route=" << route
                << " points=" << grid.point_count() << " tile_points=" << tile
                << " tiles_per_submission=" << batch.tiles
                << " point_submissions_per_evaluation=" << submissions << " max_point_ctas="
                << (std::min(batch.tiles * tile, grid.point_count()) + 31) / 32
                << " point_threads=32 tiles=" << work.tiles << " empty_tiles=" << work.empty_tiles
                << " selected_point_ao_visits=" << work.point_ao_visits
                << " selected_point_ao_square=" << work.point_ao_square_sum
                << " dense_point_ao_square=" << work.dense_point_ao_square_sum
                << " ao_jet_values_per_evaluation=" << work.point_ao_visits * plan.layout().jets
                << " density_submissions_per_evaluation=" << work.tiles - work.empty_tiles
                << " potential_contraction_submissions_per_evaluation="
                << work.tiles - work.empty_tiles
                << " empty_total_submissions_per_evaluation=" << work.empty_tiles
                << " matrix_scatter_elements_per_evaluation=" << work.point_ao_square_sum / tile
                << " geometry_prepare_s=" << geometry_seconds << " prepare_s=" << preparation[route]
                << " discovery_s=" << work.discovery_seconds
                << " batch_prepare_s=" << batch_preparation[route]
                << " original_arena_bytes=" << fixture.allocation_bytes
                << " additional_batch_bytes=" << batch.device_bytes
                << " retained_arena_bytes=" << fixture.allocation_bytes + batch.device_bytes
                << " host_map_peak_bytes=" << work.host_peak_bytes
                << " setup_h2d_bytes=" << movement.setup_h2d_bytes
                << " discovery_d2h_bytes=" << work.discovery_d2h_bytes
                << " input_h2d_bytes=" << movement.evaluations * d.size() * sizeof(double)
                << " input_upload_synchronizations=" << movement.evaluations
                << " discovery_synchronizations=" << work.tiles
                << " output_d2h_bytes=" << movement.output_d2h_bytes
                << " synchronizations=" << movement.synchronizations
                << " evaluations=" << movement.evaluations
                << " potential_calls=" << movement.potential_calls
                << " potential_summands=" << movement.potential_summands
                << " point_gathers=0 descriptor_transfers=0 concurrent_scatters=0"
                << " cold_s=" << samples[route][0] << " warm_median_s=" << ordered[3]
                << " max_energy_error=" << energy_error << " max_v_error=" << potential_error;
      for (std::size_t sample = 0; sample < samples[route].size(); ++sample)
        std::cout << " sample" << sample << "_s=" << samples[route][sample];
      std::cout << std::endl;
      fixtures[route]->canary();
    }
  }
}
