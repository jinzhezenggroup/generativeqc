/** Actual retained water geometries/bases and full production quadrature.
 * The positive diagnostic density is deliberately fixed, not a converged SCF
 * state. Timings cover complete XC H2D/evaluation/E/V publication, while map
 * discovery and provider setup are reported separately. No default promotion.
 */
namespace mapped_potential_test {
generativeqc::core::System read_system(const char* path) {
  std::ifstream input(path);
  std::size_t atoms{}, shells{};
  require(bool(input >> atoms >> shells) && atoms && shells, "missing geometry/basis header");
  generativeqc::core::System molecule;
  molecule.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  molecule.atoms.resize(atoms);
  for (auto& atom : molecule.atoms) {
    require(bool(input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >>
                 atom.position[2]),
            "missing atom");
    molecule.electron_count += atom.atomic_number;
  }
  molecule.shells.resize(shells);
  for (auto& shell : molecule.shells) {
    std::size_t count{};
    require(bool(input >> shell.atom_index >> shell.angular_momentum >> count), "missing shell");
    shell.primitives.resize(count);
    for (auto& primitive : shell.primitives)
      require(bool(input >> primitive.exponent >> primitive.coefficient), "missing primitive");
  }
  std::string detail;
  require(generativeqc::molecule::validate_and_normalize(molecule, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  return molecule;
}

double elapsed(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace mapped_potential_test

void mapped_potential_benchmark(const char* original, const char* moved) {
  using namespace mapped_potential_test;
  for (unsigned geometry = 0; geometry != 2; ++geometry) {
    const auto molecule = read_system(geometry ? moved : original);
    const auto geometry_begin = std::chrono::steady_clock::now();
    const AoBasis basis(molecule);
    const auto grid = MolecularGrid::from_cuda(molecule, {1, 48, 16, 32, 3, 1e-12}, 0, true);
    const auto geometry_seconds = elapsed(geometry_begin);
    const auto d = density(basis.nao, 1);
    for (std::size_t tile : {128U, 256U, 512U}) {
      // The retained unpruned grid has full tiles. This allows exact scatter
      // traffic from the point-weighted map census without exporting all maps.
      require(grid.point_count() % tile == 0, "scatter census requires full benchmark tiles");
      std::array<std::unique_ptr<Fixture>, 2> fixtures;
      std::array<double, 2> preparation{};
      for (unsigned route = 0; route != 2; ++route) {
        const auto begin = std::chrono::steady_clock::now();
        fixtures[route] = std::make_unique<Fixture>(
            basis, grid, 1U, false, tile, CudaXcAoPrecision::Fp64, false, 0.75, 1.0, nullptr, true);
        auto& plan = *fixtures[route]->plan;
        const auto admission = cuda_xc_ao_selection_resources(plan.layout());
        require(plan.select_local_ao(1e-16, admission.host_peak_bytes),
                "mapped benchmark discovery");
        xc_potential_indexed_qualification_for_test(route == 1);
        plan.prepare_potential(128ULL << 20);
        xc_potential_indexed_qualification_for_test(false);
        preparation[route] = elapsed(begin);
        require((plan.potential_lowering().indexed_candidate.algorithm ==
                 "indexed-symmetric-cross-rank2k") == (route == 1),
                "mapped benchmark did not execute requested qualification");
      }
      const auto& work = fixtures[0]->plan->ao_selection_work();
      const auto& trial_work = fixtures[1]->plan->ao_selection_work();
      require(work.point_ao_square_sum == trial_work.point_ao_square_sum &&
                  work.active_sum == trial_work.active_sum &&
                  work.max_active == trial_work.max_active &&
                  work.empty_tiles == trial_work.empty_tiles,
              "mapped benchmark changed scientific work");
      std::array<std::array<double, 6>, 2> samples{};
      double energy_error{}, potential_error{}, population_error{};
      for (std::size_t sample = 0; sample != samples[0].size(); ++sample) {
        std::array<CudaXcScalars, 2> scalar;
        std::array<std::vector<double>, 2> potential;
        // Alternate order so one candidate does not always inherit the other's
        // device clocks/cache state; change the initial order after geometry.
        for (unsigned order = 0; order != 2; ++order) {
          const auto route = (order + sample + geometry) % 2;
          auto& fixture = *fixtures[route];
          const auto begin = std::chrono::steady_clock::now();
          fixture.submit(d);
          scalar[route] = fixture.scalars();
          potential[route] = fixture.potential();
          samples[route][sample] = elapsed(begin);
          require(scalar[route].error == 0, "mapped benchmark finite publication");
        }
        energy_error = std::max(energy_error, std::abs(scalar[0].energy - scalar[1].energy));
        close(scalar[1].energy, scalar[0].energy, "mapped generated energy parity", 1e-8);
        for (std::size_t i = 0; i != potential[0].size(); ++i) {
          potential_error = std::max(potential_error, std::abs(potential[0][i] - potential[1][i]));
          close(potential[1][i], potential[0][i], "mapped generated potential parity", 1e-9);
        }
        population_error =
            std::max(population_error, std::abs(scalar[0].electrons[0] - scalar[1].electrons[0]));
        close(scalar[1].electrons[0], scalar[0].electrons[0], "mapped generated population parity",
              1e-8);
      }
      for (unsigned route = 0; route != 2; ++route) {
        const auto& fixture = *fixtures[route];
        const auto& plan = *fixture.plan;
        const auto& census = plan.ao_selection_work();
        const auto& resource = plan.potential_lowering();
        const auto& selected = resource.indexed_candidate;
        require(plan.layout().work_jets == 1 && plan.layout().spins == 1,
                "benchmark census assumes one packed GGA jet and one spin");
        auto ordered = samples[route];
        std::sort(ordered.begin() + 1, ordered.end());
        const auto evaluations = plan.transfers().evaluations;
        const auto squares = evaluations * census.point_ao_square_sum / tile;
        const auto triangle = (squares + evaluations * census.active_sum) / 2;
        // Logical scatter accesses: compact load, destination load and two
        // symmetric stores, including the duplicate diagonal store. Count the
        // two index loads separately; these are not measured DRAM bytes.
        std::cout << std::setprecision(12)
                  << "mapped_potential_endpoint atoms=" << molecule.atoms.size()
                  << " nao=" << basis.nao << " geometry=" << geometry << " tile=" << tile
                  << " points=" << grid.point_count() << " provider=" << selected.provider
                  << " algorithm=" << selected.algorithm
                  << " geometry_prepare_s=" << geometry_seconds
                  << " prepare_s=" << preparation[route]
                  << " discovery_s=" << census.discovery_seconds
                  << " provider_prepare_s=" << resource.prepare_seconds
                  << " cold_s=" << samples[route][0] << " warm_median_s=" << ordered[3]
                  << " evaluations=" << evaluations << " tiles=" << census.tiles
                  << " empty_tiles=" << census.empty_tiles << " min_active=" << census.min_active
                  << " max_active=" << census.max_active << " active_sum=" << census.active_sum
                  << " point_ao_square_sum=" << census.point_ao_square_sum
                  << " point_ao_visits=" << census.point_ao_visits
                  << " summands=" << plan.transfers().potential_summands
                  << " calls=" << plan.transfers().potential_calls << " dense_summands="
                  << evaluations * grid.point_count() * basis.nao * (basis.nao + 1)
                  << " rank2k_calls=" << (route ? plan.transfers().potential_calls : 0)
                  << " compact_write_bytes=" << (route ? triangle * sizeof(double) : 0)
                  << " scatter_matrix_logical_bytes=" << (route ? 4 * triangle * sizeof(double) : 0)
                  << " scatter_index_logical_bytes="
                  << (route ? 2 * triangle * sizeof(std::size_t) : 0)
                  << " logical_weighted_panel_elements=" << evaluations * census.point_ao_visits
                  << " matrix_cache_bytes=" << resource.matrix_bytes
                  << " provider_allowance=" << resource.provider_allowance
                  << " host_bytes=" << resource.host_bytes
                  << " provider_version=" << resource.provider_version
                  << " max_energy_parity_error=" << energy_error
                  << " max_v_parity_error=" << potential_error
                  << " max_population_parity_error=" << population_error;
        for (std::size_t sample = 0; sample != samples[route].size(); ++sample)
          std::cout << " sample" << sample << "_s=" << samples[route][sample];
        std::cout << std::endl;
        fixtures[route]->canary();
      }
    }
  }
}
