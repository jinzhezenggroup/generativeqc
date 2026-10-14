// Included in the native test namespace. CPU AO values independently determine
// each expected map, then full-global-AO bilinears validate the selected E/V.
void ao_discovery_cases() {
  for (bool spherical : {false, true}) {
    auto molecule = system(3, spherical);
    molecule.shells.push_back({0, 3, {{0.51, 1.0}}});
    molecule.shells.push_back({0, 3, {{0.39, 1.0}}});
    molecule.shells.push_back({1, 3, {{0.32, 1.0}}});
    molecule.shells.push_back({1, 2, {{0.27, 1.0}}});
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 7, 7, 8, 3, 1e-12});
    require(grid.point_count() > 257 && basis.nao > 32 && basis.nao % 32,
            "discovery oracle must cross both native block boundaries");
    std::vector<double> ao(4 * grid.point_count() * basis.nao);
    basis.evaluate(grid.points().data(), grid.point_count(), 1, 0, basis.nao, ao.data(), ao.size());
    for (std::size_t tile : {1U, 7U, 129U, 257U})
      for (double cutoff : {1e-16, 1e-2, 1e8}) {
        CudaXcAoTiles expected;
        expected.derivative_order = 1;
        expected.offsets.push_back(0);
        std::uint64_t point_square = 0;
        std::size_t empty = 0;
        for (std::size_t first = 0; first < grid.point_count(); first += tile) {
          const auto count = std::min(tile, grid.point_count() - first);
          for (std::size_t mu = 0; mu < basis.nao; ++mu) {
            double maximum = 0;
            for (std::size_t point = first; point < first + count; ++point)
              for (unsigned jet = 0; jet < 4; ++jet)
                maximum = std::max(
                    maximum, std::abs(ao[(jet * grid.point_count() + point) * basis.nao + mu]));
            if (maximum > cutoff) expected.indices.push_back(mu);
          }
          const auto active = expected.indices.size() - expected.offsets.back();
          point_square += count * active * active;
          empty += active == 0;
          expected.offsets.push_back(expected.indices.size());
        }
        for (auto schedule : {CudaXcAoDiscovery::HostTiles, CudaXcAoDiscovery::DeviceChunks}) {
          Fixture fixture(basis, grid, 4U, spherical, tile, CudaXcAoPrecision::Fp64, false, 1, 1,
                          nullptr, true);
          const auto bounds = cuda_xc_ao_selection_resources(fixture.layout);
          require(!fixture.plan->select_local_ao(cutoff, bounds.host_peak_bytes - 1, schedule),
                  "AO discovery exceeded its host admission");
          require(!fixture.plan->layout().local_ao, "budget miss published a partial map");
          require(fixture.plan->select_local_ao(cutoff, bounds.host_peak_bytes, schedule),
                  "exact AO discovery admission failed");
          const auto& work = fixture.plan->ao_selection_work();
          require(work.selected && work.active_sum == expected.indices.size() &&
                      work.empty_tiles == empty && work.point_ao_square_sum == point_square &&
                      work.discovery_ao_jet_values == ao.size(),
                  "AO discovery differs from independent CPU masks or work counts");
          // Inspect the published compact device inventory directly, independent
          // of its numerical consumer. This catches ordering/map permutations
          // that could be hidden by symmetric test densities or summary counts.
          std::vector<std::size_t> actual(work.active_sum);
          if (!actual.empty())
            check(cudaMemcpy(actual.data(),
                             static_cast<char*>(fixture.arena) + fixture.layout.device_bytes,
                             actual.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));
          require(actual == expected.indices, "device discovery changed sorted sampled AO IDs");
          if (schedule == CudaXcAoDiscovery::DeviceChunks) {
            require(
                work.discovery_h2d_bytes == 0 &&
                    work.discovery_d2h_bytes <= work.tiles * (sizeof(std::size_t) + sizeof(int)),
                "device discovery staged AO IDs or flags through the host");
            require(work.discovery_synchronizations < work.tiles,
                    "device discovery retained per-tile host fences");
          }
          local_ao_reference(fixture, basis, grid, expected, density(basis.nao, spherical ? 2 : 1));
          bool rejected = false;
          try {
            fixture.plan->select_local_ao(cutoff, bounds.host_peak_bytes);
          } catch (const std::invalid_argument&) {
            rejected = true;
          }
          require(rejected, "AO discovery accepted a previously evaluated/local plan");
          fixture.canary();
        }
      }
    Fixture dense(basis, grid, 4U, false, 129);
    require(!dense.plan->select_local_ao(1e-16, std::numeric_limits<std::size_t>::max()),
            "AO discovery exceeded its device admission");
    local_ao_reference(dense, basis, grid, local_maps(grid.point_count(), 129, basis.nao, 0),
                       density(basis.nao, 1));
    // A physical replay body touches scratch before its generation is
    // published. Zero submitted/evaluation counters must not admit discovery
    // into a previously captured or outstanding body.
    Fixture replay(basis, grid, 4U, false, 129, CudaXcAoPrecision::Fp64, false, 1, 1, nullptr,
                   true);
    const auto d = density(basis.nao, 1);
    check(cudaMemcpyAsync(replay.density, d.data(), d.size() * sizeof(double),
                          cudaMemcpyHostToDevice, replay.stream));
    replay.plan->enqueue_replay_body(replay.density, d.size());
    require(replay.plan->transfers().evaluations == 0, "replay probe unexpectedly published work");
    bool rejected = false;
    try {
      replay.plan->select_local_ao(1e-16, std::numeric_limits<std::size_t>::max());
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "discovery was allowed after an unpublished replay body");
    check(cudaStreamSynchronize(replay.stream));
    replay.canary();
  }

  // The smallest LDA panel has only one work jet. Discovery must fit its
  // borrowed flags/offsets as well as the larger meta-GGA scratch layout.
  const AoBasis tiny_basis(system());
  const MolecularGrid tiny_grid(system(), {1, 3, 3, 4, 3, 1e-12});
  for (auto schedule : {CudaXcAoDiscovery::HostTiles, CudaXcAoDiscovery::DeviceChunks}) {
    Fixture tiny(tiny_basis, tiny_grid, 0U, false, 1, CudaXcAoPrecision::Fp64, false, 1, 1, nullptr,
                 true);
    const auto bound = cuda_xc_ao_selection_resources(tiny.layout);
    // Inject a bad sampled coordinate after the constructor's upload. A failed
    // chunk may not publish offsets or IDs; a subsequent valid discovery must
    // reset its device cursor/error state and produce the original map.
    auto* points = static_cast<double*>(tiny.arena) + tiny_basis.packed.size();
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    check(cudaMemcpyAsync(points, &invalid, sizeof(double), cudaMemcpyHostToDevice, tiny.stream));
    bool rejected = false;
    try {
      tiny.plan->select_local_ao(1e-16, bound.host_peak_bytes, schedule);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && !tiny.plan->layout().local_ao,
            "failed AO discovery published a partial map");
    check(cudaMemcpyAsync(points, tiny_grid.points().data(), sizeof(double), cudaMemcpyHostToDevice,
                          tiny.stream));
    require(tiny.plan->select_local_ao(1e-16, bound.host_peak_bytes, schedule),
            "valid retry after discovery failure was refused");
    compare(tiny, tiny_basis, tiny_grid, density(tiny_basis.nao, 1));
    tiny.canary();
  }

  // Cross the compactor's entire 256-lane wave, not only an AO warp. Check
  // the sampled inventory directly so this large-AO control needs no dense
  // N^2 CPU contraction merely to validate an integer prefix scan.
  for (bool spherical : {false, true}) {
    auto wide_system = system(3, spherical);
    for (unsigned i = 0; i < 40; ++i)
      wide_system.shells.push_back({i % 2, 3, {{0.2 + 0.07 * i, 1.0}}});
    const AoBasis wide_basis(wide_system);
    const MolecularGrid wide_grid(wide_system, {1, 3, 3, 4, 3, 1e-12});
    require(wide_basis.nao > 256, "wide discovery control lost its multi-wave AO domain");
    const auto points = wide_grid.point_count(), n = wide_basis.nao;
    std::vector<double> ao(4 * points * n);
    wide_basis.evaluate(wide_grid.points().data(), points, 1, 0, n, ao.data(), ao.size());
    for (double cutoff : {1e-16, 1e-2, 1e8}) {
      std::vector<std::size_t> expected;
      constexpr std::size_t tile = 17;
      for (std::size_t first = 0; first < points; first += tile)
        for (std::size_t column = 0; column < n; ++column) {
          bool keep = false;
          for (std::size_t p = first; p < std::min(points, first + tile); ++p)
            for (unsigned jet = 0; jet < 4; ++jet)
              keep |= std::abs(ao[(jet * points + p) * n + column]) > cutoff;
          if (keep) expected.push_back(column);
        }
      Fixture wide(wide_basis, wide_grid, 4U, false, tile, CudaXcAoPrecision::Fp64, false, 1, 1,
                   nullptr, true);
      const auto bound = cuda_xc_ao_selection_resources(wide.layout);
      require(wide.plan->select_local_ao(cutoff, bound.host_peak_bytes,
                                         CudaXcAoDiscovery::DeviceChunks),
              "wide device AO discovery was refused");
      std::vector<std::size_t> actual(wide.plan->ao_selection_work().active_sum);
      if (!actual.empty())
        check(cudaMemcpy(actual.data(), static_cast<char*>(wide.arena) + wide.layout.device_bytes,
                         actual.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));
      require(actual == expected, "multi-wave device compaction changed the sampled AO map");
      wide.canary();
    }
  }
}
