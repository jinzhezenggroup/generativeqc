/** Complete fixed-density XC qualification; never a production promotion rule.
 * Reuse #1958's exact AO extents, point tiles and independent CPU integrator.
 * Include factor materialization, finite audit, H2D and E/Vxc publication.
 */
void density_provider_benchmark() {
  for (std::size_t nao : {384U, 768U}) {
    auto molecule = system();
    for (unsigned i = 0; i < (nao - 2) / 10; ++i)
      molecule.shells.push_back({i % 2, 3, {{0.3 + 0.015 * i, 1.0}}});
    for (unsigned i = 0; i < (nao - 2) % 10; ++i)
      molecule.shells.push_back({i % 2, 0, {{0.25 + 0.02 * i, 1.0}}});
    for (unsigned geometry = 0; geometry != 2; ++geometry) {
      if (geometry) molecule.atoms[1].position[0] += 0.02;
      const AoBasis basis(molecule);
      require(basis.nao == nao, "density crossover AO extent changed");
      const MolecularGrid grid(molecule, {1, 12, 8, 12, 3, 1e-12});
      const auto d = density(nao, 1);
      const auto reference = integrate_pbe_rks_with_tail_scaled(basis, grid, d, 128, {}, 0.75, 1.0);
      for (std::size_t tile : {128U, 256U, 512U})
        for (bool trial : {false, true}) {
          const auto start = std::chrono::steady_clock::now();
          Fixture fixture(basis, grid, 1, false, tile, CudaXcAoPrecision::Fp64, false, 0.75, 1.0);
          xc_density_provider_for_test(trial, false);
          fixture.plan->prepare_density(generativeqc::runtime::strict_fp64_precision(), 6,
                                        128ULL << 20);
          xc_density_provider_for_test(false, false);
          const auto prepared =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          const auto& selected =
              fixture.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit);
          require(selected.candidate.provider == (trial ? "cublas" : "generated.cuda"),
                  "density crossover provider unavailable");
          std::array<double, 6> samples{};
          double max_error{};
          for (auto& seconds : samples) {
            const auto begin = std::chrono::steady_clock::now();
            fixture.submit(d);
            const auto e = fixture.scalars();
            const auto v = fixture.potential();
            seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            require(e.error == 0, "density crossover finite publication");
            close(e.energy, reference.energy, "density crossover CPU energy", 2e-9);
            for (std::size_t i = 0; i != v.size(); ++i) {
              max_error = std::max(max_error, std::abs(v[i] - reference.potential[i]));
              close(v[i], reference.potential[i], "density crossover CPU potential", 2e-9);
            }
          }
          auto sorted = samples;
          std::sort(sorted.begin() + 1, sorted.end());
          const auto* resource = fixture.plan->density_provider_diagnostic();
          const auto evaluations = fixture.plan->transfers().evaluations;
          const auto tiles = (grid.point_count() + tile - 1) / tile;
          const auto products =
              evaluations * fixture.layout.spins * fixture.layout.work_jets * tiles;
          const auto summands = evaluations * fixture.layout.spins * fixture.layout.work_jets *
                                grid.point_count() * nao * nao;
          std::cout << std::setprecision(12) << "endpoint nao=" << nao << " tile=" << tile
                    << " points=" << grid.point_count() << " geometry=" << geometry
                    << " provider=" << selected.candidate.provider << " prepare_s=" << prepared
                    << " provider_prepare_s=" << resource->prepare_seconds
                    << " cold_s=" << samples[0] << " warm_median_s=" << sorted[3]
                    << " max_v_error=" << max_error << " products=" << products
                    << " summands=" << summands
                    << " gemm_calls=" << (trial ? evaluations * fixture.layout.spins * tiles : 0)
                    << " full_panel_rows=" << fixture.layout.work_jets * tile
                    << " tail_panel_rows=" << fixture.layout.work_jets * (grid.point_count() % tile)
                    << " matrix_cache_bytes=" << resource->matrix_bytes
                    << " materialized_bytes=" << evaluations * resource->matrix_bytes
                    << " provider_allowance=" << resource->provider_allowance
                    << " host_bytes=" << resource->host_bytes
                    << " provider_version=" << resource->provider_version;
          for (std::size_t i = 0; i != samples.size(); ++i)
            std::cout << " sample" << i << "_s=" << samples[i];
          std::cout << std::endl;
          fixture.canary();
        }
    }
  }
}
