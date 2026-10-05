/** Qualify the mapped recipe against independent scalar and full-AO CPU
 * bilinears, including empty/noncontiguous maps and changed captured inputs. */
void indexed_density_provider_cases() {
  density_provider_scalar_cases(true);
  auto molecule = system(3);
  molecule.shells.push_back({0, 2, {{0.4, 1.0}}});
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
  for (unsigned route = 0; route != 5; ++route) {
    const auto maps = local_maps(grid.point_count(), 7, basis.nao, route == 4 ? 2 : 1);
    Fixture fixture(basis, grid, 4U, true, 7, CudaXcAoPrecision::Fp64, false, 1.0, 1.0, &maps);
    xc_density_indexed_provider_for_test(route != 0);
    xc_density_provider_for_test(false, route == 3);
    fixture.plan->prepare_density(generativeqc::runtime::strict_fp64_precision(), 10,
                                  route == 2 ? 1 : 128ULL << 20);
    xc_density_indexed_provider_for_test(false);
    xc_density_provider_for_test(false, false);
    const auto& binding =
        fixture.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit);
    require((binding.candidate.algorithm == "bounded-matrix-panel-gemm") == (route == 1),
            "indexed density admission route");
    if (route == 4)
      require(!fixture.plan->density_provider_diagnostic(), "empty domain retained a provider");
    auto d = density(basis.nao, 2);
    local_ao_reference(fixture, basis, grid, maps, d);
    for (auto& value : d) value *= 0.73;
    local_ao_reference(fixture, basis, grid, maps, d);
  }
  xc_density_indexed_provider_for_test(true);
  density_provider_qualification_budget = 128ULL << 20;
  // Existing independent full-map LDA/PBE/meta-GGA and sparse WB97M-V
  // oracles now execute the newly selected candidate, including graph replay.
  local_ao_cases();
  // Preparing a dense factor before discovery must invalidate that recipe.
  // Reprepare against the discovered maps and compare scaled-PBE RKS/UKS
  // with the independent CPU full-AO integration for both basis conventions.
  xc_density_provider_for_test(true, false);
  pbe0_ao_discovery_cases(true);
  xc_density_provider_for_test(false, false);
  density_provider_qualification_budget = 0;
  xc_density_indexed_provider_for_test(false);
  std::cout
      << "Indexed density scalar, capture, local-map and discovered scaled-PBE gates passed\n";
}
