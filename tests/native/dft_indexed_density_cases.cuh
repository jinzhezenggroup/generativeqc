/** Exercise the production qualification without granting a test-only offer.
 * The full-AO CPU bilinear oracle independently zeros omitted columns. RKS,
 * UKS, meta-GGA derivatives, noncontiguous/empty maps and changed densities
 * must retain the original energy, potential and exported feature gates.
 */
void qualified_indexed_density_provider_cases() {
  auto molecule = system(3);
  molecule.shells.push_back({0, 2, {{0.4, 1.0}}});
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
  for (unsigned functional : {1U, 2U})
    for (bool unrestricted : {false, true})
      for (unsigned variant : {0U, 1U, 2U})
        for (bool bounded_budget : {false, true}) {
          const auto maps = local_maps(grid.point_count(), 7, basis.nao, variant);
          Fixture fixture(basis, grid, functional, unrestricted, 7, CudaXcAoPrecision::Fp64, false,
                          1.0, 1.0, &maps);
          fixture.plan->prepare_density(generativeqc::runtime::strict_fp64_precision(), 10,
                                        bounded_budget ? 1 : 128ULL << 20, true);
          const auto& binding =
              fixture.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit);
          require((binding.candidate.algorithm == "bounded-matrix-panel-gemm") ==
                      (!bounded_budget && variant != 2),
                  "explicit indexed provider or bounded fallback was not selected");
          auto values = density(basis.nao, unrestricted ? 2 : 1);
          local_ao_reference(fixture, basis, grid, maps, values);
          for (auto& value : values) value *= 0.73;
          local_ao_reference(fixture, basis, grid, maps, values);
          fixture.canary();
        }
  std::cout << "Qualified indexed PBE/r2SCAN RKS/UKS full-AO and bounded-fallback gates passed\n";
}

/** Qualify the mapped recipe against independent scalar and full-AO CPU
 * bilinears, including empty/noncontiguous maps and changed captured inputs. */
void indexed_density_provider_cases() {
  density_provider_scalar_cases(true);
  auto molecule = system(3);
  molecule.shells.push_back({0, 2, {{0.4, 1.0}}});
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
  for (unsigned route = 0; route != 7; ++route) {
    const auto maps = local_maps(grid.point_count(), 7, basis.nao, route == 4 ? 2 : 1);
    Fixture fixture(basis, grid, 4U, true, 7, CudaXcAoPrecision::Fp64, false, 1.0, 1.0, &maps);
    const bool explicit_gemm = route >= 5;
    xc_density_indexed_provider_for_test(route != 0 && !explicit_gemm);
    xc_density_provider_for_test(false, route == 3);
    fixture.plan->prepare_density(generativeqc::runtime::strict_fp64_precision(), 10,
                                  route == 2 || route == 6 ? 1 : 128ULL << 20, explicit_gemm);
    xc_density_indexed_provider_for_test(false);
    xc_density_provider_for_test(false, false);
    const auto& binding =
        fixture.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit);
    require(
        (binding.candidate.algorithm == "bounded-matrix-panel-gemm") == (route == 1 || route == 5),
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
