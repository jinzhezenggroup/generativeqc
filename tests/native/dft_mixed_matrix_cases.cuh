/** Qualify shared products without changing point algebra or the AO domain.
 * The incumbent is first checked against the independent CPU bilinear oracle;
 * strict handoff must then recover that same scheduled result bit for bit. */
void mixed_matrix_cases() {
  using generativeqc::runtime::PrecisionPhase;
  for (unsigned domain : {0U, 1U, 2U}) {
    auto molecule = system(domain ? 3 : 0, true);
    if (domain == 1)
      for (unsigned shell = 0; shell < 9; ++shell)
        molecule.shells.push_back({shell % 2, 2, {{0.2 + 0.02 * shell, 1.0}}});
    if (domain == 2)
      for (unsigned shell = 0; shell < 44; ++shell)
        molecule.shells.push_back({shell % 2, 2, {{0.11 + 0.013 * shell, 1.0}}});
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 3, 4, 6, 3, 1e-12});
    const std::size_t tile_points = domain ? 56 : 7;
    const auto maps = local_maps(grid.point_count(), tile_points, basis.nao, 1);
    for (unsigned schedule : {0U, 1U, 2U})
      for (bool indexed : {false, true})
        for (bool uks : {false, true})
          for (unsigned functional : {0U, 1U, 2U, 3U, 4U}) {
            const auto* selected = indexed ? &maps : nullptr;
            Fixture strict(basis, grid, functional, uks, tile_points, CudaXcAoPrecision::Fp64,
                           false, 1.0, 1.0, selected);
            Fixture mixed(basis, grid, functional, uks, tile_points, CudaXcAoPrecision::Fp64, false,
                          1.0, 1.0, selected);
            if (schedule) {
              strict.plan->prepare_point_batches(16, 32ULL << 20, schedule == 2, domain == 2);
              mixed.plan->prepare_point_batches(16, 32ULL << 20, schedule == 2, domain == 2);
              require(mixed.plan->point_batch_plan().tiles > 1,
                      "shared mixed products did not exercise point batching");
              if (schedule == 2) {
                require(
                    mixed.plan->point_batch_plan().compact == (domain != 0),
                    "shared mixed products changed compact admission or missed its legal domain");
                if (domain == 2)
                  require(mixed.plan->point_batch_plan().compact_large_tiles > 0,
                          "shared mixed products did not exercise ragged batching");
              }
            }
            mixed.plan->prepare_mixed_matrix_contractions();
            auto density_matrix = density(basis.nao, uks ? 2 : 1);
            for (unsigned replay = 0; replay < 2; ++replay) {
              if (indexed)
                local_ao_reference(strict, basis, grid, maps, density_matrix);
              else
                compare(strict, basis, grid, density_matrix);
              const auto reference = strict.scalars();
              const auto potential = strict.potential();
              mixed.submit(density_matrix, PrecisionPhase::Admitted);
              const auto candidate = mixed.scalars();
              require(candidate.error == 0, "shared mixed products rejected finite input");
              const auto tolerance = [](double value) { return 2e-6 + 2e-6 * std::abs(value); };
              close(candidate.energy, reference.energy, "shared mixed energy",
                    tolerance(reference.energy));
              for (unsigned spin = 0; spin < 2; ++spin)
                close(candidate.electrons[spin], reference.electrons[spin],
                      "shared mixed population", tolerance(reference.electrons[spin]));
              const auto actual = mixed.potential();
              bool changed = candidate.energy != reference.energy;
              for (std::size_t index = 0; index < actual.size(); ++index) {
                close(actual[index], potential[index], "shared mixed potential",
                      tolerance(potential[index]));
                changed = changed || actual[index] != potential[index];
              }
              require(changed, "shared mixed selection retained strict products");
              if (strict.layout.feature_terms >= 4) {
                const auto strict_features = strict.submit_density_features(density_matrix);
                const auto mixed_features =
                    mixed.submit_density_features(density_matrix, PrecisionPhase::Admitted);
                require(mixed.scalars().energy == candidate.energy && mixed.potential() == actual,
                        "physical feature capture silently changed the admitted precision");
                for (std::size_t point = 0; point < strict_features.first.size(); ++point)
                  close(mixed_features.first[point], strict_features.first[point],
                        "mixed exported density", tolerance(strict_features.first[point]));
                for (std::size_t entry = 0; entry < strict_features.second.size(); ++entry)
                  close(mixed_features.second[entry], strict_features.second[entry],
                        "mixed exported gradient", tolerance(strict_features.second[entry]));
                require(mixed.submit_density_features(density_matrix) == strict_features,
                        "independent feature capture lost its strict default");
              }
              mixed.submit(density_matrix, PrecisionPhase::StrictAudit);
              const auto restored = mixed.scalars();
              require(restored.error == 0 && restored.energy == reference.energy &&
                          restored.electrons[0] == reference.electrons[0] &&
                          restored.electrons[1] == reference.electrons[1] &&
                          mixed.potential() == potential,
                      "shared mixed products did not restore strict arithmetic");
              bool rejected = false;
              try {
                mixed.plan->prepare_mixed_matrix_contractions();
              } catch (const std::invalid_argument&) {
                rejected = true;
              }
              require(rejected && mixed.potential() == potential,
                      "late mixed preparation invalidated a completed result");
              for (auto& value : density_matrix) value *= 0.73;
            }
            strict.canary();
            mixed.canary();
          }
  }
  const auto molecule = system(0, true);
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 4, 6, 3, 1e-12});
  Fixture response(basis, grid, 1U, false, 7, CudaXcAoPrecision::Fp64, true);
  bool rejected = false;
  try {
    response.plan->prepare_mixed_matrix_contractions();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "shared mixed products admitted an unqualified response");
  response.canary();
}
