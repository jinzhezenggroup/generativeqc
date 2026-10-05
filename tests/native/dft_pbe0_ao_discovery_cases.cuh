/** Compare scaled-PBE local panels with independent full-AO CPU integration.
 * These are XC component gates, not a complete PBE0 SCF/force qualification.
 */
void pbe0_ao_discovery_cases(bool indexed_provider = false) {
  for (bool spherical : {false, true}) {
    generativeqc::core::System molecule;
    molecule.atoms = {{1, {0, 0, 0}}, {1, {0.1, 0.2, 8.0}}};
    molecule.electron_count = 2;
    molecule.basis_representation =
        spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
    for (unsigned center = 0; center < 2; ++center)
      for (unsigned angular = 0; angular < 4; ++angular)
        molecule.shells.push_back({center, angular, {{0.7 + 0.1 * angular, 1.0}}});
    molecule.shells.push_back({0, 0, {{0.23, 1.0}}});
    std::string detail;
    require(generativeqc::molecule::validate_and_normalize(molecule, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 6, 5, 8, 3, 1e-12});
    const auto matrix = basis.nao * basis.nao;
    require(basis.nao > 32 && basis.nao % 32, "scaled-PBE oracle must cross AO block tails");
    std::vector<double> alpha(matrix), beta(matrix);
    for (std::size_t index = 0; index < basis.nao; ++index) {
      alpha[index * basis.nao + index] = 0.7 / basis.nao;
      beta[index * basis.nao + index] = 0.3 / basis.nao;
    }
    // Positive rank-one updates exercise gathering off-diagonal density entries
    // without admitting negative densities to the independent XC oracle.
    for (std::size_t row = 0; row < basis.nao; ++row)
      for (std::size_t column = 0; column < basis.nao; ++column) {
        const auto alpha_row = std::sin(static_cast<double>(row + 1));
        const auto alpha_column = std::sin(static_cast<double>(column + 1));
        const auto beta_row = std::cos(static_cast<double>(row + 1));
        const auto beta_column = std::cos(static_cast<double>(column + 1));
        alpha[row * basis.nao + column] += 0.2 * alpha_row * alpha_column / basis.nao;
        beta[row * basis.nao + column] += 0.1 * beta_row * beta_column / basis.nao;
      }
    for (bool unrestricted : {false, true}) {
      std::vector<double> input(matrix * (unrestricted ? 2 : 1));
      for (std::size_t index = 0; index < matrix; ++index) {
        input[index] = unrestricted ? alpha[index] : alpha[index] + beta[index];
        if (unrestricted) input[matrix + index] = beta[index];
      }
      double energy{};
      std::vector<double> expected;
      if (unrestricted) {
        const auto reference = integrate_pbe_uks_scaled(basis, grid, alpha, beta, 19, 0.75, 1.0);
        energy = reference.energy;
        expected = reference.potential[0];
        expected.insert(expected.end(), reference.potential[1].begin(),
                        reference.potential[1].end());
      } else {
        const auto reference =
            integrate_pbe_rks_with_tail_scaled(basis, grid, input, 19, {}, 0.75, 1.0);
        energy = reference.energy;
        expected = reference.potential;
      }
      for (double cutoff : {1e-16, 1e8}) {
        Fixture selected(basis, grid, 1U, unrestricted, 19, CudaXcAoPrecision::Fp64, false, 0.75,
                         1.0, nullptr, true);
        const auto bound = cuda_xc_ao_selection_resources(selected.layout);
        require(!selected.plan->select_local_ao(cutoff, bound.host_peak_bytes - 1),
                "scaled-PBE discovery exceeded host admission");
        require(!selected.plan->layout().local_ao, "budget miss installed a partial map");
        require(selected.plan->select_local_ao(cutoff, bound.host_peak_bytes),
                "scaled-PBE local AO admission failed");
        if (indexed_provider) {
          require(!selected.plan->density_provider_diagnostic(),
                  "AO discovery retained a stale dense materialization");
          selected.plan->prepare_density(generativeqc::runtime::strict_fp64_precision(), 10,
                                         density_provider_qualification_budget);
          const auto& binding =
              selected.plan->density_binding(generativeqc::runtime::PrecisionPhase::StrictAudit);
          require((binding.candidate.algorithm == "bounded-matrix-panel-gemm") == (cutoff < 1),
                  "discovered indexed density admission");
        }
        selected.submit(input);
        const auto scalar = selected.scalars();
        const auto potential = selected.potential();
        require(scalar.error == 0, "scaled-PBE local evaluation failed");
        require(potential.size() == expected.size(), "scaled-PBE potential shape changed");
        close(scalar.energy, cutoff > 1 ? 0 : energy, "scaled-PBE CPU/local energy", 1e-10);
        for (std::size_t index = 0; index < potential.size(); ++index)
          close(potential[index], cutoff > 1 ? 0 : expected[index],
                "scaled-PBE CPU/local potential", 1e-10);
        const auto& work = selected.plan->ao_selection_work();
        require(work.selected && work.point_ao_square_sum < work.dense_point_ao_square_sum,
                "scaled-PBE gate did not exercise reduced maps");
        if (cutoff > 1) require(work.empty_tiles == work.tiles, "empty-map oracle was not empty");
        selected.canary();
      }
    }
  }
}
