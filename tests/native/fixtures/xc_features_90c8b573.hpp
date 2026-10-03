// Frozen pre-specialization body from 90c8b573c1d0b6c83157802e653e4193777a4ab7.
// Test-only operation-order comparator; not an independent scientific oracle.
std::array<double, 5> frozen_rks_features(const double* phi,
                                          const std::array<const double*, 3>& derivatives,
                                          std::size_t n, const std::vector<double>& density,
                                          const scf::OccupiedDensityFactor* factor,
                                          unsigned ingredient_mask) {
  std::array<double, 5> features{};
  const bool need_first = (ingredient_mask & 14U) != 0;
  const bool need_tau = (ingredient_mask & 8U) != 0;
  if (factor) {
    for (std::size_t o = 0; o < factor->rank(); ++o) {
      double work[4]{};
      for (std::size_t mu = 0; mu < n; ++mu) {
        const double b = factor->values()[mu * factor->rank() + o];
        work[0] += phi[mu] * b;
        if (need_first)
          for (unsigned axis = 0; axis < 3; ++axis) work[axis + 1] += derivatives[axis][mu] * b;
      }
      generated::add_features(work[0], work + 1, work, features.data(), ingredient_mask);
    }
  } else {
    // Every AO feature kernel is symmetric in (mu, nu). Preserve the accepted
    // near-symmetric density semantics by summing both off-diagonal elements,
    // but evaluate each AO pair only once.
    for (std::size_t mu = 0; mu < n; ++mu) {
      const double phi_mu = phi[mu];
      const double diagonal = density[mu * n + mu];
      features[0] += phi_mu * diagonal * phi_mu;
      if (need_first)
        for (unsigned axis = 0; axis < 3; ++axis)
          features[axis + 1] +=
              (derivatives[axis][mu] * phi_mu + phi_mu * derivatives[axis][mu]) * diagonal;
      if (need_tau)
        for (unsigned axis = 0; axis < 3; ++axis)
          features[4] += 0.5 * derivatives[axis][mu] * diagonal * derivatives[axis][mu];

      for (std::size_t nu = mu + 1; nu < n; ++nu) {
        const double pair_density = density[mu * n + nu] + density[nu * n + mu];
        features[0] += phi_mu * pair_density * phi[nu];
        if (need_first)
          for (unsigned axis = 0; axis < 3; ++axis)
            features[axis + 1] +=
                (derivatives[axis][mu] * phi[nu] + phi_mu * derivatives[axis][nu]) * pair_density;
        if (need_tau)
          for (unsigned axis = 0; axis < 3; ++axis)
            features[4] += 0.5 * derivatives[axis][mu] * pair_density * derivatives[axis][nu];
      }
    }
  }
  return features;
}
