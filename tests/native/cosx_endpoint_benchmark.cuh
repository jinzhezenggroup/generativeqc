/** Fixed-density, explicit-quadrature COSX endpoint qualification. The input
 * basis and geometry are real, but the density is a positive diagnostic matrix,
 * not a converged SCF state. No provider default is changed by this harness.
 * Timed build includes AO/ESP generation, H2D, contractions, D2H and host energy.
 */
namespace cosx_endpoint_test {
using namespace generativeqc;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

core::System read_system(const char* path) {
  std::ifstream input(path);
  std::size_t atoms{}, shells{};
  require(bool(input >> atoms >> shells) && atoms && shells, "missing geometry/basis header");
  core::System system;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  system.atoms.resize(atoms);
  for (auto& atom : system.atoms)
    require(bool(input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >>
                 atom.position[2]),
            "missing atom");
  system.shells.resize(shells);
  for (auto& shell : system.shells) {
    std::size_t count{};
    require(bool(input >> shell.atom_index >> shell.angular_momentum >> count), "missing shell");
    shell.primitives.resize(count);
    for (auto& primitive : shell.primitives)
      require(bool(input >> primitive.exponent >> primitive.coefficient), "missing primitive");
  }
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return system;
}

std::vector<double> diagnostic_density(std::size_t n) {
  // A scaled Hilbert matrix plus a positive diagonal: symmetric positive
  // definite at every dimension, without relying on a molecular SCF oracle.
  std::vector<double> density(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      density[i * n + j] = (i == j ? 0.1 : 0.0) + 0.001 / (1.0 + i + j);
  return density;
}

std::array<double, 3> errors(const dft::CosxReferenceResult& actual,
                             const dft::CosxReferenceResult& reference) {
  require(actual.nbf == reference.nbf && actual.npoint == reference.npoint &&
              actual.spec == reference.spec && actual.convention == reference.convention,
          "COSX benchmark changed mathematical identity");
  std::array<double, 3> result{};
  const std::array<const std::vector<double>*, 2> first{&actual.raw_exchange, &actual.exchange};
  const std::array<const std::vector<double>*, 2> second{&reference.raw_exchange,
                                                         &reference.exchange};
  for (unsigned field = 0; field < 2; ++field) {
    require(first[field]->size() == second[field]->size(), "COSX result size mismatch");
    for (std::size_t i = 0; i < first[field]->size(); ++i) {
      const auto a = (*first[field])[i], b = (*second[field])[i];
      require(std::isfinite(a) && std::isfinite(b), "nonfinite COSX matrix");
      const double error = std::abs(a - b);
      require(error <= 1e-9 + 1e-11 * std::abs(b), "COSX matrix parity failed");
      result[field] = std::max(result[field], error);
    }
  }
  result[2] = std::abs(actual.exchange_energy - reference.exchange_energy);
  require(std::isfinite(actual.exchange_energy) && std::isfinite(reference.exchange_energy) &&
              result[2] <= 1e-9 + 1e-11 * std::abs(reference.exchange_energy),
          "COSX energy parity failed");
  return result;
}

std::unique_ptr<dft::CudaCosxStagingPlan> prepare(const core::System& system,
                                                  std::span<const double> xyz,
                                                  std::span<const double> weights, std::size_t tile,
                                                  unsigned mask) {
  const auto base = dft::cuda_cosx_staging_diagnostic(system, weights.size(), tile).device_bytes;
  cosx_contraction_qualification_for_test(mask, false);
  // Test admission only. Both generated and library plans receive the same
  // budget; retained resources reflect what each candidate actually reserves.
  try {
    auto plan = std::make_unique<dft::CudaCosxStagingPlan>(system, xyz, weights, tile, 0,
                                                           base + (96ULL << 20));
    cosx_contraction_qualification_for_test(0, false);
    return plan;
  } catch (...) {
    cosx_contraction_qualification_for_test(0, false);
    throw;
  }
}

void site_json(const tensor::ContractionSiteDiagnostic& site) {
  const auto& r = site.resolved;
  const auto& c = site.candidate;
  std::cout << "{\"provider\":" << std::quoted(std::string(c.provider))
            << ",\"algorithm\":" << std::quoted(std::string(c.algorithm))
            << ",\"candidate\":" << std::quoted(std::string(c.identity))
            << ",\"scientific\":" << std::quoted(std::string(r.scientific_identity))
            << ",\"semantic\":" << std::quoted(std::string(r.semantic_template_identity))
            << ",\"precision\":" << std::quoted(std::string(r.precision_identity))
            << ",\"m\":" << r.m << ",\"n\":" << r.n << ",\"k\":" << r.k
            << ",\"calls\":" << site.calls << ",\"summands\":" << site.summands
            << ",\"selected\":" << site.selected << ",\"offers\":[";
  for (std::size_t i = 0; i < site.offer_count; ++i) {
    const auto& offer = site.offers[i];
    if (i) std::cout << ',';
    std::cout << "{\"identity\":" << std::quoted(std::string(offer.identity))
              << ",\"provider\":" << std::quoted(std::string(offer.provider))
              << ",\"rejection\":" << std::quoted(std::string(offer.rejection)) << '}';
  }
  std::cout << "]}";
}
}  // namespace cosx_endpoint_test

/** CLI fields: original, moved, radial, polar, azimuth, tile. Grid resolution
 * remains explicit so a coarse diagnostic receipt cannot masquerade as a
 * production PBE0/COSX quadrature or converged SCF/force comparison. */
void cosx_endpoint_benchmark(char** args) {
  using namespace cosx_endpoint_test;
  const auto radial = std::stoul(args[2]), polar = std::stoul(args[3]),
             azimuth = std::stoul(args[4]);
  const auto tile = std::stoul(args[5]);
  require(radial && radial <= 1000 && polar && polar <= 1000 && azimuth && azimuth <= 1000 &&
              tile && tile <= 4096,
          "invalid COSX benchmark grid/tile");
  constexpr std::size_t repeats = 6;
  constexpr auto convention = dft::CosxDensityConvention::rhf_spin_summed;
  for (unsigned geometry = 0; geometry < 2; ++geometry) {
    const auto geometry_begin = Clock::now();
    const auto system = read_system(args[geometry]);
    const dft::MolecularGrid grid(system,
                                  {1, static_cast<unsigned>(radial), static_cast<unsigned>(polar),
                                   static_cast<unsigned>(azimuth), 3, 1e-12});
    const auto geometry_seconds = elapsed(geometry_begin);
    const auto n = molecule::ao_count(system), points = grid.point_count();
    const auto density = diagnostic_density(n);

    // Independent CPU oracle at the actual large AO dimension, outside timing.
    // Three spread-out quadrature points and tile=2 exercise both full and tail
    // with bounded CPU materialization; the full-grid check below is paired.
    std::vector<double> oracle_xyz, oracle_weights;
    for (const auto point : {points / 6, points / 2, 5 * points / 6}) {
      oracle_xyz.insert(oracle_xyz.end(), grid.points().begin() + 3 * point,
                        grid.points().begin() + 3 * point + 3);
      oracle_weights.push_back(grid.weights()[point]);
    }
    const auto oracle =
        dft::build_cosx_reference(system, oracle_xyz, oracle_weights, density, convention);
    std::array<std::array<double, 3>, 4> oracle_errors{};
    for (unsigned mask = 0; mask < 4; ++mask) {
      auto plan = prepare(system, oracle_xyz, oracle_weights, 2, mask);
      oracle_errors[mask] = errors(plan->build(density, convention), oracle);
    }

    std::array<std::unique_ptr<dft::CudaCosxStagingPlan>, 4> plans;
    std::array<double, 4> setup{};
    for (unsigned mask = 0; mask < 4; ++mask) {
      const auto begin = Clock::now();
      plans[mask] = prepare(system, grid.points(), grid.weights(), tile, mask);
      setup[mask] = elapsed(begin);
    }
    std::array<std::array<double, repeats>, 4> samples{}, energies{};
    std::array<std::array<std::array<double, 3>, repeats>, 4> paired_errors{};
    for (std::size_t sample = 0; sample < repeats; ++sample) {
      std::array<dft::CosxReferenceResult, 4> result;
      // Rotate all four routes, including after geometry change. This reports
      // first replay, not process-cold CUDA/library initialization.
      for (unsigned order = 0; order < 4; ++order) {
        const auto mask = (order + sample + geometry) % 4;
        const auto begin = Clock::now();
        result[mask] = plans[mask]->build(density, convention);
        samples[mask][sample] = elapsed(begin);
        energies[mask][sample] = result[mask].exchange_energy;
      }
      for (unsigned mask = 0; mask < 4; ++mask)
        paired_errors[mask][sample] = errors(result[mask], result[0]);
    }
    for (unsigned mask = 0; mask < 4; ++mask) {
      const auto& info = plans[mask]->diagnostic();
      require(info.provider_allowance == (mask ? 96ULL << 20 : 0), "wrong provider allowance");
      for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& site = info.contractions[slot];
        const bool library = mask & (1U << (slot % 2));
        const auto count = slot < 2 ? points / tile : std::size_t(points % tile != 0);
        const auto extent = slot < 2 ? tile : points % tile;
        require(site.candidate.provider == (library ? "cublas" : "generated.cuda") &&
                    site.calls == repeats * count &&
                    site.summands == repeats * count * extent * n * n,
                "benchmark provider or semantic work mismatch");
      }
      std::cout << std::setprecision(17)
                << "{\"schema\":\"cosx-endpoint-v1\",\"atoms\":" << system.atoms.size()
                << ",\"nao\":" << n << ",\"geometry\":" << geometry << ",\"grid\":[" << radial
                << ',' << polar << ',' << azimuth << ']' << ",\"points\":" << points
                << ",\"tile\":" << tile << ",\"mask\":" << mask
                << ",\"geometry_prepare_s\":" << geometry_seconds
                << ",\"prepare_s\":" << setup[mask]
                << ",\"provider_prepare_s\":" << info.contraction_prepare_seconds
                << ",\"device_bytes\":" << info.device_bytes
                << ",\"provider_allowance\":" << info.provider_allowance
                << ",\"retained_provider_bytes\":" << info.retained_provider_bytes
                << ",\"host_reservation\":" << info.contraction_host_bytes
                << ",\"provider_version\":" << info.provider_version
                << ",\"runtime_version\":" << info.runtime_version << ",\"compute_capability\":\""
                << info.compute_major << '.' << info.compute_minor << '"' << ",\"oracle_errors\":["
                << oracle_errors[mask][0] << ',' << oracle_errors[mask][1] << ','
                << oracle_errors[mask][2] << "],\"samples\":[";
      for (std::size_t sample = 0; sample < repeats; ++sample) {
        if (sample) std::cout << ',';
        const auto& e = paired_errors[mask][sample];
        std::cout << "{\"seconds\":" << samples[mask][sample]
                  << ",\"energy\":" << energies[mask][sample] << ",\"errors\":[" << e[0] << ','
                  << e[1] << ',' << e[2] << "]}";
      }
      std::cout << "],\"sites\":[";
      for (unsigned slot = 0; slot < 4; ++slot) {
        if (slot) std::cout << ',';
        site_json(info.contractions[slot]);
      }
      std::cout << "]}" << std::endl;
    }
  }
}
