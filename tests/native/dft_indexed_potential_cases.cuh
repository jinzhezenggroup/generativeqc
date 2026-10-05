/** Independent mapped symmetric-bilinear oracle, including compact spin
 * strides, untouched global entries, overwrite/accumulate and captured replay. */
namespace indexed_potential_test {
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
template <class T>
struct Buffer {
  T* data{};
  explicit Buffer(std::size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
  ~Buffer() { cudaFree(data); }
};
struct Ledger {
  std::shared_ptr<generativeqc::runtime::DeviceResourceLedger> previous =
      generativeqc::runtime::active_device_resource_ledger;
  std::shared_ptr<generativeqc::runtime::DeviceResourceLedger> value =
      std::make_shared<generativeqc::runtime::DeviceResourceLedger>();
  explicit Ledger(std::size_t limit) {
    value->limit = limit;
    check(cudaGetDevice(&value->device));
    generativeqc::runtime::active_device_resource_ledger = value;
  }
  ~Ledger() { generativeqc::runtime::active_device_resource_ledger = previous; }
};
void scalar_cases() {
  for (std::size_t active : {1U, 17U})
    for (bool uks : {false, true}) {
      Stream stream;
      const auto full = 2 * active + 3;
      auto layout = cuda_xc_layout_shape(1, 1, full, 13, 1, uks, 13);
      layout.local_ao = true;
      layout.ao_map_entries = active;
      const auto spins = layout.spins;
      Ledger ledger(spins * full * full * sizeof(double));
      xc_potential_indexed_qualification_for_test(true);
      auto binding = cuda_xc_detail::prepare_potential(layout, stream.value, 128ULL << 20);
      xc_potential_indexed_qualification_for_test(false);
      require(binding->diagnostic().candidate.algorithm == "indexed-symmetric-cross-rank2k",
              "scalar gate did not admit indexed rank-2k");
      require(ledger.value->live == ledger.value->limit &&
                  binding->diagnostic().matrix_bytes == ledger.value->live,
              "compact potential cache did not charge the numeric ledger");
      Buffer<double> a(13 * active), b(spins * 13 * active), out(spins * full * full);
      Buffer<std::size_t> ids(active);
      Buffer<int> error(1);
      for (std::size_t k : {1U, 13U})
        for (bool accumulate : {false, true}) {
          generativeqc::tensor::SymmetricProductInvocation call{
              active,
              k,
              spins,
              a.data,
              b.data,
              out.data,
              error.data,
              accumulate,
              true,
              nullptr,
              [](void*) { throw std::logic_error("indexed scalar gate used generated fallback"); },
              [](void*) {},
              [](void*) {},
              ids.data,
              full};
          auto aliased = call;
          aliased.output = a.data;
          bool rejected{};
          try {
            binding->execute(stream.value, aliased);
          } catch (const std::invalid_argument&) {
            rejected = true;
          }
          require(rejected, "indexed potential accepted aliased panels");
          cudaGraph_t graph{};
          cudaGraphExec_t executable{};
          for (unsigned replay = 0; replay != 3; ++replay) {
            std::vector<double> left(k * active), right(spins * k * active),
                expected(spins * full * full), actual(expected.size());
            std::vector<std::size_t> map(active);
            for (std::size_t i = 0; i < active; ++i) map[i] = 2 * i + replay;
            for (std::size_t i = 0; i < left.size(); ++i)
              left[i] = std::sin(0.23 * (i + 1) + replay);
            for (std::size_t i = 0; i < right.size(); ++i)
              right[i] = std::cos(0.37 * (i + 1) - replay);
            for (std::size_t i = 0; i < expected.size(); ++i)
              expected[i] = 0.001 * (i / (full * full) + i / full % full + i % full + replay);
            check(cudaMemcpyAsync(a.data, left.data(), left.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream.value));
            check(cudaMemcpyAsync(b.data, right.data(), right.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream.value));
            check(cudaMemcpyAsync(ids.data, map.data(), map.size() * sizeof(std::size_t),
                                  cudaMemcpyHostToDevice, stream.value));
            check(cudaMemcpyAsync(out.data, expected.data(), expected.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream.value));
            check(cudaMemsetAsync(error.data, 0, sizeof(int), stream.value));
            check(cudaStreamSynchronize(stream.value));
            if (!replay) {
              check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
              binding->execute(stream.value, call);
              check(cudaStreamEndCapture(stream.value, &graph));
              check(cudaGraphInstantiate(&executable, graph, 0));
            }
            check(cudaGraphLaunch(executable, stream.value));
            int failure{};
            check(cudaMemcpyAsync(actual.data(), out.data, actual.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream.value));
            check(cudaMemcpyAsync(&failure, error.data, sizeof(int), cudaMemcpyDeviceToHost,
                                  stream.value));
            check(cudaStreamSynchronize(stream.value));
            require(failure == 0, "indexed potential scalar finite audit");
            for (std::size_t spin = 0; spin < spins; ++spin)
              for (std::size_t mu = 0; mu < active; ++mu)
                for (std::size_t nu = 0; nu < active; ++nu) {
                  const auto target = (spin * full + map[mu]) * full + map[nu];
                  long double value = accumulate ? expected[target] : 0;
                  for (std::size_t p = 0; p < k; ++p)
                    value += static_cast<long double>(left[p * active + mu]) *
                                 right[(spin * k + p) * active + nu] +
                             static_cast<long double>(right[(spin * k + p) * active + mu]) *
                                 left[p * active + nu];
                  expected[target] = static_cast<double>(value);
                }
            for (std::size_t i = 0; i < actual.size(); ++i)
              close(actual[i], expected[i], "indexed symmetric scalar oracle and untouched domain",
                    2e-12);
          }
          check(cudaGraphExecDestroy(executable));
          check(cudaGraphDestroy(graph));
          std::vector<double> bad(spins * k * active, std::numeric_limits<double>::quiet_NaN());
          int sticky = 7;
          check(cudaMemcpyAsync(b.data, bad.data(), bad.size() * sizeof(double),
                                cudaMemcpyHostToDevice, stream.value));
          check(cudaMemcpyAsync(error.data, &sticky, sizeof(int), cudaMemcpyHostToDevice,
                                stream.value));
          binding->execute(stream.value, call);
          std::vector<double> finite(spins * full * full);
          check(cudaMemcpyAsync(finite.data(), out.data, finite.size() * sizeof(double),
                                cudaMemcpyDeviceToHost, stream.value));
          check(cudaMemcpyAsync(&sticky, error.data, sizeof(int), cudaMemcpyDeviceToHost,
                                stream.value));
          check(cudaStreamSynchronize(stream.value));
          require(sticky == 7 && std::all_of(finite.begin(), finite.end(),
                                             [](double v) { return std::isfinite(v); }),
                  "indexed scatter lost sticky error or finite publication");
        }
      binding.reset();
      require(ledger.value->live == 0, "indexed potential cache survived provider teardown");
    }
}
}  // namespace indexed_potential_test

/** Compare scaled-PBE local panels with independent full-AO CPU integration.
 * These are XC component gates, not a complete PBE0 SCF/force qualification.
 */
void indexed_potential_discovery_cases() {
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
        selected.plan->prepare_potential(128ULL << 20);
        const auto& diagnostic = selected.plan->potential_lowering();
        require((diagnostic.indexed_candidate.algorithm == "indexed-symmetric-cross-rank2k") ==
                    (cutoff < 1),
                "discovered indexed potential admission");
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

void indexed_potential_cases() {
  indexed_potential_test::scalar_cases();
  auto molecule = system(3);
  molecule.shells.push_back({0, 2, {{0.4, 1.0}}});
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
  for (unsigned route = 0; route != 7; ++route) {
    const auto maps = local_maps(grid.point_count(), 7, basis.nao, route == 5 ? 2 : 1);
    Fixture fixture(basis, grid, 4U, true, 7, CudaXcAoPrecision::Fp64, false, 1.0, 1.0, &maps);
    xc_potential_indexed_qualification_for_test(route != 0 && route != 4);
    xc_potential_qualification_for_test(route == 4, route == 3);
    indexed_potential_test::Ledger ledger(route == 6 ? 0 : 128ULL << 20);
    fixture.plan->prepare_potential(route == 2 ? 96ULL << 20 : 128ULL << 20);
    xc_potential_indexed_qualification_for_test(false);
    xc_potential_qualification_for_test(false, false);
    const auto& diagnostic = fixture.plan->potential_lowering();
    const bool selected = route == 1;
    require((diagnostic.candidate.algorithm == "indexed-symmetric-cross-rank2k") == selected,
            "indexed potential admission route");
    require(diagnostic.indexed_candidate.provider == (selected ? "cublas" : "generated.cuda"),
            "indexed candidate provenance disagrees with execution");
    require(
        diagnostic.matrix_bytes == (selected ? 2 * basis.nao * basis.nao * sizeof(double) : 0) &&
            diagnostic.provider_allowance == (selected ? 96ULL << 20 : 0) &&
            ledger.value->live == diagnostic.matrix_bytes,
        "indexed potential resource fallback");
    if (route == 6)
      require(ledger.value->rejected == 1, "numeric ledger exhaustion was not exercised");
    auto d = density(basis.nao, 2);
    local_ao_reference(fixture, basis, grid, maps, d);
    for (auto& value : d) value *= 0.73;
    local_ao_reference(fixture, basis, grid, maps, d);
    fixture.plan.reset();
    require(ledger.value->live == 0, "indexed XC teardown retained its numeric cache");
  }
  xc_potential_indexed_qualification_for_test(true);
  potential_qualification_budget = 128ULL << 20;
  local_ao_cases();
  potential_qualification_budget = 0;
  indexed_potential_discovery_cases();
  xc_potential_indexed_qualification_for_test(false);
  std::cout << "Indexed potential scalar, capture, local-map CPU and resource gates passed\n";
}
