/** A prepared response allowance does not reserve physical CUDA memory.
 * Load the complete force path first, then retain unrelated device storage so
 * its old largest response tile cannot fit. Replanning must preserve forces
 * and the immutable value owner. This explicit GPU test is run in isolation.
 */
void response_headroom_cases() {
  System system;
  for (unsigned atom = 0; atom < 16; ++atom) {
    system.atoms.push_back({1, {0.17 * (atom % 2), 0.09 * (atom % 3), 2.3 * atom}});
    system.shells.push_back({atom, 0, {{0.8, 1.0}}});
    system.shells.push_back({atom, 0, {{0.2, 1.0}}});
    system.shells.push_back({atom, 1, {{0.4, 1.0}}});
  }
  system.electron_count = 16;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  auto auxiliary = system;
  for (unsigned atom = 0; atom < 16; ++atom) auxiliary.shells.push_back({atom, 2, {{0.7, 1.0}}});
  require(generativeqc::molecule::validate_and_normalize(auxiliary, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  spec.exchange.present = false;
  spec.derivative_order = 1;
  const auto strategy = resolve_fock_build(spec, FockBackend::Cuda, 0.0, 1e-8);
  const PreparedFockPlan plan(system, &auxiliary, strategy, 0, 128U << 20);
  const auto n = plan.one_electron().nbf;
  std::vector<double> density(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      density[i * n + j] = (i == j ? 0.3 : 0.0) + 0.005 * std::cos(double(i + j));
  const auto expected = plan.energy_derivative(density);
  const auto expected_j = plan.build(density).coulomb;
  require(cudaDeviceSynchronize() == cudaSuccess, "cannot drain pressure-test warmup");
  std::size_t free_bytes{}, total_bytes{};
  require(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess,
          "cannot query pressure-test capacity");
  constexpr std::size_t headroom = 8U << 20;
  require(free_bytes > headroom && plan.diagnostic().response_device_bytes > headroom,
          "pressure fixture does not constrain its prepared response allowance");
  {
    struct Allocation {
      void* pointer{};
      ~Allocation() {
        if (pointer) (void)cudaFree(pointer);
      }
    } unrelated;
    // Leave eight MiB plus allocation alignment. No pressure buffer is touched
    // by the solver, and its RAII lifetime also covers every failed assertion.
    const auto bytes = (free_bytes - headroom) & ~std::size_t{65535};
    require(cudaMalloc(&unrelated.pointer, bytes) == cudaSuccess,
            "cannot create isolated response memory pressure");
    matrix(plan.energy_derivative(density), expected,
           "current-headroom response changed the complete derivative");
  }
  matrix(plan.build(density).coulomb, expected_j,
         "memory-pressure response damaged the retained Coulomb owner");
}
