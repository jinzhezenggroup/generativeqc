// Included after the independent full-AO and masked-column CPU references.
// Compare schedules bitwise as well as against those independent E/V oracles.
void point_batch_cases() {
  for (bool spherical : {false, true}) {
    auto molecule = system(3, spherical);
    molecule.shells.push_back({0, 3, {{0.51, 1.0}}});
    molecule.shells.push_back({1, 3, {{0.32, 1.0}}});
    molecule.shells.push_back({1, 2, {{0.27, 1.0}}});
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
    for (bool uks : {false, true})
      for (std::size_t tile : {7U, 19U, 33U})
        for (unsigned functional : {0U, 1U, 2U, 3U, 4U}) {
          auto d = density(basis.nao, uks ? 2U : 1U);
          Fixture batched(basis, grid, functional, uks, tile);
          batched.plan->prepare_point_batches(16, 32 * 1024 * 1024);
          require(batched.plan->point_batch_plan().tiles > 1, "point batch plan was not admitted");
          compare(batched, basis, grid, d);
          Fixture serial(basis, grid, functional, uks, tile);
          serial.submit(d);
          require(batched.scalars().energy == serial.scalars().energy,
                  "point batching changed energy reduction order");
          require(batched.potential() == serial.potential(),
                  "point batching changed potential accumulation");
          for (auto& value : d) value *= 0.73;
          compare(batched, basis, grid, d);
        }
    for (bool uks : {false, true})
      for (unsigned variant : {0U, 1U, 2U}) {
        const auto maps = local_maps(grid.point_count(), 19, basis.nao, variant);
        Fixture batched(basis, grid, 4U, uks, 19, CudaXcAoPrecision::Fp64, false, 1.0, 1.0, &maps);
        batched.plan->prepare_point_batches(16, 32 * 1024 * 1024);
        require(batched.plan->point_batch_plan().tiles > 1, "mapped point batch was not admitted");
        auto d = density(basis.nao, uks ? 2U : 1U);
        local_ao_reference(batched, basis, grid, maps, d);
        Fixture serial(basis, grid, 4U, uks, 19, CudaXcAoPrecision::Fp64, false, 1.0, 1.0, &maps);
        serial.submit(d);
        require(batched.scalars().energy == serial.scalars().energy &&
                    batched.potential() == serial.potential(),
                "ragged point batching changed ordered accumulation");
        graph_capture(basis, grid, 4U, uks, 19, &maps, 16);
      }
    for (bool uks : {false, true}) {
      Fixture scaled(basis, grid, 1U, uks, 19, CudaXcAoPrecision::Fp64, false, 0.75, 1.0);
      scaled.plan->prepare_point_batches(16, 32 * 1024 * 1024);
      compare(scaled, basis, grid, density(basis.nao, uks ? 2U : 1U));
    }
    Fixture fallback(basis, grid, 1U, false, 19);
    fallback.plan->prepare_point_batches(16, 1);
    require(fallback.plan->point_batch_plan().tiles == 1 &&
                fallback.plan->point_batch_plan().device_bytes == 0,
            "bounded point batch fallback retained optional storage");
    compare(fallback, basis, grid, density(basis.nao, 1));
    bool late_rejected = false;
    try {
      fallback.plan->prepare_point_batches(16, 32 * 1024 * 1024);
    } catch (const std::logic_error&) {
      late_rejected = true;
    }
    require(late_rejected, "late point batch preparation was accepted");
    Fixture tiny(basis, grid, 1U, false, grid.point_count());
    tiny.plan->prepare_point_batches(16, 32 * 1024 * 1024);
    require(tiny.plan->point_batch_plan().tiles == 1, "tiny point domain was batched");
    compare(tiny, basis, grid, density(basis.nao, 1));
    graph_capture(basis, grid, 1U, false, 19, nullptr, 16);
  }
  // Restrict only the optional numeric allocation, not the fixture's arena.
  // Ledger rejection must not silently disable indexed maps or poison CUDA.
  const auto previous = generativeqc::runtime::active_device_resource_ledger;
  const auto ledger = std::make_shared<generativeqc::runtime::DeviceResourceLedger>();
  check(cudaGetDevice(&ledger->device));
  generativeqc::runtime::active_device_resource_ledger = ledger;
  try {
    const AoBasis basis(system());
    const MolecularGrid grid(system(), {1, 3, 3, 4, 3, 1e-12});
    {
      Fixture rejected(basis, grid, 1U, false, 7);
      rejected.plan->prepare_point_batches(16, 32 * 1024 * 1024);
      require(rejected.plan->point_batch_plan().tiles == 1 && ledger->rejected == 1,
              "point batch ledger exhaustion did not retain the incumbent");
      compare(rejected, basis, grid, density(basis.nao, 1));
    }
    ledger->limit = 32 * 1024 * 1024;
    {
      Fixture admitted(basis, grid, 1U, false, 7);
      admitted.plan->prepare_point_batches(16, ledger->limit);
      require(ledger->live == admitted.plan->point_batch_plan().device_bytes && ledger->live > 0,
              "point batch residency was not charged exactly");
      compare(admitted, basis, grid, density(basis.nao, 1));
    }
    require(ledger->live == 0, "point batch teardown retained numeric storage");
  } catch (...) {
    generativeqc::runtime::active_device_resource_ledger = previous;
    throw;
  }
  generativeqc::runtime::active_device_resource_ledger = previous;
}
