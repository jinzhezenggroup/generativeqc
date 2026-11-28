int main() {
  using namespace generativeqc;
  methods::PreparedBatch batch;
  dft::CudaKsFinalStateToken token;
  using Mode = dft::CudaKsWb97mvHessianMode;
  {
    dft::CudaKsWb97mvHessian owner;
    std::string detail;
    assert(owner.prepare(batch, 0, token, Mode::action_only, {}, detail) ==
           GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    assert(stub::allocation_attempts == 0);
  }
  token.strict_physical_fock = true;
  {
    // A screened Cartesian provider can serve a dense matrix, but must not be
    // published in an owner that promises arbitrary-direction actions.
    std::string detail;
    stub::canonical_source_limit = true;
    for (const auto mode : {Mode::action_only, Mode::dense_and_action}) {
      dft::CudaKsWb97mvHessian owner;
      assert(owner.prepare(batch, 0, token, mode, {}, detail) ==
             GENERATIVEQC_STATUS_INVALID_ARGUMENT);
      assert(stub::screened_exact_preparations == 0 && stub::live_bytes == 0);
    }
    {
      dft::CudaKsWb97mvHessian owner;
      assert(owner.prepare(batch, 0, token, Mode::dense_only, {}, detail) ==
             GENERATIVEQC_STATUS_SUCCESS);
      assert(stub::screened_exact_preparations == 1);
      assert(owner.work().frozen_exact_bytes && owner.work().device_bytes == stub::live_bytes);
    }
    assert(stub::live_bytes == 0);
    stub::canonical_source_limit = false;
  }
  // Both raw providers may fit a tape individually while starving a later
  // mandatory provider. Drive the actual Hessian owner at its streamed bound.
  stub::df = stub::range_df = true;
  {
    dft::CudaKsWb97mvHessianOptions options;
    options.max_resident_cpks_bytes = 0;
    options.grid.replay_nuclear_maps = false;
    options.one_electron.parallel_streams = 0;
    std::string detail;
    std::size_t streamed_bytes{}, mandatory_allocations{};
    {
      dft::CudaKsWb97mvHessian owner;
      stub::allocation_attempts = 0;
      assert(owner.prepare(batch, 0, token, Mode::dense_only, options, detail) ==
             GENERATIVEQC_STATUS_SUCCESS);
      streamed_bytes = owner.work().device_bytes;
      mandatory_allocations = stub::allocation_attempts;
    }
    stub::retain_df_tape = true;
    for (const auto tapes : {0U, 1U, 2U}) {
      options.max_device_bytes = streamed_bytes + tapes * stub::df_tape_bytes;
      dft::CudaKsWb97mvHessian owner;
      assert(owner.prepare(batch, 0, token, Mode::dense_only, options, detail) ==
             GENERATIVEQC_STATUS_SUCCESS);
      assert(owner.work().device_bytes == options.max_device_bytes);
      assert(owner.work().device_bytes == stub::live_bytes);
    }
    assert(stub::live_bytes == 0);
    // Simulate physical allocation failure after the budget check. The
    // mandatory state stays usable and the second tape can still be admitted.
    stub::allocation_attempts = 0;
    stub::fail_allocation = mandatory_allocations + 1;
    {
      dft::CudaKsWb97mvHessian owner;
      assert(owner.prepare(batch, 0, token, Mode::dense_only, options, detail) ==
             GENERATIVEQC_STATUS_SUCCESS);
      assert(owner.work().device_bytes == streamed_bytes + stub::df_tape_bytes);
      assert(owner.work().device_bytes == stub::live_bytes);
    }
    assert(stub::live_bytes == 0);
    stub::fail_allocation = 0;
    stub::retain_df_tape = false;
  }
  std::size_t controls{};
  for (const auto points : {8U, 32U, 256U})
    for (const auto spins : {1U, 2U})
      for (const auto fitted_sources : {0, 1, 2}) {
        const bool df = fitted_sources != 0;
        stub::points = points;
        stub::spins = spins;
        stub::df = df;
        stub::range_df = fitted_sources == 2;
        dft::primary.data.ecp_terms.resize(df ? 1 : 0);
        dft::CudaKsWb97mvHessianOptions serial;
        serial.max_frozen_exact_bytes = 0;
        serial.max_frozen_core_bytes = 0;
        serial.max_resident_cpks_bytes = 0;
        serial.one_electron.parallel_streams = 0;
        serial.grid.replay_nuclear_maps = false;
        std::size_t baseline{}, mandatory_allocations{};
        std::string detail;
        {
          dft::CudaKsWb97mvHessian owner;
          stub::allocation_attempts = 0;
          assert(owner.prepare(batch, 0, token, Mode::dense_and_action, serial, detail) ==
                 GENERATIVEQC_STATUS_SUCCESS);
          mandatory_allocations = stub::allocation_attempts;
          baseline = owner.work().device_bytes;
          assert(baseline == stub::live_bytes);
        }
        assert(stub::live_bytes == 0);
        {
          // Fit a standalone action at a budget that rejects even a dense
          // owner with its optional caches disabled. Leave those caches
          // enabled for the action to exercise mode selection, not cap=0.
          dft::CudaKsWb97mvHessian owner;
          auto action = serial;
          action.max_resident_cpks_bytes = 128ULL << 10;
          action.max_frozen_exact_bytes = 8ULL << 20;
          action.max_frozen_core_bytes = 8ULL << 20;
          action.max_device_bytes = baseline - 1;
          stub::peak_bytes = 0;
          assert(owner.prepare(batch, 0, token, Mode::action_only, action, detail) ==
                 GENERATIVEQC_STATUS_SUCCESS);
          const auto work = owner.work();
          assert(work.device_bytes == stub::live_bytes &&
                 stub::peak_bytes <= action.max_device_bytes);
          assert(!work.cpks_operator_bytes && !work.frozen_exact_bytes && !work.frozen_core_bytes);
        }
        assert(stub::live_bytes == 0 && stub::allocations.empty());
        auto options = serial;
        options.one_electron.parallel_streams = 32;
        options.grid.replay_nuclear_maps = true;
        const auto extra = stub::grid_parallel() - stub::grid_scalar();
        // Every single-byte boundary around either optional layout, including
        // the grid payload/256-byte enclosing-alignment gap, must keep scalar success.
        std::vector<std::size_t> deltas{0, 1, 255, 256, extra / 2, extra + 1024};
        for (std::size_t n = 0; n != 512; ++n) deltas.push_back(extra - 256 + n);
        for (const auto delta : deltas) {
          options.max_device_bytes = baseline + delta;
          stub::peak_bytes = 0;
          {
            dft::CudaKsWb97mvHessian owner;
            const auto status =
                owner.prepare(batch, 0, token, Mode::dense_and_action, options, detail);
            if (status != GENERATIVEQC_STATUS_SUCCESS) {
              std::cerr << "lost scalar request: points=" << points << " spins=" << spins
                        << " df=" << df << " surplus=" << delta << " " << detail << '\n';
              return 1;
            }
            const auto work = owner.work();
            assert(work.device_bytes == stub::live_bytes);
            assert(work.device_bytes <= options.max_device_bytes);
            assert(stub::peak_bytes <= options.max_device_bytes);
            if (!delta) {
              assert(work.grid.workspace_bytes == stub::grid_scalar());
              assert(work.one_electron.device_bytes == stub::core_scalar);
            }
            if (delta == extra + 1024) {
              assert(work.grid.workspace_bytes == stub::grid_parallel());
              assert(work.one_electron.device_bytes == stub::core_parallel);
            }
          }
          assert(stub::live_bytes == 0 && stub::allocations.empty());
          ++controls;
        }
        options.max_device_bytes = baseline - 1;
        {
          dft::CudaKsWb97mvHessian owner;
          assert(owner.prepare(batch, 0, token, Mode::dense_and_action, options, detail) !=
                 GENERATIVEQC_STATUS_SUCCESS);
        }
        assert(stub::live_bytes == 0);
        options.max_device_bytes = baseline + extra + 1024;
        using stub::Provider;
        auto providers = std::vector{Provider::core, Provider::induced};
        if (df)
          providers.insert(providers.end(), {Provider::raw_df, Provider::fitted, Provider::ecp});
        else
          providers.push_back(Provider::exact);
        if (stub::range_df)
          providers.insert(providers.end(), {Provider::raw_df_range, Provider::fitted_range});
        for (const auto failure : providers) {
          stub::reject_provider = failure;
          dft::CudaKsWb97mvHessian owner;
          assert(owner.prepare(batch, 0, token, Mode::dense_and_action, options, detail) !=
                 GENERATIVEQC_STATUS_SUCCESS);
        }
        stub::reject_provider = Provider::none;
        assert(stub::live_bytes == 0);
        for (std::size_t failure = 1; failure <= mandatory_allocations; ++failure) {
          stub::allocation_attempts = 0;
          stub::fail_allocation = failure;
          {
            dft::CudaKsWb97mvHessian owner;
            assert(owner.prepare(batch, 0, token, Mode::dense_and_action, serial, detail) !=
                   GENERATIVEQC_STATUS_SUCCESS);
          }
          assert(stub::live_bytes == 0);
        }
        stub::fail_allocation = 0;
        for (const auto invalid : {0, 1, 2}) {
          auto rejected = options;
          if (invalid == 0) rejected.one_electron.max_jobs = 1;
          if (invalid == 1) rejected.grid.max_nuclear_affine_terms = 1;
          if (invalid == 2) rejected.max_assembly_terms = 1;
          dft::CudaKsWb97mvHessian owner;
          assert(owner.prepare(batch, 0, token, Mode::dense_and_action, rejected, detail) !=
                 GENERATIVEQC_STATUS_SUCCESS);
        }
        assert(stub::live_bytes == 0);
      }
  std::cout << "PASS mandatory-first Hessian admission controls=" << controls << '\n';
}
