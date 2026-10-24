int main() {
  using namespace generativeqc;
  methods::PreparedBatch batch;
  dft::CudaKsFinalStateToken token;
  std::size_t controls{};
  for (const auto points : {8U, 32U, 256U})
    for (const auto spins : {1U, 2U})
      for (const auto df : {false, true}) {
        stub::points = points;
        stub::spins = spins;
        stub::df = stub::ecp = df;
        dft::primary.data.ecp_terms.resize(df ? 1 : 0);
        dft::CudaKsWb97mvHessianOptions serial;
        serial.max_frozen_exact_bytes = 0;
        serial.max_resident_cpks_bytes = 0;
        serial.one_electron.parallel_streams = 0;
        serial.grid.replay_nuclear_maps = false;
        std::size_t baseline{};
        std::string detail;
        {
          dft::CudaKsWb97mvHessian owner;
          assert(owner.prepare(batch, 0, token, serial, detail) == GENERATIVEQC_STATUS_SUCCESS);
          baseline = owner.work().device_bytes;
          assert(baseline == stub::live_bytes);
        }
        assert(stub::live_bytes == 0);
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
            const auto status = owner.prepare(batch, 0, token, options, detail);
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
          assert(owner.prepare(batch, 0, token, options, detail) != GENERATIVEQC_STATUS_SUCCESS);
        }
        assert(stub::live_bytes == 0);
        options.max_device_bytes = baseline + extra + 1024;
        for (int failure = 1; failure <= (df ? 6 : 3); ++failure) {
          stub::reject_provider = failure;
          dft::CudaKsWb97mvHessian owner;
          assert(owner.prepare(batch, 0, token, options, detail) != GENERATIVEQC_STATUS_SUCCESS);
        }
        stub::reject_provider = 0;
        assert(stub::live_bytes == 0);
        for (std::size_t failure = 1; failure <= (df ? 7U : 4U); ++failure) {
          stub::allocation_attempts = 0;
          stub::fail_allocation = failure;
          {
            dft::CudaKsWb97mvHessian owner;
            assert(owner.prepare(batch, 0, token, options, detail) != GENERATIVEQC_STATUS_SUCCESS);
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
          assert(owner.prepare(batch, 0, token, rejected, detail) != GENERATIVEQC_STATUS_SUCCESS);
        }
        assert(stub::live_bytes == 0);
      }
  std::cout << "PASS mandatory-first Hessian admission controls=" << controls << '\n';
}
