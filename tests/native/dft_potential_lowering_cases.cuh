// Included inside the native DFT test namespace. Reuse the independently
// qualified CPU integrator/variational fixtures rather than another rank-2k
// implementation as an oracle for the complete native E/Vxc endpoint.
void potential_lowering_cases() {
  auto molecule = system(3);
  molecule.shells.push_back({0, 3, {{0.51, 1.0}}});
  molecule.shells.push_back({0, 0, {{0.22, 1.0}}});
  const AoBasis basis(molecule);
  const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
  for (unsigned route = 0; route < 4; ++route) {
    for (unsigned functional : {0U, 1U, 2U})
      for (bool uks : {false, true}) {
        // Default, qualified trial, insufficient budget and provider-unavailable.
        xc_potential_qualification_for_test(false, false);
        Fixture fixture(basis, grid, functional, uks, 17);
        xc_potential_qualification_for_test(route != 0, route == 3);
        fixture.plan->prepare_potential(route == 2 ? 0 : 96ULL << 20);
        xc_potential_qualification_for_test(false, false);
        const auto& diagnostic = fixture.plan->potential_lowering();
        require((diagnostic.candidate.provider == "cublas") == (route == 1),
                "potential lowering did not select the admitted provider");
        require(diagnostic.host_bytes == 16U << 10, "potential binding host reservation");
        require(diagnostic.candidate.host_bytes == diagnostic.host_bytes &&
                    diagnostic.indexed_candidate.host_bytes == diagnostic.host_bytes,
                "candidate host charge differs from the prepared owner");
        require(diagnostic.provider_allowance == (route == 1 ? 96ULL << 20 : 0),
                "provider resource charge");
        auto d = density(basis.nao, uks ? 2 : 1);
        compare(fixture, basis, grid, d);
        const auto calls = fixture.layout.spins * ((grid.point_count() + 16) / 17);
        const auto work = fixture.layout.spins * basis.nao * (basis.nao + 1) * grid.point_count() *
                          fixture.layout.work_jets;
        require(fixture.plan->transfers().potential_calls == calls &&
                    fixture.plan->transfers().potential_summands == work,
                "semantic potential work count");
        const auto reference = fixture.potential();
        // Recording is not a physical submission. Replays read changed D from the
        // same captured storage; only explicit publication increments accounting.
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        check(cudaStreamBeginCapture(fixture.stream, cudaStreamCaptureModeGlobal));
        fixture.plan->enqueue_replay_body(fixture.density, d.size());
        check(cudaStreamEndCapture(fixture.stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        require(fixture.plan->transfers().potential_calls == calls, "capture counted as execution");
        check(cudaGraphLaunch(executable, fixture.stream));
        fixture.plan->publish_submitted_generation(++fixture.generation);
        require(fixture.scalars().error == 0, "rank-2k captured finite audit");
        const auto actual = fixture.potential();
        for (std::size_t i = 0; i < actual.size(); ++i)
          close(actual[i], reference[i], "captured potential");
        check(cudaMemsetAsync(fixture.density, 0, d.size() * sizeof(double), fixture.stream));
        check(cudaGraphLaunch(executable, fixture.stream));
        fixture.plan->publish_submitted_generation(++fixture.generation);
        require(fixture.scalars().error == 0, "rank-2k captured zero density");
        for (double value : fixture.potential()) require(value == 0.0, "stale rank-2k output");
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
        require(fixture.plan->transfers().potential_calls == 3 * calls &&
                    fixture.plan->transfers().potential_summands == 3 * work,
                "physical replay accounting");
        fixture.canary();
        // A central difference of the complete energy independently checks both
        // gradient legs and the one-half scalar/tau panel factors.
        compare(fixture, basis, grid, d);
        const auto v = fixture.potential();
        std::vector<double> direction(d.size());
        double expected = 0;
        for (std::size_t i = 0; i < d.size(); ++i) {
          direction[i] = d[i] * 0.1;
          expected += v[i] * direction[i];
        }
        const double step = 1e-4;
        auto plus = d, minus = d;
        for (std::size_t i = 0; i < d.size(); ++i) {
          plus[i] += step * direction[i];
          minus[i] -= step * direction[i];
        }
        fixture.submit(plus);
        const auto ep = fixture.scalars().energy;
        fixture.submit(minus);
        const auto em = fixture.scalars().energy;
        close((ep - em) / (2 * step), expected, "potential energy directional derivative", 2e-7);
        std::cout << "potential route=" << route << " functional=" << functional
                  << " spins=" << fixture.layout.spins << " calls=" << calls << " summands=" << work
                  << " passed\n";
      }
  }
  // Selected maps and zero-column tiles cannot write into global V through a
  // dense library operation. Exercise their real generated fallback after a
  // qualified dense binding, including unchanged nonlocal assembly.
  xc_potential_qualification_for_test(true, false);
  potential_qualification_budget = 96ULL << 20;
  local_ao_cases();
  for (unsigned functional : {0U, 1U}) matrix_response_case(basis, grid, functional, true, 19);
  potential_qualification_budget = 0;
  xc_potential_qualification_for_test(false, false);
}

/** Complete native fixed-density E/Vxc endpoints. These synthetic basis/grid
 * domains expose the actual AO/density/point/potential cost composition, but
 * do not by themselves establish a production molecule/SCF promotion profile.
 * Every warm sample includes density H2D and explicit E/Vxc exports.
 */
void potential_lowering_benchmark() {
  for (unsigned shells : {8U, 18U}) {
    auto molecule = system();
    for (unsigned i = 0; i < shells; ++i)
      molecule.shells.push_back({i % 2, 3, {{0.3 + 0.015 * i, 1.0}}});
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 12, 8, 12, 3, 1e-12});
    const auto d = density(basis.nao, 1);
    const auto reference = integrate_pbe_rks_with_tail(basis, grid, d, 128);
    for (bool trial : {false, true}) {
      xc_potential_qualification_for_test(false, false);
      Fixture fixture(basis, grid, 1, false, 128);
      xc_potential_qualification_for_test(trial, false);
      fixture.plan->prepare_potential(96ULL << 20);
      xc_potential_qualification_for_test(false, false);
      std::array<double, 6> times{};
      double max_error = 0;
      for (auto& seconds : times) {
        const auto start = std::chrono::steady_clock::now();
        fixture.submit(d);
        const auto e = fixture.scalars();
        const auto v = fixture.potential();
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        require(e.error == 0, "benchmark finite publication");
        close(e.energy, reference.energy, "benchmark independent energy", 2e-9);
        for (std::size_t i = 0; i < v.size(); ++i) {
          max_error = std::max(max_error, std::abs(v[i] - reference.potential[i]));
          close(v[i], reference.potential[i], "benchmark independent potential", 2e-9);
        }
      }
      const auto first = times[0];
      std::sort(times.begin() + 1, times.end());
      const auto& diag = fixture.plan->potential_lowering();
      std::cout << std::setprecision(12) << "endpoint nao=" << basis.nao
                << " points=" << grid.point_count() << " provider=" << diag.candidate.provider
                << " prepare_s=" << diag.prepare_seconds << " cold_s=" << first
                << " warm_median_s=" << times[3] << " max_v_error=" << max_error
                << " calls=" << fixture.plan->transfers().potential_calls
                << " summands=" << fixture.plan->transfers().potential_summands << '\n';
      fixture.canary();
    }
  }
}
