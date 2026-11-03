// Independent fixed-density RSH source qualification. Included in the native
// composition test's anonymous namespace, after its fixture/assertion helpers.
void emit_rsh_vector(std::string_view label, const std::vector<double>& values) {
  std::cout << "RSH_VECTOR " << label << " [";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << "]\n";
}

std::vector<double> independent_rsh_source_reference(const System& system, FockSpin spin,
                                                     const std::vector<double>& density,
                                                     const std::vector<double>& beta,
                                                     double omega) {
  auto full_spec = make_rsh_primary_fock_spec(spin, 0.2);
  full_spec.derivative_order = 1;
  PreparedFockPlan full(system, nullptr, resolve_fock_build(full_spec, FockBackend::Cpu, 0.0), -1,
                        0);
  const auto full_components = full.energy_derivative_components(density, beta);
  const auto correction = make_rsh_correction_fock_spec(spin, 0.2, 0.5, omega);
  const auto n = generativeqc::molecule::ao_count(system);
  const auto coordinates = 3 * system.atoms.size();
  auto range_reference = [&](FockOperator op, double coefficient, double step) {
    auto spec = full_spec;
    spec.derivative_order = 0;
    spec.coulomb.present = false;
    spec.exchange.op = op;
    spec.exchange.omega = omega;
    spec.exchange.coefficient = coefficient;
    const auto model = resolve_fock_build(spec, FockBackend::Cpu, 0.0);
    const auto range = op == FockOperator::ShortRange ? generativeqc::integrals::CoulombRange::Short
                                                      : generativeqc::integrals::CoulombRange::Long;
    std::vector<double> response(coordinates);
    constexpr std::array<int, 4> offsets{-2, -1, 1, 2};
    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      std::array<double, 4> energies{};
      for (std::size_t point = 0; point < offsets.size(); ++point) {
        auto moved = system;
        moved.atoms[coordinate / 3].position[coordinate % 3] += offsets[point] * step;
        const auto eri = generativeqc::integrals::build_range_eri(moved, range, omega);
        energies[point] = contract_fock_energy(
            model, build_exact_direct_jk(model, n, eri, density, beta), density, beta);
      }
      response[coordinate] =
          (energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]) / (12 * step);
      std::cout << "RSH_CPU_STENCIL {\"range\":\""
                << (op == FockOperator::ShortRange ? "short" : "long")
                << "\",\"coefficient\":" << coefficient << ",\"step\":" << step
                << ",\"coordinate\":" << coordinate << ",\"energies\":[" << energies[0] << ','
                << energies[1] << ',' << energies[2] << ',' << energies[3]
                << "],\"derivative\":" << response[coordinate] << "}\n";
    }
    return response;
  };
  auto result = full_components.coulomb;
  for (const auto op : {FockOperator::ShortRange, FockOperator::LongRange}) {
    const auto coefficient = full_spec.exchange.coefficient +
                             (op == FockOperator::LongRange ? correction.exchange.coefficient : 0);
    const auto coarse = range_reference(op, coefficient, 2e-4);
    const auto fine = range_reference(op, coefficient, 1e-4);
    matrix(coarse, fine, "independent CPU range-value derivative convergence");
    result.insert(result.end(), fine.begin(), fine.end());
  }
  return result;
}

void independent_rsh_source_cases() {
  std::cout.precision(17);
  for (unsigned basis = 0; basis < 3; ++basis)
    for (const auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
      auto system = fixture(true);
      if (basis != 0) {
        system.basis_representation =
            basis == 1 ? GENERATIVEQC_BASIS_CARTESIAN : GENERATIVEQC_BASIS_SPHERICAL;
        system.shells.push_back({0, 2, {{0.35, 1.0}}});
        std::string detail;
        require(generativeqc::molecule::validate_and_normalize(system, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());
      }
      const auto n = generativeqc::molecule::ao_count(system);
      const auto coordinates = 3 * system.atoms.size();
      std::vector<double> density(n * n), beta;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          density[i * n + j] =
              (i == j ? 0.6 : 0.0) + 0.1 * std::cos(0.31 * (i + 1)) * std::cos(0.31 * (j + 1));
      if (spin == FockSpin::Unrestricted) {
        beta = density;
        for (auto& value : beta) value *= 0.3;
      }
      const auto strategy =
          resolve_fock_build(make_rsh_primary_fock_spec(spin, 0.2), FockBackend::Cuda, 0.0);
      std::size_t primitives = 0;
      for (const auto& shell : system.shells) primitives += shell.primitives.size();
      const auto budget = cuda_direct_coulomb_device_bytes(1, n, system.atoms.size(),
                                                           system.shells.size(), primitives, 1);
      PreparedFockPlan plan(system, nullptr, strategy, 0, budget, 1);
      const auto binding = prepared_cuda_direct_derivative_binding(plan);
      require(static_cast<bool>(binding), "independent RSH fixture has no derivative owner");
      struct DeviceDensity {
        double* values{};
        ~DeviceDensity() {
          if (values) (void)cudaFree(values);
        }
      } device;
      require(cudaMalloc(reinterpret_cast<void**>(&device.values), 2 * n * n * sizeof(double)) ==
                  cudaSuccess,
              "independent RSH density allocation failed");
      require(cudaMemcpyAsync(device.values, density.data(), n * n * sizeof(double),
                              cudaMemcpyHostToDevice, binding.stream) == cudaSuccess,
              "independent RSH alpha upload failed");
      const double* device_beta = nullptr;
      if (!beta.empty()) {
        device_beta = device.values + n * n;
        require(cudaMemcpyAsync(device.values + n * n, beta.data(), n * n * sizeof(double),
                                cudaMemcpyHostToDevice, binding.stream) == cudaSuccess,
                "independent RSH beta upload failed");
      }
      for (const double omega : {0.3, 0.4, 0.5}) {
        // The s/p cases cover default and two exact runtime radial identities;
        // both d representations additionally cover the failing non-default gate.
        if (basis != 0 && omega != 0.4) continue;
        std::cout << "RSH_CASE {\"basis\":" << basis << ",\"nbf\":" << n << ",\"spin\":\""
                  << (spin == FockSpin::Restricted ? "RKS" : "UKS") << "\",\"omega\":" << omega
                  << "}\n";
        emit_rsh_vector("density-alpha", density);
        emit_rsh_vector("density-beta", beta);
        const auto correction = resolve_fock_build(
            make_rsh_correction_fock_spec(spin, 0.2, 0.5, omega), FockBackend::Cuda, 0.0);
        const auto expected = independent_rsh_source_reference(system, spin, density, beta, omega);
        emit_rsh_vector("cpu", expected);
        std::vector<double> host, resident;
        std::string detail;
        require(execute_prepared_cuda_direct_rsh_energy_derivatives(
                    plan, correction, density, beta, host, detail) == GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());
        // Deliberately leave a different host-uploaded density in the owner.
        // Resident/borrowed calls must honor their external device pointers.
        auto poison_density = density, poison_beta = beta, poison_expected = expected;
        for (auto& value : poison_density) value *= 0.7;
        for (auto& value : poison_beta) value *= 0.7;
        for (auto& value : poison_expected) value *= 0.49;
        auto replace_host_density = [&] {
          std::vector<double> poison;
          require(execute_prepared_cuda_direct_rsh_energy_derivatives(
                      plan, correction, poison_density, poison_beta, poison, detail) ==
                      GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          matrix(poison, poison_expected, "distinct preceding host density source derivatives");
        };
        emit_rsh_vector("host", host);
        matrix(host, expected, "host RSH independent CPU source derivatives");
        replace_host_density();
        require(execute_prepared_cuda_direct_rsh_energy_derivatives_device(
                    plan, correction, device.values, device_beta, n * n, resident, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());
        emit_rsh_vector("resident", resident);
        matrix(resident, expected, "resident RSH independent CPU source derivatives");

        if (basis == 0 && omega == 0.4) {
          // A failed callback may have queued a copy. The provider must drain
          // it, preserve its typed status, and remain reusable by the check below.
          DeviceDensity failed_device;
          require(cudaMalloc(reinterpret_cast<void**>(&failed_device.values),
                             3 * coordinates * sizeof(double)) == cudaSuccess,
                  "RSH failure copy allocation failed");
          struct CompletionEvent {
            cudaEvent_t value{};
            ~CompletionEvent() {
              if (value) (void)cudaEventDestroy(value);
            }
          } completion;
          require(
              cudaEventCreateWithFlags(&completion.value, cudaEventDisableTiming) == cudaSuccess,
              "RSH failure completion event creation failed");
          bool witnessed_pending = false;
          for (const auto status :
               {GENERATIVEQC_STATUS_CUDA_ERROR, GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                GENERATIVEQC_STATUS_NUMERICAL_FAILURE, GENERATIVEQC_STATUS_INVALID_ARGUMENT}) {
            require(consume_prepared_cuda_direct_rsh_energy_derivatives_device(
                        plan, correction, device.values, device_beta, n * n,
                        [&](const CudaDerivativeSourceView& view) {
                          require(view.channels == 3, "general RSH failure fixture lost rows");
                          // Device copies avoid pageable-host synchronization.
                          // Retain an actual not-ready witness before throwing.
                          for (unsigned copy = 0; copy < 32; ++copy)
                            require(cudaMemcpyAsync(failed_device.values, view.values,
                                                    3 * coordinates * sizeof(double),
                                                    cudaMemcpyDeviceToDevice,
                                                    binding.stream) == cudaSuccess,
                                    "RSH failure fixture device copy failed");
                          require(cudaEventRecord(completion.value, binding.stream) == cudaSuccess,
                                  "RSH failure completion event recording failed");
                          const auto ready = cudaEventQuery(completion.value);
                          require(ready == cudaSuccess || ready == cudaErrorNotReady,
                                  "RSH failure completion event query failed");
                          witnessed_pending |= ready == cudaErrorNotReady;
                          throw generativeqc::Error(status, "injected RSH callback failure");
                        },
                        detail) == status,
                    "RSH callback failure lost its typed status");
            require(cudaStreamQuery(binding.stream) == cudaSuccess,
                    "failed RSH callback returned before its copy drained");
            require(cudaEventQuery(completion.value) == cudaSuccess,
                    "failed RSH callback returned before its completion event");
            std::vector<double> failed_copy(3 * coordinates);
            require(cudaMemcpy(failed_copy.data(), failed_device.values,
                               failed_copy.size() * sizeof(double),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                    "RSH drained failure copy export failed");
            matrix(failed_copy, expected, "RSH asynchronous failure copy source derivatives");
          }
          require(witnessed_pending, "RSH failure drain lacks a pending device-work witness");
          std::cout << "RSH_FAILURE_DRAIN {\"pending_witness\":true,\"spin\":\""
                    << (spin == FockSpin::Restricted ? "RKS" : "UKS") << "\"}\n";
          require(cudaStreamBeginCapture(binding.stream, cudaStreamCaptureModeThreadLocal) ==
                      cudaSuccess,
                  "RSH capture fixture failed");
          bool published = false;
          const auto refused = consume_prepared_cuda_direct_rsh_energy_derivatives_device(
              plan, correction, device.values, device_beta, n * n,
              [&](const CudaDerivativeSourceView&) { published = true; }, detail);
          cudaGraph_t graph{};
          require(cudaStreamEndCapture(binding.stream, &graph) == cudaSuccess,
                  "RSH capture refusal invalidated the caller graph");
          if (graph)
            require(cudaGraphDestroy(graph) == cudaSuccess, "RSH capture graph cleanup failed");
          require(refused == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !published,
                  "RSH source was published during capture");
        }

        // Copies remain alive across callback return and storage reuse. Each
        // source keeps its own transport convention until the independent check.
        std::vector<double> copies(6 * coordinates), signs(3, 0);
        bool have_full = false, have_correction = false, have_rsh = false;
        replace_host_density();
        require(
            consume_prepared_cuda_direct_rsh_energy_derivatives_device(
                plan, correction, device.values, device_beta, n * n,
                [&](const CudaDerivativeSourceView& view) {
                  require(view.coordinates == coordinates && view.device == binding.device_id &&
                              view.producer_stream == reinterpret_cast<void*>(binding.stream),
                          "RSH callback has incompatible device or extent");
                  std::size_t offset = 0, slot = 0;
                  if (view.kind == CudaDerivativeSourceKind::FullRangeJk) {
                    require(!have_full && view.channels == 2, "invalid full RSH source rows");
                    have_full = true;
                  } else if (view.kind == CudaDerivativeSourceKind::RangeCorrection) {
                    require(!have_correction && view.channels == 1, "invalid LR correction row");
                    have_correction = true;
                    offset = 2 * coordinates;
                    slot = 1;
                  } else {
                    require(view.kind == CudaDerivativeSourceKind::RangeSeparatedJk && !have_rsh &&
                                view.channels == 3,
                            "invalid general RSH source rows");
                    have_rsh = true;
                    offset = 3 * coordinates;
                    slot = 2;
                  }
                  signs[slot] = view.convention == CudaDerivativeConvention::Force ? -1 : 1;
                  require(cudaMemcpyAsync(copies.data() + offset, view.values,
                                          view.channels * coordinates * sizeof(double),
                                          cudaMemcpyDeviceToHost, binding.stream) == cudaSuccess,
                          "RSH callback copy failed");
                },
                detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
        require(cudaStreamSynchronize(binding.stream) == cudaSuccess,
                "RSH callback copy drain failed");
        std::vector<double> callback(3 * coordinates);
        require(have_rsh ? (!have_full && !have_correction) : (have_full && have_correction),
                "RSH provider published an incomplete or overlapping source inventory");
        for (std::size_t i = 0; i < coordinates; ++i) {
          if (have_rsh) {
            for (unsigned row = 0; row < 3; ++row)
              callback[row * coordinates + i] = signs[2] * copies[(3 + row) * coordinates + i];
          } else {
            const auto correction_value = signs[1] * copies[2 * coordinates + i];
            const auto ratio =
                strategy.spec.exchange.coefficient / correction.spec.exchange.coefficient;
            callback[i] = signs[0] * copies[i];
            callback[coordinates + i] =
                signs[0] * copies[coordinates + i] - ratio * correction_value;
            callback[2 * coordinates + i] = (1 + ratio) * correction_value;
          }
        }
        emit_rsh_vector("borrowed", callback);
        matrix(callback, expected, "borrowed RSH independent CPU source derivatives");
      }
    }
}
