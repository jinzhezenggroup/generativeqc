#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "api/handles.hpp"
#include "dft/cuda_ks.hpp"
#include "molecule/basis.hpp"
#include "runtime/cuda_resources.cuh"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda/direct_jk_source_lease.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;

void require(bool accepted, const std::string& reason) {
  if (!accepted) throw std::runtime_error(reason);
}

void cuda_check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

/** Independent PySCF matrices are assertions only; native KS starts cold. */
struct Fixture {
  core::System system;
  std::vector<double> density, coulomb, exchange;
  std::array<std::vector<double>, 2> range_exchange;

  explicit Fixture(const char* path, bool read_oracle = true, bool read_range = false) {
    std::ifstream input(path);
    std::size_t atoms{}, shells{}, nao{};
    input >> atoms >> shells >> nao;
    system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives{};
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    if (read_oracle)
      for (auto* matrix : {&density, &coulomb, &exchange}) {
        matrix->resize(nao * nao);
        for (auto& value : *matrix) input >> value;
      }
    if (read_range)
      for (auto& matrix : range_exchange) {
        matrix.resize(nao * nao);
        for (auto& value : matrix) input >> value;
      }
    std::string detail;
    require(bool(input) &&
                molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
            "invalid independent fixture: " + detail);
    require(nao == molecule::ao_count(system) && nao >= 24, "fixture source admission domain");
  }
};

/** Query the incumbent geometry-only screened traversal, without ERI storage. */
void capacity(const Fixture& fixture) {
  scf::CudaDirectJkPlan* raw{};
  scf::CudaDirectJkDiagnostic diagnostic;
  std::string detail;
  require(scf::create_cuda_direct_jk_plan(0, {fixture.system}, 0, 1e-12, 2ULL << 30, &raw,
                                          diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "capacity preparation: " + detail);
  std::unique_ptr<scf::CudaDirectJkPlan, decltype(&scf::destroy_cuda_direct_jk_plan)> plan(
      raw, scf::destroy_cuda_direct_jk_plan);
  require(plan->canonical_row_prefix, "screened canonical rows unavailable");
  const auto& offsets = plan->canonical_pair_offsets.front();
  std::uint64_t count = 0;
  for (unsigned first = 0; first < 7; ++first)
    for (unsigned second = 0; second <= first; ++second) {
      if (offsets[first] == offsets[first + 1]) continue;
      std::uint64_t bucket = 0;
      cuda_check(cudaMemcpy(
          &bucket, plan->canonical_row_prefix + second * offsets[7] + offsets[first + 1] - 1,
          sizeof(bucket), cudaMemcpyDeviceToHost));
      count += bucket;
    }
  require(scf::prepare_cuda_direct_jk_resident_layout(plan.get(), detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              scf::cuda_direct_jk_resident_value_bytes(plan.get()) == count * sizeof(double),
          "compact source inventory differs from incumbent Schwarz rows: " + detail);
  std::cout << "{\"atoms\":" << fixture.system.atoms.size()
            << ",\"aos\":" << molecule::ao_count(fixture.system)
            << ",\"cartesian_aos\":" << molecule::cartesian_ao_count(fixture.system)
            << ",\"screened_values\":" << count << ",\"screened_value_bytes\":" << 8 * count
            << "}\n";
}

/** Exercise raw coefficients, signed density, screening and source teardown. */
void raw_oracle(const Fixture& fixture) {
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  double* matrices{};
  int* error{};
  const auto elements = fixture.density.size();
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&matrices), 3 * elements * sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(int)));
  for (const double preparation_mask : {0.0, 1e-12})
    for (const std::uint32_t buckets : {(1U << 28) - 1, 0x1555U}) {
      scf::cuda_execution::DirectJkSourceLease source(stream);
      require(
          source.prepare(0, {fixture.system}, 100, molecule::cartesian_ao_count(fixture.system),
                         3 * elements * sizeof(double), 8ULL << 30, "1", "ks-source-oracle", false,
                         false, preparation_mask, 8ULL << 30, preparation_mask == 0.0,
                         molecule::ao_count(fixture.system) < 64
                             ? scf::cuda_execution::DirectJkSourceAutoPolicy::screened_hybrid_cold
                             : scf::cuda_execution::DirectJkSourceAutoPolicy::generic_f,
                         buckets) == GENERATIVEQC_STATUS_SUCCESS &&
              source.active(),
          source.reason());
      require((source.coulomb() != nullptr) == (preparation_mask == 0.0) &&
                  (source.exchange() != nullptr) == (preparation_mask == 0.0),
              "borrowed outputs unexpectedly allocated duplicate J/K slots");
      cuda_check(cudaMemcpyAsync(matrices, fixture.density.data(), elements * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      auto* coulomb = matrices + elements;
      auto* exchange = coulomb + elements;
      std::string detail;
      for (const auto coefficient : {0.25, 0.20}) {
        const auto spec = scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, coefficient);
        require(source.enqueue(spec, matrices, coulomb, exchange, error, preparation_mask,
                               detail) == GENERATIVEQC_STATUS_SUCCESS &&
                    source.audit() == GENERATIVEQC_STATUS_SUCCESS,
                "raw phase J/K audit: " + detail);
        std::vector<double> actual(elements);
        for (const auto& pair :
             {std::pair{coulomb, &fixture.coulomb}, std::pair{exchange, &fixture.exchange}}) {
          cuda_check(cudaMemcpy(actual.data(), pair.first, elements * sizeof(double),
                                cudaMemcpyDeviceToHost));
          double maximum_error = 0.0;
          for (std::size_t index = 0; index < elements; ++index)
            maximum_error =
                std::max(maximum_error, std::abs(actual[index] - (*pair.second)[index]));
          require(maximum_error < 2e-9, "independent PySCF raw J/K parity");
        }
      }
      const auto spec = scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, 0.25);
      if (preparation_mask > 0.0)
        require(source.enqueue(spec, matrices, coulomb, exchange, error, 0.0, detail) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                "screened source accepted a lower replay threshold");
      require(source.enqueue(spec, matrices, coulomb, exchange, error, 1e12, detail) ==
                      GENERATIVEQC_STATUS_SUCCESS &&
                  source.audit() == GENERATIVEQC_STATUS_SUCCESS,
              "screened phase J/K audit");
      std::vector<double> screened(2 * elements);
      cuda_check(cudaMemcpy(screened.data(), coulomb, screened.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
      require(
          std::all_of(screened.begin(), screened.end(), [](double value) { return value == 0; }),
          "resident replay bypassed the Direct Schwarz predicate");
    }
  cuda_check(cudaFree(error));
  cuda_check(cudaFree(matrices));
  cuda_check(cudaStreamDestroy(stream));
}

/** UKS shares ERIs, not spin weights: signed beta checks J(total), K(alpha),
 * K(beta) against the independent fixture for full and partial residency. */
void raw_oracle_uks(const Fixture& fixture) {
  runtime::OwnedCudaStream stream;
  stream.create(0);
  const auto elements = fixture.density.size();
  runtime::OwnedCudaBuffer<double> matrices;
  runtime::OwnedCudaBuffer<int> error;
  matrices.allocate(0, 5 * elements, stream.get());
  error.allocate(0, 1, stream.get());
  std::vector<double> beta_density = fixture.density;
  for (auto& value : beta_density) value *= -0.5;
  auto* alpha = matrices.get();
  auto* beta = alpha + elements;
  auto* coulomb = beta + elements;
  auto* alpha_exchange = coulomb + elements;
  auto* beta_exchange = alpha_exchange + elements;
  cuda_check(cudaMemcpyAsync(alpha, fixture.density.data(), elements * sizeof(double),
                             cudaMemcpyHostToDevice, stream.get()));
  cuda_check(cudaMemcpyAsync(beta, beta_density.data(), elements * sizeof(double),
                             cudaMemcpyHostToDevice, stream.get()));
  const auto check_source = [&](double screening, std::uint32_t buckets) {
    scf::cuda_execution::DirectJkSourceLease source(stream.get());
    require(source.prepare(0, {fixture.system}, 100, molecule::cartesian_ao_count(fixture.system),
                           5 * elements * sizeof(double), 8ULL << 30, "1", "ks-uks-source-oracle",
                           false, false, screening, 8ULL << 30, false,
                           molecule::ao_count(fixture.system) < 64
                               ? scf::cuda_execution::DirectJkSourceAutoPolicy::screened_hybrid_cold
                               : scf::cuda_execution::DirectJkSourceAutoPolicy::generic_f,
                           buckets, 2) == GENERATIVEQC_STATUS_SUCCESS &&
                source.active(),
            source.reason());
    for (const bool has_exchange : {true, false}) {
      auto spec = scf::make_global_hybrid_fock_spec(scf::FockSpin::Unrestricted, 0.25);
      if (!has_exchange) spec.exchange.present = false;
      std::string detail;
      require(
          source.enqueue(spec, alpha, coulomb, has_exchange ? alpha_exchange : nullptr, error.get(),
                         screening, detail, beta,
                         has_exchange ? beta_exchange : nullptr) == GENERATIVEQC_STATUS_SUCCESS &&
              source.audit() == GENERATIVEQC_STATUS_SUCCESS,
          "UKS shared source audit: " + detail);
      std::vector<double> actual(elements);
      for (unsigned channel = 0; channel < (has_exchange ? 3U : 1U); ++channel) {
        const auto* expected = channel == 0 ? &fixture.coulomb : &fixture.exchange;
        const double scale = channel == 0 ? 0.5 : channel == 1 ? 1.0 : -0.5;
        cuda_check(cudaMemcpy(actual.data(), coulomb + channel * elements,
                              elements * sizeof(double), cudaMemcpyDeviceToHost));
        for (std::size_t index = 0; index < elements; ++index)
          require(std::abs(actual[index] - scale * expected->at(index)) < 2e-9,
                  "independent PySCF UKS J/K parity");
      }
    }
  };
  for (const double screening : {0.0, 1e-12})
    for (const std::uint32_t buckets : {(1U << 28) - 1, 0x1555U}) check_source(screening, buckets);
}

/** Independent SR/LR matrices protect radial identity, both spins and partial
 * caches. Invalid operator requests must fail before any retained value is used. */
void raw_range_oracle(const Fixture& fixture) {
  runtime::OwnedCudaStream stream;
  stream.create(0);
  const auto elements = fixture.density.size();
  runtime::OwnedCudaBuffer<double> matrices;
  runtime::OwnedCudaBuffer<int> error;
  matrices.allocate(0, 4 * elements, stream.get());
  error.allocate(0, 1, stream.get());
  auto beta_density = fixture.density;
  for (auto& value : beta_density) value *= -0.5;
  auto* alpha = matrices.get();
  auto* beta = alpha + elements;
  auto* ka = beta + elements;
  auto* kb = ka + elements;
  cuda_check(cudaMemcpyAsync(alpha, fixture.density.data(), elements * sizeof(double),
                             cudaMemcpyHostToDevice, stream.get()));
  cuda_check(cudaMemcpyAsync(beta, beta_density.data(), elements * sizeof(double),
                             cudaMemcpyHostToDevice, stream.get()));
  unsigned checks = 0;
  for (unsigned range = 0; range < 2; ++range)
    for (const unsigned spins : {1U, 2U})
      for (const double screening : {0.0, 1e-12})
        for (const std::uint32_t buckets : {(1U << 28) - 1, 0x1555U}) {
          const auto op = range == 0 ? scf::FockOperator::ShortRange : scf::FockOperator::LongRange;
          scf::cuda_execution::DirectJkSourceLease source(stream.get(), op, 0.3);
          require(
              source.prepare(0, {fixture.system}, 100, molecule::cartesian_ao_count(fixture.system),
                             4 * elements * sizeof(double), 8ULL << 30, "1", "range-source-oracle",
                             false, false, screening, 8ULL << 30, false,
                             scf::cuda_execution::DirectJkSourceAutoPolicy::screened_hybrid_cold,
                             buckets, spins) == GENERATIVEQC_STATUS_SUCCESS &&
                  source.active(),
              source.reason());
          auto spec = scf::make_global_hybrid_fock_spec(
              spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted, 0.25);
          spec.coulomb.present = false;
          spec.exchange.op = op;
          spec.exchange.omega = 0.3;
          std::string detail;
          const auto execute = [&](scf::FockBuildSpec request, double threshold) {
            return source.enqueue(request, alpha, nullptr, ka, error.get(), threshold, detail,
                                  spins == 2 ? beta : nullptr, spins == 2 ? kb : nullptr);
          };
          for (unsigned mismatch = 0; mismatch < 4; ++mismatch) {
            auto wrong = spec;
            // Keep each radial request valid in isolation so rejection tests
            // the immutable lease identity, not ordinary argument validation.
            if (mismatch == 0)
              wrong.exchange.op = range == 0 ? scf::FockOperator::LongRange
                                            : scf::FockOperator::ShortRange;
            if (mismatch == 1) wrong.exchange.omega = 0.4;
            if (mismatch == 2) wrong.coulomb.present = true;
            if (mismatch == 3) {
              wrong.exchange.op = scf::FockOperator::FullRange;
              wrong.exchange.omega = 0.0;
            }
            require(execute(wrong, screening) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                    "range source accepted a different radial consumer");
          }
          require(execute(spec, screening) == GENERATIVEQC_STATUS_SUCCESS &&
                      source.audit() == GENERATIVEQC_STATUS_SUCCESS,
                  "range resident finite/census audit: " + detail);
          std::vector<double> actual(elements);
          for (unsigned spin = 0; spin < spins; ++spin) {
            cuda_check(cudaMemcpy(actual.data(), spin == 0 ? ka : kb, elements * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            const double scale = spin == 0 ? 1.0 : -0.5;
            for (std::size_t index = 0; index < elements; ++index)
              require(std::abs(actual[index] - scale * fixture.range_exchange[range][index]) < 2e-9,
                      "independent PySCF range-K parity");
          }
          require(execute(spec, 1e12) == GENERATIVEQC_STATUS_SUCCESS &&
                      source.audit() == GENERATIVEQC_STATUS_SUCCESS,
                  "range replay changed the immutable screening predicate");
          cuda_check(
              cudaMemcpy(actual.data(), ka, elements * sizeof(double), cudaMemcpyDeviceToHost));
          require(
              std::all_of(actual.begin(), actual.end(), [](double value) { return value == 0.0; }),
              "screened range replay retained nonzero values");
          ++checks;
        }
  std::cout << "range oracle checks " << checks << "\n";
}

void solve_parity(const Fixture& fixture, bool b3lyp, const std::string& state_base) {
  const auto strategy = scf::resolve_fock_build(
      scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, b3lyp ? 0.20 : 0.25),
      scf::FockBackend::Cuda, 1e-12);
  scf::PreparedFockPlan provider(fixture.system, nullptr, strategy, 0);
  const dft::AoBasis basis(fixture.system);
  dft::GridSpec grid_spec;
  grid_spec.radial_points = 24;
  grid_spec.angular_polar = 6;
  grid_spec.angular_azimuth = 12;
  const dft::MolecularGrid grid(fixture.system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 100;
  options.energy_tolerance = 1e-11;
  options.density_tolerance = 1e-10;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.semilocal_exchange_scale = b3lyp ? 1.0 : 0.75;
  dft::CudaKsPlan plan(provider, basis, grid, options,
                       b3lyp ? dft::SemilocalFamily::B3lyp : dft::SemilocalFamily::Pbe, 257);
  const auto base_bytes = plan.resources().provider_device_bytes;
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "0", 1);
  const auto baseline = plan.run(nullptr, false);
  require(baseline.converged, "baseline Hybrid convergence");
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "1", 1);
  const auto resident = plan.run(nullptr, false);
  require(resident.converged && std::abs(resident.energy - baseline.energy) < 2e-9,
          "Hybrid resident/baseline energy parity");
  const auto* chunk = std::getenv("GENERATIVEQC_CUDA_KS_CHUNK");
  const bool chunked = chunk && std::string(chunk) == "2";
  require(
      resident.precision.effective_bits == 64 &&
          (chunked || resident.precision.operator_work_counters_valid) &&
          std::any_of(
              resident.precision_work.operators.begin(), resident.precision_work.operators.end(),
              [expected = plan.transfers().phase_resident_values_built](const auto& record) {
                return record.kind == scf::PrecisionOperatorKind::CoulombRecurrence &&
                       record.compute == scf::PrecisionDtype::Fp64 && record.count == expected;
              }),
      "strict FP64 source construction provenance");
  std::ofstream state(state_base + (b3lyp ? ".b3lyp.json" : ".pbe0.json"));
  state << std::setprecision(17) << "{\"energy\":" << resident.energy;
  for (const auto& pair :
       {std::pair{"density", &resident.density}, std::pair{"points", &grid.points()},
        std::pair{"weights", &grid.weights()}}) {
    state << ",\"" << pair.first << "\":[";
    for (std::size_t index = 0; index < pair.second->size(); ++index)
      state << (index ? "," : "") << (*pair.second)[index];
    state << "]";
  }
  state << "}";
  require(bool(state), "independent energy assertion snapshot");
  require(plan.transfers().phase_resident_values_built > 0 &&
              plan.transfers().phase_resident_fock_actions > 0 &&
              plan.resources().provider_device_bytes > base_bytes &&
              !provider.diagnostic().direct.resident_value_count,
          "final-state source work/lifetime inventory");
  auto built = plan.transfers().phase_resident_values_built;
  const auto rebases = plan.transfers().warm_energy_rebase_fock_actions;
  require(rebases == 1 && resident.fock_builds == resident.iterations + 1 &&
              resident.precision.post_scf_fock_builds == 1,
          "resident final warm handoff was not counted separately");
  require(
      std::count_if(resident.precision_work.events.begin(), resident.precision_work.events.end(),
                    [](const auto& event) {
                      return event.kind == scf::PrecisionWorkEventKind::PostScfFock &&
                             event.phase == scf::PrecisionWorkPhase::Finalization;
                    }) == 1,
      "warm handoff precision provenance");
  dft::CudaKsFinalStateToken token;
  dft::CudaKsResidentDensityBinding density_before, density_after;
  std::string source_detail;
  require(plan.final_state_token(token, source_detail) == GENERATIVEQC_STATUS_SUCCESS &&
              plan.resident_final_density(token, density_before, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
          "resident final-state value consumer binding: " + source_detail);
  dft::CudaKsFixedDensityProfile profile;
  require(plan.profile_fixed_density_components(token, profile, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              profile.present_mask == 11U &&
              plan.resident_final_density(token, density_after, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              density_before.alpha == density_after.alpha &&
              plan.transfers().phase_resident_values_built == built &&
              plan.resources().provider_device_bytes > base_bytes,
          "post-SCF value reuse lost the final density or rebuilt ERIs: " + source_detail);
  // A force handoff must leave a roomy energy-only lease intact, but reclaim
  // just the optional source when the next consumer needs more than free VRAM.
  // Asking for free+1 exercises pressure deterministically without exhausting
  // the physical device or changing the SCF source-selection policy.
  std::array<std::uint64_t, 4> memory_receipt{};
  const auto retained_bytes = plan.resources().provider_device_bytes;
  const auto retained_host_bytes = plan.resources().retained_host_numeric_bytes;
  const auto seed_before_handoff = plan.warm_density();
  require(plan.prepare_force_memory(token, 0, memory_receipt, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              memory_receipt[3] == 0 && plan.resources().provider_device_bytes == retained_bytes,
          "roomy force handoff retired reusable SCF values");
  auto stale_token = token;
  stale_token.version = 0;
  require(plan.prepare_force_memory(stale_token, memory_receipt[1] + 1, memory_receipt,
                                    source_detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              plan.resources().provider_device_bytes == retained_bytes,
          "stale force handoff changed SCF ownership");
  dft::CudaKsResidentStationaryWeightsBinding weights_before, weights_after;
  dft::KsFinalStateDiagnostic weights_diagnostic;
  require(
      plan.prepare_resident_final_stationary_weights(token, weights_before, weights_diagnostic,
                                                     source_detail) == GENERATIVEQC_STATUS_SUCCESS,
      "pre-handoff resident D/W proof: " + source_detail);
  std::size_t free_bytes{}, total_bytes{};
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
  require(plan.prepare_force_memory(token, free_bytes + 1, memory_receipt, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              memory_receipt[3] == retained_bytes - base_bytes && memory_receipt[3] > 0 &&
              plan.resources().provider_device_bytes == base_bytes &&
              plan.resources().retained_host_numeric_bytes < retained_host_bytes &&
              plan.resident_final_density(token, density_after, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              plan.resident_final_stationary_weights(token, weights_after, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              density_after.alpha == density_before.alpha &&
              weights_after.density == weights_before.density &&
              weights_after.weighted_density == weights_before.weighted_density &&
              plan.warm_density() == seed_before_handoff,
          "pressured force handoff lost state, token or resource accounting: " + source_detail);
  dft::CudaKsFinalStateToken after_handoff;
  require(plan.final_state_token(after_handoff, source_detail) == GENERATIVEQC_STATUS_SUCCESS &&
              after_handoff == token &&
              plan.prepare_force_memory(token, total_bytes, memory_receipt, source_detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              memory_receipt[3] == 0,
          "repeat force handoff changed the final token or double-retired values");
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "auto", 1);
  const auto warm = plan.run(nullptr, true);
  require(warm.converged && std::abs(warm.energy - baseline.energy) < 2e-9 &&
              plan.transfers().phase_resident_values_built == built &&
              plan.transfers().warm_energy_rebase_fock_actions == rebases,
          "automatic warm solve rebuilt a phase source");
  require(chunked || warm.fock_builds == 1,
          "resident to incumbent warm solve failed the cached energy gate");
  const auto frozen_density = plan.warm_density();
  plan.set_warm_start_updates(false);
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "1", 1);
  const auto frozen = plan.run(nullptr, true);
  require(frozen.converged && plan.warm_density() == frozen_density &&
              plan.transfers().warm_energy_rebase_fock_actions == rebases &&
              frozen.fock_builds == frozen.iterations,
          "frozen warm cache was overwritten or unnecessarily rebased");
  require(chunked || std::isinf(frozen.dft_diagnostic.history.front().energy_change),
          "forced resident warm phase compared an incumbent energy baseline");
  plan.set_warm_start_updates(true);
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "auto", 1);
  const auto after_frozen = plan.run(nullptr, true);
  require(after_frozen.converged && (chunked || after_frozen.fock_builds == 1) &&
              plan.transfers().warm_energy_rebase_fock_actions == rebases,
          "frozen solve invalidated the last-good incumbent warm baseline");
  built = plan.transfers().phase_resident_values_built;
  const auto seeded = plan.run(&baseline.density, false);
  require(seeded.converged && plan.transfers().phase_resident_values_built == built,
          "automatic seeded solve rebuilt a phase source");
  const auto automatic = plan.run(nullptr, false);
  unsigned maximum_angular = 0;
  for (const auto& shell : fixture.system.shells)
    maximum_angular = std::max(maximum_angular, shell.angular_momentum);
  const bool auto_domain = !scf::cuda_execution::direct_jk_source_auto_refusal(
      molecule::ao_count(fixture.system), molecule::cartesian_ao_count(fixture.system),
      options.max_iterations, maximum_angular, true,
      scf::cuda_direct_jk_value_census_available(provider.cuda_direct_source(), strategy.spec),
      scf::cuda_execution::DirectJkSourceAutoPolicy::screened_hybrid_cold,
      options.screening_tolerance);
  require(automatic.converged && std::abs(automatic.energy - baseline.energy) < 2e-9 &&
              (plan.transfers().phase_resident_values_built > built) == auto_domain,
          "automatic cold Hybrid source domain");
  const auto automatic_built = plan.transfers().phase_resident_values_built;
  const auto native_guess =
      plan.run(&baseline.density, false, true, dft::CudaKsSeedOrigin::NativeColdGuess);
  require(native_guess.converged && std::abs(native_guess.energy - baseline.energy) < 2e-9 &&
              (plan.transfers().phase_resident_values_built > automatic_built) == auto_domain,
          "native cold initial guess lost automatic source eligibility");
  const auto before_default = plan.transfers().phase_resident_values_built;
  unsetenv("GENERATIVEQC_KS_RESIDENT_VALUES");
  const auto default_cold = plan.run(nullptr, false);
  require(default_cold.converged && std::abs(default_cold.energy - baseline.energy) < 2e-9 &&
              (plan.transfers().phase_resident_values_built > before_default) == auto_domain,
          "unset selection lost the bounded automatic cold policy");
  const auto default_built = plan.transfers().phase_resident_values_built;
  const auto default_warm = plan.run(nullptr, true);
  require(default_warm.converged && (chunked || default_warm.fock_builds == 1) &&
              plan.transfers().phase_resident_values_built == default_built,
          "unset warm selection rebuilt values or lost the one-Fock gate");
  const auto default_seeded = plan.run(&baseline.density, false);
  require(default_seeded.converged && plan.transfers().phase_resident_values_built == default_built,
          "unset external seed rebuilt optional values");
  const auto completed_built = plan.transfers().phase_resident_values_built;
  {
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->device = 0;
    ledger->limit = 1;
    runtime::active_device_resource_ledger = ledger;
    setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "1", 1);
    const auto bounded = plan.run(nullptr, true);
    runtime::active_device_resource_ledger.reset();
    require(bounded.converged && plan.transfers().phase_resident_values_built == completed_built &&
                ledger->allocations == 0 && ledger->rejected == 0,
            "forced source bypassed the public resource plan");
  }
  auto bounded_options = options;
  bounded_options.reference_memory_budget_bytes = 1;
  for (const auto* selection : {"1", "auto", static_cast<const char*>(nullptr)}) {
    if (selection)
      setenv("GENERATIVEQC_KS_RESIDENT_VALUES", selection, 1);
    else
      unsetenv("GENERATIVEQC_KS_RESIDENT_VALUES");
    dft::CudaKsPlan bounded(provider, basis, grid, bounded_options,
                            b3lyp ? dft::SemilocalFamily::B3lyp : dft::SemilocalFamily::Pbe, 257);
    const auto refusal = bounded.run(nullptr, false);
    require(refusal.converged && std::abs(refusal.energy - baseline.energy) < 2e-9 &&
                bounded.transfers().phase_resident_values_built == 0,
            "source capacity refusal lost the exact fallback");
  }
  std::cout << "{\"functional\":\"" << (b3lyp ? "B3LYP" : "PBE0")
            << "\",\"energy\":" << resident.energy << ",\"source_values\":" << completed_built
            << ",\"source_actions\":" << plan.transfers().phase_resident_fock_actions
            << ",\"warm_focks\":" << warm.fock_builds
            << ",\"warm_energy_change\":" << warm.energy_change
            << ",\"warm_energy_rebases\":" << plan.transfers().warm_energy_rebase_fock_actions
            << ",\"source_bytes\":" << plan.transfers().phase_resident_value_bytes << "}\n";
}

/** Mixed epochs must not read strict values; refinement restores their bounded
 * lease without admitting chunk scratch or changing warm/seed provenance. */
void mixed_refinement(const Fixture& fixture, bool b3lyp, const std::string& state_base) {
  const auto strategy = scf::resolve_fock_build(
      scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, b3lyp ? 0.20 : 0.25),
      scf::FockBackend::Cuda, 1e-12);
  scf::PreparedFockPlan provider(fixture.system, nullptr, strategy, 0);
  const dft::AoBasis basis(fixture.system);
  dft::GridSpec grid_spec;
  grid_spec.radial_points = 24;
  grid_spec.angular_polar = 6;
  grid_spec.angular_azimuth = 12;
  const dft::MolecularGrid grid(fixture.system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 100;
  options.energy_tolerance = 1e-11;
  options.density_tolerance = 1e-10;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.semilocal_exchange_scale = b3lyp ? 1.0 : 0.75;
  const auto family = b3lyp ? dft::SemilocalFamily::B3lyp : dft::SemilocalFamily::Pbe;
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "0", 1);
  dft::CudaKsPlan strict(provider, basis, grid, options, family, 257);
  const auto baseline = strict.run(nullptr, false);
  require(baseline.converged, "mixed refinement strict baseline");
  options.precision_mode = GENERATIVEQC_PRECISION_AUTO;
  setenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "1", 1);
  setenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS", "1", 1);
  for (const auto* selection : {"0", "1"}) {
    setenv("GENERATIVEQC_KS_RESIDENT_VALUES", selection, 1);
    dft::CudaKsPlan plan(provider, basis, grid, options, family, 257);
    const bool cached = std::string(selection) == "1";
    plan.begin(nullptr, false);
    require(plan.transfers().phase_resident_values_built == 0,
            "mixed epoch prepared a strict value lease");
    plan.enqueue_iteration();
    require(plan.finish_iteration(), "mixed observation omitted strict refinement");
    require((plan.transfers().phase_resident_values_built > 0) == cached,
            "precision handoff lost strict resident admission");
    while (plan.active()) {
      plan.enqueue_iteration();
      plan.finish_iteration();
    }
    const auto result = plan.result();
    require(result.converged && std::abs(result.energy - baseline.energy) < 2e-9 &&
                result.precision.mixed_stage_fock_builds == 1 &&
                result.precision.strict_stage_fock_builds >= 2 &&
                result.precision.strict_refinement_applied &&
                result.precision.operator_work_counters_valid &&
                (plan.transfers().phase_resident_fock_actions > 0) == cached,
            "mixed resident/fallback scientific or provenance gate");
    if (cached) {
      std::ofstream state(state_base + (b3lyp ? ".b3lyp.json" : ".pbe0.json"));
      state << std::setprecision(17) << "{\"energy\":" << result.energy;
      for (const auto& pair :
           {std::pair{"density", &result.density}, std::pair{"points", &grid.points()},
            std::pair{"weights", &grid.weights()}}) {
        state << ",\"" << pair.first << "\":[";
        for (std::size_t index = 0; index < pair.second->size(); ++index)
          state << (index ? "," : "") << (*pair.second)[index];
        state << "]";
      }
      state << "}";
      require(bool(state), "mixed independent energy snapshot");
      const auto built = plan.transfers().phase_resident_values_built;
      setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "auto", 1);
      const auto warm = plan.run(nullptr, true);
      require(warm.converged && std::abs(warm.energy - baseline.energy) < 2e-9 &&
                  plan.transfers().phase_resident_values_built == built,
              "mixed handoff relabeled a warm seed as cold");
      const auto external = plan.run(&result.density, false);
      require(external.converged && plan.transfers().phase_resident_values_built == built,
              "mixed handoff relabeled an external density as cold");
    }
  }
  setenv("GENERATIVEQC_KS_RESIDENT_VALUES", "1", 1);
  options.reference_memory_budget_bytes = 1;
  dft::CudaKsPlan bounded(provider, basis, grid, options, family, 257);
  const auto fallback = bounded.run(nullptr, false);
  require(fallback.converged && std::abs(fallback.energy - baseline.energy) < 2e-9 &&
              bounded.transfers().phase_resident_values_built == 0,
          "mixed strict lease bypassed its budget or lost fallback");
}

/** Time a complete public cold endpoint, including provider preparation.
 * This legacy C-API owner rejects KS nuclear gradients. Complete E+F must use
 * the high-level compiled stationary endpoint, not this energy-only probe.
 */
void endpoint(const Fixture& fixture, bool b3lyp, bool compute_forces) {
  const auto started = std::chrono::steady_clock::now();
  generativeqc_context_descriptor context_spec{sizeof(generativeqc_context_descriptor),
                                               GENERATIVEQC_ABI_VERSION, 0,
                                               GENERATIVEQC_BACKEND_CUDA};
  generativeqc_context* raw_context{};
  require(generativeqc_context_create(&context_spec, &raw_context) == GENERATIVEQC_STATUS_SUCCESS,
          "public Hybrid context");
  std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)> context(
      raw_context, generativeqc_context_destroy);
  generativeqc_system system{fixture.system};
  const auto family = b3lyp ? dft::SemilocalFamily::B3lyp : dft::SemilocalFamily::Pbe;
  const auto* metadata =
      dft::semilocal_family_metadata_from_code(dft::semilocal_family_code(family));
  std::vector<generativeqc_ks_semilocal_component> components;
  for (std::size_t index = 0; index < metadata->component_count; ++index)
    components.push_back({metadata->component_ids[index],
                          !b3lyp && index == 0 ? 0.75 : metadata->component_coefficients[index]});
  const generativeqc_ks_exchange_term exact_exchange{
      GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, b3lyp ? 0.20 : 0.25, 0.0, b3lyp ? -0.10 : -0.125};
  generativeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.scf_domain = metadata->scf_domain;
  ks.grid_version = 1;
  ks.radial_points = 24;
  ks.angular_polar = 6;
  ks.angular_azimuth = 12;
  ks.partition_iterations = 3;
  ks.coincident_tolerance = 1e-12;
  ks.tile_points = 257;
  ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.spin_channels = 1;
  ks.semilocal_components = components.data();
  ks.semilocal_component_count = components.size();
  ks.exchange_terms = &exact_exchange;
  ks.exchange_term_count = 1;
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = GENERATIVEQC_METHOD_PBE_RKS;
  method.ks_options = &ks;
  method.max_iterations = 100;
  method.diis_history = 8;
  method.energy_tolerance = 1e-11;
  method.density_tolerance = 1e-10;
  method.screening_tolerance = 1e-12;
  method.precision_mode = GENERATIVEQC_PRECISION_FP64;
  method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  generativeqc_calculation* raw_calculation{};
  require(generativeqc_calculation_prepare(context.get(), &system, &method, &raw_calculation) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "public Hybrid preparation");
  std::unique_ptr<generativeqc_calculation, decltype(&generativeqc_calculation_destroy)>
      calculation(raw_calculation, generativeqc_calculation_destroy);
  std::vector<double> forces(compute_forces ? 3 * fixture.system.atoms.size() : 0);
  generativeqc_result_descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  if (compute_forces) {
    result.forces = forces.data();
    result.force_count = forces.size();
  }
  const auto status = generativeqc_calculation_execute(calculation.get(), &result);
  require(status == GENERATIVEQC_STATUS_SUCCESS && result.converged &&
              result.executed_backend == GENERATIVEQC_BACKEND_CUDA,
          "public Hybrid endpoint status=" + std::to_string(status) +
              " converged=" + std::to_string(result.converged) +
              " backend=" + std::to_string(result.executed_backend) + ": " +
              generativeqc_context_last_error(context.get()));
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  require(calculation->precision_work && calculation->precision_work->operator_inventory_complete,
          "public Hybrid SCF work inventory");
  std::uint64_t source_values = 0, fock_builds = 0;
  for (const auto& record : calculation->precision_work->operators) {
    if (record.kind == scf::PrecisionOperatorKind::CoulombRecurrence) source_values += record.count;
    if (record.kind == scf::PrecisionOperatorKind::FockAssembly) fock_builds += record.count;
  }
  // The cache handoff evaluates J/K and its energy, not another Fock matrix.
  // Assembly counts alone would hide that real post-SCF operator application.
  const auto handoff_focks = calculation->precision.post_scf_fock_builds;
  fock_builds += handoff_focks;
  std::cout << "{\"functional\":\"" << (b3lyp ? "B3LYP" : "PBE0") << "\",\"seconds\":" << seconds
            << ",\"properties\":\"" << (compute_forces ? "E+F" : "E") << "\""
            << ",\"energy\":" << result.energy << ",\"iterations\":" << result.iterations
            << ",\"source_values\":" << source_values << ",\"fock_builds\":" << fock_builds
            << ",\"warm_energy_rebase_focks\":" << handoff_focks << ",\"forces\":[";
  for (std::size_t index = 0; index < forces.size(); ++index)
    std::cout << (index ? "," : "") << forces[index];
  std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(
        argc == 2 ||
            (argc == 3 &&
             (std::string(argv[2]) == "capacity" || std::string(argv[2]) == "range-oracle" ||
              std::string(argv[2]) == "endpoint" || std::string(argv[2]) == "endpoint-forces" ||
              std::string(argv[2]) == "mixed-refinement")),
        "usage: ks-phase-values-probe INPUT [capacity|endpoint|endpoint-forces|mixed-refinement]");
    const bool range_oracle = argc == 3 && std::string(argv[2]) == "range-oracle";
    const Fixture fixture(argv[1],
                          argc == 2 || std::string(argv[2]) == "mixed-refinement" || range_oracle,
                          range_oracle);
    std::cout << std::setprecision(17);
    if (range_oracle) {
      raw_range_oracle(fixture);
    } else if (argc == 3 && std::string(argv[2]) == "capacity") {
      capacity(fixture);
    } else if (argc == 3 && std::string(argv[2]) == "mixed-refinement") {
      mixed_refinement(fixture, false, argv[1]);
      mixed_refinement(fixture, true, argv[1]);
    } else if (argc == 3) {
      const bool compute_forces = std::string(argv[2]) == "endpoint-forces";
      endpoint(fixture, false, compute_forces);
      endpoint(fixture, true, compute_forces);
    } else {
      raw_oracle(fixture);
      raw_oracle_uks(fixture);
      solve_parity(fixture, false, argv[1]);
      solve_parity(fixture, true, argv[1]);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
