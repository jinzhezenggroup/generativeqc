// Complete actual-body host-call trace; no GPU arithmetic or timing claims.
#define main provider_probe_main
#include "test_shared_eigen_provider.cpp"
#undef main

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "solver/cuda/generalized_eigen.hpp"

std::uint32_t private_generalized_phase(generativeqc::solver::GeneralizedEigenBasis, bool,
                                        generativeqc::solver::GeneralizedEigenDomain, void*,
                                        generativeqc::solver::GeneralizedEigenMatrices,
                                        shared::GeneralizedEigenPointerMatrices);

namespace orchestration {
struct Trace {
  std::string name;
  std::vector<std::uint64_t> args;
  bool fallible;
  bool operator==(const Trace& other) const {
    return name == other.name && args == other.args && fallible == other.fallible;
  }
};
std::vector<Trace> trace;
std::size_t fail = std::numeric_limits<std::size_t>::max();
int failure_variant = 0;
template <class T, std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>, int> = 0>
void append(Trace& t, T value) {
  t.args.push_back(static_cast<std::uint64_t>(value));
}
template <class T>
void append(Trace& t, T* value) {
  t.args.push_back(reinterpret_cast<std::uintptr_t>(value));
}
void append(Trace& t, double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  t.args.push_back(bits);
}
}  // namespace orchestration

#include "generalized_gfn_types.inc"

namespace orchestration {
template <class... Args>
bool record(const char* name, bool fallible, const Args&... args) {
  Trace entry{name, {}, fallible};
  (append(entry, args), ...);
  trace.push_back(std::move(entry));
  return fallible && trace.size() - 1 == fail;
}
cublasStatus_t blas_status(bool failed) {
  return failed ? (failure_variant ? CUBLAS_STATUS_ALLOC_FAILED : CUBLAS_STATUS_EXECUTION_FAILED)
                : CUBLAS_STATUS_SUCCESS;
}
cudaError_t cuda_status(bool failed) {
  return failed ? (failure_variant ? cudaErrorMemoryAllocation : cudaErrorUnknown) : cudaSuccess;
}
cusolverStatus_t solver_status(bool failed) {
  return failed
             ? (failure_variant ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_EXECUTION_FAILED)
             : CUSOLVER_STATUS_SUCCESS;
}
template <class Grid, class Block, class Bytes>
auto trace_kernel(const char* name, Grid grid, Block block, Bytes bytes, cudaStream_t stream) {
  return [=](const auto&... args) { record(name, false, grid, block, bytes, stream, args...); };
}
void reset(std::size_t stop = std::numeric_limits<std::size_t>::max(), int variant = 0) {
  trace.clear();
  calls.clear();
  fail = stop;
  failure_variant = variant;
}
void same(const std::vector<Trace>& expected, const std::vector<Trace>& actual) {
  if (expected == actual) return;
  const auto count = std::min(expected.size(), actual.size());
  for (std::size_t i = 0; i < count; ++i) {
    if (!(expected[i] == actual[i])) {
      std::cerr << "trace divergence at " << i << ": " << expected[i].name << " / "
                << actual[i].name << '\n';
      assert(false);
    }
  }
  std::cerr << "trace length " << expected.size() << " / " << actual.size() << '\n';
  assert(false);
}
}  // namespace orchestration
using orchestration::trace_kernel;

cudaError_t cudaGetLastError() {
  return orchestration::cuda_status(orchestration::record("cudaGetLastError", true));
}
cudaError_t cudaMemsetAsync(void* ptr, int value, std::size_t bytes, cudaStream_t stream) {
  return orchestration::cuda_status(
      orchestration::record("cudaMemsetAsync", true, ptr, value, bytes, stream));
}
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
  return orchestration::blas_status(orchestration::record("cublasSetStream", true, handle, stream));
}
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* ptr, std::size_t bytes) {
  return orchestration::blas_status(
      orchestration::record("cublasSetWorkspace", true, handle, ptr, bytes));
}
cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
  return orchestration::blas_status(
      orchestration::record("cublasSetPointerMode", true, handle, mode));
}
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  return orchestration::blas_status(orchestration::record("cublasSetMathMode", true, handle, mode));
}
cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t fill, cublasOperation_t operation,
                                  cublasDiagType_t diagonal, int m, int n, const double* alpha,
                                  const double* const* a, int lda, double* const* b, int ldb,
                                  int count) {
  return orchestration::blas_status(orchestration::record("cublasDtrsmBatched", true, handle, side,
                                                          fill, operation, diagonal, m, n, *alpha,
                                                          a, lda, b, ldb, count));
}
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t opa,
                                         cublasOperation_t opb, int m, int n, int k,
                                         const double* alpha, const double* a, int lda,
                                         long long stride_a, const double* b, int ldb,
                                         long long stride_b, const double* beta, double* c, int ldc,
                                         long long stride_c, int count) {
  return orchestration::blas_status(
      orchestration::record("cublasDgemmStridedBatched", true, handle, opa, opb, m, n, k, *alpha, a,
                            lda, stride_a, b, ldb, stride_b, *beta, c, ldc, stride_c, count));
}

cusolverStatus_t cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t fill,
                                             int n, double* matrix, int lda, double* d, double* e,
                                             double* tau, int* elements) {
  *elements = 17;
  return orchestration::solver_status(orchestration::record(
      "cusolverDnDsytrd_bufferSize", true, handle, fill, n, matrix, lda, d, e, tau));
}
cusolverStatus_t cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                                             cublasFillMode_t fill, cublasOperation_t operation,
                                             int m, int n, double* a, int lda, double* tau,
                                             double* c, int ldc, int* elements) {
  *elements = 23;
  return orchestration::solver_status(orchestration::record("cusolverDnDormtr_bufferSize", true,
                                                            handle, side, fill, operation, m, n, a,
                                                            lda, tau, c, ldc));
}
cusolverStatus_t cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t fill, int n,
                                  double* matrix, int lda, double* d, double* e, double* tau,
                                  double* work, int elements, int* info) {
  return orchestration::solver_status(orchestration::record(
      "cusolverDnDsytrd", true, handle, fill, n, matrix, lda, d, e, tau, work, elements, info));
}
cusolverStatus_t cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t fill, cublasOperation_t operation, int m, int n,
                                  double* a, int lda, double* tau, double* c, int ldc, double* work,
                                  int elements, int* info) {
  return orchestration::solver_status(orchestration::record("cusolverDnDormtr", true, handle, side,
                                                            fill, operation, m, n, a, lda, tau, c,
                                                            ldc, work, elements, info));
}

// Surrounding full-file token comparison pins real trace scope ownership; these
// host trace services have no GPU side effects in the ABI harness.
namespace runtime::cuda_trace {
struct TraceRegion {
  TraceRegion(const char*, cudaStream_t) {}
};
}  // namespace runtime::cuda_trace
namespace scf {
generativeqc_status launch_solver(const EigensolverResources&, CudaEigensolverFamily, int, int,
                                  double*, double*, double*, int, int*, const std::uint8_t*,
                                  const EigensolverProfileLaunch* = nullptr);
}
struct CanonicalState {
  df::DeviceSolver solver;
};
struct CanonicalBuffers {
  double *d_fock{}, *d_orthogonalizer{}, *d_temporary{}, *d_eigenvalues{};
  int* d_info{};
  double *d_alpha_fock{}, *d_beta_fock{}, *d_alpha_eigenvalues{}, *d_beta_eigenvalues{};
  int *d_alpha_info{}, *d_beta_info{};
  const std::int32_t *d_alpha_occupied{}, *d_beta_occupied{};
  double *d_next_alpha{}, *d_next_beta{};
};
template <class... Args>
void launch_build_device_density_kernel(const Args&... args) {
  orchestration::record("launch_build_device_density_kernel", false, args...);
}
void store_scf_factor(CudaDensityFittingJkPlan& plan, CanonicalState& state, const double* matrix,
                      bool beta) {
  orchestration::record("store_scf_factor", false, &plan, &state, matrix, beta);
}
void store_scf_final_frame(CudaDensityFittingJkPlan& plan, CanonicalState& state,
                           const double* matrix, bool beta) {
  orchestration::record("store_scf_final_frame", false, &plan, &state, matrix, beta);
}
#include "generalized_gfn_bodies.inc"

namespace {
struct GfnFixture {
  // Heterogeneous buckets make physical, spin, matrix and orbital offsets differ.
  std::array<gfn2::Gfn2EigensolverBucket, 2> buckets{
      {{5, 2, 0, 0, 0, 3, 0, 0, 0}, {3, 1, 2, 50, 10, 2, 3, 75, 15}}};
  std::array<std::int64_t, 4> orbital_offsets{{0, 5, 10, 13}}, matrix_offsets{{0, 25, 50, 59}};
  std::array<std::int64_t, 4> spin_offsets{{0, 1, 3, 5}}, spin_orbitals{{0, 5, 15, 21}},
      spin_matrices{{0, 25, 75, 93}};
  std::array<std::int32_t, 3> systems{{0, 1, 2}}, spins{{1, 2, 2}};
  std::array<std::uint8_t, 3> active{{1, 0, 1}};
  std::array<double, 128> h{}, a{}, b{}, factors{}, coefficients{};
  std::array<double, 32> values{}, eigenvalues{};
  std::array<double*, 7> factor_pointers{}, matrix_pointers{};
  std::array<int, 7> info_a{}, info_b{};
  std::array<std::uint8_t, 7> eligible{};
  std::array<std::uint64_t, 3> generations{{11, 11, 11}};
  std::array<std::uint32_t, 3> factor_status{}, system_errors{};
  std::array<std::int32_t, 3> compact_systems{}, compact_slots{};
  std::array<gfn2::Gfn2EigensolverBucketActivity, 2> activity{};
  std::uint32_t sequence = 1, device_error = 0;
  std::uint64_t generation = 11;
  alignas(256) std::array<double, 256> device_work{};
  alignas(std::max_align_t) std::array<unsigned char, 192> host_work{};
  int solver_token{}, parameter_token{}, jacobi_token{}, blas_token{}, stream_token{};
  cusolverDnHandle_t solver = reinterpret_cast<cusolverDnHandle_t>(&solver_token);
  cusolverDnParams_t parameters = reinterpret_cast<cusolverDnParams_t>(&parameter_token);
  cublasHandle_t blas = reinterpret_cast<cublasHandle_t>(&blas_token);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(&stream_token);
  gfn2::Gfn2EigensolverDeviceBatch batch{};
  gfn2::Gfn2EigensolverOverlapCache cache{};
  gfn2::Gfn2WavefunctionLayoutView layout{};
  gfn2::Gfn2GeometryEpochDevice epoch{&generation, 1, 41};
  gfn2::Gfn2EigensolverDeviceWorkspace workspace{};
  gfn2::Gfn2EigensolverDeviceResults output{};
  gfn2::Gfn2EigensolverOptions options{};
  GfnFixture() {
    batch = {3,
             13,
             59,
             4,
             4,
             3,
             3,
             41,
             orbital_offsets.data(),
             matrix_offsets.data(),
             systems.data(),
             active.data()};
    cache = {factors.data(), 128, generations.data(), 3, factor_status.data(), 3, 41};
    output = {eigenvalues.data(), 32, coefficients.data(), 128, 41};
    layout.memory_space = gfn2::Gfn2PlanMemorySpace::kCudaDevice;
    layout.plan_token = 41;
    layout.batch_size = 3;
    layout.layout_fingerprint = 43;
    layout.total_spin_channels = 5;
    layout.total_spin_orbitals = 21;
    layout.total_spin_matrix_elements = 93;
    layout.spin_channel_count = 3;
    layout.spin_channel_offset_count = layout.spin_orbital_offset_count =
        layout.spin_matrix_offset_count = 4;
    layout.spin_channels = spins.data();
    layout.spin_channel_offsets = spin_offsets.data();
    layout.spin_orbital_offsets = spin_orbitals.data();
    layout.spin_matrix_offsets = spin_matrices.data();
    workspace.matrix_scratch_a = a.data();
    workspace.matrix_a_elements = a.size();
    workspace.matrix_scratch_b = b.data();
    workspace.matrix_b_elements = b.size();
    workspace.eigenvalue_scratch = values.data();
    workspace.eigenvalue_elements = values.size();
    workspace.factor_pointers = factor_pointers.data();
    workspace.factor_pointer_elements = 7;
    workspace.matrix_pointers = matrix_pointers.data();
    workspace.matrix_pointer_elements = 7;
    workspace.info_a = info_a.data();
    workspace.info_a_elements = 7;
    workspace.info_b = info_b.data();
    workspace.info_b_elements = 7;
    workspace.eligible = eligible.data();
    workspace.eligible_elements = 7;
    workspace.sequence_active = &sequence;
    workspace.sequence_active_elements = 1;
    workspace.solver_device_workspace = device_work.data();
    workspace.solver_device_workspace_bytes = sizeof(device_work);
    workspace.solver_host_workspace = host_work.data();
    workspace.solver_host_workspace_bytes = host_work.size();
    workspace.plan_token = 41;
    workspace.compact_systems = compact_systems.data();
    workspace.compact_system_elements = 3;
    workspace.compact_source_slots = compact_slots.data();
    workspace.compact_source_slot_elements = 3;
    workspace.bucket_activity = activity.data();
    workspace.bucket_activity_elements = 2;
    options.jacobi = reinterpret_cast<syevjInfo_t>(&jacobi_token);
  }
  gfn2::Gfn2EigensolverLaunchResult run(const std::string& scenario, bool baseline, bool dynamic,
                                        std::size_t bucket_index, unsigned capacity) {
    if (scenario == "restricted") {
      const auto fn =
          baseline ? gfn2::baseline_solve_eigensystems_impl : gfn2::solve_eigensystems_impl;
      return fn(batch, buckets.data(), 2, cache, dynamic ? 0 : generation,
                dynamic ? &epoch : nullptr, h.data(), options, solver, parameters, blas, workspace,
                output, system_errors.data(), &device_error, stream);
    }
    if (scenario == "spin") {
      const auto fn = baseline ? gfn2::baseline_solve_spin_eigensystems_impl
                               : gfn2::solve_spin_eigensystems_impl;
      return fn(batch, layout, buckets.data(), 2, cache, dynamic ? 0 : generation,
                dynamic ? &epoch : nullptr, h.data(), options, solver, parameters, blas, workspace,
                output, system_errors.data(), &device_error, stream);
    }
    if (scenario == "capacity-solve") {
      const auto fn = baseline ? gfn2::baseline_enqueue_capacity_eigensolver_body
                               : gfn2::enqueue_capacity_eigensolver_body;
      return fn(stream, capacity, batch, buckets[bucket_index], bucket_index, cache, h.data(),
                solver, parameters, blas, workspace, system_errors.data(), &device_error, options);
    }
    assert(scenario == "capacity-recover");
    const auto fn = baseline ? gfn2::baseline_enqueue_capacity_backtransform_body
                             : gfn2::enqueue_capacity_backtransform_body;
    return fn(stream, capacity, batch, buckets[bucket_index], bucket_index, blas, workspace, output,
              system_errors.data(), &device_error, options.deterministic_debug);
  }
};

void check_status(const gfn2::Gfn2EigensolverLaunchResult& a,
                  const gfn2::Gfn2EigensolverLaunchResult& b) {
  assert(a.status == b.status && a.cuda_status == b.cuda_status &&
         a.cublas_status == b.cublas_status && a.cusolver_status == b.cusolver_status);
}
void test_orchestration(const std::string& scenario) {
  GfnFixture f;
  for (auto strategy : {gfn2::Gfn2EigensolverStrategy::kBatchedDivideAndConquer,
                        gfn2::Gfn2EigensolverStrategy::kBatchedJacobi,
                        gfn2::Gfn2EigensolverStrategy::kTridiagonalBisection}) {
    f.options.strategy = strategy;
    for (bool dynamic : {false, true})
      for (bool deterministic : {false, true}) {
        f.options.deterministic_debug = deterministic;
        for (unsigned bucket = 0; bucket != 2; ++bucket)
          for (unsigned capacity = 1;
               capacity <= static_cast<unsigned>(f.buckets[bucket].system_count); ++capacity) {
            orchestration::reset();
            const auto before = f.run(scenario, true, dynamic, bucket, capacity);
            const auto expected = orchestration::trace;
            assert(before.success() && !expected.empty());
            orchestration::reset();
            const auto after = f.run(scenario, false, dynamic, bucket, capacity);
            check_status(before, after);
            orchestration::same(expected, orchestration::trace);
            for (std::size_t stop = 0; stop < expected.size(); ++stop) {
              if (!expected[stop].fallible) continue;
              for (int variant : {0, 1}) {
                orchestration::reset(stop, variant);
                const auto old_failed = f.run(scenario, true, dynamic, bucket, capacity);
                const auto prefix = orchestration::trace;
                assert(!old_failed.success() && prefix.size() == stop + 1);
                assert(std::equal(prefix.begin(), prefix.end(), expected.begin()));
                orchestration::reset(stop, variant);
                const auto new_failed = f.run(scenario, false, dynamic, bucket, capacity);
                check_status(old_failed, new_failed);
                orchestration::same(prefix, orchestration::trace);
              }
            }
          }
      }
  }
}

void test_canonical() {
  using namespace generativeqc::solver;
  GfnFixture f;
  const GeneralizedEigenDomain domain{5, 3, 5, GeneralizedEigenLayout::column_major, 2};
  const GeneralizedEigenMatrices matrices{f.h.data(),
                                          f.factors.data(),
                                          f.a.data(),
                                          f.b.data(),
                                          f.coefficients.data(),
                                          128,
                                          128,
                                          128,
                                          128,
                                          128};
  const shared::GeneralizedEigenLowering lowering{domain, f.blas, matrices};
  const double one = 1, zero = 0;
  for (bool recovery : {false, true}) {
    for (std::size_t stop :
         {std::numeric_limits<std::size_t>::max(), std::size_t(0), std::size_t(1)}) {
      orchestration::reset(stop);
      cublasStatus_t expected;
      if (recovery) {
        expected = cublasDgemmStridedBatched(f.blas, CUBLAS_OP_N, CUBLAS_OP_N, 5, 5, 5, &one,
                                             f.factors.data(), 5, 25, f.b.data(), 5, 25, &zero,
                                             f.coefficients.data(), 5, 25, 3);
      } else {
        expected =
            cublasDgemmStridedBatched(f.blas, CUBLAS_OP_N, CUBLAS_OP_N, 5, 5, 5, &one, f.h.data(),
                                      5, 25, f.factors.data(), 5, 25, &zero, f.a.data(), 5, 25, 3);
        if (expected == CUBLAS_STATUS_SUCCESS)
          expected = cublasDgemmStridedBatched(f.blas, CUBLAS_OP_T, CUBLAS_OP_N, 5, 5, 5, &one,
                                               f.factors.data(), 5, 25, f.a.data(), 5, 25, &zero,
                                               f.b.data(), 5, 25, 3);
      }
      const auto wanted = orchestration::trace;
      orchestration::reset(stop);
      const auto actual =
          recovery ? recover_generalized_eigen(GeneralizedEigenBasis::canonical_x, lowering)
                   : reduce_generalized_eigen(GeneralizedEigenBasis::canonical_x, lowering);
      assert(actual == expected);
      orchestration::same(wanted, orchestration::trace);
    }
  }
}
void test_canonical_adapter() {
  GfnFixture f;
  CudaDensityFittingJkPlan plan;
  plan.blas = f.blas;
  for (bool recovery : {false, true}) {
    for (std::size_t batch : {std::size_t(1), std::size_t(3)}) {
      for (int variant : {0, 1}) {
        for (std::size_t stop :
             {std::numeric_limits<std::size_t>::max(), std::size_t(0), std::size_t(1)}) {
          std::string expected_detail = "unchanged", actual_detail = "unchanged";
          orchestration::reset(stop, variant);
          generativeqc_status expected;
          if (recovery) {
            expected = df::scf_gemm(plan, false, batch, 5, f.factors.data(), f.b.data(), f.a.data(),
                                    expected_detail);
          } else {
            expected = df::scf_gemm(plan, false, batch, 5, f.b.data(), f.factors.data(), f.a.data(),
                                    expected_detail);
            if (expected == GENERATIVEQC_STATUS_SUCCESS)
              expected = df::scf_gemm(plan, true, batch, 5, f.factors.data(), f.a.data(),
                                      f.b.data(), expected_detail);
          }
          const auto wanted = orchestration::trace;
          orchestration::reset(stop, variant);
          const auto actual = df::scf_generalized_transform(
              plan, recovery, batch, 5, f.b.data(), f.factors.data(), f.a.data(), actual_detail);
          assert(actual == expected && actual_detail == expected_detail);
          orchestration::same(wanted, orchestration::trace);
        }
      }
    }
  }
}
void test_canonical_sequences(const std::string& scenario) {
  GfnFixture f;
  CudaDensityFittingJkPlan plan;
  plan.blas = f.blas;
  plan.stream = f.stream;
  handle_tokens = {f.solver, f.parameters, reinterpret_cast<syevjInfo_t>(&f.jacobi_token)};
  assert(plan.eigen_handles.create() == 0 && plan.eigen_handles.create_parameters() == 0);
  CanonicalState state;
  assert(state.solver.handles.create() == 0 && state.solver.handles.create_parameters() == 0);
  assert(state.solver.handles.configure_jacobi(1e-13, 100, 1) == 0);
  state.solver.workspace = f.device_work.data();
  state.solver.workspace_bytes = sizeof(f.device_work);
  state.solver.lwork = 23;
  state.solver.host_workspace = std::malloc(192);
  state.solver.host_workspace_bytes = 192;
  assert(state.solver.host_workspace != nullptr);
  std::array<double, 128> beta_density{};
  CanonicalBuffers buffers{f.h.data(),      f.factors.data(),      f.a.data(),
                           f.values.data(), f.info_a.data(),       f.h.data(),
                           f.b.data(),      f.values.data(),       f.eigenvalues.data(),
                           f.info_a.data(), f.info_b.data(),       f.spins.data(),
                           f.spins.data(),  f.coefficients.data(), beta_density.data()};
  dfao::OrdinaryEigensystem ordinary;
  ordinary.matrix = f.h.data();
  ordinary.x = f.factors.data();
  ordinary.temporary = f.a.data();
  ordinary.values = f.values.data();
  ordinary.info = f.info_a.data();
  ordinary.active = f.active.data();
  ordinary.workspace = f.device_work.data();
  ordinary.workspace_bytes = sizeof(f.device_work);
  ordinary.host_workspace.resize(192);
  ordinary.host_workspace_bytes = 192;
  for (bool xsyev : {false, true})
    for (bool occupied_exchange : {false, true}) {
      state.solver.xsyev = xsyev;
      for (bool orthogonalizer : {false, true}) {
        const auto run = [&](bool baseline, std::string& detail, double*& published) {
          if (scenario == "canonical-ordinary") {
            const auto fn =
                baseline ? canonical::baseline_df_eigensystem : canonical::live_df_eigensystem;
            return fn(&plan, &ordinary, orthogonalizer, published, detail);
          }
          const auto fn =
              scenario == "canonical-rhf"
                  ? (baseline ? canonical::baseline_df_rhf_scf : canonical::live_df_rhf_scf)
                  : (baseline ? canonical::baseline_df_uhf_scf : canonical::live_df_uhf_scf);
          return fn(&plan, &state, buffers, occupied_exchange, detail);
        };
        orchestration::reset();
        std::string expected_detail = "unchanged", actual_detail = "unchanged";
        double *expected_output = nullptr, *actual_output = nullptr;
        const auto expected_status = run(true, expected_detail, expected_output);
        const auto expected = orchestration::trace;
        assert(expected_status == GENERATIVEQC_STATUS_SUCCESS && !expected.empty());
        orchestration::reset();
        const auto actual_status = run(false, actual_detail, actual_output);
        assert(actual_status == expected_status && expected_detail == actual_detail &&
               expected_output == actual_output);
        orchestration::same(expected, orchestration::trace);
        for (std::size_t stop = 0; stop < expected.size(); ++stop) {
          if (!expected[stop].fallible) continue;
          for (int variant : {0, 1}) {
            expected_detail = actual_detail = "unchanged";
            expected_output = actual_output = nullptr;
            orchestration::reset(stop, variant);
            const auto old_failed = run(true, expected_detail, expected_output);
            const auto prefix = orchestration::trace;
            assert(old_failed != GENERATIVEQC_STATUS_SUCCESS && prefix.size() == stop + 1);
            orchestration::reset(stop, variant);
            const auto new_failed = run(false, actual_detail, actual_output);
            assert(new_failed == old_failed && expected_detail == actual_detail &&
                   expected_output == actual_output);
            orchestration::same(prefix, orchestration::trace);
          }
        }
      }
    }
}
void test_unused_capacity() {
  using namespace generativeqc::solver;
  GfnFixture f;
  const auto above_vendor = std::size_t(std::numeric_limits<int>::max()) + 1;
  // The pointer-table byte count fits, while treating its unused tail as
  // contiguous 2x2 matrices would overflow the active matrix byte limit.
  const auto overflow_tail = std::numeric_limits<std::size_t>::max() / sizeof(double) / 4 + 1;
  for (const auto capacity : {above_vendor, overflow_tail}) {
    // Exercise the real GFN adapter with a padded descriptor too. Kernel and
    // vendor stand-ins observe only the used prefix; no huge allocation occurs.
    f.workspace.factor_pointer_elements = static_cast<std::int64_t>(capacity);
    f.workspace.matrix_pointer_elements = static_cast<std::int64_t>(capacity);
    for (const std::string scenario : {"restricted", "spin"}) {
      orchestration::reset();
      const auto expected = f.run(scenario, true, false, 0, 1);
      const auto wanted = orchestration::trace;
      orchestration::reset();
      const auto actual = f.run(scenario, false, false, 0, 1);
      assert(expected.success());
      check_status(expected, actual);
      orchestration::same(wanted, orchestration::trace);
    }
    const GeneralizedEigenDomain domain{2, 1, capacity, GeneralizedEigenLayout::column_major, 1};
    assert(domain.valid() && domain.matrix_extent() == 4 && domain.value_extent() == 2);
    const GeneralizedEigenMatrices matrices{
        f.h.data(), f.factors.data(), f.a.data(), f.b.data(), f.coefficients.data(), 4, 4, 4, 4, 4};
    const shared::GeneralizedEigenPointerMatrices pointers{
        f.factor_pointers.data(), f.matrix_pointers.data(), capacity, capacity};
    for (const auto basis :
         {GeneralizedEigenBasis::canonical_x, GeneralizedEigenBasis::lower_cholesky}) {
      const auto lowering = basis == GeneralizedEigenBasis::canonical_x
                                ? shared::GeneralizedEigenLowering{domain, f.blas, matrices}
                                : shared::GeneralizedEigenLowering{domain, f.blas, pointers};
      for (bool recovery : {false, true}) {
        orchestration::reset();
        const double one = 1, zero = 0;
        if (basis == GeneralizedEigenBasis::canonical_x) {
          if (recovery) {
            cublasDgemmStridedBatched(f.blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &one,
                                      f.factors.data(), 2, 4, f.b.data(), 2, 4, &zero,
                                      f.coefficients.data(), 2, 4, 1);
          } else {
            cublasDgemmStridedBatched(f.blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &one, f.h.data(),
                                      2, 4, f.factors.data(), 2, 4, &zero, f.a.data(), 2, 4, 1);
            cublasDgemmStridedBatched(f.blas, CUBLAS_OP_T, CUBLAS_OP_N, 2, 2, 2, &one,
                                      f.factors.data(), 2, 4, f.a.data(), 2, 4, &zero, f.b.data(),
                                      2, 4, 1);
          }
        } else {
          const auto factors = reinterpret_cast<const double* const*>(f.factor_pointers.data());
          cublasDtrsmBatched(f.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                             recovery ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, &one,
                             factors, 2, f.matrix_pointers.data(), 2, 1);
          if (!recovery)
            cublasDtrsmBatched(f.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                               CUBLAS_DIAG_NON_UNIT, 2, 2, &one, factors, 2,
                               f.matrix_pointers.data(), 2, 1);
        }
        const auto wanted = orchestration::trace;
        orchestration::reset();
        const auto actual = recovery ? recover_generalized_eigen(basis, lowering)
                                     : reduce_generalized_eigen(basis, lowering);
        assert(actual == CUBLAS_STATUS_SUCCESS);
        orchestration::same(wanted, orchestration::trace);
      }
    }
    auto short_span = matrices;
    short_span.reduced_elements = 3;
    orchestration::reset();
    const shared::GeneralizedEigenLowering invalid_span{domain, f.blas, short_span};
    assert(reduce_generalized_eigen(GeneralizedEigenBasis::canonical_x, invalid_span) ==
           CUBLAS_STATUS_INVALID_VALUE);
    assert(recover_generalized_eigen(GeneralizedEigenBasis::canonical_x, invalid_span) ==
           CUBLAS_STATUS_INVALID_VALUE);
    assert(orchestration::trace.empty());
  }
  for (const auto invalid_work :
       {GeneralizedEigenDomain{2, above_vendor, std::numeric_limits<std::size_t>::max(),
                               GeneralizedEigenLayout::column_major},
        GeneralizedEigenDomain{50000, std::size_t(std::numeric_limits<int>::max()),
                               std::numeric_limits<std::size_t>::max(),
                               GeneralizedEigenLayout::column_major}}) {
    const shared::GeneralizedEigenPointerMatrices pointers{
        f.factor_pointers.data(), f.matrix_pointers.data(), invalid_work.capacity,
        invalid_work.capacity};
    const shared::GeneralizedEigenLowering lowering{invalid_work, f.blas, pointers};
    orchestration::reset();
    assert(!invalid_work.valid());
    assert(reduce_generalized_eigen(GeneralizedEigenBasis::lower_cholesky, lowering) ==
           CUBLAS_STATUS_INVALID_VALUE);
    assert(recover_generalized_eigen(GeneralizedEigenBasis::lower_cholesky, lowering) ==
           CUBLAS_STATUS_INVALID_VALUE);
    assert(orchestration::trace.empty());
  }
}
void test_private_abi() {
  using namespace generativeqc::solver;
  GfnFixture f;
  // Only the actual solve prefix crosses the vendor ABI, including when the
  // borrowed capacity exceeds its integer range or matrix-byte bound.
  for (const auto capacity : {std::size_t(std::numeric_limits<int>::max()) + 1,
                              std::numeric_limits<std::size_t>::max() / sizeof(double)}) {
    const GeneralizedEigenDomain domain{2, 1, capacity, GeneralizedEigenLayout::column_major, 1};
    const shared::GeneralizedEigenPointerMatrices pointers{
        f.factor_pointers.data(), f.matrix_pointers.data(), capacity, capacity};
    for (const auto basis :
         {GeneralizedEigenBasis::canonical_x, GeneralizedEigenBasis::lower_cholesky,
          GeneralizedEigenBasis::identity}) {
      GeneralizedEigenMatrices matrices{
          f.h.data(), f.factors.data(), f.a.data(), f.b.data(), f.coefficients.data(), 4, 4, 4, 4,
          4};
      if (basis == GeneralizedEigenBasis::identity)
        matrices.input = matrices.coefficients = matrices.reduced;
      const auto lowering = basis == GeneralizedEigenBasis::lower_cholesky
                                ? shared::GeneralizedEigenLowering{domain, f.blas, pointers}
                                : shared::GeneralizedEigenLowering{domain, f.blas, matrices};
      for (const bool recovery : {false, true}) {
        for (const auto stop :
             {std::numeric_limits<std::size_t>::max(), std::size_t(0), std::size_t(1)}) {
          for (const int variant : {0, 1}) {
            orchestration::reset(stop, variant);
            const auto expected = recovery ? recover_generalized_eigen(basis, lowering)
                                           : reduce_generalized_eigen(basis, lowering);
            const auto wanted = orchestration::trace;
            orchestration::reset(stop, variant);
            const auto actual =
                private_generalized_phase(basis, recovery, domain, f.blas, matrices, pointers);
            assert(actual == expected);
            orchestration::same(wanted, orchestration::trace);
          }
        }
        // Invalid bindings must return the private ABI's exact invalid status
        // without calling the provider or reading any deferred numerical info.
        for (const int invalid : {0, 1, 2, 3}) {
          auto bad_domain = domain;
          auto bad_matrices = matrices;
          auto bad_pointers = pointers;
          auto handle = f.blas;
          if (invalid == 0) handle = nullptr;
          if (invalid == 1) bad_domain.solves = capacity;
          if (invalid == 2) bad_domain.layout = GeneralizedEigenLayout::row_major;
          if (invalid == 3) {
            bad_matrices.reduced_elements = 3;
            bad_pointers.matrix_capacity = 0;
          }
          orchestration::reset();
          assert(private_generalized_phase(basis, recovery, bad_domain, handle, bad_matrices,
                                           bad_pointers) == CUBLAS_STATUS_INVALID_VALUE);
          assert(orchestration::trace.empty());
        }
      }
    }
  }
}
void test_identity() {
  using namespace generativeqc::solver;
  GfnFixture f;
  const GeneralizedEigenDomain domain{5, 3, 5, GeneralizedEigenLayout::column_major, 2};
  GeneralizedEigenMatrices matrices{f.b.data(), nullptr, nullptr, f.b.data(), f.b.data(),
                                    128,        0,       0,       128,        128};
  orchestration::reset();
  shared::GeneralizedEigenLowering lowering{domain, f.blas, matrices};
  assert(reduce_generalized_eigen(GeneralizedEigenBasis::identity, lowering) ==
         CUBLAS_STATUS_SUCCESS);
  assert(recover_generalized_eigen(GeneralizedEigenBasis::identity, lowering) ==
         CUBLAS_STATUS_SUCCESS);
  assert(orchestration::trace.empty());
}
void test_invalid() {
  // The public capacity entry points retain zero-work and overflow admission.
  // A zero-capacity graph arm must never reach the positive-work phase domain.
  {
    GfnFixture f;
    for (unsigned capacity : {0U, 3U}) {
      orchestration::reset();
      const auto solve = gfn2::enqueue_gfn2_eigensolver_capacity_cuda(
          f.stream, capacity, f.batch, f.buckets[0], 0, f.cache, f.h.data(), f.solver, f.parameters,
          f.blas, f.workspace, f.system_errors.data(), &f.device_error, f.options);
      const auto recover = gfn2::enqueue_gfn2_backtransform_capacity_cuda(
          f.stream, capacity, f.batch, f.buckets[0], 0, f.blas, f.workspace, f.output,
          f.system_errors.data(), &f.device_error, false);
      assert(solve.success() == (capacity == 0) && recover.success() == (capacity == 0));
      assert(orchestration::trace.empty());
    }
  }
  for (const std::string scenario : {"restricted", "spin"})
    for (int bad = 0; bad != 13; ++bad) {
      GfnFixture f;
      switch (bad) {
        case 0:
          f.generation = 0;
          break;
        case 1:
          f.epoch.value = nullptr;
          break;
        case 2:
          f.epoch.plan_token = 42;
          break;
        case 3:
          f.batch.plan_token = 0;
          break;
        case 4:
          f.workspace.matrix_pointer_elements = 1;
          break;
        case 5:
          f.output.coefficient_elements = 1;
          break;
        case 6:
          f.workspace.matrix_scratch_a = f.workspace.matrix_scratch_b;
          break;
        case 7:
          f.options.symmetry_tolerance = -1;
          break;
        case 8:
          f.solver = nullptr;
          break;
        case 9:
          f.blas = nullptr;
          break;
        case 10:
          f.cache.plan_token = 42;
          break;
        case 11:
          f.buckets[1].system_index_offset = 0;
          break;
        case 12:
          f.parameters = nullptr;
          break;
      }
      const bool dynamic = bad == 1 || bad == 2;
      orchestration::reset();
      const auto before = f.run(scenario, true, dynamic, 0, 1);
      assert(!before.success() && orchestration::trace.empty());
      const auto after = f.run(scenario, false, dynamic, 0, 1);
      check_status(before, after);
      assert(orchestration::trace.empty());
    }
}
}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  provider_result_hook = [] {
    const auto& c = calls.back();
    return orchestration::solver_status(orchestration::record(
        "symmetric_eigen", true, c.operation, c.solver, c.parameters, c.jacobi, c.vectors,
        c.triangle, c.n, c.lda, c.batch, c.matrix, c.values, c.matrix_type, c.values_type,
        c.compute_type, c.device, c.device_bytes, c.host, c.host_bytes, c.info, c.lwork,
        c.matrix_stride, c.values_stride));
  };
  provider_stream_hook = [](cusolverDnHandle_t solver, cudaStream_t stream) {
    return orchestration::solver_status(
        orchestration::record("cusolverDnSetStream", true, solver, stream));
  };
  gfn2::tridiagonal_hook = gfn2::actual_tridiagonal_symmetric_eigensolve;
  const std::string scenario = argv[1];
  if (scenario == "unused-capacity")
    test_unused_capacity();
  else if (scenario == "private-abi")
    test_private_abi();
  else if (scenario == "canonical")
    test_canonical();
  else if (scenario == "canonical-ordinary" || scenario == "canonical-rhf" ||
           scenario == "canonical-uhf")
    test_canonical_sequences(scenario);
  else if (scenario == "canonical-adapter")
    test_canonical_adapter();
  else if (scenario == "identity")
    test_identity();
  else if (scenario == "invalid")
    test_invalid();
  else
    test_orchestration(scenario);
  std::cout << "Preserved " << scenario << " host-call arguments and every first-failure prefix\n";
}
