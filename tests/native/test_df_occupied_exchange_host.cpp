// Reuse the numerical host cuBLAS double and the actual extracted provider.
// Its small borrowed-handle plan is a base of this caller's plan stand-in.
#define CudaDensityFittingJkPlan BorrowedBlasPlan
#define main borrowed_provider_test_main
#include "test_df_matrix_product_view.cpp"
#undef main
#undef CudaDensityFittingJkPlan

#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include "scf/df_streamed_k_policy.hpp"

using cudaError_t = int;
constexpr auto cudaSuccess = 0, cudaMemcpyHostToDevice = 1;
constexpr auto kThreads = 128;
std::size_t cleared_bytes{};
int cudaMemsetAsync(void* data, int value, std::size_t bytes, cudaStream_t) {
  cleared_bytes += bytes;
  std::memset(data, value, bytes);
  return cudaSuccess;
}
int cudaMemcpyAsync(void* target, const void* source, std::size_t bytes, int, cudaStream_t) {
  std::memcpy(target, source, bytes);
  return cudaSuccess;
}
int cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
int cudaPeekAtLastError() { return cudaSuccess; }
int blocks_for(std::size_t) { return 1; }
generativeqc_status cuda_failure(int, const char*, std::string&) {
  return GENERATIVEQC_STATUS_CUDA_ERROR;
}
namespace runtime::cuda_trace {
struct TraceOperation {
  TraceOperation(const char*, cudaStream_t, std::initializer_list<std::size_t>) {}
};
struct TraceRegion {
  TraceRegion(const char*, cudaStream_t) {}
};
void trace_counter(const char*, std::size_t) {}
template <class Submit>
auto trace_call(const char*, cudaStream_t, Submit submit) {
  return submit();
}
}  // namespace runtime::cuda_trace
namespace runtime::df_progress {
void number(const char*, std::size_t) {}
}  // namespace runtime::df_progress
struct CudaDensityFittingJkPlan : BorrowedBlasPlan {
  std::size_t nbf{}, naux{}, batch_size{1}, matrix_elements{}, panel_capacity{};
  std::size_t row_tile{}, auxiliary_tile{}, ao_pair_tile{}, stored_pair_count{};
  std::size_t stored_tensor_elements_per_system{}, tensor_elements_per_system{};
  std::size_t completed_occupied_projection_rank{};
  bool streamed{}, resident_exchange_enabled{}, triangular_exchange{}, resident_raw_coulomb{};
  void* integral_source{};
  std::vector<bool> metric_full_rank;
  struct {
    int pairs{};
    std::size_t rank_capacity{};
  } value_storage;
  double *three_center{}, *auxiliary_tile_values{}, *exchange_contributions{};
  double *exchange_intermediate{}, *exchange_tile_output{};
  std::vector<double> streamed_raw_three_center, streamed_inverse_square_roots;
  void revoke_projection_leases() { completed_occupied_projection_rank = 0; }
};
bool df_packed_pairs(int pairs) { return pairs != 0; }
using generativeqc::scf::df_streamed_k_panel;
struct ProjectedSchedule {
  std::size_t rows{};
};
ProjectedSchedule df_projected_exchange_schedule(std::size_t, std::size_t, std::size_t, std::size_t,
                                                 bool) {
  return {2};
}
namespace generated {
std::size_t df_occupied_gram_splits(std::size_t, std::size_t, std::size_t) { return 0; }
}  // namespace generated
// Unexercised integral-source and packed GPU kernels must not silently succeed.
template <class... Args>
void launch_project_packed_df(Args...) {
  std::abort();
}
template <class... Args>
void launch_unpack_df_values(Args...) {
  std::abort();
}
template <class... Args>
int launch_split_occupied_gram(Args...) {
  std::abort();
}
template <class... Args>
void launch_transpose_streamed_df_tile_kernel(Args...) {
  std::abort();
}
template <class... Args>
generativeqc_status generate_metric_panel(Args...) {
  std::abort();
}
struct ProjectedCall {
  CudaDensityFittingJkPlan* plan{};
  const double* coefficients{};
  double* output{};
  std::size_t rank{};
  bool column_major{}, fail{}, submitted{};
} projected;
// Exercise the caller's fast-path delegation and traffic contract. Integral
// generation and projected numerical kernels remain outside this host probe.
generativeqc_status build_streamed_projected_exchange(CudaDensityFittingJkPlan& plan,
                                                      std::size_t system,
                                                      const double* coefficients, std::size_t rank,
                                                      bool column_major, double weight,
                                                      std::size_t rows, double* exchange,
                                                      std::string& detail) {
  assert(&plan == projected.plan && system == 0 && coefficients == projected.coefficients);
  assert(rank == projected.rank && column_major == projected.column_major && weight == 1.25);
  assert(rows == 2 && exchange == projected.output);
  projected.submitted = true;
  if (projected.fail) {
    detail = "projected provider failure";
    return GENERATIVEQC_STATUS_CUDA_ERROR;
  }
  std::fill_n(exchange, plan.matrix_elements, 17.0);
  return GENERATIVEQC_STATUS_SUCCESS;
}
void launch_mirror_exchange_triangle(int, int, cudaStream_t, std::size_t n, double* output) {
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < j; ++i) output[i + j * n] = output[j + i * n];
}
void launch_gather_auxiliary_tile_kernel(int, int, int, cudaStream_t, std::size_t pairs,
                                         std::size_t auxiliaries, std::size_t system,
                                         std::size_t begin, std::size_t count, const double* values,
                                         double* output) {
  for (std::size_t q = 0; q < count; ++q)
    for (std::size_t pair = 0; pair < pairs; ++pair)
      output[q * pairs + pair] = values[(system * pairs + pair) * auxiliaries + begin + q];
}
void launch_reduce_exchange_row_tile_kernel(int, int, int, cudaStream_t, std::size_t n,
                                            std::size_t rb, std::size_t nr, std::size_t cb,
                                            std::size_t nc, std::size_t nq, std::size_t system,
                                            const double* values, double* output) {
  for (std::size_t q = 0; q < nq; ++q)
    for (std::size_t j = 0; j < nc; ++j)
      for (std::size_t i = 0; i < nr; ++i)
        output[system * n * n + rb + i + (cb + j) * n] += values[(q * nc + j) * nr + i];
}
#include "df_occupied_exchange.inc"

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  const bool is_projected = mode == "projected" || mode == "projected_failure";
  constexpr std::size_t n = 3, a = 3, rank = 2;
  const double c[n][rank] = {{1, 2}, {3, -1}, {0.5, 4}};
  std::vector<double> factors(n * n * a);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu)
      for (std::size_t q = 0; q < a; ++q)
        factors[(mu * n + nu) * a + q] = 0.25 + mu + 2 * nu - 0.5 * q;
  for (bool column_major : {false, true}) {
    for (bool triangular : {false, true}) {
      cublasContext handle{};
      auto stream = reinterpret_cast<cudaStream_t>(0x1234);
      handle.stream = stream;
      CudaDensityFittingJkPlan plan{};
      plan.blas = &handle;
      plan.stream = stream;
      plan.nbf = n;
      plan.naux = a;
      plan.matrix_elements = n * n;
      plan.tensor_elements_per_system = plan.stored_tensor_elements_per_system = factors.size();
      plan.row_tile = mode == "streamed" ? 2 : n;
      plan.auxiliary_tile = mode == "resident" ? a : 1;
      plan.resident_exchange_enabled = mode == "resident";
      plan.streamed = mode == "streamed" || is_projected;
      plan.resident_raw_coulomb = mode == "raw";
      if (is_projected) {
        plan.integral_source = &plan;
        plan.metric_full_rank = {true};
      }
      plan.triangular_exchange = triangular;
      plan.panel_capacity = n * n * a;
      plan.three_center = factors.data();
      plan.streamed_raw_three_center = factors;
      plan.streamed_inverse_square_roots.assign(a * a, 0);
      for (std::size_t q = 0; q < a; ++q) plan.streamed_inverse_square_roots[q * a + q] = 1;
      std::vector<double> u(n * n * a), v(u.size()), w(u.size()), x(u.size());
      plan.auxiliary_tile_values = u.data();
      plan.exchange_contributions = v.data();
      plan.exchange_intermediate = w.data();
      plan.exchange_tile_output = x.data();
      double coefficients[n * rank];
      for (std::size_t nu = 0; nu < n; ++nu)
        for (std::size_t i = 0; i < rank; ++i)
          coefficients[column_major ? nu + i * n : nu * rank + i] = c[nu][i];
      const auto requested_rank = mode == "zero" ? 0 : rank;
      // NaNs ensure beta=0 really overwrites and cannot accidentally accumulate.
      std::vector<double> output(n * n + 2, sentinel);
      std::fill(output.begin() + 1, output.end() - 1, std::numeric_limits<double>::quiet_NaN());
      cleared_bytes = 0;
      std::string detail;
      projected = {&plan,          coefficients, output.data() + 1,
                   requested_rank, column_major, mode == "projected_failure",
                   false};
      const auto expected_status = plan.resident_raw_coulomb ? GENERATIVEQC_STATUS_INVALID_ARGUMENT
                                   : projected.fail          ? GENERATIVEQC_STATUS_CUDA_ERROR
                                                             : GENERATIVEQC_STATUS_SUCCESS;
      assert(build_occupied_exchange(plan, 0, coefficients, requested_rank, column_major, 1.25,
                                     output.data() + 1, detail) == expected_status);
      assert(output.front() == sentinel && output.back() == sentinel);
      if (plan.resident_raw_coulomb) {
        // Rejection must precede writes or provider submission: A is not B.
        assert(cleared_bytes == 0 && !projected.submitted && !detail.empty());
        for (std::size_t i = 1; i <= n * n; ++i) assert(std::isnan(output[i]));
        continue;
      }
      assert(cleared_bytes == (mode == "resident" || is_projected ? 0 : n * n * sizeof(double)));
      if (is_projected) {
        assert(projected.submitted);
        assert(detail == (projected.fail ? "projected provider failure" : ""));
        for (std::size_t i = 1; i <= n * n; ++i)
          assert(projected.fail ? std::isnan(output[i]) : output[i] == 17.0);
        assert(plan.blas == &handle && plan.stream == stream && handle.stream == stream);
        continue;
      }
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          double expected = 0;
          for (std::size_t q = 0; q < a; ++q) {
            for (std::size_t i = 0; i < requested_rank; ++i) {
              double left = 0, right = 0;
              for (std::size_t j = 0; j < n; ++j) {
                left += factors[(mu * n + j) * a + q] * c[j][i];
                right += factors[(nu * n + j) * a + q] * c[j][i];
              }
              expected += 1.25 * left * right;
            }
          }
          assert(std::abs(output[1 + mu + nu * n] - expected) < 1e-10);
        }
      }
      assert(plan.blas == &handle && plan.stream == stream && handle.stream == stream);
      if (mode == "resident") assert(plan.completed_occupied_projection_rank == rank);
    }
  }
}
