#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include "cc/df_triples.hpp"
#include "cc/df_triples_traversal.hpp"
#include "generated_df_occupied_triples_cuda.cuh"
#include "posthf/capacity.hpp"
#include "tensor/cuda_runtime.cuh"

#if defined(GENERATIVEQC_TEST_HOOKS)
namespace {
thread_local bool reject_w_library_for_test = false;
}
extern "C" void df_triples_reject_library_for_test_v1(bool reject) {
  reject_w_library_for_test = reject;
}
extern "C" bool tensor_cutensor_qualification_for_test_v1(bool enable, int reject_after) {
#if GENERATIVEQC_HAS_CUTENSOR
  // Synthetic resource limits belong to endpoint qualification, never to the
  // production profile registry or the method's scientific admission API.
  generativeqc::tensor::cutensor_reservation_for_test =
      enable ? generativeqc::tensor::ContractionProviderReservation{64ULL << 20, 256ULL << 20,
                                                                    64ULL << 20}
             : generativeqc::tensor::ContractionProviderReservation{};
  generativeqc::tensor::cutensor_preparations_before_rejection_for_test = reject_after;
  return true;
#else
  (void)enable;
  (void)reject_after;
  return false;
#endif
}
#endif

namespace generativeqc::cc::triples {
namespace {
using posthf::checked_add;
using posthf::checked_mul;
using Clock = std::chrono::steady_clock;
constexpr std::size_t provider_allowance = 96ULL << 20;
constexpr std::size_t blas_workspace = 4ULL << 20;
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t align256(std::size_t n) { return n % 256 ? checked_add(n, 256 - n % 256) : n; }
std::size_t reserve(std::size_t& cursor, std::size_t n) {
  cursor = align256(cursor);
  const auto offset = cursor;
  cursor = checked_add(cursor, n);
  return offset;
}

struct Layout {
  std::array<std::size_t, 9> sizes{}, inputs{};
  std::size_t panels{}, moments{}, partials{}, energies{}, energy{}, error{}, library{};
  std::size_t execution_storage{}, paired_pointers{}, paired_pointer_bytes{};
  std::size_t arena{}, total{}, panel_capacity{}, v3{}, tiles{};
  unsigned blocks{};
};

Layout layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels,
              std::size_t execution_bytes = 0, std::size_t host_binding_bytes = 0,
              std::size_t execution_provider_bytes = provider_allowance,
              std::size_t library_bytes = blas_workspace, bool external_inputs = false,
              std::size_t paired_pointer_bytes = 0) {
  const auto oo = checked_mul(o, o), vv = checked_mul(v, v), ov = checked_mul(o, v);
  const auto ovv = checked_mul(o, vv);
  // Check every dimension and physical leading dimension before reading inputs
  // or allocating a provider. All subsequent pointer products fit these sizes.
  if (std::max({o, v, q, oo, vv, ov, ovv}) >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF triples exceed BLAS indexing");
  Layout p;
  p.sizes = {checked_mul(q, ov),
             checked_mul(q, vv),
             checked_mul(ov, oo),
             checked_mul(ov, ov),
             ov,
             ov,
             checked_mul(oo, vv),
             o,
             v};
  p.v3 = checked_mul(v, vv);
  p.tiles = checked_mul(checked_mul(o, checked_add(o, 1)), checked_add(o, 2)) / 6;
  p.blocks = static_cast<unsigned>(std::min<std::size_t>(1 + (p.v3 - 1) / 256, 65535));
  p.panel_capacity = std::min(o, panels);
  // Reject unrepresentable complete work ledgers before any allocation or launch.
  (void)checked_mul(p.tiles, p.v3);
  const auto moment_terms =
      checked_mul(checked_mul(6, p.tiles), checked_add(checked_mul(p.v3, v), checked_mul(p.v3, o)));
  (void)checked_add(moment_terms, checked_mul(checked_mul(3, p.tiles), checked_mul(q, p.v3)));
  std::size_t cursor = 0;
  if (!external_inputs)
    for (std::size_t x = 0; x < p.sizes.size(); ++x)
      p.inputs[x] = reserve(cursor, bytes(p.sizes[x]));
  p.panels = reserve(cursor, bytes(checked_mul(p.panel_capacity, p.v3)));
  p.moments = reserve(cursor, bytes(checked_mul(6, p.v3)));
  p.execution_storage = reserve(cursor, execution_bytes);
  p.paired_pointer_bytes = paired_pointer_bytes;
  p.paired_pointers = reserve(cursor, paired_pointer_bytes);
  p.partials = reserve(cursor, bytes(p.blocks));
  p.energies = reserve(cursor, bytes(p.tiles));
  p.energy = reserve(cursor, sizeof(double));
  p.error = reserve(cursor, sizeof(int));
  p.library = reserve(cursor, library_bytes);
  p.arena = align256(cursor);
  p.total = checked_add(checked_add(p.arena, execution_provider_bytes), host_binding_bytes);
  return p;
}

double validate_inputs(std::size_t o, std::size_t v, std::size_t q, const Layout& p,
                       const std::array<const double*, 9>& host, double threshold) {
  const auto* bvv = host[1];
  const auto* eps_o = host[7];
  const auto* eps_v = host[8];
  for (std::size_t x = 0; x < host.size(); ++x) {
    if (!host[x]) throw std::invalid_argument("null DF triples input");
    if (!std::all_of(host[x], host[x] + p.sizes[x],
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("nonfinite DF triples input");
  }
  for (std::size_t Q = 0; Q < q; ++Q)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < a; ++b)
        if (std::abs(bvv[(Q * v + a) * v + b] - bvv[(Q * v + b) * v + a]) > 1e-10)
          throw std::invalid_argument("DF triples require symmetric B_vv pairs");
  const auto max_occ = *std::max_element(eps_o, eps_o + o);
  const auto min_vir = *std::min_element(eps_v, eps_v + v);
  const double minimum = 3.0 * (min_vir - max_occ);
  if (!(max_occ < min_vir) || !std::isfinite(minimum) || minimum <= threshold)
    throw std::invalid_argument("noncanonical or near-zero DF triples denominator");
  return minimum;
}

std::size_t prepared_input_storage_bytes(const Layout& p) {
  std::size_t cursor = 0;
  for (const auto size : p.sizes) (void)reserve(cursor, bytes(size));
  return align256(cursor);
}

struct PreparedInputs {
  unsigned char* base{};
  std::size_t storage_bytes{}, h2d_bytes{};
  generated_df::Inputs values;

  PreparedInputs(const Layout& p, const std::array<const double*, 9>& host) {
    std::size_t cursor = 0;
    std::array<std::size_t, 9> offsets{};
    for (std::size_t x = 0; x < p.sizes.size(); ++x)
      offsets[x] = reserve(cursor, bytes(p.sizes[x]));
    storage_bytes = align256(cursor);
    generativeqc_tensor::cuda_check(cudaMalloc(reinterpret_cast<void**>(&base), storage_bytes));
    try {
      const std::array<const double**, 9> fields{&values.bov,  &values.bvv,   &values.ovoo,
                                                 &values.ovov, &values.fov,   &values.t1,
                                                 &values.t2,   &values.eps_o, &values.eps_v};
      for (std::size_t x = 0; x < host.size(); ++x) {
        auto* destination = base + offsets[x];
        *fields[x] = reinterpret_cast<const double*>(destination);
        generativeqc_tensor::cuda_check(
            cudaMemcpy(destination, host[x], bytes(p.sizes[x]), cudaMemcpyHostToDevice));
        h2d_bytes = checked_add(h2d_bytes, bytes(p.sizes[x]));
      }
    } catch (...) {
      cudaFree(base);
      base = nullptr;
      throw;
    }
  }

  PreparedInputs(const PreparedInputs&) = delete;
  PreparedInputs& operator=(const PreparedInputs&) = delete;
  ~PreparedInputs() {
    if (base) cudaFree(base);
  }
};

void validate_response_work(std::size_t o, std::size_t v, std::size_t q, const Layout& p,
                            bool include_gap_response) {
  const auto scalar_outputs = include_gap_response
                                  ? generated_df::response_scalar_outputs_with_gap
                                  : generated_df::response_scalar_outputs_without_gap;
  (void)checked_mul(scalar_outputs, checked_mul(p.tiles, p.v3));
  // At most three forward and three reverse-regenerated panels per tile, plus
  // two factor products per distinct occupied index. Charge complete work
  // before input access even for logical shapes too large to execute.
  const auto panel_work = checked_mul(checked_mul(12, p.tiles), checked_mul(q, p.v3));
  const auto w_work = checked_mul(checked_mul(18, p.tiles),
                                  checked_add(checked_mul(v, p.v3), checked_mul(o, p.v3)));
  const auto v_work = checked_mul(checked_mul(24, p.tiles), p.v3);
  (void)checked_add(panel_work, checked_add(w_work, v_work));
}

struct ResponseLayout {
  Layout value;
  std::array<std::size_t, 9> outputs{};
  std::size_t bar_w{}, bar_v{}, bar_panel{}, packed{}, gap{}, host_bytes{}, output_bytes{},
      complete{};
};
ResponseLayout response_layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels,
                               std::size_t caller_bytes, bool parallel_gap_reduction,
                               bool include_gap_response, bool external_inputs = false) {
  ResponseLayout p;
  p.value = layout(o, v, q, panels, 0, 0, provider_allowance, blas_workspace, external_inputs);
  auto cursor = p.value.arena;
  std::size_t values = 0;
  for (std::size_t x = 0; x < p.outputs.size(); ++x) {
    p.host_bytes = checked_add(p.host_bytes, bytes(p.value.sizes[x]));
    if (!include_gap_response && x >= 7) continue;
    p.outputs[x] = reserve(cursor, bytes(p.value.sizes[x]));
    values = checked_add(values, p.value.sizes[x]);
  }
  p.bar_w = reserve(cursor, bytes(checked_mul(6, p.value.v3)));
  p.bar_v = reserve(cursor, bytes(p.value.v3));
  p.bar_panel = reserve(cursor, bytes(p.value.v3));
  p.packed = reserve(cursor, bytes(checked_mul(v, v)));
  if (include_gap_response)
    p.gap = reserve(cursor,
                    bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction)));
  p.value.arena = align256(cursor);
  p.value.total = checked_add(p.value.arena, provider_allowance);
  p.output_bytes = bytes(values);
  p.complete = checked_add(caller_bytes,
                           checked_add(p.value.total, checked_add(p.host_bytes, p.output_bytes)));
  validate_response_work(o, v, q, p.value, include_gap_response);
  return p;
}

struct FockLayout {
  Layout value;
  std::size_t foo{}, fvv{}, xl{}, yl{}, xr{}, yr{}, block{};
  std::size_t capacity{}, pages{}, pairs{}, cubes{}, page_builds{}, page_pairs{};
  std::size_t host_bytes{}, output_bytes{}, complete{}, contraction_summands_bound{};
};
FockLayout fock_layout(std::size_t o, std::size_t v, std::size_t q, std::size_t capacity,
                       std::size_t panels, std::size_t caller_bytes, bool external_inputs = false) {
  FockLayout r;
  r.value = layout(o, v, q, panels, 0, 0, provider_allowance, blas_workspace, external_inputs);
  auto& p = r.value;
  // Cross-page oo contracts a flattened complete virtual cube as BLAS k.
  if (p.v3 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF triples Fock response exceeds BLAS indexing");
  r.capacity = capacity;
  r.pages = 1 + (o - 1) / capacity;
  r.pairs = checked_mul(o, checked_add(o, 1)) / 2;
  r.page_pairs = checked_mul(r.pages, checked_add(r.pages, 1)) / 2;
  r.page_builds = checked_mul(r.pairs, r.page_pairs);
  const auto tail = o - checked_mul(r.pages - 1, capacity);
  const auto prefix = checked_mul(checked_mul(capacity, r.pages), r.pages - 1) / 2;
  r.cubes = checked_mul(r.pairs, checked_add(prefix, checked_mul(tail, r.pages)));
  // Charge the complete replay ledger before touching input pointers, including
  // zero-padded cross-page products. Scalar evaluations are separately reported.
  (void)checked_mul(r.cubes, p.v3);
  const auto panel_work = checked_mul(checked_mul(3, r.cubes), checked_mul(q, p.v3));
  const auto w_work =
      checked_mul(checked_mul(6, r.cubes), checked_add(checked_mul(v, p.v3), checked_mul(o, p.v3)));
  const auto vv_work = checked_mul(checked_mul(2, checked_mul(r.pairs, o)), checked_mul(v, p.v3));
  const auto oo_work = checked_mul(checked_mul(2, r.page_builds),
                                   checked_mul(checked_mul(capacity, capacity), p.v3));
  r.contraction_summands_bound =
      checked_add(checked_add(panel_work, w_work), checked_add(vv_work, oo_work));
  auto cursor = p.arena;
  r.foo = reserve(cursor, bytes(checked_mul(o, o)));
  r.fvv = reserve(cursor, bytes(checked_mul(v, v)));
  const auto page_bytes = bytes(checked_mul(capacity, p.v3));
  r.xl = reserve(cursor, page_bytes);
  r.yl = reserve(cursor, page_bytes);
  // A full occupied page needs only one X/Y pair. Under a smaller budget the
  // right pair is independently owned and reused across cross-page moments.
  r.xr = r.pages == 1 ? r.xl : reserve(cursor, page_bytes);
  r.yr = r.pages == 1 ? r.yl : reserve(cursor, page_bytes);
  r.block = reserve(cursor, bytes(checked_mul(capacity, capacity)));
  p.arena = align256(cursor);
  p.total = checked_add(p.arena, provider_allowance);
  for (auto size : p.sizes) r.host_bytes = checked_add(r.host_bytes, bytes(size));
  r.output_bytes = bytes(checked_add(checked_mul(o, o), checked_mul(v, v)));
  r.complete =
      checked_add(caller_bytes, checked_add(p.total, checked_add(r.host_bytes, r.output_bytes)));
  return r;
}

struct JointResponseLayout {
  FockLayout fock;
  std::array<std::size_t, 9> response_outputs{};
  std::size_t bar_w{}, bar_v{}, bar_panel{}, packed{}, gap{};
  std::size_t response_output_bytes{}, complete{};
};

JointResponseLayout joint_response_layout(std::size_t o, std::size_t v, std::size_t q,
                                          std::size_t capacity, std::size_t panels,
                                          std::size_t caller_bytes, bool parallel_gap_reduction,
                                          bool include_gap_response,
                                          bool fused_scalar_response = false) {
  JointResponseLayout r;
  r.fock = fock_layout(o, v, q, capacity, panels, 0, true);
  auto& p = r.fock.value;
  validate_response_work(o, v, q, p, include_gap_response);
  const auto scalar_points = checked_mul(p.tiles, p.v3);
  (void)checked_mul(scalar_points, generated_df::scalar_unfused_arithmetic_ops);
  (void)checked_mul(scalar_points, generated_df::scalar_unfused_value_reads);
  (void)checked_add(checked_mul(12, scalar_points), checked_mul(p.tiles, p.blocks));
  // A pullback can regenerate up to three evicted panels per triangular tile.
  // Those products share the Fock ledger, including its right-page replay work.
  const auto regenerated_panel_work = checked_mul(checked_mul(3, p.tiles), checked_mul(q, p.v3));
  (void)checked_add(r.fock.contraction_summands_bound, regenerated_panel_work);
  auto cursor = p.arena;
  std::size_t values = 0;
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;
  for (std::size_t x = 0; x < requested_outputs; ++x) {
    r.response_outputs[x] = reserve(cursor, bytes(p.sizes[x]));
    values = checked_add(values, p.sizes[x]);
  }
  r.bar_w = reserve(cursor, bytes(checked_mul(6, p.v3)));
  r.bar_v = reserve(cursor, bytes(checked_mul(fused_scalar_response ? 6 : 1, p.v3)));
  r.bar_panel = reserve(cursor, bytes(p.v3));
  r.packed = reserve(cursor, bytes(checked_mul(v, v)));
  if (include_gap_response)
    r.gap = reserve(cursor,
                    bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction)));
  p.arena = align256(cursor);
  p.total = checked_add(p.arena, provider_allowance);
  r.response_output_bytes = bytes(values);
  r.complete = checked_add(
      caller_bytes,
      checked_add(p.total, checked_add(r.fock.host_bytes,
                                       checked_add(r.fock.output_bytes, r.response_output_bytes))));
  return r;
}

// A cross-page moment is already symmetric in its two factors. Scatter each
// matrix block once and mirror only off-diagonal page pairs; discard padded rows.
__global__ void scatter_fock_block(std::size_t o, std::size_t capacity, std::size_t left,
                                   std::size_t right, const double* block, double* foo,
                                   int* error) {
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < capacity * capacity; flat += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = left + flat / capacity, j = right + flat % capacity;
    if (i >= o || j >= o) continue;
    foo[i * o + j] = generativeqc_tensor::finite(foo[i * o + j] + block[flat], error, 10);
    if (left != right)
      foo[j * o + i] = generativeqc_tensor::finite(foo[j * o + i] + block[flat], error, 10);
  }
}

// Audit only the logical output of a strided BLAS view. Padding can contain
// unrelated valid tensor elements and is not part of this contraction's result.
__global__ void audit_matrix(const double* c, std::size_t m, std::size_t n, std::size_t ldc,
                             int* error) {
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; flat < m * n;
       flat += std::size_t(blockDim.x) * gridDim.x)
    if (!isfinite(c[(flat % m) + (flat / m) * ldc])) atomicCAS(error, 0, 7);
}

/** Existing explicit-stride FP64 matrix product boundary.
 *
 * This centralizes the legacy method-local call while #1890 migrates the
 * generated triples views to the shared provider-neutral contraction binding.
 * Scientific callers still supply the compiler-derived transpose/stride data.
 */
inline void execute_strided_fp64_product(generativeqc_tensor::Context& context, char ta, char tb,
                                         std::size_t m, std::size_t n, std::size_t k, double alpha,
                                         const double* a, std::size_t lda, const double* b,
                                         std::size_t ldb, double beta, double* c, std::size_t ldc) {
  generativeqc_tensor::blas_check(cublasDgemm(
      context.handle, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), &alpha, a,
      static_cast<int>(lda), b, static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
}

// Pure view packing/scatter. The scientific V products stay compiler-owned.
template <bool Scatter>
__global__ void ovov_view(std::size_t o, std::size_t v, std::size_t i, std::size_t j,
                          const double* source, double* destination, int* error) {
  for (std::size_t x = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < v * v;
       x += std::size_t(blockDim.x) * gridDim.x) {
    const auto global = ((i * v + x / v) * o + j) * v + x % v;
    if constexpr (Scatter)
      destination[global] = generativeqc_tensor::finite(destination[global] + source[x], error, 8);
    else
      destination[x] = source[global];
  }
}

__global__ void scatter_gap(std::size_t v, std::size_t i, std::size_t j, std::size_t k,
                            generated_df::GapOutputs values, double* occupied, double* virtuals,
                            int* error) {
  // One thread owns all three occupied writes, including repeated indices.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    occupied[i] = generativeqc_tensor::finite(occupied[i] + *values.bar_eps_i, error, 9);
    occupied[j] = generativeqc_tensor::finite(occupied[j] + *values.bar_eps_j, error, 9);
    occupied[k] = generativeqc_tensor::finite(occupied[k] + *values.bar_eps_k, error, 9);
  }
  for (std::size_t x = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < v;
       x += std::size_t(blockDim.x) * gridDim.x)
    virtuals[x] = generativeqc_tensor::finite(virtuals[x] + values.bar_eps_v[x], error, 9);
}

// Deterministic generic reduction, shared by each tile and final publication.
__global__ void reduce(const double* values, std::size_t count, double* output, int* error) {
  double sum = 0.0;
  for (std::size_t x = threadIdx.x; x < count; x += blockDim.x) sum += values[x];
  __shared__ double sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = generativeqc_tensor::finite(sums[0], error, 3);
}
}  // namespace

DFCudaResult evaluate_df_cuda(std::size_t o, std::size_t v, std::size_t q, const double* bov,
                              const double* bvv, const double* ovoo, const double* ovov,
                              const double* fov, const double* t1, const double* t2,
                              const double* eps_o, const double* eps_v, double threshold,
                              std::size_t max_bytes, int device, std::size_t max_panel_buffers,
                              runtime::PrecisionDirective admitted_w) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples dimensions, threshold or panel limit");
  bool library_available = true;
#if defined(GENERATIVEQC_TEST_HOOKS)
  library_available = !reject_w_library_for_test;
#endif
  auto execution_plan = generated_df::prepare_w_plan(admitted_w, library_available);
  const auto planned_layout = [&](std::size_t panels) {
    return layout(o, v, q, panels, execution_plan.storage_bytes(o, v, std::min(o, panels)),
                  generated_df::WExecution::host_bytes(execution_plan),
                  execution_plan.provider_bytes(), 0);
  };
  auto p = planned_layout(max_panel_buffers);
  if (p.total > max_bytes) p = planned_layout(1);
  const bool resource_fallback = p.total > max_bytes;
  while (p.total > max_bytes) {
    const auto fallback = generated_df::lower_resource_w_plan(execution_plan);
    if (!fallback) break;
    execution_plan = *fallback;
    p = planned_layout(max_panel_buffers);
    if (p.total > max_bytes) p = planned_layout(1);
  }
  if (p.total > max_bytes) throw std::length_error("DF triples exceed numeric memory budget");
  // Keep the original panel/provider choices when optional pointer storage
  // cannot fit; paired dispatch is not a reason to reduce those earlier choices.
  const auto unpaired_layout = p;
  if (p.panel_capacity == 3 && execution_plan.pair_pointer_bytes()) {
    const auto paired =
        layout(o, v, q, p.panel_capacity, execution_plan.storage_bytes(o, v, p.panel_capacity),
               generated_df::WExecution::host_bytes(execution_plan),
               execution_plan.provider_bytes(), 0, false, execution_plan.pair_pointer_bytes());
    if (paired.total <= max_bytes) p = paired;
  }
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);

  // Borrowed D2H destinations outlive Context's exception-path stream drain.
  DFCudaResult result;
  result.resource_fallback = resource_fallback;
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    std::optional<generativeqc_tensor::Context> context_owner;
    const auto prepare_context = [&](const Layout& requested) {
      context_owner.emplace();
      context_owner->prepare(device, properties.major, properties.minor, requested.arena,
                             requested.error, requested.library, 0, 0, false);
    };
    try {
      prepare_context(p);
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      if (!p.paired_pointer_bytes) throw;
      context_owner.reset();
      p = unpaired_layout;
      prepare_context(p);
    }
    auto& context = *context_owner;
    generated_df::WExecution execution(execution_plan, o, v, q, context,
                                       context.arena + p.execution_storage, p.panel_capacity,
                                       result.fp64_gemms, result.fp32_gemms,
                                       result.contraction_summands, result.precision_cast_elements);
    const auto& selected = execution.plan();
    result.host_binding_bytes = generated_df::WExecution::host_bytes(selected);
    result.resource_fallback =
        result.resource_fallback || selected.selected != execution_plan.selected;
    result.precision = selected.precision.arithmetic;
    result.w_contraction_storage_bits =
        result.precision.storage_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
    result.w_contraction_compute_bits =
        result.precision.compute_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
    result.w_contraction_accumulation_bits =
        result.precision.accumulation_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
    const auto& candidate = generated_df::w_lowering_candidates[selected.selected];
    result.w_scientific_identity = generated_df::w_lowering_request.scientific_identity;
    result.w_semantic_identity = generated_df::w_lowering_request.semantic_identity;
    result.w_candidate_identity = candidate.identity;
    result.w_provider = candidate.provider;
    result.w_algorithm = candidate.algorithm;
    result.retained_incumbent = selected.retained_incumbent;
    result.w_provider_version = execution.provider_version();
    result.cuda_runtime_version = execution.runtime_version();
    result.w_precision_identity = selected.precision.identity;
    result.w_codegen_precision_schedule_identity = selected.schedule_identity;
    generated_df::Inputs in;
    const std::array<const double**, 9> fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                               &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    for (std::size_t x = 0; x < host.size(); ++x) {
      *fields[x] = reinterpret_cast<const double*>(context.arena + p.inputs[x]);
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(context.arena + p.inputs[x], host[x],
                                                      bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                      context.stream));
      result.h2d_bytes = checked_add(result.h2d_bytes, bytes(p.sizes[x]));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    auto* panels = reinterpret_cast<double*>(context.arena + p.panels);
    auto* moments = reinterpret_cast<double*>(context.arena + p.moments);
    auto* partials = reinterpret_cast<double*>(context.arena + p.partials);
    auto* energies = reinterpret_cast<double*>(context.arena + p.energies);
    auto* energy = reinterpret_cast<double*>(context.arena + p.energy);
    execution.initialize(context, in);
    const bool distinct_moments = result.w_contraction_storage_bits == 64 &&
                                  result.w_contraction_compute_bits == 64 &&
                                  result.w_contraction_accumulation_bits == 64;
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0, tile = 0;
    auto panel_slot_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        execution.build_panel(context, in, occupied, panels + slot * p.v3, slot);
        ++result.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return slot;
    };
    auto visit_tile = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t canonical_tile) {
      const std::array<std::size_t, 3> occupied{i, j, k};
      const bool alias_tile = distinct_moments && (i == j || j == k);
      const auto sources = alias_tile ? generated_df::occupied_moment_sources(i, j, k)
                                      : generated_df::MomentSourceMap{};
      // Group W seeds by their integral-panel index. A single-panel fallback
      // consumes every dependent GEMM before that storage is reused. All
      // producer/consumer work is ordered on Context's one owned stream.
      for (std::size_t index = 0; index < occupied.size(); ++index) {
        if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
            occupied.begin() + index)
          continue;
        const auto slot = panel_slot_for(occupied[index]);
        const auto* panel = panels + slot * p.v3;
        // Equal-index non-FP64 tiles still require all six original seeds.
        std::array<std::size_t, 6> seed_permutations{};
        std::size_t seed_count = 0;
        for (std::size_t permutation = 0; permutation < 6; ++permutation) {
          const auto* order = generated_df::permutations[permutation];
          if (occupied[order[0]] != occupied[index]) continue;
          // Equal physical tuples have identical W seeds; every original
          // energy contribution still loads its canonical stored seed.
          if (sources.index[permutation] != permutation) continue;
          seed_permutations[seed_count++] = permutation;
        }
        if (seed_count == 2 && p.paired_pointer_bytes) {
          const auto* first_order = generated_df::permutations[seed_permutations[0]];
          const auto* second_order = generated_df::permutations[seed_permutations[1]];
          const bool swapped = occupied[second_order[0]] == occupied[first_order[0]] &&
                               occupied[second_order[1]] == occupied[first_order[2]] &&
                               occupied[second_order[2]] == occupied[first_order[1]];
          if (swapped && execution.build_w_pair(context, in, occupied[first_order[0]],
                                                occupied[first_order[1]], occupied[first_order[2]],
                                                panel, moments + seed_permutations[0] * p.v3,
                                                moments + seed_permutations[1] * p.v3,
                                                context.arena + p.paired_pointers)) {
            result.moment_gemms += 4;
            continue;
          }
        }
        for (std::size_t seed = 0; seed < seed_count; ++seed) {
          const auto permutation = seed_permutations[seed];
          const auto* order = generated_df::permutations[permutation];
          execution.build_w(context, in, occupied[order[0]], occupied[order[1]], occupied[order[2]],
                            panel, moments + permutation * p.v3, slot);
          result.moment_gemms += 2;
        }
      }
      const double degeneracy = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
      if (alias_tile)
        generated_df::energy_distinct_tile(o, v, i, j, k, degeneracy, threshold, in, moments,
                                           sources, p.blocks, partials, context.error,
                                           context.stream);
      else
        generated_df::energy_tile(o, v, i, j, k, degeneracy, threshold, in, moments, p.blocks,
                                  partials, context.error, context.stream);
      ++result.epilogue_kernels;
      reduce<<<1, 256, 0, context.stream>>>(partials, p.blocks, energies + canonical_tile,
                                            context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.reduction_kernels;
      ++tile;
    };
    // Only the qualified three-panel FP64 domain changes its visit order;
    // canonical tile addresses preserve the original final reduction inputs.
    detail::visit_occupied_tiles(o, distinct_moments && p.panel_capacity == 3, visit_tile);
    if (tile != p.tiles) throw std::logic_error("DF triples occupied work mismatch");
    reduce<<<1, 256, 0, context.stream>>>(energies, p.tiles, energy, context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++result.reduction_kernels;
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&result.energy, energy, sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(result.energy))
      throw std::runtime_error("nonfinite or unsafe generated DF triples arithmetic");
    result.minimum_absolute_denominator = minimum;
    result.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    result.occupied_tiles = p.tiles;
    result.epilogue_points = checked_mul(p.tiles, p.v3);
    result.workspace_bytes = p.total;
    result.arena_bytes = p.arena;
    result.provider_retained_bytes = execution.provider_bytes();
    result.panel_capacity = p.panel_capacity;
    result.d2h_bytes = sizeof(double) + sizeof(int);
  }  // Drain/destroy the stream, buffers and BLAS provider inside endpoint timing.
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}
static DFCudaResponseResult pullback_df_cuda_impl(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double threshold, std::size_t max_bytes, int device,
    std::size_t caller_bytes, std::size_t max_panel_buffers, bool parallel_gap_reduction,
    bool include_gap_response, const generated_df::Inputs* prepared_inputs) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples response dimensions, threshold or panel limit");
  auto r = response_layout(o, v, q, max_panel_buffers, caller_bytes, parallel_gap_reduction,
                           include_gap_response, prepared_inputs != nullptr);
  if (r.complete > max_bytes)
    r = response_layout(o, v, q, 1, caller_bytes, parallel_gap_reduction, include_gap_response,
                        prepared_inputs != nullptr);
  if (r.complete > max_bytes)
    throw std::length_error("DF triples response exceeds complete numeric budget");
  const auto& p = r.value;
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);
  // Detached output destinations outlive Context and its exception-path drain.
  DFCudaResponseResult result;
  auto& d = result.diagnostic;
  const std::array<std::vector<double>*, 9> outputs{&result.bov,  &result.bvv,   &result.ovoo,
                                                    &result.ovov, &result.fov,   &result.t1,
                                                    &result.t2,   &result.eps_o, &result.eps_v};
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;
  for (std::size_t x = 0; x < requested_outputs; ++x) outputs[x]->resize(p.sizes[x]);
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    generated_df::Inputs in;
    generated_df::ResponseOutputs out;
    const std::array<const double**, 9> input_fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                                     &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    const std::array<double**, 9> output_fields{&out.bov, &out.bvv, &out.ovoo,  &out.ovov, &out.fov,
                                                &out.t1,  &out.t2,  &out.eps_o, &out.eps_v};
    const std::array<const double*, 9> prepared{prepared_inputs ? prepared_inputs->bov : nullptr,
                                                prepared_inputs ? prepared_inputs->bvv : nullptr,
                                                prepared_inputs ? prepared_inputs->ovoo : nullptr,
                                                prepared_inputs ? prepared_inputs->ovov : nullptr,
                                                prepared_inputs ? prepared_inputs->fov : nullptr,
                                                prepared_inputs ? prepared_inputs->t1 : nullptr,
                                                prepared_inputs ? prepared_inputs->t2 : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_o : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_v : nullptr};
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };
    for (std::size_t x = 0; x < 9; ++x) {
      if (prepared_inputs) {
        *input_fields[x] = prepared[x];
      } else {
        *input_fields[x] = pointer(p.inputs[x]);
        generativeqc_tensor::cuda_check(cudaMemcpyAsync(pointer(p.inputs[x]), host[x],
                                                        bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                        context.stream));
        d.h2d_bytes = checked_add(d.h2d_bytes, bytes(p.sizes[x]));
      }
      *output_fields[x] = x < requested_outputs ? pointer(r.outputs[x]) : nullptr;
      if (x < requested_outputs)
        generativeqc_tensor::cuda_check(
            cudaMemsetAsync(pointer(r.outputs[x]), 0, bytes(p.sizes[x]), context.stream));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    auto* panels = pointer(p.panels);
    auto* moments = pointer(p.moments);
    auto* bar_w = pointer(r.bar_w);
    auto* bar_v = pointer(r.bar_v);
    auto* bar_panel = pointer(r.bar_panel);
    auto* packed = pointer(r.packed);
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.audit_kernels;
      d.contraction_summands = checked_add(d.contraction_summands, checked_mul(count, kk));
    };
    auto reverse_gemm = [&](auto... args) {
      gemm(args...);
      ++result.reverse_gemms;
    };
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0, tile = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        generated_df::build_panel(o, v, q, occupied, in, panels + slot * p.v3, gemm);
        ++d.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return panels + slot * p.v3;
    };
    const auto small_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (v * v - 1) / 256, 65535));
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
          const std::array<std::size_t, 3> occupied{i, j, k};
          for (std::size_t index = 0; index < 3; ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto* panel = panel_for(occupied[index]);
            for (std::size_t perm = 0; perm < 6; ++perm) {
              const auto* order = generated_df::permutations[perm];
              if (occupied[order[0]] != occupied[index]) continue;
              generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                    occupied[order[2]], in, panel, moments + perm * p.v3, gemm);
              d.moment_gemms += 2;
            }
          }
          const double multiplicity = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
          generated_df::energy_tile(o, v, i, j, k, multiplicity, threshold, in, moments, p.blocks,
                                    pointer(p.partials), context.error, context.stream);
          ++d.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(pointer(p.partials), p.blocks,
                                                pointer(p.energies) + tile, context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++d.reduction_kernels;
          generated_df::response_w_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                        p.blocks, bar_w, context.error, context.stream);
          ++result.reverse_kernels;
          for (unsigned perm = 0; perm < 6; ++perm) {
            const auto* order = generated_df::permutations[perm];
            const auto I = occupied[order[0]], J = occupied[order[1]], K = occupied[order[2]];
            generated_df::response_v_tile(o, v, i, j, k, perm, multiplicity, threshold, in, moments,
                                          p.blocks, bar_v, context.error, context.stream);
            ovov_view<false><<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, in.ovov, packed,
                                                                       context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            // Generated V reverse consumes packed ovov in its first product and
            // overwrites it only in its last product. One stream preserves this reuse.
            generated_df::pullback_v(o, v, I, J, K, in, packed, bar_v, packed, out, reverse_gemm);
            ovov_view<true><<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, packed, out.ovov,
                                                                      context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            result.reverse_kernels += 3;
          }
          if (include_gap_response) {
            generated_df::response_gap_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                            p.blocks, bar_v, context.error, context.stream);
            generated_df::GapCudaState gap{
                o, v, bar_v, pointer(r.gap), context.error, context.stream, parallel_gap_reduction};
            const auto gaps = generated_df::gap_response_cuda(gap);
            scatter_gap<<<small_blocks, 256, 0, context.stream>>>(v, i, j, k, gaps, out.eps_o,
                                                                  out.eps_v, context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            const auto gap_kernels =
                generated_df::gap_response_kernel_count(o, v, parallel_gap_reduction);
            result.reverse_kernels += 2 + gap_kernels;
            result.gap.kernels += gap_kernels;
          }
          for (std::size_t index = 0; index < 3; ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto I = occupied[index];
            const auto* panel = panel_for(I);
            generativeqc_tensor::cuda_check(
                cudaMemsetAsync(bar_panel, 0, bytes(p.v3), context.stream));
            for (std::size_t perm = 0; perm < 6; ++perm) {
              const auto* order = generated_df::permutations[perm];
              if (occupied[order[0]] != I) continue;
              generated_df::pullback_w(o, v, I, occupied[order[1]], occupied[order[2]], in, panel,
                                       bar_w + perm * p.v3, bar_panel, out, reverse_gemm);
            }
            generated_df::pullback_panel(o, v, q, I, in, bar_panel, out, reverse_gemm);
          }
          ++tile;
        }
    if (tile != p.tiles) throw std::logic_error("DF triples response occupied work mismatch");
    result.gap.requested = include_gap_response;
    if (include_gap_response) {
      result.gap.parallel = generated_df::gap_response_parallel_selected(v, parallel_gap_reduction);
      result.gap.schedule = result.gap.parallel
                                ? generated_df::gap_response_parallel_schedule_identity
                                : "serial-source-major";
      result.gap.workspace_bytes =
          bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction));
      result.gap.materialized_elements = checked_mul(
          p.tiles, generated_df::gap_response_materialized_elements(o, v, parallel_gap_reduction));
      result.gap.value_reads = checked_mul(
          p.tiles, generated_df::gap_response_value_reads(o, v, parallel_gap_reduction));
      result.gap.value_writes = checked_mul(
          p.tiles, generated_df::gap_response_value_writes(o, v, parallel_gap_reduction));
      result.gap.reduction_summands = checked_mul(
          p.tiles, generated_df::gap_response_reduction_summands(o, v, parallel_gap_reduction));
    } else {
      result.gap.schedule = generated_df::response_without_gap_identity;
    }
    reduce<<<1, 256, 0, context.stream>>>(pointer(p.energies), p.tiles, pointer(p.energy),
                                          context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++d.reduction_kernels;
    for (std::size_t x = 0; x < requested_outputs; ++x)
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(outputs[x]->data(), pointer(r.outputs[x]),
                                                      bytes(p.sizes[x]), cudaMemcpyDeviceToHost,
                                                      context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&d.energy, pointer(p.energy), sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(d.energy))
      throw std::runtime_error("nonfinite or unsafe generated DF triples response arithmetic");
    d.minimum_absolute_denominator = minimum;
    d.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    d.occupied_tiles = p.tiles;
    d.workspace_bytes = p.total;
    d.arena_bytes = p.arena;
    d.provider_retained_bytes = context.metrics.provider_retained_bytes;
    d.panel_capacity = p.panel_capacity;
    d.epilogue_points = checked_mul(p.tiles, p.v3);
    d.d2h_bytes = checked_add(r.output_bytes, sizeof(double) + sizeof(int));
    result.borrowed_host_bytes = r.host_bytes;
    result.numeric_capacity_bytes = r.complete;
    result.scalar_response_evaluations =
        checked_mul(include_gap_response ? generated_df::response_scalar_outputs_with_gap
                                         : generated_df::response_scalar_outputs_without_gap,
                    d.epilogue_points);
  }
  d.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

DFCudaResponseResult pullback_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                      const double* bov, const double* bvv, const double* ovoo,
                                      const double* ovov, const double* fov, const double* t1,
                                      const double* t2, const double* eps_o, const double* eps_v,
                                      double threshold, std::size_t max_bytes, int device,
                                      std::size_t caller_bytes, std::size_t max_panel_buffers,
                                      bool parallel_gap_reduction, bool include_gap_response) {
  return pullback_df_cuda_impl(o, v, q, bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v, threshold,
                               max_bytes, device, caller_bytes, max_panel_buffers,
                               parallel_gap_reduction, include_gap_response, nullptr);
}

static DFCudaFockResult fock_response_df_cuda_impl(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double threshold, std::size_t max_bytes, int device,
    std::size_t caller_bytes, std::size_t max_page_rows, std::size_t max_panel_buffers,
    const generated_df::Inputs* prepared_inputs) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples Fock dimensions, threshold or panel limit");
  auto capacity = max_page_rows ? std::min(o, max_page_rows) : o;
  auto r =
      fock_layout(o, v, q, capacity, max_panel_buffers, caller_bytes, prepared_inputs != nullptr);
  while (r.complete > max_bytes) {
    // Prefer retaining more resolvent rows over extra cached integral panels:
    // a full page eliminates every cross-occupied recomputation.
    if (r.value.panel_capacity > 1) {
      r = fock_layout(o, v, q, capacity, 1, caller_bytes, prepared_inputs != nullptr);
      if (r.complete <= max_bytes) break;
    }
    if (capacity == 1)
      throw std::length_error("DF triples Fock response exceeds complete numeric budget");
    --capacity;
    r = fock_layout(o, v, q, capacity, max_panel_buffers, caller_bytes, prepared_inputs != nullptr);
  }
  const auto& p = r.value;
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);
  // Detached destinations and error status outlive the device owner's drain.
  DFCudaFockResult result;
  result.foo.resize(o * o);
  result.fvv.resize(v * v);
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };
    generated_df::Inputs in;
    const std::array<const double**, 9> fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                               &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    const std::array<const double*, 9> prepared{prepared_inputs ? prepared_inputs->bov : nullptr,
                                                prepared_inputs ? prepared_inputs->bvv : nullptr,
                                                prepared_inputs ? prepared_inputs->ovoo : nullptr,
                                                prepared_inputs ? prepared_inputs->ovov : nullptr,
                                                prepared_inputs ? prepared_inputs->fov : nullptr,
                                                prepared_inputs ? prepared_inputs->t1 : nullptr,
                                                prepared_inputs ? prepared_inputs->t2 : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_o : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_v : nullptr};
    for (std::size_t x = 0; x < 9; ++x) {
      if (prepared_inputs) {
        *fields[x] = prepared[x];
      } else {
        *fields[x] = pointer(p.inputs[x]);
        generativeqc_tensor::cuda_check(cudaMemcpyAsync(pointer(p.inputs[x]), host[x],
                                                        bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                        context.stream));
      }
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(r.foo), 0, bytes(o * o), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(r.fvv), 0, bytes(v * v), context.stream));
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.audit_kernels;
      result.contraction_summands =
          checked_add(result.contraction_summands, checked_mul(count, kk));
    };
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        generated_df::build_panel(o, v, q, occupied, in, pointer(p.panels) + slot * p.v3, gemm);
        ++result.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return pointer(p.panels) + slot * p.v3;
    };
    auto page = [&](std::size_t start, std::size_t j, std::size_t k, double* x, double* y) {
      ++result.page_builds;
      const auto page_bytes = bytes(checked_mul(capacity, p.v3));
      // Explicit padding protects oo tail products even when the right buffer
      // was previously used for a full page in a different (j,k) pair.
      generativeqc_tensor::cuda_check(cudaMemsetAsync(x, 0, page_bytes, context.stream));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(y, 0, page_bytes, context.stream));
      for (std::size_t lane = 0; lane < capacity && start + lane < o; ++lane) {
        const auto i = start + lane;
        const std::array<std::size_t, 3> occupied{i, j, k};
        for (std::size_t index = 0; index < 3; ++index) {
          if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
              occupied.begin() + index)
            continue;
          const auto* panel = panel_for(occupied[index]);
          for (std::size_t perm = 0; perm < 6; ++perm) {
            const auto* order = generated_df::permutations[perm];
            if (occupied[order[0]] != occupied[index]) continue;
            generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]], occupied[order[2]],
                                  in, panel, pointer(p.moments) + perm * p.v3, gemm);
            result.w_gemms += 2;
          }
        }
        generated_df::resolvent_tile(o, v, i, j, k, threshold, in, pointer(p.moments), p.blocks,
                                     x + lane * p.v3, y + lane * p.v3, context.error,
                                     context.stream);
        ++result.vector_cubes;
      }
    };
    const auto scatter_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (capacity * capacity - 1) / 256, 65535));
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k <= j; ++k) {
        const double weight = j == k ? 1.0 : 2.0;
        for (std::size_t left = 0; left < o; left += capacity) {
          page(left, j, k, pointer(r.xl), pointer(r.yl));
          for (std::size_t lane = 0; lane < capacity && left + lane < o; ++lane) {
            generated_df::fock_vv(v, capacity, pointer(r.xl) + lane * p.v3,
                                  pointer(r.yl) + lane * p.v3, weight, pointer(r.fvv), gemm);
            result.fock_gemms += 2;
          }
          for (std::size_t right = left; right < o; right += capacity) {
            if (left != right) page(right, j, k, pointer(r.xr), pointer(r.yr));
            generated_df::fock_oo(v, capacity, pointer(r.xl), pointer(left == right ? r.xl : r.xr),
                                  pointer(r.yl), pointer(left == right ? r.yl : r.yr), weight,
                                  pointer(r.block), gemm);
            result.fock_gemms += 2;
            scatter_fock_block<<<scatter_blocks, 256, 0, context.stream>>>(
                o, capacity, left, right, pointer(r.block), pointer(r.foo), context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            ++result.scatter_kernels;
          }
        }
      }
    if (result.vector_cubes != r.cubes || result.page_builds != r.page_builds)
      throw std::logic_error("DF triples Fock replay work mismatch");
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(result.foo.data(), pointer(r.foo), bytes(o * o),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(result.fvv.data(), pointer(r.fvv), bytes(v * v),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed)
      throw std::runtime_error("nonfinite or unsafe generated DF triples Fock arithmetic");
    result.provider_retained_bytes = context.metrics.provider_retained_bytes;
  }
  result.minimum_absolute_denominator = minimum;
  result.numeric_capacity_bytes = r.complete;
  result.borrowed_host_bytes = r.host_bytes;
  result.workspace_bytes = p.total;
  result.arena_bytes = p.arena;
  result.page_capacity = capacity;
  result.page_count = r.pages;
  result.panel_capacity = p.panel_capacity;
  result.occupied_pairs = r.pairs;
  result.unique_vector_cubes = checked_mul(r.pairs, o);
  result.scalar_evaluations = checked_mul(r.cubes, p.v3);
  result.h2d_bytes = prepared_inputs ? 0 : r.host_bytes;
  result.d2h_bytes = checked_add(r.output_bytes, sizeof(int));
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

DFCudaFockResult fock_response_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                       const double* bov, const double* bvv, const double* ovoo,
                                       const double* ovov, const double* fov, const double* t1,
                                       const double* t2, const double* eps_o, const double* eps_v,
                                       double threshold, std::size_t max_bytes, int device,
                                       std::size_t caller_bytes, std::size_t max_page_rows,
                                       std::size_t max_panel_buffers) {
  return fock_response_df_cuda_impl(o, v, q, bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v,
                                    threshold, max_bytes, device, caller_bytes, max_page_rows,
                                    max_panel_buffers, nullptr);
}

DFCudaCombinedResponseResult pullback_and_fock_df_cuda(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double threshold, std::size_t max_bytes, int device,
    std::size_t caller_bytes, std::size_t max_page_rows, std::size_t max_panel_buffers,
    bool parallel_gap_reduction, bool include_gap_response, bool fused_scalar_response,
    runtime::PrecisionDirective admitted_w) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid combined DF triples response dimensions or controls");

  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const auto staging_layout = layout(o, v, q, 1);
  const auto prepared_storage = prepared_input_storage_bytes(staging_layout);
  const auto prepared_caller = checked_add(caller_bytes, prepared_storage);

  // Joint execution keeps the complete Fock pages plus the pullback outputs and
  // reverse scratch in one owner. Try the same resource order as the standalone
  // Fock owner: preserve page rows, then give up extra cached integral panels,
  // then shrink the page. Complete work/capacity is proven before input access.
  auto page_capacity = max_page_rows ? std::min(o, max_page_rows) : o;
  bool fused_selected = fused_scalar_response && !include_gap_response;
  // Validate precision before touching input pointers, even if storage later
  // refuses the candidate. Optional W ownership supplements the reverse BLAS
  // owner, so charge both providers and all cast/cache buffers simultaneously.
  std::optional<generated_df::WPlan> w_plan;
  bool w_resource_fallback = false;
  const auto requested_w_plan = generated_df::prepare_w_plan(admitted_w);
  if (!admitted_w.is_strict_fp64()) w_plan = requested_w_plan;
  auto admitted_layout = [&](std::size_t capacity, std::size_t panels) {
    auto candidate =
        joint_response_layout(o, v, q, capacity, panels, prepared_caller, parallel_gap_reduction,
                              include_gap_response, fused_selected);
    if (fused_selected && candidate.complete > max_bytes) {
      // Never sacrifice a previously admitted page/panel policy for optional
      // seed storage. The unchanged unfused schedule remains the bounded oracle.
      fused_selected = false;
      candidate = joint_response_layout(o, v, q, capacity, panels, prepared_caller,
                                        parallel_gap_reduction, include_gap_response);
    }
    if (w_plan) {
      auto& value = candidate.fock.value;
      auto cursor = value.arena;
      const auto offset = reserve(cursor, w_plan->storage_bytes(o, v, value.panel_capacity));
      const auto arena_bytes = align256(cursor);
      const auto extra = checked_add(
          arena_bytes - value.arena,
          checked_add(w_plan->provider_bytes(), generated_df::WExecution::host_bytes(*w_plan)));
      if (checked_add(candidate.complete, extra) > max_bytes) {
        // Preserve the strict page/panel admission rather than reducing
        // residency just to fit an unqualified precision experiment.
        w_plan.reset();
        w_resource_fallback = true;
      } else {
        value.execution_storage = offset;
        value.arena = arena_bytes;
        value.total = checked_add(value.total, extra);
        candidate.complete = checked_add(candidate.complete, extra);
      }
    }
    return candidate;
  };
  auto plan = admitted_layout(page_capacity, max_panel_buffers);
  while (plan.complete > max_bytes) {
    if (plan.fock.value.panel_capacity > 1) {
      plan = admitted_layout(page_capacity, 1);
      if (plan.complete <= max_bytes) break;
    }
    if (page_capacity == 1)
      throw std::length_error("joint DF triples response exceeds complete numeric budget");
    --page_capacity;
    plan = admitted_layout(page_capacity, max_panel_buffers);
  }
  const double minimum = validate_inputs(o, v, q, staging_layout, host, threshold);

  runtime::CudaDeviceScope device_scope(device);
  PreparedInputs prepared(staging_layout, host);

  DFCudaCombinedResponseResult result;
  auto& response = result.pullback;
  auto& d = response.diagnostic;
  d.resource_fallback = w_resource_fallback;
  auto& fock = result.fock;
  auto& p = plan.fock.value;
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;
  const std::array<std::vector<double>*, 9> response_outputs{
      &response.bov, &response.bvv, &response.ovoo,  &response.ovov, &response.fov,
      &response.t1,  &response.t2,  &response.eps_o, &response.eps_v};
  for (std::size_t x = 0; x < requested_outputs; ++x) response_outputs[x]->resize(p.sizes[x]);
  fock.foo.resize(checked_mul(o, o));
  fock.fvv.resize(checked_mul(v, v));

  int failed = 0;
  std::size_t response_tiles = 0;
  {
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };

    const generated_df::Inputs in = prepared.values;
    std::unique_ptr<generated_df::WExecution> w_execution;
    if (w_plan) {
      w_execution = std::make_unique<generated_df::WExecution>(
          *w_plan, o, v, q, context, context.arena + p.execution_storage, p.panel_capacity,
          d.fp64_gemms, d.fp32_gemms, fock.contraction_summands, d.precision_cast_elements);
      const auto& selected = w_execution->plan();
      const auto& candidate = generated_df::w_lowering_candidates[selected.selected];
      d.resource_fallback = d.resource_fallback || selected.selected != w_plan->selected;
      d.precision = selected.precision.arithmetic;
      d.w_contraction_storage_bits =
          d.precision.storage_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
      d.w_contraction_compute_bits =
          d.precision.compute_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
      d.w_contraction_accumulation_bits =
          d.precision.accumulation_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
      d.w_scientific_identity = generated_df::w_lowering_request.scientific_identity;
      d.w_semantic_identity = generated_df::w_lowering_request.semantic_identity;
      d.w_candidate_identity = candidate.identity;
      d.w_provider = candidate.provider;
      d.w_algorithm = candidate.algorithm;
      d.retained_incumbent = selected.retained_incumbent;
      d.w_provider_version = w_execution->provider_version();
      d.cuda_runtime_version = w_execution->runtime_version();
      d.w_precision_identity = selected.precision.identity;
      d.w_codegen_precision_schedule_identity = selected.schedule_identity;
      d.host_binding_bytes = generated_df::WExecution::host_bytes(selected);
      w_execution->initialize(context, in);
    }
    generated_df::ResponseOutputs out;
    const std::array<double**, 9> output_fields{&out.bov, &out.bvv, &out.ovoo,  &out.ovov, &out.fov,
                                                &out.t1,  &out.t2,  &out.eps_o, &out.eps_v};
    for (std::size_t x = 0; x < requested_outputs; ++x) {
      *output_fields[x] = pointer(plan.response_outputs[x]);
      generativeqc_tensor::cuda_check(
          cudaMemsetAsync(pointer(plan.response_outputs[x]), 0, bytes(p.sizes[x]), context.stream));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(plan.fock.foo), 0, bytes(checked_mul(o, o)), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(plan.fock.fvv), 0, bytes(checked_mul(v, v)), context.stream));

    auto* panels = pointer(p.panels);
    auto* moments = pointer(p.moments);
    auto* bar_w = pointer(plan.bar_w);
    auto* bar_v = pointer(plan.bar_v);
    auto* bar_panel = pointer(plan.bar_panel);
    auto* packed = pointer(plan.packed);

    // Shared forward panel/W contractions are charged once to the Fock work
    // ledger. Pullback-only reverse contractions have a separate ledger below.
    auto forward_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk,
                            double alpha, const double* a, std::size_t lda, const double* b,
                            std::size_t ldb, double beta, double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++fock.audit_kernels;
      fock.contraction_summands = checked_add(fock.contraction_summands, checked_mul(count, kk));
    };
    auto reverse_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk,
                            double alpha, const double* a, std::size_t lda, const double* b,
                            std::size_t ldb, double beta, double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++response.audit_kernels;
      ++response.reverse_gemms;
      d.contraction_summands = checked_add(d.contraction_summands, checked_mul(count, kk));
    };

    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        if (w_execution)
          w_execution->build_panel(context, in, occupied, panels + slot * p.v3, slot);
        else
          generated_df::build_panel(o, v, q, occupied, in, panels + slot * p.v3, forward_gemm);
        ++fock.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return panels + slot * p.v3;
    };

    const auto small_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (v * v - 1) / 256, 65535));
    const auto response_index = [&](std::size_t i, std::size_t j, std::size_t k) {
      const auto prefix = checked_mul(checked_mul(i, checked_add(i, 1)), checked_add(i, 2)) / 6;
      const auto row = checked_mul(j, checked_add(j, 1)) / 2;
      return checked_add(prefix, checked_add(row, k));
    };
    const auto run_pullback = [&](std::size_t i, std::size_t j, std::size_t k) {
      const std::array<std::size_t, 3> occupied{i, j, k};
      const double multiplicity = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
      const auto tile = response_index(i, j, k);
      if (tile >= p.tiles) throw std::logic_error("joint triples response tile index overflow");

      if (fused_selected)
        generated_df::fused_response_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                          p.blocks, bar_w, bar_v, pointer(p.partials),
                                          context.error, context.stream);
      else
        generated_df::energy_tile(o, v, i, j, k, multiplicity, threshold, in, moments, p.blocks,
                                  pointer(p.partials), context.error, context.stream);
      ++d.epilogue_kernels;
      reduce<<<1, 256, 0, context.stream>>>(pointer(p.partials), p.blocks,
                                            pointer(p.energies) + tile, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++d.reduction_kernels;

      if (!fused_selected) {
        generated_df::response_w_tile(o, v, i, j, k, multiplicity, threshold, in, moments, p.blocks,
                                      bar_w, context.error, context.stream);
        ++response.reverse_kernels;
      }
      for (unsigned perm = 0; perm < 6; ++perm) {
        const auto* order = generated_df::permutations[perm];
        const auto I = occupied[order[0]], J = occupied[order[1]], K = occupied[order[2]];
        if (!fused_selected)
          generated_df::response_v_tile(o, v, i, j, k, perm, multiplicity, threshold, in, moments,
                                        p.blocks, bar_v, context.error, context.stream);
        ovov_view<false>
            <<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, in.ovov, packed, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        generated_df::pullback_v(o, v, I, J, K, in, packed,
                                 bar_v + (fused_selected ? perm * p.v3 : 0), packed, out,
                                 reverse_gemm);
        ovov_view<true>
            <<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, packed, out.ovov, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        response.reverse_kernels += fused_selected ? 2 : 3;
      }

      if (include_gap_response) {
        generated_df::response_gap_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                        p.blocks, bar_v, context.error, context.stream);
        generated_df::GapCudaState gap{
            o, v, bar_v, pointer(plan.gap), context.error, context.stream, parallel_gap_reduction};
        const auto gaps = generated_df::gap_response_cuda(gap);
        scatter_gap<<<small_blocks, 256, 0, context.stream>>>(v, i, j, k, gaps, out.eps_o,
                                                              out.eps_v, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        const auto gap_kernels =
            generated_df::gap_response_kernel_count(o, v, parallel_gap_reduction);
        response.reverse_kernels += 2 + gap_kernels;
        response.gap.kernels += gap_kernels;
      }

      for (std::size_t index = 0; index < 3; ++index) {
        if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
            occupied.begin() + index)
          continue;
        const auto I = occupied[index];
        const auto* panel = panel_for(I);
        generativeqc_tensor::cuda_check(cudaMemsetAsync(bar_panel, 0, bytes(p.v3), context.stream));
        for (std::size_t perm = 0; perm < 6; ++perm) {
          const auto* order = generated_df::permutations[perm];
          if (occupied[order[0]] != I) continue;
          generated_df::pullback_w(o, v, I, occupied[order[1]], occupied[order[2]], in, panel,
                                   bar_w + perm * p.v3, bar_panel, out, reverse_gemm);
        }
        generated_df::pullback_panel(o, v, q, I, in, bar_panel, out, reverse_gemm);
      }
      ++response_tiles;
    };

    auto page = [&](std::size_t start, std::size_t j, std::size_t k, double* x, double* y,
                    bool run_response) {
      ++fock.page_builds;
      const auto page_bytes = bytes(checked_mul(page_capacity, p.v3));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(x, 0, page_bytes, context.stream));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(y, 0, page_bytes, context.stream));
      for (std::size_t lane = 0; lane < page_capacity && start + lane < o; ++lane) {
        const auto i = start + lane;
        const std::array<std::size_t, 3> occupied{i, j, k};
        for (std::size_t index = 0; index < 3; ++index) {
          if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
              occupied.begin() + index)
            continue;
          const auto* panel = panel_for(occupied[index]);
          for (std::size_t perm = 0; perm < 6; ++perm) {
            const auto* order = generated_df::permutations[perm];
            if (occupied[order[0]] != occupied[index]) continue;
            if (w_execution) {
              const auto slot = static_cast<std::size_t>((panel - panels) / p.v3);
              w_execution->build_w(context, in, occupied[order[0]], occupied[order[1]],
                                   occupied[order[2]], panel, moments + perm * p.v3, slot);
            } else {
              generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                    occupied[order[2]], in, panel, moments + perm * p.v3,
                                    forward_gemm);
            }
            fock.w_gemms += 2;
          }
        }
        generated_df::resolvent_tile(o, v, i, j, k, threshold, in, moments, p.blocks,
                                     x + lane * p.v3, y + lane * p.v3, context.error,
                                     context.stream);
        ++fock.vector_cubes;
        // k<=j is fixed by the outer traversal. i>=j selects every unordered
        // occupied triple exactly once, so its W moments can serve pullback too.
        if (run_response && i >= j) run_pullback(i, j, k);
      }
    };

    const auto scatter_blocks = static_cast<unsigned>(
        std::min<std::size_t>(1 + (page_capacity * page_capacity - 1) / 256, 65535));
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k <= j; ++k) {
        const double weight = j == k ? 1.0 : 2.0;
        for (std::size_t left = 0; left < o; left += page_capacity) {
          // Every page becomes a left page exactly once. Run pullback only here;
          // right-page replay below is purely a bounded Fock-storage fallback.
          page(left, j, k, pointer(plan.fock.xl), pointer(plan.fock.yl), true);
          for (std::size_t lane = 0; lane < page_capacity && left + lane < o; ++lane) {
            generated_df::fock_vv(v, page_capacity, pointer(plan.fock.xl) + lane * p.v3,
                                  pointer(plan.fock.yl) + lane * p.v3, weight,
                                  pointer(plan.fock.fvv), forward_gemm);
            fock.fock_gemms += 2;
          }
          for (std::size_t right = left; right < o; right += page_capacity) {
            if (left != right)
              page(right, j, k, pointer(plan.fock.xr), pointer(plan.fock.yr), false);
            generated_df::fock_oo(v, page_capacity, pointer(plan.fock.xl),
                                  pointer(left == right ? plan.fock.xl : plan.fock.xr),
                                  pointer(plan.fock.yl),
                                  pointer(left == right ? plan.fock.yl : plan.fock.yr), weight,
                                  pointer(plan.fock.block), forward_gemm);
            fock.fock_gemms += 2;
            scatter_fock_block<<<scatter_blocks, 256, 0, context.stream>>>(
                o, page_capacity, left, right, pointer(plan.fock.block), pointer(plan.fock.foo),
                context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            ++fock.scatter_kernels;
          }
        }
      }

    if (response_tiles != p.tiles)
      throw std::logic_error("joint DF triples pullback work mismatch");
    auto& fusion = response.scalar_fusion;
    fusion.requested = fused_scalar_response;
    fusion.selected = fused_selected;
    fusion.resource_fallback = fused_scalar_response && !include_gap_response && !fused_selected;
    fusion.schedule = fused_selected ? generated_df::scalar_fusion_identity : "unfused-primal-w-v";
    fusion.tiles = p.tiles;
    fusion.kernels = checked_mul(fused_selected ? 1 : 8, p.tiles);
    fusion.avoided_kernels = fused_selected ? checked_mul(7, p.tiles) : 0;
    fusion.workspace_bytes = bytes(checked_mul(fused_selected ? 12 : 7, p.v3));
    const auto points = checked_mul(p.tiles, p.v3);
    fusion.value_reads =
        checked_mul(points, fused_selected ? generated_df::scalar_fused_value_reads
                                           : generated_df::scalar_unfused_value_reads);
    fusion.arithmetic_ops =
        checked_mul(points, fused_selected ? generated_df::scalar_fused_arithmetic_ops
                                           : generated_df::scalar_unfused_arithmetic_ops);
    fusion.value_writes = checked_add(checked_mul(12, points), checked_mul(p.tiles, p.blocks));
    if (fock.vector_cubes != plan.fock.cubes || fock.page_builds != plan.fock.page_builds)
      throw std::logic_error("joint DF triples Fock replay work mismatch");

    response.gap.requested = include_gap_response;
    if (include_gap_response) {
      response.gap.parallel =
          generated_df::gap_response_parallel_selected(v, parallel_gap_reduction);
      response.gap.schedule = response.gap.parallel
                                  ? generated_df::gap_response_parallel_schedule_identity
                                  : "serial-source-major";
      response.gap.workspace_bytes =
          bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction));
      response.gap.materialized_elements = checked_mul(
          p.tiles, generated_df::gap_response_materialized_elements(o, v, parallel_gap_reduction));
      response.gap.value_reads = checked_mul(
          p.tiles, generated_df::gap_response_value_reads(o, v, parallel_gap_reduction));
      response.gap.value_writes = checked_mul(
          p.tiles, generated_df::gap_response_value_writes(o, v, parallel_gap_reduction));
      response.gap.reduction_summands = checked_mul(
          p.tiles, generated_df::gap_response_reduction_summands(o, v, parallel_gap_reduction));
    } else {
      response.gap.schedule = generated_df::response_without_gap_identity;
    }

    reduce<<<1, 256, 0, context.stream>>>(pointer(p.energies), p.tiles, pointer(p.energy),
                                          context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++d.reduction_kernels;
    for (std::size_t x = 0; x < requested_outputs; ++x)
      generativeqc_tensor::cuda_check(
          cudaMemcpyAsync(response_outputs[x]->data(), pointer(plan.response_outputs[x]),
                          bytes(p.sizes[x]), cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&d.energy, pointer(p.energy), sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(fock.foo.data(), pointer(plan.fock.foo),
                                                    bytes(checked_mul(o, o)),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(fock.fvv.data(), pointer(plan.fock.fvv),
                                                    bytes(checked_mul(v, v)),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(d.energy))
      throw std::runtime_error("nonfinite or unsafe joint DF triples response arithmetic");

    d.minimum_absolute_denominator = minimum;
    d.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    d.occupied_tiles = p.tiles;
    d.epilogue_points = checked_mul(p.tiles, p.v3);
    d.workspace_bytes = p.total;
    d.arena_bytes = p.arena;
    d.provider_retained_bytes = checked_add(context.metrics.provider_retained_bytes,
                                            w_execution ? w_execution->provider_bytes() : 0);
    d.panel_capacity = p.panel_capacity;
    d.h2d_bytes = prepared.h2d_bytes;
    d.d2h_bytes = checked_add(plan.response_output_bytes, sizeof(double) + sizeof(int));
    response.borrowed_host_bytes = plan.fock.host_bytes;
    response.numeric_capacity_bytes = plan.complete;
    response.scalar_response_evaluations =
        checked_mul(include_gap_response ? generated_df::response_scalar_outputs_with_gap
                                         : generated_df::response_scalar_outputs_without_gap,
                    d.epilogue_points);

    fock.minimum_absolute_denominator = minimum;
    fock.numeric_capacity_bytes = plan.complete;
    fock.borrowed_host_bytes = plan.fock.host_bytes;
    fock.workspace_bytes = p.total;
    fock.arena_bytes = p.arena;
    fock.provider_retained_bytes = d.provider_retained_bytes;
    fock.page_capacity = page_capacity;
    fock.page_count = plan.fock.pages;
    fock.panel_capacity = p.panel_capacity;
    fock.occupied_pairs = plan.fock.pairs;
    fock.unique_vector_cubes = checked_mul(plan.fock.pairs, o);
    fock.scalar_evaluations = checked_mul(plan.fock.cubes, p.v3);
    fock.h2d_bytes = 0;
    // The shared sticky status is attributed to pullback, so the Fock D2H count
    // contains only its detached matrices.
    fock.d2h_bytes = plan.fock.output_bytes;
  }

  result.shared_input_device_bytes = prepared.storage_bytes;
  result.shared_h2d_bytes = prepared.h2d_bytes;
  result.shared_response_cubes = response_tiles;
  result.avoided_w_gemms = checked_mul(12, response_tiles);
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  d.seconds = result.seconds;
  fock.seconds = 0.0;
  return result;
}

}  // namespace generativeqc::cc::triples
