#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "cc/df_triples.hpp"
#include "generated_df_occupied_triples_cuda.cuh"
#include "posthf/capacity.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc::triples {
namespace {
using posthf::checked_add;
using posthf::checked_mul;
using Clock = std::chrono::steady_clock;
constexpr std::size_t provider_allowance = 96ULL << 20;
constexpr std::size_t blas_workspace = 4ULL << 20;
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t float_bytes(std::size_t n) { return checked_mul(n, sizeof(float)); }
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
  std::size_t fp32_ovoo{}, fp32_t2{}, fp32_panels{}, fp32_w_scratch{};
  std::size_t arena{}, total{}, panel_capacity{}, v3{}, tiles{};
  unsigned blocks{};
};

Layout layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels,
              bool mixed_w = false) {
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
  for (std::size_t x = 0; x < p.sizes.size(); ++x) p.inputs[x] = reserve(cursor, bytes(p.sizes[x]));
  p.panels = reserve(cursor, bytes(checked_mul(p.panel_capacity, p.v3)));
  p.moments = reserve(cursor, bytes(checked_mul(6, p.v3)));
  if (mixed_w) {
    p.fp32_ovoo = reserve(cursor, float_bytes(p.sizes[2]));
    p.fp32_t2 = reserve(cursor, float_bytes(p.sizes[6]));
    p.fp32_panels =
        reserve(cursor, float_bytes(checked_mul(p.panel_capacity, p.v3)));
    p.fp32_w_scratch = reserve(cursor, float_bytes(p.v3));
  }
  p.partials = reserve(cursor, bytes(p.blocks));
  p.energies = reserve(cursor, bytes(p.tiles));
  p.energy = reserve(cursor, sizeof(double));
  p.error = reserve(cursor, sizeof(int));
  p.library = reserve(cursor, blas_workspace);
  p.arena = align256(cursor);
  p.total = checked_add(p.arena, provider_allowance);
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

struct ResponseLayout {
  Layout value;
  std::array<std::size_t, 9> outputs{};
  std::size_t bar_w{}, bar_v{}, bar_panel{}, packed{}, gap{}, host_bytes{}, complete{};
};
ResponseLayout response_layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels,
                               std::size_t caller_bytes) {
  ResponseLayout p;
  p.value = layout(o, v, q, panels);
  auto cursor = p.value.arena;
  std::size_t values = 0;
  for (std::size_t x = 0; x < p.outputs.size(); ++x) {
    p.outputs[x] = reserve(cursor, bytes(p.value.sizes[x]));
    values = checked_add(values, p.value.sizes[x]);
  }
  p.bar_w = reserve(cursor, bytes(checked_mul(6, p.value.v3)));
  p.bar_v = reserve(cursor, bytes(p.value.v3));
  p.bar_panel = reserve(cursor, bytes(p.value.v3));
  p.packed = reserve(cursor, bytes(checked_mul(v, v)));
  p.gap = reserve(cursor, bytes(generated_df::gap_response_arena_elements(o, v)));
  p.value.arena = align256(cursor);
  p.value.total = checked_add(p.value.arena, provider_allowance);
  p.host_bytes = bytes(values);
  p.complete = checked_add(caller_bytes, checked_add(p.value.total, checked_mul(2, p.host_bytes)));
  (void)checked_mul(43, checked_mul(p.value.tiles, p.value.v3));
  // At most three forward and three reverse-regenerated panels per tile, plus
  // two factor products per distinct occupied index. Charge complete work
  // before input access even for logical shapes too large to execute.
  const auto panel_work = checked_mul(checked_mul(12, p.value.tiles), checked_mul(q, p.value.v3));
  const auto w_work =
      checked_mul(checked_mul(18, p.value.tiles),
                  checked_add(checked_mul(v, p.value.v3), checked_mul(o, p.value.v3)));
  const auto v_work = checked_mul(checked_mul(24, p.value.tiles), p.value.v3);
  (void)checked_add(panel_work, checked_add(w_work, v_work));
  return p;
}

struct FockLayout {
  Layout value;
  std::size_t foo{}, fvv{}, xl{}, yl{}, xr{}, yr{}, block{};
  std::size_t capacity{}, pages{}, pairs{}, cubes{}, page_builds{}, page_pairs{};
  std::size_t host_bytes{}, output_bytes{}, complete{};
};
FockLayout fock_layout(std::size_t o, std::size_t v, std::size_t q, std::size_t capacity,
                       std::size_t panels, std::size_t caller_bytes) {
  FockLayout r;
  r.value = layout(o, v, q, panels);
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
  (void)checked_add(checked_add(panel_work, w_work), checked_add(vv_work, oo_work));
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
                              DFTriplesPrecision precision) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples dimensions, threshold or panel limit");
  if (precision != DFTriplesPrecision::Fp64 && precision != DFTriplesPrecision::WFp32)
    throw std::invalid_argument("invalid DF triples precision mode");
  const bool mixed_w = precision == DFTriplesPrecision::WFp32;
  auto p = layout(o, v, q, max_panel_buffers, mixed_w);
  if (p.total > max_bytes) p = layout(o, v, q, 1, mixed_w);
  if (p.total > max_bytes) throw std::length_error("DF triples exceed numeric memory budget");
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);

  // Borrowed D2H destinations outlive Context's exception-path stream drain.
  DFCudaResult result;
  result.precision = precision;
  if (mixed_w) {
    result.w_storage_bits = result.w_compute_bits = result.w_accumulation_bits = 32;
    std::copy_n(generated_df::w_fp32_precision_schedule_identity, 64,
                result.precision_schedule_identity.begin());
  }
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    if (mixed_w)
      generativeqc_tensor::blas_check(cublasSetMathMode(context.handle, CUBLAS_PEDANTIC_MATH));
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
    auto* fp32_ovoo =
        mixed_w ? reinterpret_cast<float*>(context.arena + p.fp32_ovoo) : nullptr;
    auto* fp32_t2 = mixed_w ? reinterpret_cast<float*>(context.arena + p.fp32_t2) : nullptr;
    auto* fp32_panels =
        mixed_w ? reinterpret_cast<float*>(context.arena + p.fp32_panels) : nullptr;
    auto* fp32_w_scratch =
        mixed_w ? reinterpret_cast<float*>(context.arena + p.fp32_w_scratch) : nullptr;
    if (mixed_w) {
      generativeqc_tensor::convert_fp64_to_fp32(context, in.ovoo, fp32_ovoo,
                                                static_cast<generativeqc_tensor::I>(p.sizes[2]), 20);
      generativeqc_tensor::convert_fp64_to_fp32(context, in.t2, fp32_t2,
                                                static_cast<generativeqc_tensor::I>(p.sizes[6]), 21);
      result.precision_cast_elements =
          checked_add(p.sizes[2], p.sizes[6]);
    }
    auto gemm64 = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      generativeqc_tensor::blas_check(
          cublasDgemm(context.handle, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                      tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, static_cast<int>(m),
                      static_cast<int>(n), static_cast<int>(k), &alpha, a, static_cast<int>(lda), b,
                      static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
      result.contraction_summands =
          checked_add(result.contraction_summands, checked_mul(checked_mul(m, n), k));
      ++result.fp64_gemms;
    };
    auto gemm32 = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k, double alpha,
                      const float* a, std::size_t lda, const float* b, std::size_t ldb, double beta,
                      float* out, std::size_t ldc) {
      const float alpha32 = static_cast<float>(alpha), beta32 = static_cast<float>(beta);
      generativeqc_tensor::blas_check(
          cublasSgemm(context.handle, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                      tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, static_cast<int>(m),
                      static_cast<int>(n), static_cast<int>(k), &alpha32, a, static_cast<int>(lda),
                      b, static_cast<int>(ldb), &beta32, out, static_cast<int>(ldc)));
      result.contraction_summands =
          checked_add(result.contraction_summands, checked_mul(checked_mul(m, n), k));
      ++result.fp32_gemms;
    };
    auto accumulate32 = [&](const float* source, std::size_t count, double alpha, double beta,
                            double* target) {
      generativeqc_tensor::accumulate_fp32_into_fp64(
          context, source, target, static_cast<generativeqc_tensor::I>(count), alpha, beta, 22);
      result.precision_cast_elements = checked_add(result.precision_cast_elements, count);
    };
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
        generated_df::build_panel(o, v, q, occupied, in, panels + slot * p.v3, gemm64);
        ++result.panel_gemms;
        if (mixed_w) {
          generativeqc_tensor::convert_fp64_to_fp32(
              context, panels + slot * p.v3, fp32_panels + slot * p.v3,
              static_cast<generativeqc_tensor::I>(p.v3), 23);
          result.precision_cast_elements = checked_add(result.precision_cast_elements, p.v3);
        }
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return slot;
    };
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
          const std::array<std::size_t, 3> occupied{i, j, k};
          // Group W seeds by their integral-panel index. A single-panel fallback
          // consumes every dependent GEMM before that storage is reused. All
          // producer/consumer work is ordered on Context's one owned stream.
          for (std::size_t index = 0; index < occupied.size(); ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto slot = panel_slot_for(occupied[index]);
            const auto* panel = panels + slot * p.v3;
            for (std::size_t permutation = 0; permutation < 6; ++permutation) {
              const auto* order = generated_df::permutations[permutation];
              if (occupied[order[0]] != occupied[index]) continue;
              if (mixed_w) {
                generated_df::build_w_fp32(
                    o, v, occupied[order[0]], occupied[order[1]], occupied[order[2]], fp32_ovoo,
                    fp32_t2, fp32_panels + slot * p.v3, fp32_w_scratch,
                    moments + permutation * p.v3, gemm32, accumulate32);
              } else {
                generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                      occupied[order[2]], in, panel,
                                      moments + permutation * p.v3, gemm64);
              }
              result.moment_gemms += 2;
            }
          }
          const double degeneracy = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
          generated_df::energy_tile(o, v, i, j, k, degeneracy, threshold, in, moments, p.blocks,
                                    partials, context.error, context.stream);
          ++result.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(partials, p.blocks, energies + tile, context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++result.reduction_kernels;
          ++tile;
        }
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
    result.provider_retained_bytes = context.metrics.provider_retained_bytes;
    result.panel_capacity = p.panel_capacity;
    result.d2h_bytes = sizeof(double) + sizeof(int);
  }  // Drain/destroy the stream, buffers and BLAS provider inside endpoint timing.
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}
DFCudaResponseResult pullback_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                      const double* bov, const double* bvv, const double* ovoo,
                                      const double* ovov, const double* fov, const double* t1,
                                      const double* t2, const double* eps_o, const double* eps_v,
                                      double threshold, std::size_t max_bytes, int device,
                                      std::size_t caller_bytes, std::size_t max_panel_buffers) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples response dimensions, threshold or panel limit");
  auto r = response_layout(o, v, q, max_panel_buffers, caller_bytes);
  if (r.complete > max_bytes) r = response_layout(o, v, q, 1, caller_bytes);
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
  for (std::size_t x = 0; x < 9; ++x) outputs[x]->resize(p.sizes[x]);
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
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };
    for (std::size_t x = 0; x < 9; ++x) {
      *input_fields[x] = pointer(p.inputs[x]);
      *output_fields[x] = pointer(r.outputs[x]);
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(pointer(p.inputs[x]), host[x],
                                                      bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                      context.stream));
      generativeqc_tensor::cuda_check(
          cudaMemsetAsync(pointer(r.outputs[x]), 0, bytes(p.sizes[x]), context.stream));
      d.h2d_bytes = checked_add(d.h2d_bytes, bytes(p.sizes[x]));
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
      generativeqc_tensor::blas_check(
          cublasDgemm(context.handle, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                      tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, static_cast<int>(m),
                      static_cast<int>(n), static_cast<int>(kk), &alpha, a, static_cast<int>(lda),
                      b, static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
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
          generated_df::response_gap_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                          p.blocks, bar_v, context.error, context.stream);
          generated_df::GapCudaState gap{
              o, v, bar_v, pointer(r.gap), context.error, context.stream};
          const auto gaps = generated_df::gap_response_cuda(gap);
          scatter_gap<<<small_blocks, 256, 0, context.stream>>>(v, i, j, k, gaps, out.eps_o,
                                                                out.eps_v, context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          result.reverse_kernels += 2 + generated_df::gap_response_operations;
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
    reduce<<<1, 256, 0, context.stream>>>(pointer(p.energies), p.tiles, pointer(p.energy),
                                          context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++d.reduction_kernels;
    for (std::size_t x = 0; x < 9; ++x)
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
    d.d2h_bytes = checked_add(r.host_bytes, sizeof(double) + sizeof(int));
    result.borrowed_host_bytes = r.host_bytes;
    result.numeric_capacity_bytes = r.complete;
    result.scalar_response_evaluations = checked_mul(43, d.epilogue_points);
  }
  d.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}
DFCudaFockResult fock_response_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                       const double* bov, const double* bvv, const double* ovoo,
                                       const double* ovov, const double* fov, const double* t1,
                                       const double* t2, const double* eps_o, const double* eps_v,
                                       double threshold, std::size_t max_bytes, int device,
                                       std::size_t caller_bytes, std::size_t max_page_rows,
                                       std::size_t max_panel_buffers) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples Fock dimensions, threshold or panel limit");
  auto capacity = max_page_rows ? std::min(o, max_page_rows) : o;
  auto r = fock_layout(o, v, q, capacity, max_panel_buffers, caller_bytes);
  while (r.complete > max_bytes) {
    // Prefer retaining more resolvent rows over extra cached integral panels:
    // a full page eliminates every cross-occupied recomputation.
    if (r.value.panel_capacity > 1) {
      r = fock_layout(o, v, q, capacity, 1, caller_bytes);
      if (r.complete <= max_bytes) break;
    }
    if (capacity == 1)
      throw std::length_error("DF triples Fock response exceeds complete numeric budget");
    --capacity;
    r = fock_layout(o, v, q, capacity, max_panel_buffers, caller_bytes);
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
    for (std::size_t x = 0; x < 9; ++x) {
      *fields[x] = pointer(p.inputs[x]);
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(pointer(p.inputs[x]), host[x],
                                                      bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                      context.stream));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(r.foo), 0, bytes(o * o), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(r.fvv), 0, bytes(v * v), context.stream));
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      generativeqc_tensor::blas_check(
          cublasDgemm(context.handle, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                      tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, static_cast<int>(m),
                      static_cast<int>(n), static_cast<int>(kk), &alpha, a, static_cast<int>(lda),
                      b, static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
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
  result.h2d_bytes = r.host_bytes;
  result.d2h_bytes = checked_add(r.output_bytes, sizeof(int));
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}
}  // namespace generativeqc::cc::triples
