#pragma once

// Included only by the native generated_grid_policy.cu, after cuda_grid.cu.
// Reuse compiler-emitted AO traversal, XC point algebra and dense contractions;
// this header keeps only validation and launch/runtime glue.
#include "dft/cuda_xc.hpp"

namespace generativeqc::dft::cuda_xc_detail {
namespace {
using generativeqc_tensor::blocks;
using generativeqc_tensor::cuda_check;
using generativeqc_tensor::I;

__global__ void validate_density(const double* density, I n, I spins, int* error) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * n * n;
       i += I(blockDim.x) * gridDim.x) {
    const I transpose = (i / (n * n) * n + i % n) * n + i / n % n;
    const double a = density[i], b = density[transpose];
    if (!isfinite(a) || fabs(a - b) > 1e-12 + 1e-10 * fmax(fabs(a), fabs(b)))
      atomicCAS(error, 0, 5);
  }
}

__global__ void add_nonlocal_energy(const double* energy, double* totals, int* error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const double value = *energy;
  if (!isfinite(value) || !isfinite(totals[0])) {
    atomicCAS(error, 0, 6);
    return;
  }
  totals[0] += value;
}

/** Stream-ordered stable compaction of one tile's already computed predicate.
 * Warp ballots implement a prefix over increasing AO IDs. Every lane reaches
 * every barrier, including tail lanes and empty tiles. No floating-point AO
 * arithmetic or selection threshold is duplicated here.
 */
__global__ void compact_ao_columns(const unsigned* flags, std::size_t nao, std::size_t* indices,
                                   std::size_t capacity, std::size_t initial_offset,
                                   std::size_t tile, std::size_t* offsets, int* error) {
  constexpr unsigned width = 32, warps = 8;
  __shared__ unsigned warp_prefix[warps];
  __shared__ std::size_t base;
  const unsigned lane = threadIdx.x % width, warp = threadIdx.x / width;
  if (threadIdx.x == 0) base = tile ? offsets[tile - 1] : initial_offset;
  __syncthreads();
  for (std::size_t first = 0; first < nao; first += width * warps) {
    const auto column = first + threadIdx.x;
    const bool keep = column < nao && flags[column] != 0;
    const unsigned mask = __ballot_sync(0xffffffffU, keep);
    if (lane == 0) warp_prefix[warp] = __popc(mask);
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned prefix = 0;
      for (unsigned w = 0; w < warps; ++w) {
        const auto count = warp_prefix[w];
        warp_prefix[w] = prefix;
        prefix += count;
      }
    }
    __syncthreads();
    if (keep) {
      const auto rank = warp_prefix[warp] + __popc(mask & ((1U << lane) - 1U));
      if (base > capacity || rank >= capacity - base)
        atomicCAS(error, 0, 7);
      else
        indices[base + rank] = column;
    }
    // Retire readers before publishing the next wave's base/prefix storage.
    __syncthreads();
    if (threadIdx.x == width * warps - 1) base += warp_prefix[warp] + __popc(mask);
    __syncthreads();
  }
  if (threadIdx.x == 0) offsets[tile] = base;
}

}  // namespace

void select_ao(const CudaXcLayout& l, cudaStream_t stream, const double* basis,
               const double* points, std::size_t count, double cutoff, double* ao, double* work,
               int* error, unsigned* host_flags) {
  // Before first evaluation the dense work panels are dead. Even the smallest
  // panel holds N doubles, enough for N flags, with no extra device allocation.
  auto* flags = reinterpret_cast<unsigned*>(work);
  int failure = 0;
  try {
    cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(flags, 0, l.nao * sizeof(unsigned), stream));
    scheduled_ao(stream, basis, l.natom, l.nprimitive, l.nao, points, count, l.jets, ao, error,
                 nullptr);
    cuda_check(cudaGetLastError());
    const auto point_blocks = std::min(std::size_t{65535}, (count + 127) / 128);
    active_ao_columns<<<dim3((l.nao + 31) / 32, point_blocks), 128, 0, stream>>>(
        ao, count, l.nao, l.jets, cutoff, flags, error);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(host_flags, flags, l.nao * sizeof(unsigned), cudaMemcpyDeviceToHost,
                               stream));
    cuda_check(cudaMemcpyAsync(&failure, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }
  if (failure) throw std::runtime_error("nonfinite native XC AO selection output");
}

void select_ao_chunk(const CudaXcLayout& l, cudaStream_t stream, const double* basis,
                     const double* points, std::size_t first_tile, std::size_t tile_count,
                     double cutoff, double* ao, double* work, double* point_totals, int* error,
                     std::size_t* indices, std::size_t capacity, std::size_t initial_offset,
                     std::size_t* host_offsets) {
  static_assert(sizeof(std::size_t) <= sizeof(double));
  const auto tiles = 1 + (l.npoint - 1) / l.tile_points;
  if (!tile_count || tile_count > 3 * l.tile_points || first_tile >= tiles ||
      tile_count > tiles - first_tile || initial_offset > capacity)
    throw std::invalid_argument("CUDA XC discovery chunk exceeds borrowed scratch");
  auto* flags = reinterpret_cast<unsigned*>(work);
  auto* offsets = reinterpret_cast<std::size_t*>(point_totals);
  int failure = 0;
  try {
    cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
      const auto begin = (first_tile + tile) * l.tile_points;
      const auto count = std::min(l.tile_points, l.npoint - begin);
      cuda_check(cudaMemsetAsync(flags, 0, l.nao * sizeof(unsigned), stream));
      scheduled_ao(stream, basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, l.jets,
                   ao, error, nullptr);
      cuda_check(cudaGetLastError());
      const auto point_blocks = std::min(std::size_t{65535}, (count + 127) / 128);
      active_ao_columns<<<dim3((l.nao + 31) / 32, point_blocks), 128, 0, stream>>>(
          ao, count, l.nao, l.jets, cutoff, flags, error);
      cuda_check(cudaGetLastError());
      compact_ao_columns<<<1, 256, 0, stream>>>(flags, l.nao, indices, capacity, initial_offset,
                                                tile, offsets, error);
      cuda_check(cudaGetLastError());
    }
    cuda_check(cudaMemcpyAsync(host_offsets, offsets, tile_count * sizeof(std::size_t),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&failure, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }
  if (failure) throw std::runtime_error("invalid native XC device AO discovery output");
}

bool mixed_matrix_contractions_supported(const CudaXcLayout& layout) {
  return !layout.response && layout.ao_precision == CudaXcAoPrecision::Fp64 && layout.npoint != 0 &&
         fp32_tiled_xc_admitted(layout.nao, layout.tile_points, layout.spins, layout.work_jets);
}

void enqueue(const CudaXcLayout& l, CudaXcPointLauncher point_launcher, cudaStream_t stream,
             const double* basis, const double* points, const double* weights,
             const double* density, double* ao, double* work, double* features,
             double* coefficients, double* point_totals, double* potential, double* totals,
             int* error, const std::array<CudaXcDensityBinding, 2>& density_bindings,
             const std::vector<CudaXcDensityLauncher>& local_density_launchers,
             const double* direction, double* delta_features, double* total_density,
             double* total_gradient, const std::vector<std::size_t>& ao_offsets,
             const std::size_t* ao_ids, const tensor::PreparedPanelProduct* density_provider,
             const tensor::PreparedSymmetricProduct* potential_binding,
             const CudaXcPointBatchPlan& point_batch_plan,
             CudaXcPointBatchLauncher point_batch_launcher, double* point_batch_arena) {
  const I matrices = l.spins * l.nao * l.nao;
  if ((total_density == nullptr) != (total_gradient == nullptr))
    throw std::invalid_argument("CUDA XC total-density capture requires rho and gradient together");
  if (total_density != nullptr && l.feature_terms < 4)
    throw std::invalid_argument("CUDA XC total-density gradient capture requires GGA ingredients");
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(density, l.nao, l.spins, error);
  cuda_check(cudaGetLastError());
  if (direction) {
    validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(direction, l.nao, l.spins, error);
    cuda_check(cudaGetLastError());
  }
  const bool compact_candidate =
      point_batch_plan.compact && !density_provider &&
      !(potential_binding && potential_binding->diagnostic().provider_allowance);
  const bool compensate_totals = !l.fp32_point_totals;
  if (compensate_totals)
    cuda_check(cudaMemsetAsync(totals, 0, kCudaXcTotalStorageElements * sizeof(double), stream));
  if (l.local_ao || compact_candidate) {
    // The first selected tile cannot initialize matrix entries absent from its
    // map. Clear once for the complete evaluation; dense initialization stays
    // fused with the first potential tile as before.
    cuda_check(cudaMemsetAsync(potential, 0, matrices * sizeof(double), stream));
    cuda_check(cudaMemsetAsync(totals, 0, 3 * sizeof(double), stream));
  }
  if (density_provider && !l.local_ao) {
    // The compiler-owned symmetric factor is invariant across point tiles.
    // Rebuild it on every physical/captured body, never by density pointer identity.
    materialize_density_factor<<<blocks(matrices, 128), 128, 0, stream>>>(
        density, l.nao, l.spins, density_provider->materialized_matrices(), error);
    cuda_check(cudaGetLastError());
  }
  const auto batch_tiles = point_batch_plan.tiles;
  auto* batch_ao = batch_tiles > 1 ? point_batch_arena : ao;
  auto* batch_features = batch_tiles > 1 ? batch_ao + point_batch_plan.ao_elements : features;
  auto* batch_coefficients =
      batch_tiles > 1 ? batch_features + point_batch_plan.feature_elements : coefficients;
  auto* batch_totals =
      batch_tiles > 1 ? batch_coefficients + point_batch_plan.feature_elements : point_totals;
  auto* batch_work = batch_totals + point_batch_plan.total_elements;
  auto* batch_potential = batch_work + point_batch_plan.work_elements;
  const auto* batch_descriptors = reinterpret_cast<const CudaXcCompactTile*>(
      batch_potential + point_batch_plan.potential_elements);
  for (std::size_t batch_begin = 0; batch_begin < l.npoint;
       batch_begin += batch_tiles * l.tile_points) {
    const auto batch_count = std::min(batch_tiles * l.tile_points, l.npoint - batch_begin);
    const auto batch_end = batch_begin + batch_count;
    const bool compact =
        compact_candidate &&
        compact_point_batch_admitted(l, ao_offsets, batch_begin / l.tile_points,
                                     1 + (batch_end - 1) / l.tile_points, point_batch_plan.ragged);
    std::size_t ao_offset = 0;
    // Empty groups still launch feature/point-total bodies with a legal grid.
    std::size_t max_active = 1;
    // Retain only AO panels. Density scratch and optional provider factors are
    // consumed immediately and reused; no AO/jet/density work is repeated.
    for (std::size_t begin = batch_begin; begin < batch_end; begin += l.tile_points) {
      const auto block = bind_native_ao_grid_block(l, ao_offsets, ao_ids, begin);
      const I count = block.npoint, active = block.nactive;
      max_active = std::max(max_active, std::size_t(active));
      const auto* ids = block.ao_ids;
      auto* ao = batch_ao + ao_offset;
      auto* features = batch_features + (begin - batch_begin) * l.spins * l.feature_terms;
      // Route B changes only AO arithmetic. The AO panel and all downstream
      // density/XC reductions stay FP64 so this is a clean precision ablation.
      if (active)
        scheduled_ao(stream, basis, l.natom, l.nprimitive, active, points + 3 * begin, count,
                     l.jets, ao, error, ids,
                     l.ao_precision == CudaXcAoPrecision::Fp32ComputeFp64Storage);
      cuda_check(cudaGetLastError());
      if (compact) {
        ao_offset += count * active * l.jets;
        continue;
      }
      const auto density_launcher = l.local_ao
                                        ? local_density_launchers[block.point_start / l.tile_points]
                                        : density_bindings[count == l.tile_points ? 0 : 1].launch;
      if (l.fp32_contractions && fp32_tiled_xc_admitted(active, count, l.spins, l.work_jets)) {
        launch_density_product<false, true, float>(stream, density, ao, active, count, l.spins,
                                                   l.work_jets, work, error, ids, l.nao);
      } else if (density_provider && active) {
        if (l.local_ao) {
          // Preserve the mapped scientific domain. Only its compact symmetric
          // factor is packed; all tiles reuse the same bounded provider cache.
          gather_density_factor<<<blocks(l.spins * active * active, 128), 128, 0, stream>>>(
              density, l.nao, active, l.spins, ids, density_provider->materialized_matrices(),
              error);
          cuda_check(cudaGetLastError());
        }
        density_provider->execute(stream, active, count * l.work_jets, ao, work, error);
      } else
        density_launcher(stream, density, ao, active, count, l.spins, l.work_jets, work, error, ids,
                         l.nao);
      cuda_check(cudaGetLastError());
      scheduled_density_features(stream, ao, work, active, count, l.spins, l.jets, l.work_jets,
                                 l.feature_terms, l.functional, features, error);
      cuda_check(cudaGetLastError());
      if (total_density) {
        scheduled_total_density_features(stream, features, count, l.spins, l.feature_terms, begin,
                                         total_density, total_gradient, error);
        cuda_check(cudaGetLastError());
      }
      if (direction) {
        // AO panels are shared; work is scratch and can be reused after the
        // reference features are retained. No host AO/feature staging occurs.
        density_launcher(stream, direction, ao, l.nao, count, l.spins, l.work_jets, work, error,
                         nullptr, l.nao);
        cuda_check(cudaGetLastError());
        scheduled_density_features(stream, ao, work, active, count, l.spins, l.jets, l.work_jets,
                                   l.feature_terms, l.functional, delta_features, error);
        cuda_check(cudaGetLastError());
      }
      ao_offset += count * active * l.jets;
    }
    const auto tile_count = 1 + (batch_count - 1) / l.tile_points;
    const auto* descriptors = compact ? batch_descriptors + batch_begin / l.tile_points : nullptr;
    if (compact) {
      launch_batch_density(stream, l, descriptors, tile_count, max_active, density, batch_ao,
                           batch_work, batch_features, batch_begin, ao_ids, error);
      cuda_check(cudaGetLastError());
      if (total_density)
        for (std::size_t begin = batch_begin; begin < batch_end; begin += l.tile_points) {
          const auto count = std::min(l.tile_points, l.npoint - begin);
          const auto offset = (begin - batch_begin) * l.spins * l.feature_terms;
          scheduled_total_density_features(stream, batch_features + offset, count, l.spins,
                                           l.feature_terms, begin, total_density, total_gradient,
                                           error);
          cuda_check(cudaGetLastError());
        }
    }
    if (batch_tiles > 1)
      point_batch_launcher(stream, batch_features, weights + batch_begin, batch_count, l.spins,
                           batch_coefficients, batch_totals, error, l.functional, l.exchange_scale,
                           l.correlation_scale, l.tile_points);
    else
      point_launcher(stream, batch_features, weights + batch_begin, batch_count, l.spins,
                     batch_coefficients, batch_totals, error, l.functional, l.exchange_scale,
                     l.correlation_scale, delta_features);
    cuda_check(cudaGetLastError());
    if (compact) {
      launch_batch_potential(stream, l, descriptors, tile_count, max_active, batch_ao,
                             batch_coefficients, weights, batch_work, batch_potential,
                             compensate_totals ? nullptr : batch_totals, batch_begin, ao_ids,
                             potential, totals, error);
      cuda_check(cudaGetLastError());
      if (compensate_totals) {
        accumulate_compensated_xc_totals<<<1, 32, 0, stream>>>(batch_totals, batch_count,
                                                               l.tile_points, totals, error);
        cuda_check(cudaGetLastError());
      }
      continue;
    }
    ao_offset = 0;
    // Contractions and scatters retain their historical stream/tile order.
    // Overlapping indexed matrix entries never race. FP64 point totals retain
    // their low words across batches instead of rounding every tile into E_xc.
    for (std::size_t begin = batch_begin; begin < batch_end; begin += l.tile_points) {
      const auto block = bind_native_ao_grid_block(l, ao_offsets, ao_ids, begin);
      const I count = block.npoint, active = block.nactive;
      const auto feature_offset = (begin - batch_begin) * l.spins * l.feature_terms;
      scheduled_potential(stream, batch_ao + ao_offset, batch_coefficients + feature_offset,
                          weights + begin, active, count, l.spins, l.feature_terms, l.work_jets,
                          work,
                          compensate_totals ? nullptr : batch_totals + 3 * (begin - batch_begin),
                          potential, totals, begin != 0 || l.local_ao, error, block.ao_ids, l.nao,
                          potential_binding, l.fp32_contractions);
      cuda_check(cudaGetLastError());
      ao_offset += count * active * l.jets;
    }
    if (compensate_totals) {
      accumulate_compensated_xc_totals<<<1, 32, 0, stream>>>(batch_totals, batch_count,
                                                             l.tile_points, totals, error);
      cuda_check(cudaGetLastError());
    }
  }
  if (compensate_totals) {
    fold_compensated_xc_totals<<<1, 32, 0, stream>>>(totals, error);
    cuda_check(cudaGetLastError());
  }
}

void enqueue_nonlocal_potential(const CudaXcLayout& l, cudaStream_t stream, const double* basis,
                                const double* points, const double* effective_weights,
                                const double* total_gradient, const double* vrho,
                                const double* vsigma, const double* nonlocal_energy, double* ao,
                                double* coefficients, double* potential, double* totals, int* error,
                                double* work, const std::vector<std::size_t>& ao_offsets,
                                const std::size_t* ao_ids) {
  if (l.feature_terms < 4 || l.ao_precision != CudaXcAoPrecision::Fp64)
    throw std::invalid_argument("CUDA nonlocal AO assembly requires strict-FP64 GGA ingredients");
  for (std::size_t begin = 0; begin < l.npoint; begin += l.tile_points) {
    const auto block = bind_native_ao_grid_block(l, ao_offsets, ao_ids, begin);
    const I count = block.npoint, active = block.nactive;
    const auto* ids = block.ao_ids;
    if (active)
      scheduled_ao(stream, basis, l.natom, l.nprimitive, active, points + 3 * begin, count, l.jets,
                   ao, error, ids);
    cuda_check(cudaGetLastError());
    scheduled_nonlocal_feature_coefficients(stream, total_gradient, vrho, vsigma, begin, count,
                                            l.spins, l.feature_terms, coefficients, error);
    cuda_check(cudaGetLastError());
    if (l.local_ao) {
      // Reuse the semilocal compact bilinear with nonlocal coefficients. The
      // energy is supplied once below, so no point-total reduction is needed.
      scheduled_potential(stream, ao, coefficients, effective_weights + begin, active, count,
                          l.spins, l.feature_terms, l.work_jets, work, nullptr, potential, totals,
                          true, error, ids, l.nao);
    } else {
      assemble_potential<<<blocks(l.spins * l.nao * l.nao, 128), 128, 0, stream>>>(
          ao, coefficients, effective_weights + begin, l.nao, count, l.spins, l.feature_terms,
          potential, error);
    }
    cuda_check(cudaGetLastError());
  }
  add_nonlocal_energy<<<1, 1, 0, stream>>>(nonlocal_energy, totals, error);
  cuda_check(cudaGetLastError());
}
}  // namespace generativeqc::dft::cuda_xc_detail
