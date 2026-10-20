#pragma once

#include <algorithm>
#include <type_traits>

#include "generated_md_j_engine.cuh"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda/direct_md_j.hpp"

// Native ownership/launch adapter for the reference-aligned compiler backend.
// GPU4PySCF source correspondence and the exact-mask compatibility contract:
// docs/developer/md_j_engine.md. No PySCF/CuPy code runs in this provider.
namespace generativeqc::scf::cuda_execution {
namespace {

/** Each output AO pair has target_count primitive products and at most
 * primitive_count sources. Apportioning by that actual target multiplicity
 * bounds the sum of omitted absolute contributions by min(tolerance,1e-12).
 * Source-prefix admission uses the global maximum multiplicity; target and
 * tile admission reuse the actual-target thresholds. This bounds
 * screening error, independently of FP64 accumulation error. */
__host__ __device__ inline double md_j_primitive_allowance(MdJView md, double screening,
                                                           std::size_t target_count = 0) {
  if (!target_count) target_count = md.maximum_pair_primitives;
#ifdef __CUDA_ARCH__
  return __ddiv_rd(__ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(target_count)),
                   double(md.primitive_count));
#else
  return std::nextafter(
      std::nextafter(std::fmin(screening, kMdJDensityErrorCap) / double(target_count), 0.0) /
          double(md.primitive_count),
      0.0);
#endif
}

/** Sorted primitive descriptors are the decontracted pair mapping. Each row
 * receives one packed Hermite interval; original shell/primitive identity and
 * sparse E offsets stay intact for residual J, K and public-AO reconstruction. */
__global__ void md_j_engine_pair_loc(MdJView md, unsigned angular) {
  const auto position = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = md.class_offsets[angular + 1] - md.class_offsets[angular];
  if (position >= count) return;
  auto& primitive = md.primitives[md.ordered_primitives[md.class_offsets[angular] + position]];
  primitive.hermite_offset = md.pair_loc_offsets[angular] + position * md_j_hermite_count(angular);
}

/** Divide outward once per target. A rejected source has q_source below a
 * downward-rounded allowance/q_target, so its contribution is below allowance.
 * Zero tolerance and unknown output norms retain every nonzero source. */
__device__ inline double md_j_engine_source_threshold(double output_norm, double allowance) {
  if (allowance == 0.0 || !(output_norm >= 0.0) || output_norm == kMdJBoundInfinity) return 0.0;
  if (output_norm == 0.0) return INFINITY;
  return __ddiv_rd(allowance, output_norm);
}

/** One warp constructs every level for 32 leaves, including neutral padding.
 * Primitive nodes retain minimum target thresholds and maximum source norms;
 * reference nodes retain maxima of all fields. Refresh on the owning stream
 * for every density/window. No reduction reads another warp's scratch. */
__global__ void md_j_engine_hierarchy(MdJView md, unsigned angular, std::size_t begin,
                                      std::size_t end, double screening = 0.0) {
  const auto position = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto layout = md.hierarchy_layout.buckets[angular];
  if (position >= layout.aligned) return;  // padded size is a whole number of warps
  auto* output = md.hierarchy + md.hierarchy_layout.bucket_offsets[angular];
  MdJScreenBound bound{};
  // Primitive nodes reduce the minimum source threshold. Infinite padding is
  // its neutral element; reference nodes retain their legacy maximum minimum.
  if (md.primitive_bounds) bound.minimum = INFINITY;
  if (position < layout.count) {
    const auto index = md.ordered_primitives[md.class_offsets[angular] + position];
    const auto primitive = md.primitives[index];
    const auto system = md.pairs[primitive.pair].system;
    if (system >= begin && system < end) {
      bound = md_j_screen_bound(
          md.primitive_bounds ? md.primitive_bounds[index] : md.maximum_bounds[primitive.pair],
          md.density_bounds[primitive.pair], md.minimum_bounds[primitive.pair]);
      if (md.primitive_screen) {
        // The preceding source-norm pass has initialized every leaf. qD now
        // bounds the actual precontracted Hermite source, not q times a shell
        // density envelope. Either direction can keep this q-sorted position.
        bound.qd = output[position].qd;
        const auto pair = md.pairs[primitive.pair];
        bound.minimum = md_j_engine_source_threshold(
            bound.q,
            md_j_primitive_allowance(md, screening, pair.primitive_end - pair.primitive_begin));
        const auto allowance = md_j_primitive_allowance(md, screening);
        const bool target = md.primitive_screen->maximum_source_norm != 0.0 &&
                            md.primitive_screen->maximum_source_norm >= bound.minimum;
        const bool source = bound.qd != 0.0 && md_j_bound_product_up(md.primitive_maximum_bound,
                                                                     bound.qd) >= allowance;
        if (target || source)
          atomicMax(md.primitive_screen->active + angular, static_cast<unsigned>(position + 1));
      }
    }
  }
  output[position] = bound;
  for (unsigned level = 1; level < kMdJHierarchyLevels; ++level) {
    const unsigned stride = 1U << (level - 1);
    bound.q = fmax(bound.q, __shfl_down_sync(0xffffffffU, bound.q, stride));
    bound.qd = fmax(bound.qd, __shfl_down_sync(0xffffffffU, bound.qd, stride));
    const double neighbor_minimum = __shfl_down_sync(0xffffffffU, bound.minimum, stride);
    bound.minimum = md.primitive_bounds ? fmin(bound.minimum, neighbor_minimum)
                                        : fmax(bound.minimum, neighbor_minimum);
    bound.density = fmax(bound.density, __shfl_down_sync(0xffffffffU, bound.density, stride));
    if (position % (2 * stride) == 0)
      output[layout.offsets[level] + position / (2 * stride)] = bound;
  }
}

/** Exact legacy leaf decision. qD hierarchy is never substituted for this
 * left-associated, upward-rounded contribution bound or the public minimum
 * product. A single global contracted-pair census apportions the allowance. */
__device__ inline bool md_j_engine_direction(MdJView md, std::uint32_t output, std::uint32_t input,
                                             double screening) {
  if (md.minimum_bounds[output] * md.minimum_bounds[input] < screening) return false;
  const auto density = md.density_bounds[input];
  const auto contribution =
      __dmul_ru(__dmul_ru(md.maximum_bounds[output], md.maximum_bounds[input]), density);
  const auto allowance = __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
  return density != 0.0 && contribution >= allowance;
}

/** The complete primitive recipe bypasses the public-AO mask explicitly.
 * Both signed density orientations already occur in density_bounds and E D. */
__device__ inline bool md_j_engine_primitive_direction(double source_norm,
                                                       double source_threshold) {
  return source_norm != 0.0 && source_norm >= source_threshold;
}

__device__ inline bool md_j_engine_tile_admitted(MdJView md, unsigned bra_angular,
                                                 unsigned ket_angular,
                                                 const MdJTileSchedule& schedule,
                                                 const MdJTileTask& task, double screening) {
  const auto& bra_layout = md.hierarchy_layout.buckets[bra_angular];
  const auto& ket_layout = md.hierarchy_layout.buckets[ket_angular];
  const auto bra =
      md.hierarchy[md.hierarchy_layout.bucket_offsets[bra_angular] +
                   bra_layout.offsets[md_j_hierarchy_level(schedule.bra_width)] + task.bra_tile];
  const auto ket =
      md.hierarchy[md.hierarchy_layout.bucket_offsets[ket_angular] +
                   ket_layout.offsets[md_j_hierarchy_level(schedule.ket_width)] + task.ket_tile];
  // Reference node minima are maxima of leaf minima: multiplication is monotone, so
  // even the same nearest-rounded comparison conservatively encloses leaves.
  if (!md.primitive_bounds && bra.minimum * ket.minimum < screening) return false;
  if (md.primitive_bounds)
    // A tile can contribute only if some source reaches some target threshold.
    // These max-source/min-threshold reductions conservatively enclose every
    // leaf comparison, including unknown bounds and a zero screening cutoff.
    return (ket.qd != 0.0 && ket.qd >= bra.minimum) || (bra.qd != 0.0 && bra.qd >= ket.minimum);
  const double allowance = __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
  return (ket.density != 0.0 && md_j_screen_density_bound(bra, ket) >= allowance) ||
         (bra.density != 0.0 && md_j_screen_density_bound(ket, bra) >= allowance);
}

/** Tile-local source metadata. The last double pads each record to nine
 * double words: adjacent bra lanes then visit distinct shared-memory banks
 * instead of the eight-word stride aliasing their coordinate/bound loads.
 * Transform identity and all original descriptors remain in the owner. */
struct MdJEngineCachedPair {
  double exponent;
  Vec3<double> product;
  std::size_t hermite_offset;
  std::uint32_t pair;
  std::int32_t system;
  double source_norm, source_threshold, bank_padding;
};

/** Stage the fields consumed by every source/component lane once per tile.
 * Original primitive/transform identity remains in its resident owner. */
__device__ inline MdJEngineCachedPair md_j_engine_cached_pair(MdJView md, unsigned angular,
                                                              std::size_t position) {
  const auto primitive = md.primitives[md.ordered_primitives[md.class_offsets[angular] + position]];
  const auto pair = md.pairs[primitive.pair];
  const auto bound = md.hierarchy[md.hierarchy_layout.bucket_offsets[angular] + position];
  return {primitive.exponent,
          primitive.product,
          primitive.hermite_offset,
          primitive.pair,
          pair.system,
          bound.qd,
          md.primitive_bounds ? bound.minimum : 0.0,
          0.0};
}

/** Bound registers to the blocks that the actual shared footprint can retain.
 * A 96-KiB scheduling envelope leaves headroom in the measured sm_120 SM's
 * 100-KiB shared pool. The eight-block ceiling avoids an aggressive register
 * target on tiny sources. Only sm_120 uses this qualified compiler hint; other
 * architectures retain the compiler default. It does not change admission. */
template <unsigned BraAngular, unsigned KetAngular>
__host__ __device__ constexpr unsigned md_j_engine_resident_blocks() {
  constexpr auto schedule = md_j_tile_schedule(BraAngular, KetAngular);
  constexpr auto bra = md_j_hermite_count(BraAngular), ket = md_j_hermite_count(KetAngular);
  constexpr auto bytes = sizeof(MdJEngineCachedPair) * (schedule.bra_width + schedule.ket_width) +
                         sizeof(double) * (schedule.bra_width * bra + 2 * schedule.ket_width * ket +
                                           md_j_source_lanes(BraAngular, KetAngular) *
                                               md_j_hermite_count(BraAngular + KetAngular)) +
                         kMdJSourceControlBytes;
  constexpr auto blocks = (96U << 10) / bytes;
  static_assert(blocks > 0, "MD-J shared footprint exceeds the scheduling envelope");
  return blocks < 8 ? blocks : 8;
}

/** Bounded reciprocal consumer with upstream-style quartet/component lanes.
 * Adjacent lanes evaluate different primitive sources; only larger angular
 * domains reserve additional lanes for output components. A bra accumulator
 * stays in registers across ket batches, while a shuffle gives each shared ket
 * output one writer. Thus no per-source shared floating-point atomic is needed.
 * All barriers remain outside divergent source/screening branches. */
template <unsigned BraAngular, unsigned KetAngular, bool CountWork>
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
__launch_bounds__(kMdJSourceThreads, (md_j_engine_resident_blocks<BraAngular, KetAngular>()))
#endif
    void md_j_engine_potential(MdJView md, MdJTileInventory inventory, std::size_t begin,
                               std::size_t end, double screening) {
  constexpr auto schedule = md_j_tile_schedule(BraAngular, KetAngular);
  constexpr auto bra_width = schedule.bra_width, ket_width = schedule.ket_width;
  constexpr auto bra_components = md_j_hermite_count(BraAngular);
  constexpr auto ket_components = md_j_hermite_count(KetAngular);
  constexpr auto radial_components = md_j_hermite_count(BraAngular + KetAngular);
  constexpr auto sources = md_j_source_lanes(BraAngular, KetAngular);
  constexpr auto components = md_j_component_lanes(BraAngular, KetAngular);
  constexpr auto bra_lanes = md_j_bra_lanes(sources), ket_lanes = sources / bra_lanes;
  constexpr auto bra_outputs = (bra_components + components - 1) / components;
  constexpr auto ket_outputs = (ket_components + components - 1) / components;
  static_assert(schedule.valid && sizeof(MdJEngineCachedPair) <= kMdJPrimitiveDescriptorBytes);
  static_assert(std::is_trivially_default_constructible_v<MdJEngineCachedPair> &&
                std::is_trivially_default_constructible_v<MdJTileTask>);
  static_assert(schedule.shared_bytes <= kMdJSharedCapacity);
  if (md.primitive_screen)
    inventory = md_j_tile_inventory(md.primitive_screen->active[BraAngular],
                                    md.primitive_screen->active[KetAngular], bra_width, ket_width,
                                    BraAngular == KetAngular);
  __shared__ MdJEngineCachedPair bra_cache[bra_width], ket_cache[ket_width];
  __shared__ double bra_density[bra_width * bra_components],
      ket_density[ket_width * ket_components];
  __shared__ double ket_potential[ket_width * ket_components];
  // Structure-of-arrays radial storage coalesces adjacent independent sources.
  // After the last ket batch this same allocation holds bra reduction partials.
  __shared__ double radial[sources * radial_components];
  __shared__ MdJTileTask task;
  __shared__ bool live, admitted;
  const auto source_lane = threadIdx.x % sources, component_lane = threadIdx.x / sources;
  const auto bra_lane = source_lane % bra_lanes, ket_lane = source_lane / bra_lanes;
  unsigned long long roots = 0, forwards = 0, reverses = 0, tiles = 0, screened = 0;
  while (true) {
    if (threadIdx.x == 0) {
      const auto index = atomicAdd(md.source_cursor, 1ULL);
      live = md_j_decode_tile(inventory, index, task);
      admitted =
          live && md_j_engine_tile_admitted(md, BraAngular, KetAngular, schedule, task, screening);
      if constexpr (CountWork) {
        tiles += live;
        screened += live && !admitted;
      }
    }
    __syncthreads();
    if (!live) break;
    if (!admitted) {
      // Every warp must consume this decision before thread zero overwrites
      // the shared task/control fields for the next cursor value.
      __syncthreads();
      continue;
    }
    for (unsigned slot = threadIdx.x; slot < task.bra_count; slot += blockDim.x)
      bra_cache[slot] = md_j_engine_cached_pair(md, BraAngular, task.bra_begin + slot);
    for (unsigned slot = threadIdx.x; slot < task.ket_count; slot += blockDim.x)
      ket_cache[slot] = md_j_engine_cached_pair(md, KetAngular, task.ket_begin + slot);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < bra_width * bra_components; i += blockDim.x) {
      const auto slot = i / bra_components;
      bra_density[i] =
          slot < task.bra_count && bra_cache[slot].system >= begin && bra_cache[slot].system < end
              ? md.density[bra_cache[slot].hermite_offset + i % bra_components]
              : 0.0;
    }
    for (unsigned i = threadIdx.x; i < ket_width * ket_components; i += blockDim.x) {
      const auto slot = i / ket_components;
      ket_density[i] =
          slot < task.ket_count && ket_cache[slot].system >= begin && ket_cache[slot].system < end
              ? md.density[ket_cache[slot].hermite_offset + i % ket_components]
              : 0.0;
      ket_potential[i] = 0.0;
    }
    __syncthreads();
    for (unsigned bra_begin = 0; bra_begin < task.bra_count; bra_begin += bra_lanes) {
      const auto bra_slot = bra_begin + bra_lane;
      double bra_values[bra_outputs]{};
      for (unsigned ket_begin = 0; ket_begin < task.ket_count; ket_begin += ket_lanes) {
        const auto ket_slot = ket_begin + ket_lane;
        bool forward = false, reverse = false;
        double prefactor = 0.0;
        if (md_j_tile_source_valid(inventory, task, bra_slot, ket_slot)) {
          const auto bra = bra_cache[bra_slot], ket = ket_cache[ket_slot];
          if (bra.system >= begin && bra.system < end && bra.system == ket.system) {
            forward = md.primitive_bounds
                          ? md_j_engine_primitive_direction(ket.source_norm, bra.source_threshold)
                          : md_j_engine_direction(md, bra.pair, ket.pair, screening);
            const bool distinct =
                BraAngular != KetAngular || task.bra_begin + bra_slot != task.ket_begin + ket_slot;
            reverse = distinct &&
                      (md.primitive_bounds
                           ? md_j_engine_primitive_direction(bra.source_norm, ket.source_threshold)
                           : md_j_engine_direction(md, ket.pair, bra.pair, screening));
            if (forward || reverse) {
              if (component_lane == 0) {
                const auto rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
                md_engine_radial<BraAngular + KetAngular, sources>(rho, bra.product, ket.product,
                                                                   radial + source_lane);
                if constexpr (CountWork) {
                  ++roots;
                  forwards += forward;
                  reverses += reverse;
                }
              }
              prefactor = 2.0 * pow(kPi, 2.5) /
                          (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
            }
          }
        }
        __syncthreads();
        if (forward)
          for (unsigned slot = 0; slot < bra_outputs; ++slot) {
            const auto component = slot * components + component_lane;
            if (component < bra_components)
              bra_values[slot] +=
                  prefactor *
                  md_engine_forward_component<BraAngular, KetAngular, sources>(
                      radial + source_lane, ket_density + ket_slot * ket_components, component);
          }
        // Every physical lane participates in each shuffle, including padded
        // tails and screened sources. Only bra_lane==0 writes the shared sum;
        // different ket/component groups own disjoint elements.
        for (unsigned slot = 0; slot < ket_outputs; ++slot) {
          const auto component = slot * components + component_lane;
          double value = 0.0;
          if (reverse && component < ket_components)
            value = prefactor *
                    md_engine_reverse_component<BraAngular, KetAngular, sources>(
                        radial + source_lane, bra_density + bra_slot * bra_components, component);
          for (unsigned stride = bra_lanes / 2; stride; stride /= 2)
            value += __shfl_down_sync(0xffffffffU, value, stride, bra_lanes);
          if (bra_lane == 0 && ket_slot < task.ket_count && component < ket_components)
            ket_potential[ket_slot * ket_components + component] += value;
        }
        // All component readers finish before source lanes reuse radial data.
        __syncthreads();
      }
      for (unsigned slot = 0; slot < bra_outputs; ++slot) {
        const auto component = slot * components + component_lane;
        if (component < bra_components)
          radial[component * sources + source_lane] = bra_values[slot];
      }
      __syncthreads();
      for (unsigned i = threadIdx.x; i < bra_lanes * bra_components; i += blockDim.x) {
        const auto component = i / bra_lanes, lane = i % bra_lanes;
        const auto output_slot = bra_begin + lane;
        if (output_slot < task.bra_count) {
          double value = 0.0;
          for (unsigned ket = 0; ket < ket_lanes; ++ket)
            value += radial[component * sources + ket * bra_lanes + lane];
          if (value != 0.0)
            atomicAdd(md.potential + bra_cache[output_slot].hermite_offset + component, value);
        }
      }
      __syncthreads();
    }
    for (unsigned i = threadIdx.x; i < task.ket_count * ket_components; i += blockDim.x) {
      const double value = ket_potential[i];
      if (value != 0.0)
        atomicAdd(md.potential + ket_cache[i / ket_components].hermite_offset + i % ket_components,
                  value);
    }
    __syncthreads();
  }
  if constexpr (CountWork) {
    if (threadIdx.x == 0) {
      atomicAdd(md.work_counts->engine_tiles + 5 * BraAngular + KetAngular, tiles);
      atomicAdd(md.work_counts->engine_screened_tiles + 5 * BraAngular + KetAngular, screened);
    }
    if (component_lane == 0) {
      constexpr auto forward_index = 5 * BraAngular + KetAngular;
      constexpr auto reverse_index = 5 * KetAngular + BraAngular;
      atomicAdd(md.work_counts->uniform_roots + forward_index, roots);
      atomicAdd(md.work_counts->uniform_directions + forward_index, forwards);
      atomicAdd(md.work_counts->uniform_directions + reverse_index, reverses);
      atomicAdd(md.work_counts->uniform_summands + forward_index,
                forwards * bra_components * ket_components);
      atomicAdd(md.work_counts->uniform_summands + reverse_index,
                reverses * bra_components * ket_components);
    }
  }
}

template <unsigned BraAngular, unsigned KetAngular>
void launch_md_j_engine_class(cudaStream_t stream, MdJView md, std::size_t begin, std::size_t end,
                              double screening) {
  constexpr auto schedule = md_j_tile_schedule(BraAngular, KetAngular);
  const auto inventory = md_j_tile_inventory(
      md.hierarchy_layout.buckets[BraAngular].count, md.hierarchy_layout.buckets[KetAngular].count,
      schedule.bra_width, schedule.ket_width, BraAngular == KetAngular);
  if (!inventory.valid) throw cudaErrorInvalidValue;
  if (!inventory.tasks) return;
  runtime::cuda_resource_check(
      cudaMemsetAsync(md.source_cursor, 0, sizeof(unsigned long long), stream));
  const auto workers = static_cast<unsigned>(std::min<std::uint64_t>(4096, inventory.tasks));
  if (md.work_counts)
    md_j_engine_potential<BraAngular, KetAngular, true>
        <<<workers, schedule.threads, 0, stream>>>(md, inventory, begin, end, screening);
  else
    md_j_engine_potential<BraAngular, KetAngular, false>
        <<<workers, schedule.threads, 0, stream>>>(md, inventory, begin, end, screening);
  runtime::cuda_resource_check(cudaPeekAtLastError());
}

void launch_md_j_engine(cudaStream_t stream, MdJView md, std::size_t begin, std::size_t end,
                        double screening) {
  for (unsigned angular = 0; angular < 5; ++angular) {
    const auto count = md.hierarchy_layout.buckets[angular].count;
    const auto aligned = (count + 31) / 32 * 32;
    if (!aligned) continue;
    md_j_engine_hierarchy<<<static_cast<unsigned>((aligned + 127) / 128), 128, 0, stream>>>(
        md, angular, begin, end, screening);
    runtime::cuda_resource_check(cudaPeekAtLastError());
  }
  launch_md_j_engine_class<0, 0>(stream, md, begin, end, screening);
  launch_md_j_engine_class<0, 1>(stream, md, begin, end, screening);
  launch_md_j_engine_class<0, 2>(stream, md, begin, end, screening);
  launch_md_j_engine_class<0, 3>(stream, md, begin, end, screening);
  launch_md_j_engine_class<0, 4>(stream, md, begin, end, screening);
  launch_md_j_engine_class<1, 1>(stream, md, begin, end, screening);
  launch_md_j_engine_class<1, 2>(stream, md, begin, end, screening);
  launch_md_j_engine_class<1, 3>(stream, md, begin, end, screening);
  launch_md_j_engine_class<1, 4>(stream, md, begin, end, screening);
  launch_md_j_engine_class<2, 2>(stream, md, begin, end, screening);
  launch_md_j_engine_class<2, 3>(stream, md, begin, end, screening);
  launch_md_j_engine_class<2, 4>(stream, md, begin, end, screening);
  launch_md_j_engine_class<3, 3>(stream, md, begin, end, screening);
  launch_md_j_engine_class<3, 4>(stream, md, begin, end, screening);
  launch_md_j_engine_class<4, 4>(stream, md, begin, end, screening);
}

}  // namespace

cudaError_t prepare_md_j_engine_layout(cudaStream_t stream, MdJView md) {
  for (unsigned angular = 0; angular < 5; ++angular) {
    const auto count = md.class_offsets[angular + 1] - md.class_offsets[angular];
    if (!count) continue;
    md_j_engine_pair_loc<<<static_cast<unsigned>((count + 127) / 128), 128, 0, stream>>>(md,
                                                                                         angular);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}
}  // namespace generativeqc::scf::cuda_execution
