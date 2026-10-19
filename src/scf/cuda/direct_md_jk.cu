#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/direct_md_j.hpp"
#include "scf/cuda/md_hermite_index.cuh"
#include "scf/cuda/md_source_tiles.cuh"

#include "generated_md_j_masked.cuh"

namespace generativeqc::scf::cuda_execution {
namespace {

constexpr unsigned kSourceCandidates = 128;
constexpr unsigned kSourceWorkers = 4096;

// Let direct_jk_guard translate and report a CUDA error while draining its stream.
inline void check_md_j_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw error;
}

/** One tiny contracted source uses eight lanes, not an otherwise idle block.
 * Higher orders retain a whole warp and their shared recurrence workspace. */
template <unsigned BraAngular, unsigned KetAngular>
struct SourceSchedule {
  static constexpr unsigned group_threads = BraAngular + KetAngular <= 2 ? 8 : 32;
  static constexpr unsigned block_threads = BraAngular + KetAngular <= 2 ? 128 : 32;
  static constexpr unsigned groups = block_threads / group_threads;
};

template <unsigned GroupThreads>
__device__ __forceinline__ void source_sync() {
  constexpr unsigned group_mask = 0xffffffffU >> (32 - GroupThreads);
  const unsigned group_begin = (threadIdx.x % 32) / GroupThreads * GroupThreads;
  __syncwarp(group_mask << group_begin);
}

/** Bound the public-AO component width for s/p/d pairs at one angular order. */
__host__ __device__ constexpr unsigned source_components(unsigned angular) {
  return angular == 0 ? 1 : angular == 1 ? 3 : angular == 2 ? 9 : angular == 3 ? 18 : 36;
}

template <unsigned BraAngular, unsigned KetAngular, bool Masked, bool Preprojected = false>
__host__ __device__ constexpr unsigned source_workspace_width() {
  constexpr unsigned incumbent =
      CoulombAuxiliary<double, BraAngular + KetAngular>::kStateCount +
      source_components(BraAngular) * md_j_hermite_count(KetAngular) +
      source_components(BraAngular) * source_components(KetAngular) + 1;
  if constexpr (Masked) {
    constexpr unsigned masked =
        (sizeof(MdMaskedWorkspace<BraAngular, KetAngular, Preprojected>) + sizeof(double) - 1) / sizeof(double);
    return incumbent > masked ? incumbent : masked;
  }
  return incumbent;
}

/** Two small E/R contractions reuse one existing Coulomb recurrence per
 * primitive product, instead of rebuilding it for every AO quartet/orientation.
 * The scratch is block-local and independent of molecular AO count. */
template <unsigned BraAngular, unsigned KetAngular, bool Masked, bool CountWork,
          bool Preprojected = false,
          unsigned GroupThreads = SourceSchedule<BraAngular, KetAngular>::group_threads>
__device__ __forceinline__ void contract_source_task(
    DeviceBatch batch, MdJView md, MdSourceTask task, bool want_j, bool want_k, bool unrestricted,
    double screening, const double* bounds, const double* density, const double* beta,
    double* coulomb, double* alpha_exchange, double* beta_exchange, double* workspace) {
  if constexpr (Masked) {
    if (md_masked_contract_source<BraAngular, KetAngular, CountWork, Preprojected>(
            batch, md, task, unrestricted, screening, bounds, density, beta, coulomb, workspace))
      return;
  }
  constexpr auto bra_hermites = md_j_hermite_count(BraAngular);
  constexpr auto ket_hermites = md_j_hermite_count(KetAngular);
  constexpr auto group_threads = GroupThreads;
  const unsigned source_rank = threadIdx.x % group_threads;
  const auto bra_pair = md.pairs[task.bra], ket_pair = md.pairs[task.ket];
  const unsigned bra_components = bra_pair.first_count * bra_pair.second_count;
  const unsigned ket_components = ket_pair.first_count * ket_pair.second_count;
  auto& auxiliary =
      *reinterpret_cast<CoulombAuxiliary<double, BraAngular + KetAngular>*>(workspace);
  double* intermediate = workspace + CoulombAuxiliary<double, BraAngular + KetAngular>::kStateCount;
  double* integrals = intermediate + bra_components * ket_hermites;
  double* prefactor = integrals + bra_components * ket_components;
  for (unsigned component = source_rank; component < bra_components * ket_components;
       component += group_threads)
    integrals[component] = 0.0;
  source_sync<group_threads>();
  if constexpr (BraAngular + KetAngular <= 2) {
    // Contracted s/p sources have tiny component vectors but many primitive
    // products. Parallelize those products instead of serializing three block
    // barriers per product around a single R-producing lane.
    double partial[9]{};
    const auto ket_count = ket_pair.primitive_end - ket_pair.primitive_begin;
    const auto product_count =
        static_cast<std::size_t>(bra_pair.primitive_end - bra_pair.primitive_begin) * ket_count;
    for (auto product_index = static_cast<std::size_t>(source_rank); product_index < product_count;
         product_index += group_threads) {
      const auto bra = md.primitives[bra_pair.primitive_begin + product_index / ket_count];
      const auto ket = md.primitives[ket_pair.primitive_begin + product_index % ket_count];
      CoulombAuxiliary<double, BraAngular + KetAngular> local_auxiliary;
      const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
      fill_coulomb<BraAngular + KetAngular>(rho, bra.product, ket.product, local_auxiliary);
      const double factor =
          2.0 * pow(kPi, 2.5) / (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
      for (unsigned component = 0; component < bra_components * ket_components; ++component) {
        const auto bra_component = component / ket_components,
                   ket_component = component % ket_components;
        double value = 0.0;
        for (unsigned ket_derivative = 0; ket_derivative < ket_hermites; ++ket_derivative) {
          const auto ket_angular = md_hermite_angular(KetAngular, ket_derivative);
          const double ket_coefficient =
              md_j_transform(md, ket, ket_pair, ket_component, ket_derivative);
          if (ket_coefficient == 0.0) continue;
          double projected = 0.0;
          for (unsigned bra_derivative = 0; bra_derivative < bra_hermites; ++bra_derivative) {
            const double coefficient =
                md_j_transform(md, bra, bra_pair, bra_component, bra_derivative);
            if (coefficient == 0.0) continue;
            const auto bra_angular = md_hermite_angular(BraAngular, bra_derivative);
            projected += coefficient * local_auxiliary.at(0, bra_angular.x + ket_angular.x,
                                                          bra_angular.y + ket_angular.y,
                                                          bra_angular.z + ket_angular.z);
          }
          const double sign = (ket_angular.x + ket_angular.y + ket_angular.z) & 1U ? -1.0 : 1.0;
          value += sign * ket_coefficient * projected;
        }
        partial[component] += factor * value;
      }
    }
    for (unsigned component = 0; component < bra_components * ket_components; ++component) {
      double value = partial[component];
      constexpr unsigned group_mask = 0xffffffffU >> (32 - group_threads);
      const unsigned group_begin = (threadIdx.x % 32) / group_threads * group_threads;
      for (unsigned stride = group_threads / 2; stride; stride /= 2)
        value += __shfl_down_sync(group_mask << group_begin, value, stride, group_threads);
      if (source_rank == 0) integrals[component] = value;
    }
    source_sync<group_threads>();
  } else {
    for (auto bra_index = bra_pair.primitive_begin; bra_index < bra_pair.primitive_end;
         ++bra_index) {
      const auto bra = md.primitives[bra_index];
      for (auto ket_index = ket_pair.primitive_begin; ket_index < ket_pair.primitive_end;
           ++ket_index) {
        const auto ket = md.primitives[ket_index];
        if (source_rank == 0) {
          const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
          fill_coulomb<BraAngular + KetAngular>(rho, bra.product, ket.product, auxiliary);
          *prefactor = 2.0 * pow(kPi, 2.5) /
                       (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
        }
        source_sync<group_threads>();
        for (unsigned element = source_rank; element < bra_components * ket_hermites;
             element += group_threads) {
          const auto component = element / ket_hermites, ket_derivative = element % ket_hermites;
          const auto ket_angular = md_hermite_angular(KetAngular, ket_derivative);
          double value = 0.0;
          for (unsigned bra_derivative = 0; bra_derivative < bra_hermites; ++bra_derivative) {
            const double coefficient = md_j_transform(md, bra, bra_pair, component, bra_derivative);
            if (coefficient == 0.0) continue;
            const auto bra_angular = md_hermite_angular(BraAngular, bra_derivative);
            value += coefficient * auxiliary.at(0, bra_angular.x + ket_angular.x,
                                                bra_angular.y + ket_angular.y,
                                                bra_angular.z + ket_angular.z);
          }
          const double sign = (ket_angular.x + ket_angular.y + ket_angular.z) & 1U ? -1.0 : 1.0;
          intermediate[element] = sign * *prefactor * value;
        }
        source_sync<group_threads>();
        for (unsigned component = source_rank; component < bra_components * ket_components;
             component += group_threads) {
          const auto bra_component = component / ket_components,
                     ket_component = component % ket_components;
          double value = 0.0;
          for (unsigned derivative = 0; derivative < ket_hermites; ++derivative)
            value += intermediate[bra_component * ket_hermites + derivative] *
                     md_j_transform(md, ket, ket_pair, ket_component, derivative);
          integrals[component] += value;
        }
        source_sync<group_threads>();
      }
    }
  }
  const std::size_t matrix = batch.nbf * batch.nbf;
  const std::size_t offset = static_cast<std::size_t>(bra_pair.system) * matrix;
  const bool contract_j =
      want_j && (want_k || md.minimum_bounds[task.bra] * md.minimum_bounds[task.ket] < screening);
  for (unsigned component = source_rank; component < bra_components * ket_components;
       component += group_threads) {
    const auto bra_component = component / ket_components,
               ket_component = component % ket_components;
    const unsigned first = bra_pair.first_ao + bra_component / bra_pair.second_count;
    const unsigned second = bra_pair.second_ao + bra_component % bra_pair.second_count;
    const unsigned third = ket_pair.first_ao + ket_component / ket_pair.second_count;
    const unsigned fourth = ket_pair.second_ao + ket_component % ket_pair.second_count;
    if (first < second || third < fourth ||
        (task.bra == task.ket && first * batch.nbf + second < third * batch.nbf + fourth))
      continue;
    const double integral = integrals[component];
    for (unsigned exchange_pairs = 0; exchange_pairs < 2; ++exchange_pairs) {
      if (exchange_pairs && first == third && second == fourth) continue;
      for (unsigned swap_bra = 0; swap_bra < 2; ++swap_bra) {
        if (swap_bra && first == second) continue;
        for (unsigned swap_ket = 0; swap_ket < 2; ++swap_ket) {
          if (swap_ket && third == fourth) continue;
          const auto bra_first = swap_bra ? second : first, bra_second = swap_bra ? first : second;
          const auto ket_first = swap_ket ? fourth : third, ket_second = swap_ket ? third : fourth;
          const auto output_first = exchange_pairs ? ket_first : bra_first;
          const auto output_second = exchange_pairs ? ket_second : bra_second;
          const auto density_first = exchange_pairs ? bra_first : ket_first;
          const auto density_second = exchange_pairs ? bra_second : ket_second;
          if (bounds[offset + output_first * batch.nbf + output_second] *
                  bounds[offset + density_first * batch.nbf + density_second] <
              screening)
            continue;
          if (contract_j) {
            const auto index = offset + density_first * batch.nbf + density_second;
            atomicAdd(coulomb + offset + output_first * batch.nbf + output_second,
                      integral * (density[index] + (unrestricted ? beta[index] : 0.0)));
          }
          if (want_k) {
            const auto index = offset + output_second * batch.nbf + density_second;
            const auto output = offset + output_first * batch.nbf + density_first;
            atomicAdd(alpha_exchange + output, integral * density[index]);
            if (unrestricted) atomicAdd(beta_exchange + output, integral * beta[index]);
          }
        }
      }
    }
  }
  source_sync<group_threads>();
}

template <bool Coarse>
struct MdMaskedTileFallback {
  template <bool CountWork>
  __device__ static bool tile_admitted(MdJView md, const unsigned* pairs, unsigned candidates) {
    if constexpr (!Coarse) return true;
    bool zero = true;
    for (unsigned index = 0; index < 6; ++index)
      if (pairs[index] != ~0U && md.density_bounds[pairs[index]] != 0.0) zero = false;
    if constexpr (CountWork) {
      atomicAdd(&md.work_counts->residual_tiles[11], 1ULL);
      if (zero) {
        atomicAdd(&md.work_counts->residual_screened_tiles[11], 1ULL);
        atomicAdd(&md.work_counts->residual_screened_candidates[11],
                  static_cast<unsigned long long>(candidates));
      }
    }
    return !zero;
  }

  __device__ static bool admitted(MdJView md, MdSourceTask task, std::size_t system_begin,
                                  std::size_t system_end, double screening) {
    const auto bra = task.bra, ket = task.ket;
    const auto system = md.pairs[bra].system;
    if (system < system_begin || system >= system_end || system != md.pairs[ket].system ||
        !(md.maximum_bounds[bra] * md.maximum_bounds[ket] >= screening) ||
        !(md.minimum_bounds[bra] * md.minimum_bounds[ket] < screening)) return false;
    const double allowance = __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
    const double bound = __dmul_ru(md.maximum_bounds[bra], md.maximum_bounds[ket]);
    return !((md.density_bounds[bra] == 0.0 && md.density_bounds[ket] == 0.0) ||
        (__dmul_ru(bound, md.density_bounds[bra]) < allowance &&
         __dmul_ru(bound, md.density_bounds[ket]) < allowance));
  }

  __device__ void operator()(DeviceBatch batch, MdJView md, MdSourceTask task,
                             bool unrestricted, double screening, const double* bounds,
                             const double* density, const double* beta, double* coulomb,
                             double* workspace) const {
    contract_source_task<2, 1, false, false>(batch, md, task, true, false, unrestricted,
        screening, bounds, density, beta, coulomb, nullptr, nullptr, workspace);
  }
};

static_assert(sizeof(MdMaskedTileWorker) >= source_workspace_width<2, 1, false>() * sizeof(double));

/** Each resident worker compacts a tiny candidate tile into block-local tasks.
 * The 64-bit cursor replaces per-page host launches; neither the task inventory
 * nor the scratch grows with the molecular quartet count. Angular dispatch is
 * a launch boundary, not a device switch: low-order tasks cannot inherit the
 * high-order recurrence's register footprint and stack frame. Default J-only
 * tiles reject whole batches before individual tests without changing the
 * surviving task predicates or primitive contraction. */
template <unsigned BraAngular, unsigned KetAngular, bool CountWork, bool Tiled, bool Masked,
          bool Preprojected = false>
__global__ void consume_source_tasks(DeviceBatch batch, MdJView md, std::size_t system_begin,
                                     std::size_t system_end, bool want_j, bool want_k,
                                     bool unrestricted, double screening, const double* bounds,
                                     const double* density, const double* beta, double* coulomb,
                                     double* alpha_exchange, double* beta_exchange) {
  extern __shared__ double workspace[];
  __shared__ MdSourceTask tasks[kSourceCandidates];
  __shared__ unsigned task_count;
  __shared__ unsigned long long candidate_begin;
  __shared__ MdSourceTile tile;
  __shared__ std::uint32_t tile_pairs[2 * kMdSourceTileWidth];
  __shared__ bool tile_rejected;
  const auto bra_begin = md.active_pair_offsets[BraAngular];
  const auto ket_begin = md.active_pair_offsets[KetAngular];
  const auto bra_count = md.active_pair_offsets[BraAngular + 1] - bra_begin;
  const auto ket_count = md.active_pair_offsets[KetAngular + 1] - ket_begin;
  const auto candidates = BraAngular == KetAngular
                              ? static_cast<unsigned long long>(bra_count) * (bra_count + 1ULL) / 2
                              : static_cast<unsigned long long>(bra_count) * ket_count;
  const auto tiles = md_source_tile_count(bra_count, ket_count, BraAngular == KetAngular);
  const double density_allowance =
      __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
  unsigned long long tested = 0, admitted = 0, roots = 0;
  unsigned long long visited_tiles = 0, screened_tiles = 0, screened_candidates = 0;
  while (true) {
    if (threadIdx.x == 0) {
      candidate_begin = atomicAdd(
          md.source_cursor, Tiled ? 1ULL : static_cast<unsigned long long>(kSourceCandidates));
      if constexpr (Tiled) {
        if (candidate_begin < tiles)
          tile = md_source_tile(candidate_begin, ket_count, BraAngular == KetAngular);
      }
      task_count = 0;
    }
    __syncthreads();
    if (candidate_begin >= (Tiled ? tiles : candidates)) {
      if constexpr (CountWork) {
        for (unsigned stride = 16; stride; stride /= 2) {
          tested += __shfl_down_sync(0xffffffff, tested, stride);
          admitted += __shfl_down_sync(0xffffffff, admitted, stride);
          roots += __shfl_down_sync(0xffffffff, roots, stride);
          visited_tiles += __shfl_down_sync(0xffffffff, visited_tiles, stride);
          screened_tiles += __shfl_down_sync(0xffffffff, screened_tiles, stride);
          screened_candidates += __shfl_down_sync(0xffffffff, screened_candidates, stride);
        }
        if (threadIdx.x % 32 == 0) {
          constexpr auto angular = 5 * BraAngular + KetAngular;
          atomicAdd(&md.work_counts->residual_candidates[angular], tested);
          atomicAdd(&md.work_counts->residual_tasks[angular], admitted);
          atomicAdd(&md.work_counts->residual_roots[angular], roots);
          atomicAdd(&md.work_counts->residual_tiles[angular], visited_tiles);
          atomicAdd(&md.work_counts->residual_screened_tiles[angular], screened_tiles);
          atomicAdd(&md.work_counts->residual_screened_candidates[angular], screened_candidates);
        }
      }
      return;
    }
    if constexpr (Tiled) {
      if (threadIdx.x < 32) {
        const auto lane = threadIdx.x;
        const bool bra_side = lane < kMdSourceTileWidth;
        const auto local_index = bra_side ? tile.bra + lane : tile.ket + lane - kMdSourceTileWidth;
        const bool valid =
            lane < 2 * kMdSourceTileWidth && local_index < (bra_side ? bra_count : ket_count);
        double minimum = INFINITY, maximum = 0.0, density_maximum = 0.0;
        if (valid) {
          const auto pair = md.active_pairs[(bra_side ? bra_begin : ket_begin) + local_index];
          tile_pairs[lane] = pair;
          minimum = md.minimum_bounds[pair];
          maximum = md.maximum_bounds[pair];
          density_maximum = md.density_bounds[pair];
        }
        for (unsigned stride = kMdSourceTileWidth / 2; stride; stride /= 2) {
          minimum =
              fmin(minimum, __shfl_down_sync(0xffffffff, minimum, stride, kMdSourceTileWidth));
          maximum =
              fmax(maximum, __shfl_down_sync(0xffffffff, maximum, stride, kMdSourceTileWidth));
          density_maximum = fmax(density_maximum, __shfl_down_sync(0xffffffff, density_maximum,
                                                                   stride, kMdSourceTileWidth));
        }
        const double ket_minimum = __shfl_sync(0xffffffff, minimum, kMdSourceTileWidth);
        const double ket_maximum = __shfl_sync(0xffffffff, maximum, kMdSourceTileWidth);
        const double ket_density = __shfl_sync(0xffffffff, density_maximum, kMdSourceTileWidth);
        if (lane == 0) {
          const double bound = __dmul_ru(maximum, ket_maximum);
          tile_rejected = md_source_tile_rejected(
              bound, __dmul_rd(minimum, ket_minimum), __dmul_ru(bound, density_maximum),
              __dmul_ru(bound, ket_density), density_maximum == 0.0 && ket_density == 0.0,
              screening, density_allowance);
          if constexpr (CountWork) {
            ++visited_tiles;
            if (tile_rejected) {
              ++screened_tiles;
              const auto bra_size = min(kMdSourceTileWidth, bra_count - tile.bra);
              const auto ket_size = min(kMdSourceTileWidth, ket_count - tile.ket);
              screened_candidates += BraAngular == KetAngular && tile.bra == tile.ket
                                         ? bra_size * (bra_size + 1) / 2
                                         : bra_size * ket_size;
            }
          }
        }
      }
      __syncthreads();
      if (tile_rejected) continue;
    }
    const auto candidate_end = Tiled ? kMdSourceTileWidth * kMdSourceTileWidth
                                     : min(candidates, candidate_begin + kSourceCandidates);
    for (auto candidate = (Tiled ? 0ULL : candidate_begin) + threadIdx.x; candidate < candidate_end;
         candidate += blockDim.x) {
      unsigned long long bra_index{}, ket_index{};
      if constexpr (Tiled) {
        bra_index = tile.bra + candidate / kMdSourceTileWidth;
        ket_index = tile.ket + candidate % kMdSourceTileWidth;
        if (bra_index >= bra_count || ket_index >= ket_count ||
            (BraAngular == KetAngular && bra_index < ket_index))
          continue;
      } else if constexpr (BraAngular == KetAngular) {
        bra_index = static_cast<unsigned long long>((sqrt(8.0 * candidate + 1.0) - 1.0) * 0.5);
        while (bra_index * (bra_index + 1ULL) / 2 > candidate) --bra_index;
        while ((bra_index + 1ULL) * (bra_index + 2ULL) / 2 <= candidate) ++bra_index;
        ket_index = candidate - bra_index * (bra_index + 1ULL) / 2;
      } else {
        bra_index = candidate / ket_count;
        ket_index = candidate % ket_count;
      }
      if constexpr (CountWork) ++tested;
      {
        const auto bra = Tiled ? tile_pairs[candidate / kMdSourceTileWidth]
                               : md.active_pairs[bra_begin + bra_index];
        const auto ket = Tiled ? tile_pairs[kMdSourceTileWidth + candidate % kMdSourceTileWidth]
                               : md.active_pairs[ket_begin + ket_index];
        const auto system = md.pairs[bra].system;
        if (system >= system_begin && system < system_end && system == md.pairs[ket].system &&
            md.maximum_bounds[bra] * md.maximum_bounds[ket] >= screening &&
            (want_k || md.minimum_bounds[bra] * md.minimum_bounds[ket] < screening)) {
          if (!want_k) {
            const double bound = __dmul_ru(md.maximum_bounds[bra], md.maximum_bounds[ket]);
            if ((md.density_bounds[bra] == 0.0 && md.density_bounds[ket] == 0.0) ||
                (__dmul_ru(bound, md.density_bounds[bra]) < density_allowance &&
                 __dmul_ru(bound, md.density_bounds[ket]) < density_allowance))
              continue;
          }
          tasks[atomicAdd(&task_count, 1U)] = {bra, ket};
          if constexpr (CountWork) ++admitted;
        }
      }
    }
    __syncthreads();
    constexpr auto group_threads = SourceSchedule<BraAngular, KetAngular>::group_threads;
    constexpr auto groups = SourceSchedule<BraAngular, KetAngular>::groups;
    constexpr auto workspace_width = source_workspace_width<BraAngular, KetAngular, Masked, Preprojected>();
    const unsigned group_index = threadIdx.x / group_threads;
    for (unsigned task_index = group_index; task_index < task_count; task_index += groups) {
      const auto task = tasks[task_index];
      contract_source_task<BraAngular, KetAngular, Masked, CountWork, Preprojected>(
          batch, md, task, want_j, want_k, unrestricted, screening, bounds, density, beta, coulomb,
          alpha_exchange, beta_exchange, workspace + group_index * workspace_width);
      if constexpr (CountWork) {
        if (threadIdx.x % group_threads == 0)
          roots += static_cast<unsigned long long>(md.pairs[task.bra].primitive_end -
                                                   md.pairs[task.bra].primitive_begin) *
                   (md.pairs[task.ket].primitive_end - md.pairs[task.ket].primitive_begin);
      }
    }
    __syncthreads();
  }
}

template <unsigned BraAngular, unsigned KetAngular, bool Tiled, bool Masked,
          bool Preprojected = false>
void launch_source_kernel(cudaStream_t stream, DeviceBatch batch, MdJView md,
                         std::size_t system_begin, std::size_t system_count, bool want_j,
                         bool want_k, bool unrestricted, double screening, const double* bounds,
                         const double* density, const double* beta, double* coulomb,
                         double* alpha_exchange, double* beta_exchange) {
  if (md.active_pair_offsets[BraAngular + 1] == md.active_pair_offsets[BraAngular] ||
      md.active_pair_offsets[KetAngular + 1] == md.active_pair_offsets[KetAngular])
    return;
  constexpr auto shared_bytes =
      source_workspace_width<BraAngular, KetAngular, Masked, Preprojected>() *
      sizeof(double) * SourceSchedule<BraAngular, KetAngular>::groups;
  const auto bra_count = static_cast<unsigned long long>(md.active_pair_offsets[BraAngular + 1] -
                                                         md.active_pair_offsets[BraAngular]);
  const auto ket_count = static_cast<unsigned long long>(md.active_pair_offsets[KetAngular + 1] -
                                                         md.active_pair_offsets[KetAngular]);
  const auto candidates =
      BraAngular == KetAngular ? bra_count * (bra_count + 1ULL) / 2 : bra_count * ket_count;
  if (!candidates) return;
  const unsigned workers = static_cast<unsigned>(std::min<unsigned long long>(
      kSourceWorkers, Tiled ? md_source_tile_count(bra_count, ket_count, BraAngular == KetAngular)
                            : (candidates + kSourceCandidates - 1) / kSourceCandidates));
  check_md_j_cuda(cudaMemsetAsync(md.source_cursor, 0, sizeof(unsigned long long), stream));
  if (md.work_counts && !want_k)
    consume_source_tasks<BraAngular, KetAngular, true, Tiled, Masked, Preprojected>
        <<<workers, SourceSchedule<BraAngular, KetAngular>::block_threads, shared_bytes, stream>>>(
            batch, md, system_begin, system_begin + system_count, want_j, want_k, unrestricted,
            screening, bounds, density, beta, coulomb, alpha_exchange, beta_exchange);
  else
    consume_source_tasks<BraAngular, KetAngular, false, Tiled, Masked, Preprojected>
        <<<workers, SourceSchedule<BraAngular, KetAngular>::block_threads, shared_bytes, stream>>>(
            batch, md, system_begin, system_begin + system_count, want_j, want_k, unrestricted,
            screening, bounds, density, beta, coulomb, alpha_exchange, beta_exchange);
  check_md_j_cuda(cudaGetLastError());
}

template <bool Coarse = true, bool Primitive = false>
void launch_masked_tile_source(cudaStream_t stream, DeviceBatch batch, MdJView md,
                               std::size_t system_begin, std::size_t system_count,
                               bool unrestricted, double screening, const double* bounds,
                               const double* density, const double* beta, double* coulomb) {
  const auto bra_count = md.active_pair_offsets[3] - md.active_pair_offsets[2];
  const auto ket_count = md.active_pair_offsets[2] - md.active_pair_offsets[1];
  const auto tiles = (bra_count + 1ULL) / 2 * ((ket_count + 3ULL) / 4);
  if (!tiles) return;
  const unsigned workers = static_cast<unsigned>(std::min<unsigned long long>(kSourceWorkers, tiles));
  check_md_j_cuda(cudaMemsetAsync(md.source_cursor, 0, sizeof(unsigned long long), stream));
  if (md.work_counts)
    md_masked_tiles<true, MdMaskedTileFallback<Coarse>, Primitive><<<workers, 128, 0, stream>>>(
        batch, md, system_begin, system_begin + system_count, unrestricted, screening,
        bounds, density, beta, coulomb);
  else
    md_masked_tiles<false, MdMaskedTileFallback<Coarse>, Primitive><<<workers, 128, 0, stream>>>(
        batch, md, system_begin, system_begin + system_count, unrestricted, screening,
        bounds, density, beta, coulomb);
  check_md_j_cuda(cudaGetLastError());
}

/** A scalar fallback owns private scratch and names only its own sync lane. */
template <unsigned KetAngular>
struct MdMaskedPackedFallback : MdMaskedTileFallback<false> {
  __device__ static void scalar(DeviceBatch batch, MdJView md, MdSourceTask task,
                                bool unrestricted, double screening, const double* bounds,
                                const double* density, const double* beta, double* coulomb) {
    double scratch[source_workspace_width<2, KetAngular, false>()];
    contract_source_task<2, KetAngular, false, false, false, 1>(batch, md, task, true, false,
        unrestricted, screening, bounds, density, beta, coulomb, nullptr, nullptr, scratch);
  }
};

template <unsigned KetAngular>
void launch_masked_packed_source(cudaStream_t stream, DeviceBatch batch, MdJView md,
                                 std::size_t system_begin, std::size_t system_count,
                                 bool unrestricted, double screening, const double* bounds,
                                 const double* density, const double* beta, double* coulomb) {
  const auto bra_count = md.active_pair_offsets[3] - md.active_pair_offsets[2];
  const auto ket_count = md.active_pair_offsets[KetAngular + 1] - md.active_pair_offsets[KetAngular];
  const auto candidates = KetAngular == 2 ? static_cast<unsigned long long>(bra_count) * (bra_count + 1ULL) / 2
      : static_cast<unsigned long long>(bra_count) * ket_count;
  if (!candidates) return;
  const unsigned workers = static_cast<unsigned>(std::min<unsigned long long>(kSourceWorkers, (candidates + 1023) / 1024));
  check_md_j_cuda(cudaMemsetAsync(md.source_cursor, 0, sizeof(unsigned long long), stream));
  if (md.work_counts)
    md_masked_packed_sources<KetAngular, true, MdMaskedPackedFallback<KetAngular>><<<workers, 128, 0, stream>>>(
        batch, md, system_begin, system_begin + system_count, unrestricted, screening, bounds, density, beta, coulomb);
  else
    md_masked_packed_sources<KetAngular, false, MdMaskedPackedFallback<KetAngular>><<<workers, 128, 0, stream>>>(
        batch, md, system_begin, system_begin + system_count, unrestricted, screening, bounds, density, beta, coulomb);
  check_md_j_cuda(cudaGetLastError());
}

template <unsigned BraAngular, unsigned KetAngular, bool Tiled>
void launch_source_class(cudaStream_t stream, DeviceBatch batch, MdJView md,
                         std::size_t system_begin, std::size_t system_count, bool want_j,
                         bool want_k, bool unrestricted, double screening, const double* bounds,
                         const double* density, const double* beta, double* coulomb,
                         double* alpha_exchange, double* beta_exchange) {
  if constexpr (BraAngular == 2 && KetAngular == 1) {
    if (md.masked_tiles == 2 && want_j && !want_k) {
      launch_masked_packed_source<KetAngular>(stream, batch, md, system_begin, system_count,
          unrestricted, screening, bounds, density, beta, coulomb);
      return;
    }
  }
  if constexpr (BraAngular == 2 && KetAngular == 1) {
    if (md.masked_tiles == 1 && want_j && !want_k) {
      launch_masked_tile_source<Tiled>(stream, batch, md, system_begin, system_count, unrestricted,
          screening, bounds, density, beta, coulomb);
      return;
    }
    if (md.masked_pairs && want_j && !want_k) {
      launch_source_kernel<BraAngular, KetAngular, Tiled, true, true>(
          stream, batch, md, system_begin, system_count, want_j, want_k, unrestricted, screening,
          bounds, density, beta, coulomb, alpha_exchange, beta_exchange);
      return;
    }
  }
  if constexpr (BraAngular == 2 && (KetAngular == 1 || KetAngular == 2)) {
    if (md.masked_residual && want_j && !want_k) {
      launch_source_kernel<BraAngular, KetAngular, Tiled, true>(
          stream, batch, md, system_begin, system_count, want_j, want_k, unrestricted, screening,
          bounds, density, beta, coulomb, alpha_exchange, beta_exchange);
      return;
    }
  }
  launch_source_kernel<BraAngular, KetAngular, Tiled, false>(
      stream, batch, md, system_begin, system_count, want_j, want_k, unrestricted, screening,
      bounds, density, beta, coulomb, alpha_exchange, beta_exchange);
}

}  // namespace

cudaError_t prepare_md_j_masked(cudaStream_t stream, DeviceBatch batch, MdJView md,
                                const double* bounds) {
  md_masked_prepare_pairs<<<static_cast<unsigned>((md.pair_count + 127) / 128), 128, 0, stream>>>(
      batch, md, bounds);
  return cudaGetLastError();
}

void launch_md_source_jk(cudaStream_t stream, DeviceBatch batch, MdJView md,
                         std::size_t system_begin, std::size_t system_count, bool want_j,
                         bool want_k, bool unrestricted, double screening, const double* bounds,
                         const double* density, const double* beta, double* coulomb,
                         double* alpha_exchange, double* beta_exchange) {
  const auto begin = system_begin * batch.nbf * batch.nbf;
  const auto elements = system_count * batch.nbf * batch.nbf;
  if (want_j)
    check_md_j_cuda(cudaMemsetAsync(coulomb + begin, 0, elements * sizeof(double), stream));
  if (want_k) {
    check_md_j_cuda(cudaMemsetAsync(alpha_exchange + begin, 0, elements * sizeof(double), stream));
    if (unrestricted)
      check_md_j_cuda(cudaMemsetAsync(beta_exchange + begin, 0, elements * sizeof(double), stream));
  }
  if (!want_k && !md.residual_candidate_count) return;
  if (want_j && !want_k && md.masked_pairs) {
    md_masked_refresh<<<static_cast<unsigned>(md.pair_count), 128, 0, stream>>>(
        batch, md, system_begin, system_begin + system_count, unrestricted, screening, bounds, density, beta);
    check_md_j_cuda(cudaGetLastError());
  }
#define MD_SOURCE_CLASS(bra, ket)                                                                \
  if (md.residual_tiled && !want_k)                                                              \
    launch_source_class<bra, ket, true>(stream, batch, md, system_begin, system_count, want_j,   \
                                        want_k, unrestricted, screening, bounds, density, beta,  \
                                        coulomb, alpha_exchange, beta_exchange);                 \
  else                                                                                           \
    launch_source_class<bra, ket, false>(stream, batch, md, system_begin, system_count, want_j,  \
                                         want_k, unrestricted, screening, bounds, density, beta, \
                                         coulomb, alpha_exchange, beta_exchange)
  MD_SOURCE_CLASS(0, 0);
  MD_SOURCE_CLASS(1, 0);
  MD_SOURCE_CLASS(1, 1);
  MD_SOURCE_CLASS(2, 0);
  MD_SOURCE_CLASS(2, 1);
  MD_SOURCE_CLASS(2, 2);
  MD_SOURCE_CLASS(3, 0);
  MD_SOURCE_CLASS(3, 1);
  MD_SOURCE_CLASS(3, 2);
  MD_SOURCE_CLASS(3, 3);
  MD_SOURCE_CLASS(4, 0);
  MD_SOURCE_CLASS(4, 1);
  MD_SOURCE_CLASS(4, 2);
  MD_SOURCE_CLASS(4, 3);
  MD_SOURCE_CLASS(4, 4);
#undef MD_SOURCE_CLASS
}

}  // namespace generativeqc::scf::cuda_execution
