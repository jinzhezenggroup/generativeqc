#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/direct_md_j.hpp"
#include "scf/cuda/md_hermite_index.cuh"

namespace generativeqc::scf::cuda_execution {
namespace {

constexpr unsigned kSourceCandidates = 128;
constexpr unsigned kSourceWorkers = 4096;

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

/** Two small E/R contractions reuse one existing Coulomb recurrence per
 * primitive product, instead of rebuilding it for every AO quartet/orientation.
 * The scratch is block-local and independent of molecular AO count. */
template <unsigned BraAngular, unsigned KetAngular>
__device__ __forceinline__ void contract_source_task(
    DeviceBatch batch, MdJView md, MdSourceTask task, bool want_j, bool want_k, bool unrestricted,
    double screening, const double* bounds, const double* density, const double* beta,
    double* coulomb, double* alpha_exchange, double* beta_exchange, double* workspace) {
  constexpr auto bra_hermites = md_j_hermite_count(BraAngular);
  constexpr auto ket_hermites = md_j_hermite_count(KetAngular);
  constexpr auto group_threads = SourceSchedule<BraAngular, KetAngular>::group_threads;
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
              md.transforms[ket.transform_offset + ket_component * ket_hermites + ket_derivative];
          if (ket_coefficient == 0.0) continue;
          double projected = 0.0;
          for (unsigned bra_derivative = 0; bra_derivative < bra_hermites; ++bra_derivative) {
            const double coefficient =
                md.transforms[bra.transform_offset + bra_component * bra_hermites + bra_derivative];
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
            const double coefficient =
                md.transforms[bra.transform_offset + component * bra_hermites + bra_derivative];
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
            value +=
                intermediate[bra_component * ket_hermites + derivative] *
                md.transforms[ket.transform_offset + ket_component * ket_hermites + derivative];
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

/** Each resident worker compacts a tiny candidate tile into block-local tasks.
 * The 64-bit cursor replaces per-page host launches; neither the task inventory
 * nor the scratch grows with the molecular quartet count. Angular dispatch is
 * a launch boundary, not a device switch: low-order tasks cannot inherit the
 * high-order recurrence's register footprint and stack frame. */
template <unsigned BraAngular, unsigned KetAngular>
__global__ void consume_source_tasks(DeviceBatch batch, MdJView md, std::size_t system_begin,
                                     std::size_t system_end, bool want_j, bool want_k,
                                     bool unrestricted, double screening, const double* bounds,
                                     const double* density, const double* beta, double* coulomb,
                                     double* alpha_exchange, double* beta_exchange) {
  extern __shared__ double workspace[];
  __shared__ MdSourceTask tasks[kSourceCandidates];
  __shared__ unsigned task_count;
  __shared__ unsigned long long candidate_begin;
  const auto bra_begin = md.active_pair_offsets[BraAngular];
  const auto ket_begin = md.active_pair_offsets[KetAngular];
  const auto bra_count = md.active_pair_offsets[BraAngular + 1] - bra_begin;
  const auto ket_count = md.active_pair_offsets[KetAngular + 1] - ket_begin;
  const auto candidates = BraAngular == KetAngular
                              ? static_cast<unsigned long long>(bra_count) * (bra_count + 1ULL) / 2
                              : static_cast<unsigned long long>(bra_count) * ket_count;
  const double density_allowance =
      __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
  while (true) {
    if (threadIdx.x == 0) {
      candidate_begin =
          atomicAdd(md.source_cursor, static_cast<unsigned long long>(kSourceCandidates));
      task_count = 0;
    }
    __syncthreads();
    if (candidate_begin >= candidates) return;
    for (auto candidate = candidate_begin + threadIdx.x;
         candidate < candidates && candidate < candidate_begin + kSourceCandidates;
         candidate += blockDim.x) {
      unsigned long long bra_index{}, ket_index{};
      if constexpr (BraAngular == KetAngular) {
        bra_index = static_cast<unsigned long long>((sqrt(8.0 * candidate + 1.0) - 1.0) * 0.5);
        while (bra_index * (bra_index + 1ULL) / 2 > candidate) --bra_index;
        while ((bra_index + 1ULL) * (bra_index + 2ULL) / 2 <= candidate) ++bra_index;
        ket_index = candidate - bra_index * (bra_index + 1ULL) / 2;
      } else {
        bra_index = candidate / ket_count;
        ket_index = candidate % ket_count;
      }
      {
        const auto bra = md.active_pairs[bra_begin + bra_index],
                   ket = md.active_pairs[ket_begin + ket_index];
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
        }
      }
    }
    __syncthreads();
    constexpr auto group_threads = SourceSchedule<BraAngular, KetAngular>::group_threads;
    constexpr auto groups = SourceSchedule<BraAngular, KetAngular>::groups;
    constexpr auto workspace_width =
        CoulombAuxiliary<double, BraAngular + KetAngular>::kStateCount +
        source_components(BraAngular) * md_j_hermite_count(KetAngular) +
        source_components(BraAngular) * source_components(KetAngular) + 1;
    const unsigned group_index = threadIdx.x / group_threads;
    for (unsigned task_index = group_index; task_index < task_count; task_index += groups) {
      const auto task = tasks[task_index];
      contract_source_task<BraAngular, KetAngular>(
          batch, md, task, want_j, want_k, unrestricted, screening, bounds, density, beta, coulomb,
          alpha_exchange, beta_exchange, workspace + group_index * workspace_width);
    }
    __syncthreads();
  }
}

template <unsigned BraAngular, unsigned KetAngular>
void launch_source_class(cudaStream_t stream, DeviceBatch batch, MdJView md,
                         std::size_t system_begin, std::size_t system_count, bool want_j,
                         bool want_k, bool unrestricted, double screening, const double* bounds,
                         const double* density, const double* beta, double* coulomb,
                         double* alpha_exchange, double* beta_exchange) {
  if (md.active_pair_offsets[BraAngular + 1] == md.active_pair_offsets[BraAngular] ||
      md.active_pair_offsets[KetAngular + 1] == md.active_pair_offsets[KetAngular])
    return;
  constexpr auto shared_bytes =
      (CoulombAuxiliary<double, BraAngular + KetAngular>::kStateCount +
       source_components(BraAngular) * md_j_hermite_count(KetAngular) +
       source_components(BraAngular) * source_components(KetAngular) + 1) *
      sizeof(double) * SourceSchedule<BraAngular, KetAngular>::groups;
  const auto bra_count = static_cast<unsigned long long>(md.active_pair_offsets[BraAngular + 1] -
                                                         md.active_pair_offsets[BraAngular]);
  const auto ket_count = static_cast<unsigned long long>(md.active_pair_offsets[KetAngular + 1] -
                                                         md.active_pair_offsets[KetAngular]);
  const auto candidates =
      BraAngular == KetAngular ? bra_count * (bra_count + 1ULL) / 2 : bra_count * ket_count;
  if (!candidates) return;
  const unsigned workers = static_cast<unsigned>(std::min<unsigned long long>(
      kSourceWorkers, (candidates + kSourceCandidates - 1) / kSourceCandidates));
  cudaMemsetAsync(md.source_cursor, 0, sizeof(unsigned long long), stream);
  consume_source_tasks<BraAngular, KetAngular>
      <<<workers, SourceSchedule<BraAngular, KetAngular>::block_threads, shared_bytes, stream>>>(
          batch, md, system_begin, system_begin + system_count, want_j, want_k, unrestricted,
          screening, bounds, density, beta, coulomb, alpha_exchange, beta_exchange);
}

}  // namespace

void launch_md_source_jk(cudaStream_t stream, DeviceBatch batch, MdJView md,
                         std::size_t system_begin, std::size_t system_count, bool want_j,
                         bool want_k, bool unrestricted, double screening, const double* bounds,
                         const double* density, const double* beta, double* coulomb,
                         double* alpha_exchange, double* beta_exchange) {
  const auto begin = system_begin * batch.nbf * batch.nbf;
  const auto elements = system_count * batch.nbf * batch.nbf;
  if (want_j) cudaMemsetAsync(coulomb + begin, 0, elements * sizeof(double), stream);
  if (want_k) {
    cudaMemsetAsync(alpha_exchange + begin, 0, elements * sizeof(double), stream);
    if (unrestricted) cudaMemsetAsync(beta_exchange + begin, 0, elements * sizeof(double), stream);
  }
  if (!want_k && !md.residual_candidate_count) return;
#define MD_SOURCE_CLASS(bra, ket)                                                              \
  launch_source_class<bra, ket>(stream, batch, md, system_begin, system_count, want_j, want_k, \
                                unrestricted, screening, bounds, density, beta, coulomb,       \
                                alpha_exchange, beta_exchange)
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
