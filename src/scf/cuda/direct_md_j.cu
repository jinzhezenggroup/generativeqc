#include <cmath>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/direct_md_j.hpp"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/md_hermite_index.cuh"
#include "generated_md_j_reciprocal.cuh"

namespace generativeqc::scf::cuda_execution {
namespace {

constexpr unsigned kMdThreads = 32;

/** A shell pair contributes at most B_bra * B_ket * sum(abs(D_ket)) to any
 * public AO output. Dividing the configured tolerance by the complete pair
 * census bounds the sum of discarded J contributions, rather than granting
 * every primitive or component its own full error allowance. */
__global__ void md_j_density_bounds(DeviceBatch batch, MdJView md, std::size_t begin,
                                    std::size_t end, bool unrestricted, const double* density,
                                    const double* beta) {
  const auto pair = md.pairs[blockIdx.x];
  if (pair.system < begin || pair.system >= end) return;
  const std::size_t offset = static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf;
  double maximum = 0.0;
  const unsigned components = pair.first_count * pair.second_count;
  for (unsigned component = threadIdx.x; component < components; component += blockDim.x) {
    const auto first = pair.first_ao + component / pair.second_count;
    const auto second = pair.second_ao + component % pair.second_count;
    const auto forward = offset + first * batch.nbf + second;
    double value = fabs(density[forward]);
    if (unrestricted) value = __dadd_ru(value, fabs(beta[forward]));
    maximum = fmax(maximum, value);
    if (pair.first != pair.second) {
      const auto reverse = offset + second * batch.nbf + first;
      value = fabs(density[reverse]);
      if (unrestricted) value = __dadd_ru(value, fabs(beta[reverse]));
      maximum = fmax(maximum, value);
    }
  }
  for (unsigned stride = kMdThreads / 2; stride; stride /= 2)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffff, maximum, stride));
  if (threadIdx.x == 0)
    md.density_bounds[blockIdx.x] =
        __dmul_ru(maximum, components * (pair.first == pair.second ? 1.0 : 2.0));
}

__global__ void md_j_bounds(DeviceBatch batch, MdJView md, const double* bounds) {
  const auto pair = md.pairs[blockIdx.x];
  double minimum = INFINITY;
  double maximum = 0.0;
  const std::size_t offset = static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf;
  for (int first = 0; first < pair.first_count; ++first) {
    for (int second = 0; second < pair.second_count; ++second) {
      minimum = fmin(
          minimum, bounds[offset + (pair.first_ao + first) * batch.nbf + pair.second_ao + second]);
      minimum = fmin(
          minimum, bounds[offset + (pair.second_ao + second) * batch.nbf + pair.first_ao + first]);
      maximum = fmax(
          maximum, bounds[offset + (pair.first_ao + first) * batch.nbf + pair.second_ao + second]);
      maximum = fmax(
          maximum, bounds[offset + (pair.second_ao + second) * batch.nbf + pair.first_ao + first]);
    }
  }
  md.minimum_bounds[blockIdx.x] = minimum;
  md.maximum_bounds[blockIdx.x] = maximum;
  atomicMax(reinterpret_cast<unsigned long long*>(md.maximum_bound),
            static_cast<unsigned long long>(__double_as_longlong(maximum)));
}

/** A pair excluded by this global upper bound cannot pass any AO quartet mask. */
__global__ void md_active_pairs(MdJView md, double screening, unsigned angular) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= md.pair_count || md.pairs[index].angular != angular ||
      md.maximum_bounds[index] * *md.maximum_bound < screening)
    return;
  md.active_pairs[atomicAdd(md.active_count, 1U)] = static_cast<std::uint32_t>(index);
}

/** Reuse the charged primitive-order buffer for uniform MD eligibility. Source
 * J/K keeps its separate maximum-bound pair list for partially screened work. */
__global__ void md_active_primitives(MdJView md, double screening, unsigned angular) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= md.primitive_count) return;
  const auto pair = md.primitives[index].pair;
  if (md.pairs[pair].angular != angular || md.minimum_bounds[pair] * *md.maximum_bound < screening)
    return;
  auto* cursor = reinterpret_cast<std::uint32_t*>(md.source_cursor);
  md.ordered_primitives[atomicAdd(cursor, 1U)] = static_cast<std::uint32_t>(index);
}

__global__ void md_j_transforms(DeviceBatch batch, MdJView md) {
  auto& primitive = md.primitives[blockIdx.x];
  const auto pair = md.pairs[primitive.pair];
  const double alpha = batch.primitive_exponents[primitive.first];
  const double beta = batch.primitive_exponents[primitive.second];
  const auto first = atom_position<double>(batch, batch.shell_atoms[pair.first], -1);
  const auto second = atom_position<double>(batch, batch.shell_atoms[pair.second], -1);
  const auto product = product_center(alpha, first, beta, second);
  if (threadIdx.x == 0) {
    primitive.exponent = alpha + beta;
    primitive.product = product;
  }
  HermiteCoefficients<double> coefficients[3];
  for (unsigned axis = 0; axis < 3; ++axis)
    fill_hermite(batch.shell_angular[pair.first], batch.shell_angular[pair.second],
                 vec_axis(product, axis), vec_axis(first, axis), vec_axis(second, axis), alpha,
                 beta, coefficients[axis]);
  const double weight = batch.primitive_coefficients[primitive.first] *
                        batch.primitive_coefficients[primitive.second];
  const auto base = static_cast<std::int64_t>(pair.system) * batch.nbf;
  for (unsigned component = threadIdx.x;
       component < static_cast<unsigned>(pair.first_count * pair.second_count);
       component += blockDim.x) {
    const auto ao_first = base + pair.first_ao + component / pair.second_count;
    const auto ao_second = base + pair.second_ao + component % pair.second_count;
    for (unsigned index = 0; index < pair.hermites; ++index) {
      const auto derivative = md_hermite_angular(pair.angular, index);
      double value = 0.0;
      for (unsigned first_term = 0; first_term < batch.ao_term_counts[ao_first]; ++first_term) {
        const auto first_angular = ao_angular(batch, ao_first, first_term);
        for (unsigned second_term = 0; second_term < batch.ao_term_counts[ao_second];
             ++second_term) {
          const auto second_angular = ao_angular(batch, ao_second, second_term);
          if (derivative.x > first_angular.x + second_angular.x ||
              derivative.y > first_angular.y + second_angular.y ||
              derivative.z > first_angular.z + second_angular.z)
            continue;
          value += ao_term_coefficient(batch, ao_first, first_term) *
                   ao_term_coefficient(batch, ao_second, second_term) *
                   coefficients[0].at(first_angular.x, second_angular.x, derivative.x) *
                   coefficients[1].at(first_angular.y, second_angular.y, derivative.y) *
                   coefficients[2].at(first_angular.z, second_angular.z, derivative.z);
        }
      }
      md.transforms[primitive.transform_offset + component * pair.hermites + index] =
          weight * value;
    }
  }
}

__global__ void md_j_density(DeviceBatch batch, MdJView md, std::size_t begin, std::size_t end,
                             bool unrestricted, const double* density, const double* beta) {
  const auto primitive = md.primitives[blockIdx.x];
  const auto pair = md.pairs[primitive.pair];
  if (pair.system < begin || pair.system >= end) return;
  const std::size_t offset = static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf;
  for (unsigned index = threadIdx.x; index < pair.hermites; index += blockDim.x) {
    double value = 0.0;
    for (int first = 0; first < pair.first_count; ++first) {
      for (int second = 0; second < pair.second_count; ++second) {
        const auto forward = offset + (pair.first_ao + first) * batch.nbf + pair.second_ao + second;
        double total = density[forward] + (unrestricted ? beta[forward] : 0.0);
        // Unordered off-diagonal shell pairs must preserve nonsymmetric input.
        // Diagonal pairs already enumerate both component orientations.
        if (pair.first != pair.second) {
          const auto reverse =
              offset + (pair.second_ao + second) * batch.nbf + pair.first_ao + first;
          total += density[reverse] + (unrestricted ? beta[reverse] : 0.0);
        }
        value +=
            total * md.transforms[primitive.transform_offset +
                                  (first * pair.second_count + second) * pair.hermites + index];
      }
    }
    md.density[primitive.hermite_offset + index] = value;
    md.potential[primitive.hermite_offset + index] = 0.0;
  }
}

/** Reuse the existing MD/Boys mathematics, contracting density before any AO
 * output expansion. Every ordered bra/ket primitive product is evaluated once.
 */
template <unsigned BraAngular, unsigned KetAngular>
__device__ __forceinline__ void contract_potential(const MdJPrimitive& bra, const MdJPrimitive& ket,
                                                   MdJView md, double* potential) {
  CoulombAuxiliary<double, BraAngular + KetAngular> auxiliary;
  const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
  fill_coulomb<BraAngular + KetAngular>(rho, bra.product, ket.product, auxiliary);
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
  for (unsigned first = 0; first < md_j_hermite_count(BraAngular); ++first) {
    const auto bra_angular = md_hermite_angular(BraAngular, first);
    double value = 0.0;
    for (unsigned second = 0; second < md_j_hermite_count(KetAngular); ++second) {
      const auto ket_angular = md_hermite_angular(KetAngular, second);
      const double sign = (ket_angular.x + ket_angular.y + ket_angular.z) & 1U ? -1.0 : 1.0;
      value += sign * md.density[ket.hermite_offset + second] *
               auxiliary.at(0, bra_angular.x + ket_angular.x, bra_angular.y + ket_angular.y,
                            bra_angular.z + ket_angular.z);
    }
    potential[first] += prefactor * value;
  }
}

/** Fixed angular products keep small recurrences from inheriting the largest
 * ket class's local storage. Sequential class launches own disjoint bra outputs
 * within each launch, so potential accumulation needs no floating atomics. */
template <unsigned BraAngular, unsigned KetAngular, bool CountWork>
__global__ void md_j_potential(MdJView md, std::size_t class_begin, std::size_t begin,
                               std::size_t end, double screening) {
  const auto bra_index = md.ordered_primitives[class_begin + blockIdx.x];
  const auto bra = md.primitives[bra_index];
  const auto pair = md.pairs[bra.pair];
  if (pair.system < begin || pair.system >= end) return;
  double potential[md_j_hermite_count(BraAngular)]{};
  unsigned long long admitted = 0;
  const double bound = md.minimum_bounds[bra.pair];
  // The compact pair upper bound is conservative for the existing minimum
  // mask. Inactive bras still clear their potential before AO projection.
  if (bound * *md.maximum_bound < screening) {
    return;
  }
  const double output_bound = md.maximum_bounds[bra.pair];
  const double allowance = __ddiv_rd(fmin(screening, kMdJDensityErrorCap), double(md.pair_count));
  for (auto primitive_index = md.class_offsets[KetAngular] + threadIdx.x;
       primitive_index < md.class_offsets[KetAngular + 1]; primitive_index += blockDim.x) {
    const auto ket = md.primitives[md.ordered_primitives[primitive_index]];
    const auto ket_pair_index = ket.pair;
    const auto ket_pair = md.pairs[ket_pair_index];
    if (ket_pair.system != pair.system || bound * md.minimum_bounds[ket_pair_index] < screening)
      continue;
    const double density_bound = md.density_bounds[ket_pair_index];
    const double contribution_bound =
        __dmul_ru(__dmul_ru(output_bound, md.maximum_bounds[ket_pair_index]), density_bound);
    if (density_bound == 0.0 || contribution_bound < allowance) continue;
    contract_potential<BraAngular, KetAngular>(bra, ket, md, potential);
    if constexpr (CountWork) ++admitted;
  }
  if constexpr (CountWork) {
    for (unsigned stride = kMdThreads / 2; stride; stride /= 2)
      admitted += __shfl_down_sync(0xffffffff, admitted, stride);
    if (threadIdx.x == 0) {
      constexpr auto angular = 5 * BraAngular + KetAngular;
      atomicAdd(&md.work_counts->uniform_roots[angular], admitted);
      atomicAdd(&md.work_counts->uniform_directions[angular], admitted);
      atomicAdd(&md.work_counts->uniform_summands[angular],
                admitted * md_j_hermite_count(BraAngular) * md_j_hermite_count(KetAngular));
    }
  }
  for (unsigned index = 0; index < md_j_hermite_count(BraAngular); ++index) {
    double value = potential[index];
    for (unsigned stride = kMdThreads / 2; stride; stride /= 2)
      value += __shfl_down_sync(0xffffffff, value, stride);
    if (threadIdx.x == 0) md.potential[bra.hermite_offset + index] += value;
  }
}

__global__ void md_j_project(DeviceBatch batch, MdJView md, std::size_t begin, std::size_t end,
                             double* coulomb) {
  const auto pair = md.pairs[blockIdx.x];
  if (pair.system < begin || pair.system >= end) return;
  const std::size_t offset = static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf;
  for (unsigned component = threadIdx.x;
       component < static_cast<unsigned>(pair.first_count * pair.second_count);
       component += blockDim.x) {
    double value = 0.0;
    for (auto index = pair.primitive_begin; index < pair.primitive_end; ++index) {
      const auto primitive = md.primitives[index];
      for (unsigned derivative = 0; derivative < pair.hermites; ++derivative)
        value +=
            md.transforms[primitive.transform_offset + component * pair.hermites + derivative] *
            md.potential[primitive.hermite_offset + derivative];
    }
    const auto first = pair.first_ao + component / pair.second_count;
    const auto second = pair.second_ao + component % pair.second_count;
    coulomb[offset + first * batch.nbf + second] += value;
    if (pair.first != pair.second) coulomb[offset + second * batch.nbf + first] += value;
  }
}

template <unsigned Angular, unsigned KetAngular>
void launch_potential(cudaStream_t stream, MdJView md, std::size_t begin, std::size_t end,
                      double screening) {
  const auto count = md.class_offsets[Angular + 1] - md.class_offsets[Angular];
  if (!count || md.class_offsets[KetAngular] == md.class_offsets[KetAngular + 1]) return;
  if (md.work_counts)
    md_j_potential<Angular, KetAngular, true><<<static_cast<unsigned>(count), kMdThreads, 0, stream>>>(
        md, md.class_offsets[Angular], begin, end, screening);
  else
    md_j_potential<Angular, KetAngular, false><<<static_cast<unsigned>(count), kMdThreads, 0, stream>>>(
        md, md.class_offsets[Angular], begin, end, screening);
}

template <unsigned Angular>
void launch_potential_classes(cudaStream_t stream, MdJView md, std::size_t begin, std::size_t end,
                              double screening) {
  launch_potential<Angular, 0>(stream, md, begin, end, screening);
  launch_potential<Angular, 1>(stream, md, begin, end, screening);
  launch_potential<Angular, 2>(stream, md, begin, end, screening);
  launch_potential<Angular, 3>(stream, md, begin, end, screening);
  launch_potential<Angular, 4>(stream, md, begin, end, screening);
}

}  // namespace

cudaError_t prepare_md_j(cudaStream_t stream, DeviceBatch batch, MdJView& md, const double* bounds,
                         double screening) {
  auto error = cudaMemsetAsync(md.maximum_bound, 0, sizeof(double), stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(md.active_count, 0, sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) return error;
  md_j_bounds<<<static_cast<unsigned>(md.pair_count), 1, 0, stream>>>(batch, md, bounds);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  md_j_transforms<<<static_cast<unsigned>(md.primitive_count), kMdThreads, 0, stream>>>(batch, md);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  md.active_pair_offsets[0] = 0;
  for (unsigned angular = 0; angular <= 4; ++angular) {
    md_active_pairs<<<static_cast<unsigned>((md.pair_count + 127) / 128), 128, 0, stream>>>(
        md, screening, angular);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = cudaMemcpyAsync(md.active_pair_offsets + angular + 1, md.active_count,
                            sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return error;
    // This fixed, preparation-only download freezes the segment boundaries.
    // Neither replay nor a density-dependent source traversal reads the host.
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return error;
  }
  md.active_pair_count = md.active_pair_offsets[5];
  error = cudaMemsetAsync(md.source_cursor, 0, sizeof(unsigned long long), stream);
  if (error != cudaSuccess) return error;
  md.class_offsets[0] = 0;
  for (unsigned angular = 0; angular <= 4; ++angular) {
    md_active_primitives<<<static_cast<unsigned>((md.primitive_count + 127) / 128), 128, 0,
                           stream>>>(md, screening, angular);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    std::uint32_t primitive_count = 0;
    error = cudaMemcpyAsync(&primitive_count, md.source_cursor, sizeof(primitive_count),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) {
      cudaStreamSynchronize(stream);
      return error;
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return error;
    md.class_offsets[angular + 1] = primitive_count;
  }
  md.residual_candidate_count =
      screening > 0.0 ? md.active_pair_count * (md.active_pair_count + 1ULL) / 2 : 0;
  return cudaSuccess;
}

void launch_md_j_density_bounds(cudaStream_t stream, DeviceBatch batch, MdJView md,
                                std::size_t system_begin, std::size_t system_count,
                                bool unrestricted, const double* density, const double* beta) {
  md_j_density_bounds<<<static_cast<unsigned>(md.pair_count), kMdThreads, 0, stream>>>(
      batch, md, system_begin, system_begin + system_count, unrestricted, density, beta);
}

void launch_md_j(cudaStream_t stream, DeviceBatch batch, MdJView md, std::size_t system_begin,
                 std::size_t system_count, bool unrestricted, double screening,
                 const double* density, const double* beta, double* coulomb) {
  const auto end = system_begin + system_count;
  md_j_density_bounds<<<static_cast<unsigned>(md.pair_count), kMdThreads, 0, stream>>>(
      batch, md, system_begin, end, unrestricted, density, beta);
  md_j_density<<<static_cast<unsigned>(md.primitive_count), kMdThreads, 0, stream>>>(
      batch, md, system_begin, end, unrestricted, density, beta);
  if (md.reciprocal) {
    launch_md_reciprocal_potential<0, 0>(stream, md, system_begin, end, screening);
    launch_md_reciprocal_potential<0, 1>(stream, md, system_begin, end, screening);
    launch_md_reciprocal_potential<0, 2>(stream, md, system_begin, end, screening);
    launch_md_reciprocal_potential<1, 1>(stream, md, system_begin, end, screening);
    launch_md_reciprocal_potential<1, 2>(stream, md, system_begin, end, screening);
    launch_md_reciprocal_potential<2, 2>(stream, md, system_begin, end, screening);
    launch_potential<0, 3>(stream, md, system_begin, end, screening);
    launch_potential<0, 4>(stream, md, system_begin, end, screening);
    launch_potential<1, 3>(stream, md, system_begin, end, screening);
    launch_potential<1, 4>(stream, md, system_begin, end, screening);
    launch_potential<2, 3>(stream, md, system_begin, end, screening);
    launch_potential<2, 4>(stream, md, system_begin, end, screening);
  } else {
    launch_potential_classes<0>(stream, md, system_begin, end, screening);
    launch_potential_classes<1>(stream, md, system_begin, end, screening);
    launch_potential_classes<2>(stream, md, system_begin, end, screening);
  }
  launch_potential_classes<3>(stream, md, system_begin, end, screening);
  launch_potential_classes<4>(stream, md, system_begin, end, screening);
  md_j_project<<<static_cast<unsigned>(md.pair_count), kMdThreads, 0, stream>>>(
      batch, md, system_begin, end, coulomb);
}

}  // namespace generativeqc::scf::cuda_execution
