#include <cmath>
#include <cstddef>
#include <cstdint>

#include "generated_df_md_j.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/df_metric_shell_derivatives.cuh"

namespace generativeqc::scf {
namespace {
using cuda_execution::Vec3;
constexpr unsigned kThreads = 128;
constexpr unsigned kMaximumAngular = 4;

__host__ __device__ constexpr unsigned monomials(unsigned l) { return (l + 1) * (l + 2) / 2; }
__host__ __device__ constexpr unsigned table_offset(unsigned l) {
  unsigned offset = 0;
  for (unsigned k = 0; k < l; ++k) offset += (2 * k + 1) * monomials(k);
  return offset;
}
static_assert(table_offset(kMaximumAngular + 1) == DfMetricHermiteTable::kCoefficients);

struct Power {
  unsigned x, y, z;
};
/** Degree-l monomials in CCA order: x descending, then y descending. */
__host__ __device__ constexpr Power monomial(unsigned l, unsigned k) {
  for (unsigned a = l + 1; a-- > 0;) {
    const unsigned span = l - a + 1;
    if (k < span) return {a, l - a - k, k};
    k -= span;
  }
  return {0, 0, 0};
}
/** Address of R(t,u,v) in df_md_radial's ascending t/u/v simplex of one order. */
__host__ __device__ constexpr unsigned radial_index(unsigned order, unsigned t, unsigned u,
                                                    unsigned v) {
  unsigned index = 0;
  for (unsigned s = 0; s < t; ++s) index += (order - s + 1) * (order - s + 2) / 2;
  for (unsigned s = 0; s < u; ++s) index += order - t - s + 1;
  return index + v;
}

template <unsigned Power>
__device__ inline double integer_power(double value) {
  double result = 1;
  if constexpr (Power > 0)
    for (unsigned k = 0; k < Power; ++k) result *= value;
  return result;
}

/** Raised addresses R(T+U+e_axis) for every first/second top-order monomial. */
template <unsigned First, unsigned Second>
struct Addresses {
  static constexpr unsigned order = First + Second + 1;
  unsigned short value[monomials(First) * monomials(Second) * 3]{};
  __host__ __device__ constexpr Addresses() {
    for (unsigned t = 0; t < monomials(First); ++t)
      for (unsigned u = 0; u < monomials(Second); ++u) {
        const auto a = monomial(First, t), b = monomial(Second, u);
        const unsigned x = a.x + b.x, y = a.y + b.y, z = a.z + b.z;
        auto* row = value + 3 * (t * monomials(Second) + u);
        row[0] = radial_index(order, x + 1, y, z);
        row[1] = radial_index(order, x, y + 1, z);
        row[2] = radial_index(order, x, y, z + 1);
      }
  }
};

template <unsigned First, unsigned Second>
__global__ void __launch_bounds__(kThreads)
    metric_shell_derivative(DfShellBasisView view, DfMetricHermiteTable table,
                            const double* positions, const double* weights, double* gradient) {
  constexpr unsigned order = First + Second + 1;
  constexpr unsigned radial_count = (order + 1) * (order + 2) * (order + 3) / 6;
  constexpr unsigned first_aos = 2 * First + 1, second_aos = 2 * Second + 1;
  constexpr unsigned first_monomials = monomials(First), second_monomials = monomials(Second);
  constexpr unsigned first_offset = table_offset(First), second_offset = table_offset(Second);
  constexpr Addresses<First, Second> addresses{};
  const auto& basis = view.basis;
  const auto first = view.shell_ids[view.begin[First] + blockIdx.x];
  const auto first_atom = basis.shell_atoms[first];
  const auto index = std::size_t{blockIdx.y} * kThreads + threadIdx.x;
  double force[3]{};
  if (index < view.count[Second]) {
    const auto second = view.shell_ids[view.begin[Second] + index];
    const auto second_atom = basis.shell_atoms[second];
    if (second_atom != first_atom) {
      const Vec3<double> a{positions[3 * first_atom], positions[3 * first_atom + 1],
                           positions[3 * first_atom + 2]};
      const Vec3<double> b{positions[3 * second_atom], positions[3 * second_atom + 1],
                           positions[3 * second_atom + 2]};
      double radial[radial_count]{};
      for (auto i = basis.primitive_offsets[first]; i < basis.primitive_offsets[first + 1]; ++i)
        for (auto j = basis.primitive_offsets[second]; j < basis.primitive_offsets[second + 1];
             ++j) {
          const double alpha = basis.exponents[i], beta = basis.exponents[j];
          const double sum = alpha + beta;
          // Hermite Coulomb prefactor 2 pi^(5/2) / (a b sqrt(a+b)), the top-order
          // expansion factors (2a)^-l (2b)^-l', and the second-center parity.
          double scale = basis.coefficients[i] * basis.coefficients[j] * 34.986836655249725 /
                         (alpha * beta * sqrt(sum));
          scale /= integer_power<First>(2 * alpha) * integer_power<Second>(2 * beta);
          if constexpr (Second & 1) scale = -scale;
          double values[radial_count];
          cuda_execution::df_md_radial<order>(alpha * (beta / sum), a, b, values, scale);
#pragma unroll
          for (unsigned k = 0; k < radial_count; ++k) radial[k] += values[k];
        }
      const auto n = basis.nbf;
      const auto first_ao = static_cast<std::size_t>(view.ao_offsets[first]);
      const auto second_ao = static_cast<std::size_t>(view.ao_offsets[second]);
#pragma unroll
      for (unsigned p = 0; p < first_aos; ++p) {
        // Project the symmetric weight row onto the second shell's monomials.
        double projected[second_monomials]{};
#pragma unroll
        for (unsigned q = 0; q < second_aos; ++q) {
          const double w = weights[(first_ao + p) * n + second_ao + q] +
                           weights[(second_ao + q) * n + first_ao + p];
#pragma unroll
          for (unsigned u = 0; u < second_monomials; ++u)
            projected[u] += table.coefficients[second_offset + q * second_monomials + u] * w;
        }
#pragma unroll
        for (unsigned t = 0; t < first_monomials; ++t) {
          const double c = table.coefficients[first_offset + p * first_monomials + t];
          if (c == 0) continue;  // Uniform: every thread shares this class table.
          double sums[3]{};
#pragma unroll
          for (unsigned u = 0; u < second_monomials; ++u)
#pragma unroll
            for (unsigned axis = 0; axis < 3; ++axis)
              sums[axis] +=
                  projected[u] * radial[addresses.value[3 * (t * second_monomials + u) + axis]];
#pragma unroll
          for (unsigned axis = 0; axis < 3; ++axis) force[axis] += c * sums[axis];
        }
      }
    }
  }
  // Every thread of the block differentiates the same first shell's center.
  __shared__ double partial[kThreads / 32][3];
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (unsigned axis = 0; axis < 3; ++axis) {
    double value = force[axis];
    for (unsigned stride = 16; stride; stride /= 2)
      value += __shfl_down_sync(0xffffffffU, value, stride);
    if (!lane) partial[warp][axis] = value;
  }
  __syncthreads();
  if (threadIdx.x < 3) {
    double value = 0;
    for (unsigned w = 0; w < kThreads / 32; ++w) value += partial[w][threadIdx.x];
    if (value != 0) atomicAdd(gradient + 3 * std::size_t(first_atom) + threadIdx.x, value);
  }
}

template <unsigned First, unsigned Second>
cudaError_t launch_class(const DfShellBasisView& view, const DfMetricHermiteTable& table,
                         const double* positions, const double* weights, double* gradient,
                         cudaStream_t stream) {
  if (!view.count[First] || !view.count[Second]) return cudaSuccess;
  const auto tiles = (view.count[Second] - 1) / kThreads + 1;
  if (view.count[First] > 0x7fffffffU || tiles > 65535) return cudaErrorInvalidValue;
  metric_shell_derivative<First, Second>
      <<<dim3(static_cast<unsigned>(view.count[First]), static_cast<unsigned>(tiles)), kThreads, 0,
         stream>>>(view, table, positions, weights, gradient);
  return cudaPeekAtLastError();
}

template <unsigned First>
cudaError_t launch_row(const DfShellBasisView& view, const DfMetricHermiteTable& table,
                       const double* positions, const double* weights, double* gradient,
                       cudaStream_t stream) {
  auto status = launch_class<First, 0>(view, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_class<First, 1>(view, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_class<First, 2>(view, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_class<First, 3>(view, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_class<First, 4>(view, table, positions, weights, gradient, stream);
  return status;
}
}  // namespace

bool prepare_df_metric_hermite_table(const core::System& auxiliary, DfMetricHermiteTable& table) {
  table = {};
  bool present[kMaximumAngular + 1]{};
  for (const auto& shell : auxiliary.shells) {
    if (shell.angular_momentum > kMaximumAngular) return false;
    present[shell.angular_momentum] = true;
  }
  for (unsigned l = 0; l <= kMaximumAngular; ++l) {
    if (!present[l]) continue;
    const auto expansions = molecule::ao_expansions(l, auxiliary.basis_representation);
    if (expansions.size() != 2 * l + 1) return false;
    for (std::size_t p = 0; p < expansions.size(); ++p) {
      double* row = table.coefficients + table_offset(l) + p * monomials(l);
      double scale = 0;
      for (const auto& term : expansions[p]) {
        const auto& power = term.component;
        if (power[0] + power[1] + power[2] != l) return false;
        // The derivative view stores exactly this product per Cartesian term.
        const double value =
            term.coefficient * molecule::cartesian_component_normalization(term.component);
        unsigned k = 0;
        while (k < monomials(l) && !(monomial(l, k).x == power[0] && monomial(l, k).y == power[1] &&
                                     monomial(l, k).z == power[2]))
          ++k;
        if (k == monomials(l)) return false;
        row[k] += value;
        scale = std::fmax(scale, std::fabs(row[k]));
      }
      // Laplacian of the polynomial: only harmonic AOs have no lower Hermite terms.
      if (l >= 2) {
        double laplacian[monomials(kMaximumAngular - 2)]{};
        for (unsigned k = 0; k < monomials(l); ++k) {
          const auto m = monomial(l, k);
          const unsigned powers[3]{m.x, m.y, m.z};
          for (unsigned axis = 0; axis < 3; ++axis) {
            if (powers[axis] < 2) continue;
            unsigned lowered[3]{powers[0], powers[1], powers[2]};
            lowered[axis] -= 2;
            unsigned target = 0;
            while (!(monomial(l - 2, target).x == lowered[0] &&
                     monomial(l - 2, target).y == lowered[1] &&
                     monomial(l - 2, target).z == lowered[2]))
              ++target;
            laplacian[target] += powers[axis] * (powers[axis] - 1) * row[k];
          }
        }
        for (unsigned k = 0; k < monomials(l - 2); ++k)
          if (!(std::fabs(laplacian[k]) <= 1e-12 * l * l * scale)) return false;
      }
    }
  }
  return true;
}

cudaError_t launch_df_metric_shell_derivative(const DfShellBasisView& auxiliary,
                                              const DfMetricHermiteTable& table,
                                              const double* positions, const double* weights,
                                              double* gradient, cudaStream_t stream) {
  if (!positions || !weights || !gradient || !auxiliary.basis.nbf || auxiliary.range_omega != 0.0)
    return cudaErrorInvalidValue;
  auto status = launch_row<0>(auxiliary, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_row<1>(auxiliary, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_row<2>(auxiliary, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_row<3>(auxiliary, table, positions, weights, gradient, stream);
  if (status == cudaSuccess)
    status = launch_row<4>(auxiliary, table, positions, weights, gradient, stream);
  return status;
}
}  // namespace generativeqc::scf
