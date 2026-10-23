// Execution template included after generated primitive/normalization functions.
// Scientific values and partials belong to the compiler. This schedule streams
// primitive pullbacks and never materializes a coordinate-by-ERI Jacobian.
#include <algorithm>
#include <cstddef>
#include <vector>
#ifndef GQC_AD_FN
#define GQC_AD_FN static
#endif

namespace integral_ad {
GQC_AD_FN bool finite(double x) { return std::isfinite(x); }
GQC_AD_FN bool zero(double x) { return x == 0; }
// Generated topology: nat, nshell, nprim, nao, ncomponent, shell_atom,
// shell_begin, ao_shell, ao_component, ao_scale, normalize(shell, x, seed, out).
constexpr std::size_t n2 = nao * nao;
constexpr std::size_t n4 = n2 * n2;
constexpr std::size_t value_count = 2 * n2 + n4 + 1;
constexpr std::size_t scratch_count = 7 * nprim + 2 * parameter_tape_nodes;
constexpr std::size_t parameter_count = 3 * nat + 2 * nprim;

/** Scatter one ordered primitive tuple, including coincident physical centers.
 * Product-except-slot avoids dividing by a possibly zero contraction
 * coefficient; repeated shell/primitive slots accumulate independently. */
template <class Scalar>
GQC_AD_FN void consume(unsigned slots, unsigned centers, const int* primitives, const int* atoms,
                       double scale, const Scalar* normalized, const Scalar* values, Scalar weight,
                       Scalar* gradient, Scalar* coefficient_cotangent) {
  Scalar factor = scale;
  for (unsigned i = 0; i < slots; ++i) factor *= normalized[primitives[i]];
  for (unsigned i = 0; i < slots; ++i) {
    gradient[3 * nat + primitives[i]] += weight * factor * values[1 + i];
    Scalar partial = weight * scale * values[0];
    for (unsigned j = 0; j < slots; ++j)
      if (j != i) partial *= normalized[primitives[j]];
    coefficient_cotangent[primitives[i]] += partial;
  }
  for (unsigned i = 0; i < centers; ++i)
    for (unsigned axis = 0; axis < 3; ++axis)
      gradient[3 * atoms[i] + axis] += weight * factor * values[1 + slots + 3 * i + axis];
}

template <class Scalar>
GQC_AD_FN void coordinate_inputs(Scalar* x, unsigned slots, unsigned centers, const int* atoms,
                                 const Scalar* coordinates) {
  for (unsigned i = 0; i < centers; ++i)
    for (unsigned axis = 0; axis < 3; ++axis)
      x[slots + 3 * i + axis] = coordinates[3 * atoms[i] + axis];
}

template <class Scalar>
GQC_AD_FN int execute(const Scalar* coordinates, const Scalar* alpha, const Scalar* coefficients,
                      const Scalar* seeds, Scalar* output, Scalar* scratch, std::size_t lane = 0,
                      std::size_t lanes = 1) {
  const bool derivative = seeds != nullptr;
  const auto output_size = derivative ? parameter_count : value_count;
  if (derivative || lanes == 1)
    for (std::size_t i = 0; i < output_size; ++i) output[i] = 0;
  Scalar* normalized = scratch;
  Scalar* cotangent = scratch + nprim;
  Scalar* normalization_input = scratch + 2 * nprim;
  Scalar* normalization_output = scratch + 4 * nprim;
  for (int p = 0; p < nprim; ++p) cotangent[p] = 0;
  for (int shell = 0; shell < nshell; ++shell) {
    const int begin = shell_begin[shell], size = shell_begin[shell + 1] - begin;
    for (int p = 0; p < size; ++p) normalization_input[p] = alpha[begin + p];
    for (int p = 0; p < size; ++p) normalization_input[size + p] = coefficients[begin + p];
    normalize<Scalar>(shell, normalization_input, nullptr, normalization_output);
    for (int p = 0; p < size; ++p) {
      // A nonzero primitive may not disappear through underflow. This matches
      // native basis admission while retaining exact zero coefficient inputs.
      if (!finite(normalization_output[p]) ||
          (!zero(coefficients[begin + p]) && zero(normalization_output[p])))
        return 1;
      normalized[begin + p] = normalization_output[p];
    }
  }

  Scalar x[16], values[17];
  for (int a = 0; a < nao; ++a)
    for (int b = 0; b < nao; ++b) {
      const int sa = ao_shell[a], sb = ao_shell[b];
      int atoms[3] = {shell_atom[sa], shell_atom[sb], 0};
      const auto index = std::size_t(a) * nao + b;
      if (index % lanes != lane) continue;
      const int component = ao_component[a] * ncomponent + ao_component[b];
      const double scale = ao_scale[a] * ao_scale[b];
      coordinate_inputs(x, 2, 2, atoms, coordinates);
      for (int p = shell_begin[sa]; p < shell_begin[sa + 1]; ++p)
        for (int q = shell_begin[sb]; q < shell_begin[sb + 1]; ++q) {
          const int primitives[2] = {p, q};
          x[0] = alpha[p];
          x[1] = alpha[q];
          const Scalar factor = scale * normalized[p] * normalized[q];
          overlap(component, x, values, derivative, scratch + 7 * nprim);
          if (derivative)
            consume(2, 2, primitives, atoms, scale, normalized, values, seeds[index], output,
                    cotangent);
          else
            output[index] += factor * values[0];
          kinetic(component, x, values, derivative, scratch + 7 * nprim);
          if (derivative)
            consume(2, 2, primitives, atoms, scale, normalized, values, seeds[n2 + index], output,
                    cotangent);
          else
            output[n2 + index] += factor * values[0];
          for (int nucleus = 0; nucleus < nat; ++nucleus) {
            atoms[2] = nucleus;
            coordinate_inputs(x, 2, 3, atoms, coordinates);
            nuclear_attraction(component, x, values, derivative, scratch + 7 * nprim);
            if (derivative)
              consume(2, 3, primitives, atoms, scale, normalized, values,
                      charges[nucleus] * seeds[n2 + index], output, cotangent);
            else
              output[n2 + index] += charges[nucleus] * factor * values[0];
          }
        }
    }

  // Full ordered AO tuples match both the native dense Fock provider and the
  // generated stationary objective; no guessed symmetry multiplicities enter.
  for (int a = 0; a < nao; ++a)
    for (int b = 0; b < nao; ++b)
      for (int c = 0; c < nao; ++c)
        for (int d = 0; d < nao; ++d) {
          if ((((std::size_t(a) * nao + b) * nao + c) * nao + d + n2) % lanes != lane) continue;
          const int shells[4] = {ao_shell[a], ao_shell[b], ao_shell[c], ao_shell[d]};
          const int atoms[4] = {shell_atom[shells[0]], shell_atom[shells[1]], shell_atom[shells[2]],
                                shell_atom[shells[3]]};
          const int component =
              ((ao_component[a] * ncomponent + ao_component[b]) * ncomponent + ao_component[c]) *
                  ncomponent +
              ao_component[d];
          const double scale = ao_scale[a] * ao_scale[b] * ao_scale[c] * ao_scale[d];
          const auto index = 2 * n2 + ((std::size_t(a) * nao + b) * nao + c) * nao + d;
          coordinate_inputs(x, 4, 4, atoms, coordinates);
          for (int p = shell_begin[shells[0]]; p < shell_begin[shells[0] + 1]; ++p)
            for (int q = shell_begin[shells[1]]; q < shell_begin[shells[1] + 1]; ++q)
              for (int r = shell_begin[shells[2]]; r < shell_begin[shells[2] + 1]; ++r)
                for (int s = shell_begin[shells[3]]; s < shell_begin[shells[3] + 1]; ++s) {
                  const int primitives[4] = {p, q, r, s};
                  x[0] = alpha[p];
                  x[1] = alpha[q];
                  x[2] = alpha[r];
                  x[3] = alpha[s];
                  eri(component, x, values, derivative, scratch + 7 * nprim);
                  if (derivative)
                    consume(4, 4, primitives, atoms, scale, normalized, values, seeds[index],
                            output, cotangent);
                  else
                    output[index] += scale * normalized[p] * normalized[q] * normalized[r] *
                                     normalized[s] * values[0];
                }
        }

  for (int a = 0; a < nat; ++a)
    for (int b = 0; b < a; ++b) {
      if ((n2 + n4 + std::size_t(a) * (a - 1) / 2 + b) % lanes != lane) continue;
      const int atoms[2] = {a, b};
      coordinate_inputs(x, 0, 2, atoms, coordinates);
      nuclear_pair(x, values, derivative);
      const Scalar factor = charges[a] * charges[b];
      if (derivative) {
        for (int i = 0; i < 2; ++i)
          for (int axis = 0; axis < 3; ++axis)
            output[3 * atoms[i] + axis] +=
                factor * seeds[value_count - 1] * values[1 + 3 * i + axis];
      } else
        output[value_count - 1 + lane] += factor * values[0];
    }
  if (derivative) {
    for (int shell = 0; shell < nshell; ++shell) {
      const int begin = shell_begin[shell], size = shell_begin[shell + 1] - begin;
      for (int p = 0; p < size; ++p) normalization_input[p] = alpha[begin + p];
      for (int p = 0; p < size; ++p) normalization_input[size + p] = coefficients[begin + p];
      normalize(shell, normalization_input, cotangent + begin, normalization_output);
      for (int p = 0; p < size; ++p) {
        output[3 * nat + begin + p] += normalization_output[size + p];
        output[3 * nat + nprim + begin + p] = normalization_output[2 * size + p];
      }
    }
  }
  if (derivative || lanes == 1)
    for (std::size_t i = 0; i < output_size; ++i)
      if (!finite(output[i])) return 1;
  return 0;
}
}  // namespace integral_ad

#ifndef GQC_INTEGRAL_AD_CUDA
extern "C" int integral_ad_execute(const double* coordinates, const double* alpha,
                                   const double* coefficients, const double* seeds,
                                   double* output) {
  try {
    std::vector<double> scratch(integral_ad::scratch_count);
    return integral_ad::execute(coordinates, alpha, coefficients, seeds, output, scratch.data());
  } catch (...) {
    return 2;
  }
}

#ifdef GQC_INTEGRAL_AD_SECOND
/** Forward-over-reverse on the same schedule. Seeds may move, allowing the
 * converged electronic response to compose with the fixed-source integral HVP. */
extern "C" int integral_ad_directional(const double* coordinates, const double* alpha,
                                       const double* coefficients, const double* direction,
                                       const double* seeds, const double* seed_direction,
                                       double* output) {
  using namespace integral_ad;
  try {
    std::vector<Directional> xyz(3 * nat), exponents(nprim), raw(nprim);
    for (int i = 0; i < 3 * nat; ++i) xyz[i] = {coordinates[i], direction[i]};
    for (int i = 0; i < nprim; ++i) {
      exponents[i] = {alpha[i], direction[3 * nat + i]};
      raw[i] = {coefficients[i], direction[3 * nat + nprim + i]};
    }
    const auto count = seeds ? parameter_count : value_count;
    std::vector<Directional> weights(seeds ? value_count : 0), result(count),
        scratch(scratch_count);
    if (seeds)
      for (std::size_t i = 0; i < value_count; ++i)
        weights[i] = {seeds[i], seed_direction ? seed_direction[i] : 0.0};
    const int status =
        execute<Directional>(xyz.data(), exponents.data(), raw.data(),
                             seeds ? weights.data() : nullptr, result.data(), scratch.data());
    if (status) return status;
    for (std::size_t i = 0; i < count; ++i) output[i] = result[i].tangent;
    return 0;
  } catch (...) {
    return 2;
  }
}
#endif

#endif  // GQC_INTEGRAL_AD_CUDA
