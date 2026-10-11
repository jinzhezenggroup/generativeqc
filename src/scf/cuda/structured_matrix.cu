#include <cmath>
#include <limits>

#include "scf/cuda/basis_transform_kernels.hpp"
#include "scf/cuda/runtime_support.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda/structured_matrix.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {
using Operation = tensor::StructuredOperation;

bool shape(MatrixLibraryResources resources, int systems, int spins, int dimension,
           std::size_t& elements) {
  if (!resources.structured_binding_ ||
      !resources.structured_binding_->compatible(resources.stream_, resources.blas_, dimension,
                                                 systems, spins))
    return false;
  elements = static_cast<std::size_t>(dimension) * dimension * systems * spins;
  return (elements + 127) / 128 <= std::numeric_limits<unsigned>::max();
}

unsigned blocks(std::size_t elements) {
  return static_cast<unsigned>(std::min<std::size_t>((elements + 127) / 128, 65535));
}

bool overlap(const void* first, std::size_t first_bytes, const void* second,
             std::size_t second_bytes) {
  const auto begin = reinterpret_cast<std::uintptr_t>(first);
  const auto other = reinterpret_cast<std::uintptr_t>(second);
  const auto limit = std::numeric_limits<std::uintptr_t>::max();
  return begin > limit - first_bytes || other > limit - second_bytes ||
         (begin < other + second_bytes && other < begin + first_bytes);
}

bool disjoint(double* scratch, std::size_t capacity, std::size_t elements, const double* input,
              std::size_t input_elements, const double* other, std::size_t other_elements,
              double* output, const std::uint8_t* active, int systems) {
  const auto bytes = elements * sizeof(double);
  return scratch && capacity >= elements && input && output &&
         !overlap(scratch, bytes, input, input_elements * sizeof(double)) &&
         (!other || !overlap(scratch, bytes, other, other_elements * sizeof(double))) &&
         !overlap(scratch, bytes, output, bytes) &&
         (!active || !overlap(scratch, bytes, active, systems));
}

__global__ void scale_columns(std::size_t elements, int dimension, int spins,
                              const std::int32_t* occupied, const double* coefficients,
                              const double* energies, const std::uint8_t* active,
                              std::uint8_t* failed, double* scaled, double* unweighted) {
  const auto matrix = static_cast<std::size_t>(dimension) * dimension;
  for (std::size_t element = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; element < elements;
       element += std::size_t(blockDim.x) * gridDim.x) {
    const auto state = element / matrix, orbital = element % matrix / dimension;
    const auto system = state / spins;
    double value = 0;
    double coefficient = 0;
    if (!active || active[system]) {
      if (failed) {
        const auto eigenvalue = energies[state * dimension + orbital];
        if (!(eigenvalue > 1.e-10) || !isfinite(eigenvalue))
          failed[system] = 1;
        else
          value = coefficients[element] / sqrt(eigenvalue);
      } else if (orbital < static_cast<std::size_t>(max(occupied[state], 0))) {
        coefficient = coefficients[element];
        value = coefficient;
        if (energies) value *= energies[state * dimension + orbital];
      }
    }
    scaled[element] = value;
    if (unweighted) unweighted[element] = coefficient;
  }
}

__global__ void publish(std::size_t elements, std::size_t state_elements, int spins, int dimension,
                        bool mirror, const double* scratch, const double* hcore,
                        const std::uint8_t* active, const std::uint8_t* failed, double* output) {
  for (std::size_t element = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; element < elements;
       element += std::size_t(blockDim.x) * gridDim.x) {
    const auto state = element / state_elements, local = element % state_elements;
    const auto system = state / spins;
    if ((active && !active[system]) || (failed && failed[system])) continue;
    const auto row = local % dimension, column = local / dimension;
    const auto source = mirror && row < column ? column + row * dimension : local;
    output[element] = scratch[state * state_elements + source] +
                      (hcore ? hcore[system * state_elements + local] : 0.0);
  }
}

generativeqc_status gram(MatrixLibraryResources resources, int systems, int spins, int dimension,
                         const std::int32_t* occupied, const double* coefficients,
                         const double* energies, const std::uint8_t* active, double* output,
                         std::uint8_t* failed, double* scaled, std::size_t scaled_elements,
                         double* unweighted, std::size_t unweighted_elements) {
  std::size_t elements{};
  if (!shape(resources, systems, spins, dimension, elements) || !coefficients || !output ||
      (!failed && !occupied) || (failed && !energies))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto operation = failed     ? Operation::InverseRoot
                         : energies ? Operation::SignedGram
                                    : Operation::Gram;
  if (!resources.structured_binding_->library(operation)) {
    const auto grid = static_cast<unsigned>((elements + 127) / 128);
    if (failed)
      launch_build_orthogonalizer_kernel(grid, 128, 0, resources.stream_, systems, dimension,
                                         coefficients, energies, active, output, failed);
    else if (energies && spins == 1)
      launch_build_weighted_density_kernel(grid, 128, 0, resources.stream_, systems, dimension,
                                           occupied, coefficients, energies, active, output);
    else if (energies && spins == 2)
      launch_build_spin_weighted_density_kernel(grid, 128, 0, resources.stream_, systems, dimension,
                                                occupied, coefficients, energies, active, output);
    else if (energies)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    else if (spins == 1)
      launch_build_density_kernel(grid, 128, 0, resources.stream_, systems, dimension, occupied,
                                  coefficients, active, output);
    else
      launch_build_spin_density_kernel(grid, 128, 0, resources.stream_, systems, spins, dimension,
                                       occupied, coefficients, active, output);
    return cuda_status(cudaPeekAtLastError());
  }
  const auto weights = static_cast<std::size_t>(systems) * spins * dimension;
  const auto states = static_cast<std::size_t>(systems) * spins;
  // Zero-weight virtual orbitals cannot enter GEMM as 0 * poisoned coefficients.
  // Both signed panels are sanitized independently before any publication.
  if (energies && !failed &&
      (!disjoint(unweighted, unweighted_elements, elements, coefficients, elements, energies,
                 weights, output, active, systems) ||
       overlap(unweighted, elements * sizeof(double), scaled, elements * sizeof(double)) ||
       overlap(unweighted, elements * sizeof(double), resources.masked_output_,
               elements * sizeof(double)) ||
       overlap(unweighted, elements * sizeof(double), occupied, states * sizeof(std::int32_t))))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!disjoint(scaled, scaled_elements, elements, coefficients, elements, energies, weights,
                output, active, systems) ||
      !disjoint(resources.masked_output_, resources.masked_output_elements_, elements, coefficients,
                elements, scaled, elements, output, active, systems) ||
      (energies && overlap(resources.masked_output_, elements * sizeof(double), energies,
                           weights * sizeof(double))) ||
      (occupied &&
       (overlap(scaled, elements * sizeof(double), occupied, states * sizeof(std::int32_t)) ||
        overlap(resources.masked_output_, elements * sizeof(double), occupied,
                states * sizeof(std::int32_t)))) ||
      (failed && (overlap(scaled, elements * sizeof(double), failed, systems) ||
                  overlap(resources.masked_output_, elements * sizeof(double), failed, systems))))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  scale_columns<<<blocks(elements), 128, 0, resources.stream_>>>(
      elements, dimension, spins, occupied, coefficients, energies, active, failed, scaled,
      energies && !failed ? unweighted : nullptr);
  auto status = cuda_status(cudaPeekAtLastError());
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  status = tensor::cuda::structured_gram(resources.blas_, dimension, systems * spins,
                                         energies && !failed ? unweighted : coefficients, scaled,
                                         resources.masked_output_,
                                         failed       ? 1.0
                                         : spins == 1 ? 2.0
                                                      : 1.0,
                                         operation == Operation::Gram);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  publish<<<blocks(elements), 128, 0, resources.stream_>>>(
      elements, static_cast<std::size_t>(dimension) * dimension, spins, dimension, true,
      resources.masked_output_, nullptr, active, failed, output);
  return cuda_status(cudaPeekAtLastError());
}
}  // namespace

generativeqc_status launch_structured_density(MatrixLibraryResources resources, int systems,
                                              int spins, int dimension,
                                              const std::int32_t* occupied,
                                              const double* coefficients, const double* energies,
                                              const std::uint8_t* active, double* output,
                                              double* scaled, std::size_t scaled_elements,
                                              double* unweighted, std::size_t unweighted_elements) {
  return gram(resources, systems, spins, dimension, occupied, coefficients, energies, active,
              output, nullptr, scaled, scaled_elements, unweighted, unweighted_elements);
}

generativeqc_status launch_structured_orthogonalizer(MatrixLibraryResources resources, int systems,
                                                     int dimension, const double* vectors,
                                                     const double* eigenvalues,
                                                     const std::uint8_t* active, double* output,
                                                     std::uint8_t* failed, double* scaled,
                                                     std::size_t scaled_elements) {
  if (!failed) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return gram(resources, systems, 1, dimension, nullptr, vectors, eigenvalues, active, output,
              failed, scaled, scaled_elements, nullptr, 0);
}

generativeqc_status launch_structured_basis(MatrixLibraryResources resources, Operation operation,
                                            int systems, int spins, int dimension,
                                            int direct_dimension, const double* transform,
                                            const double* input, const double* hcore,
                                            const std::uint8_t* active, double* output,
                                            double* scratch, std::size_t scratch_elements,
                                            const std::int32_t* shell_spans) {
  std::size_t elements{};
  if (!shape(resources, systems, spins, dimension, elements) || !transform || !input || !output ||
      direct_dimension != resources.structured_binding_->direct_dimension() ||
      (operation == Operation::FockRight && !hcore))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto states = static_cast<std::size_t>(systems) * spins;
  const auto rectangle = static_cast<std::size_t>(dimension) * direct_dimension;
  const auto matrix = static_cast<std::size_t>(dimension) * dimension;
  const auto direct_matrix = static_cast<std::size_t>(direct_dimension) * direct_dimension;
  const auto count = operation == Operation::DensityLeft ? direct_matrix
                     : operation == Operation::FockRight ? matrix
                                                         : rectangle;
  const auto input_count = operation == Operation::DensityRight ? matrix
                           : operation == Operation::FockLeft   ? direct_matrix
                                                                : rectangle;
  if ((count * states + 127) / 128 > std::numeric_limits<unsigned>::max())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (operation < Operation::DensityRight || operation > Operation::FockRight)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (shell_spans || !resources.structured_binding_->library(operation)) {
    const auto grid = static_cast<unsigned>((count * states + 127) / 128);
    if (operation == Operation::DensityRight)
      launch_transform_density_to_direct_right_kernel(grid, 128, 0, resources.stream_, systems,
                                                      spins, dimension, direct_dimension, transform,
                                                      input, active, output, shell_spans);
    else if (operation == Operation::DensityLeft)
      launch_transform_density_to_direct_left_kernel(grid, 128, 0, resources.stream_, systems,
                                                     spins, dimension, direct_dimension, transform,
                                                     input, active, output, shell_spans);
    else if (operation == Operation::FockLeft)
      launch_transform_direct_fock_left_kernel(grid, 128, 0, resources.stream_, systems, spins,
                                               dimension, direct_dimension, transform, input,
                                               active, output, shell_spans);
    else
      launch_transform_direct_fock_right_kernel(grid, 128, 0, resources.stream_, systems, spins,
                                                dimension, direct_dimension, transform, input,
                                                hcore, active, output, shell_spans);
    return cuda_status(cudaPeekAtLastError());
  }
  if (!disjoint(scratch, scratch_elements, count * states, input, input_count * states, transform,
                rectangle * systems, output, active, systems) ||
      (hcore &&
       overlap(scratch, count * states * sizeof(double), hcore, matrix * systems * sizeof(double))))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto status =
      tensor::cuda::basis_stage(resources.blas_, operation, dimension, direct_dimension, systems,
                                spins, transform, input, scratch);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  publish<<<blocks(count * states), 128, 0, resources.stream_>>>(
      count * states, count, spins,
      operation == Operation::DensityLeft ? direct_dimension : dimension, false, scratch,
      operation == Operation::FockRight ? hcore : nullptr, active, nullptr, output);
  return cuda_status(cudaPeekAtLastError());
}
}  // namespace generativeqc::scf::cuda_execution
