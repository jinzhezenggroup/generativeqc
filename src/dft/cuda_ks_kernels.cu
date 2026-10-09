#include <math_constants.h>

#include <algorithm>
#include <cmath>

#include "dft/cuda_ks_kernels.hpp"
#include "dft/energy_change.hpp"

namespace generativeqc::dft::cuda_ks_detail {
namespace {
__global__ void reset_control_kernel(unsigned spins, int occupied_alpha, int occupied_beta,
                                     Control* control, std::uint8_t* enabled,
                                     std::uint8_t* spin_enabled) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  *control = {};
  control->previous_energy = CUDART_INF;
  control->active = 1;
  *enabled = 1;
  spin_enabled[0] = static_cast<std::uint8_t>(occupied_alpha > 0);
  if (spins == 2) spin_enabled[1] = static_cast<std::uint8_t>(occupied_beta > 0);
}

__global__ void fock_kernel(std::size_t matrix, unsigned spins, const double* hcore,
                            const double* coulomb, const double* exchange,
                            double exchange_coefficient, const double* range_exchange,
                            double range_exchange_coefficient, const double* potential,
                            const std::uint8_t* enabled, double* fock) {
  if (enabled != nullptr && *enabled == 0) return;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * matrix;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const double exact = exchange != nullptr ? exchange_coefficient * exchange[i] : 0.0;
    const double range =
        range_exchange != nullptr ? range_exchange_coefficient * range_exchange[i] : 0.0;
    fock[i] = hcore[i % matrix] + coulomb[i % matrix] + exact + range + potential[i];
  }
}

__global__ void stabilize_uks_kernel(std::size_t matrix, const double* overlap,
                                     const double* occupied_projector, const std::uint8_t* enabled,
                                     double* proposal_fock) {
  if (enabled != nullptr && *enabled == 0) return;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * matrix;
       i += std::size_t(blockDim.x) * gridDim.x)
    proposal_fock[i] += 0.1 * (overlap[i % matrix] - occupied_projector[i]);
}

__device__ void accumulate_diagnostic_trace(double value, double& sum, double& correction) {
  const double next = __dadd_rn(sum, value);
  correction += fabs(sum) >= fabs(value) ? (sum - next) + value : (value - next) + sum;
  sum = next;
}

// One fixed block owns the complete reduction. This uses bounded shared storage,
// needs no arena allocation, and has no arrival-order-dependent floating atomics.
// Both words of each compensated trace survive the ordered lane merge.
constexpr unsigned kDiagnosticThreads = 256;
struct DiagnosticPartial {
  double energy[3], correction[3];
  double error2[2], change2[2], electrons[2];
  double maximum_residual;
};

__global__ void diagnostic_kernel(std::size_t matrix, unsigned spins, const double* density,
                                  const double* proposal, const double* residual,
                                  const double* hcore, const double* overlap, const double* coulomb,
                                  const double* exchange, double exchange_coefficient,
                                  const double* range_exchange, double range_exchange_coefficient,
                                  const double* xc_totals, const int* xc_error, const int* jk_error,
                                  const int* range_jk_error, const int* nonlocal_domain_error,
                                  const int* nonlocal_pair_error, const int* solver_info,
                                  const std::uint8_t* enabled, Scalars* output) {
  // The mask is immutable for this launch, so inactive blocks exit collectively.
  if (enabled != nullptr && *enabled == 0) return;
  __shared__ DiagnosticPartial partials[kDiagnosticThreads];
  DiagnosticPartial lane{};
  // Balanced contiguous partitions retain matrix source order within each spin.
  // Quotient/remainder bounds avoid overflowing matrix * lane for large inputs.
  const std::size_t quotient = matrix / blockDim.x;
  const std::size_t remainder = matrix % blockDim.x;
  const std::size_t begin =
      quotient * threadIdx.x + (threadIdx.x < remainder ? threadIdx.x : remainder);
  const std::size_t end = begin + quotient + (threadIdx.x < remainder ? 1 : 0);
  for (unsigned spin = 0; spin < spins; ++spin) {
    const auto offset = spin * matrix;
    for (std::size_t index = begin; index < end; ++index) {
      const double density_value = density[offset + index];
      const double change = proposal[offset + index] - density_value;
      accumulate_diagnostic_trace(density_value * hcore[index], lane.energy[0], lane.correction[0]);
      accumulate_diagnostic_trace(0.5 * density_value * coulomb[index], lane.energy[1],
                                  lane.correction[1]);
      if (exchange != nullptr)
        accumulate_diagnostic_trace(
            0.5 * density_value * exchange_coefficient * exchange[offset + index], lane.energy[2],
            lane.correction[2]);
      if (range_exchange != nullptr)
        accumulate_diagnostic_trace(
            0.5 * density_value * range_exchange_coefficient * range_exchange[offset + index],
            lane.energy[2], lane.correction[2]);
      lane.electrons[spin] += density_value * overlap[index];
      lane.error2[spin] += residual[offset + index] * residual[offset + index];
      lane.maximum_residual = fmax(lane.maximum_residual, fabs(residual[offset + index]));
      lane.change2[spin] += change * change;
    }
  }
  partials[threadIdx.x] = lane;
  __syncthreads();
  if (threadIdx.x != 0) return;

  DiagnosticPartial total{};
  for (unsigned index = 0; index < blockDim.x; ++index) {
    const auto& part = partials[index];
    for (unsigned term = 0; term < 3; ++term) {
      accumulate_diagnostic_trace(part.energy[term], total.energy[term], total.correction[term]);
      accumulate_diagnostic_trace(part.correction[term], total.energy[term],
                                  total.correction[term]);
    }
    for (unsigned spin = 0; spin < spins; ++spin) {
      total.electrons[spin] += part.electrons[spin];
      total.error2[spin] += part.error2[spin];
      total.change2[spin] += part.change2[spin];
    }
    total.maximum_residual = fmax(total.maximum_residual, part.maximum_residual);
  }
  Scalars result{};
  result.failure = (*xc_error != 0 ? 1 : 0) | (*jk_error != 0 ? 2 : 0) |
                   (range_jk_error != nullptr && *range_jk_error != 0 ? 16 : 0) |
                   (nonlocal_domain_error != nullptr && *nonlocal_domain_error != 0 ? 32 : 0) |
                   (nonlocal_pair_error != nullptr && *nonlocal_pair_error != 0 ? 64 : 0);
  result.xc = xc_totals[0];
  result.grid_electrons[0] = xc_totals[1];
  result.grid_electrons[1] = xc_totals[2];
  result.one_electron = total.energy[0] + total.correction[0];
  result.hartree = total.energy[1] + total.correction[1];
  result.exact_exchange = total.energy[2] + total.correction[2];
  // Do not round the individual traces before summing the electronic energy.
  // Their low words can decide a strict energy gate even at stationary density.
  for (unsigned term = 0; term < 3; ++term) {
    accumulate_diagnostic_trace(total.energy[term], result.electronic_energy,
                                result.electronic_energy_correction);
    accumulate_diagnostic_trace(total.correction[term], result.electronic_energy,
                                result.electronic_energy_correction);
  }
  accumulate_diagnostic_trace(result.xc, result.electronic_energy,
                              result.electronic_energy_correction);
  result.maximum_residual = total.maximum_residual;
  for (unsigned spin = 0; spin < spins; ++spin) {
    const double error2 = total.error2[spin], change2 = total.change2[spin];
    const double electrons = total.electrons[spin];
    if (solver_info[spin] != 0) result.failure |= 4;
    if (!isfinite(error2) || !isfinite(change2) || !isfinite(electrons)) result.failure |= 8;
    result.residual = fmax(result.residual, sqrt(error2 / matrix));
    result.density_change = fmax(result.density_change, sqrt(change2 / matrix));
    result.residual_rms += error2 / matrix / spins;
    result.density_rms += change2 / matrix / spins;
    if (spins == 1)
      result.electrons[0] = result.electrons[1] = 0.5 * electrons;
    else
      result.electrons[spin] = electrons;
  }
  result.residual_rms = sqrt(result.residual_rms);
  result.density_rms = sqrt(result.density_rms);
  if (!isfinite(result.one_electron) || !isfinite(result.hartree) ||
      !isfinite(result.exact_exchange) || !isfinite(result.xc))
    result.failure |= 8;
  *output = result;
}

__global__ void advance_kernel(std::size_t matrix, unsigned spins, double nuclear_repulsion,
                               int occupied_alpha, int occupied_beta, double energy_tolerance,
                               double density_tolerance, unsigned max_iterations, bool warm_updates,
                               Scalars* current, Control* control, const double* proposal,
                               double* density, double* warm, std::uint8_t* enabled,
                               std::uint8_t* spin_enabled) {
  __shared__ int active_at_entry;
  __shared__ int copy_density;
  __shared__ int publish_warm;
  // All threads must observe one entry decision before thread zero can make
  // this iteration terminal; late warps still owe their warm-density copies.
  if (threadIdx.x == 0) active_at_entry = control->active;
  __syncthreads();
  if (active_at_entry == 0) return;
  if (threadIdx.x == 0) {
    copy_density = 0;
    publish_warm = 0;
    const auto iteration = control->iterations + 1U;
    const double energy = nuclear_repulsion + current->one_electron + current->hartree +
                          current->exact_exchange + current->xc;
    const double change = detail::electronic_energy_change(
        current->electronic_energy, current->electronic_energy_correction, control->previous_energy,
        control->previous_energy_correction);
    current->energy_change = change;
    control->iterations = iteration;
    bool failed = current->failure != 0 || !isfinite(energy);
    if (fabs(current->electrons[0] - occupied_alpha) > 1e-8 ||
        fabs(current->electrons[1] - occupied_beta) > 1e-8)
      failed = true;
    if (failed) {
      control->failed = 1;
      control->converged = 0;
      control->active = 0;
    } else {
      const bool converged = iteration > 1 && change < energy_tolerance &&
                             current->density_change < density_tolerance &&
                             current->residual < fmin(1e-9, density_tolerance) &&
                             current->maximum_residual < fmin(1e-9, density_tolerance);
      control->converged = converged ? 1 : 0;
      control->active = (!converged && iteration < max_iterations) ? 1 : 0;
      copy_density = control->active;
      publish_warm = control->converged && warm_updates;
      control->previous_energy = current->electronic_energy;
      control->previous_energy_correction = current->electronic_energy_correction;
    }
    *enabled = static_cast<std::uint8_t>(control->active != 0);
    spin_enabled[0] = static_cast<std::uint8_t>(control->active != 0 && occupied_alpha > 0);
    if (spins == 2)
      spin_enabled[1] = static_cast<std::uint8_t>(control->active != 0 && occupied_beta > 0);
  }
  __syncthreads();
  const std::size_t elements = spins * matrix;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    if (publish_warm) warm[i] = density[i];
    if (copy_density) density[i] = proposal[i];
  }
}
}  // namespace

void reset_control(cudaStream_t stream, unsigned spins, int occupied_alpha, int occupied_beta,
                   Control* control, std::uint8_t* enabled, std::uint8_t* spin_enabled) {
  reset_control_kernel<<<1, 1, 0, stream>>>(spins, occupied_alpha, occupied_beta, control, enabled,
                                            spin_enabled);
}

void assemble_fock(cudaStream_t stream, std::size_t n, unsigned spins, const double* hcore,
                   const double* coulomb, const double* exchange, double exchange_coefficient,
                   const double* range_exchange, double range_exchange_coefficient,
                   const double* potential, const std::uint8_t* enabled, double* fock) {
  const auto blocks = std::min<std::size_t>((spins * n * n + 127) / 128, 65535);
  fock_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      n * n, spins, hcore, coulomb, exchange, exchange_coefficient, range_exchange,
      range_exchange_coefficient, potential, enabled, fock);
}

void stabilize_uks_proposal(cudaStream_t stream, std::size_t n, const double* overlap,
                            const double* occupied_projector, const std::uint8_t* enabled,
                            double* proposal_fock) {
  const auto blocks = std::min<std::size_t>((2 * n * n + 127) / 128, 65535);
  stabilize_uks_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      n * n, overlap, occupied_projector, enabled, proposal_fock);
}

void diagnostics(cudaStream_t stream, std::size_t n, unsigned spins, const double* density,
                 const double* proposal, const double* residual, const double* hcore,
                 const double* overlap, const double* coulomb, const double* exchange,
                 double exchange_coefficient, const double* range_exchange,
                 double range_exchange_coefficient, const double* xc_totals, const int* xc_error,
                 const int* jk_error, const int* range_jk_error, const int* nonlocal_domain_error,
                 const int* nonlocal_pair_error, const int* solver_info,
                 const std::uint8_t* enabled, Scalars* output) {
  diagnostic_kernel<<<1, kDiagnosticThreads, 0, stream>>>(
      n * n, spins, density, proposal, residual, hcore, overlap, coulomb, exchange,
      exchange_coefficient, range_exchange, range_exchange_coefficient, xc_totals, xc_error,
      jk_error, range_jk_error, nonlocal_domain_error, nonlocal_pair_error, solver_info, enabled,
      output);
}

void advance(cudaStream_t stream, std::size_t n, unsigned spins, double nuclear_repulsion,
             int occupied_alpha, int occupied_beta, double energy_tolerance,
             double density_tolerance, unsigned max_iterations, bool warm_updates, Scalars* current,
             Control* control, const double* proposal, double* density, double* warm,
             std::uint8_t* enabled, std::uint8_t* spin_enabled) {
  advance_kernel<<<1, 256, 0, stream>>>(n * n, spins, nuclear_repulsion, occupied_alpha,
                                        occupied_beta, energy_tolerance, density_tolerance,
                                        max_iterations, warm_updates, current, control, proposal,
                                        density, warm, enabled, spin_enabled);
}
}  // namespace generativeqc::dft::cuda_ks_detail
