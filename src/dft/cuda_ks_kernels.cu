#include <math_constants.h>

#include <algorithm>
#include <cmath>

#include "dft/cuda_ks_kernels.hpp"
#include "dft/energy_change.hpp"

namespace generativeqc::dft::cuda_ks_detail {
namespace {
__global__ void core_guess_frontier_tie_kernel(std::size_t dimension, int occupied_beta,
                                               const double* energies, int* tied) {
  *tied = 0;
  if (occupied_beta <= 0 || static_cast<std::size_t>(occupied_beta) >= dimension) return;
  const auto first = occupied_beta > 1 ? static_cast<std::size_t>(occupied_beta - 2) : 0U;
  const auto last = min(dimension - 2, static_cast<std::size_t>(occupied_beta));
  for (auto orbital = first; orbital <= last; ++orbital) {
    const double gap = energies[orbital + 1] - energies[orbital];
    if (gap >= 0.0 && gap <= 1e-10) *tied = 1;
  }
}

__global__ void break_core_guess_ties_kernel(std::size_t dimension, double* core_guess) {
  const auto row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < dimension)
    core_guess[row * (dimension + 1)] += 1e-8 * static_cast<double>(row) / dimension;
}

__global__ void orient_core_guess_orbitals_kernel(std::size_t dimension, double* coefficients) {
  const auto column = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= dimension) return;
  double maximum = 0.0;
  for (std::size_t row = 0; row < dimension; ++row)
    maximum = fmax(maximum, fabs(coefficients[column * dimension + row]));
  for (std::size_t pivot = 0; pivot < dimension; ++pivot) {
    const double value = coefficients[column * dimension + pivot];
    if (fabs(value) < maximum * (1.0 - 1e-12)) continue;
    if (value < 0.0)
      for (std::size_t row = 0; row < dimension; ++row)
        coefficients[column * dimension + row] = -coefficients[column * dimension + row];
    return;
  }
}

__global__ void merge_grid_contributions_kernel(std::size_t elements, double* potential,
                                                double* totals, int* error,
                                                const double* nonlocal_potential,
                                                const double* nonlocal_totals,
                                                const int* nonlocal_error) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += std::size_t(blockDim.x) * gridDim.x)
    potential[index] += nonlocal_potential[index];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    totals[0] += nonlocal_totals[0];
    if (*nonlocal_error != 0) *error = *nonlocal_error;
  }
}

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

__global__ void cold_start_fock_kernel(std::size_t elements, const double* physical_fock,
                                       double* previous_proposal, double* proposal, bool damp) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const double value =
        damp ? 0.5 * (physical_fock[index] + previous_proposal[index]) : physical_fock[index];
    previous_proposal[index] = proposal[index] = value;
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
                                  const std::uint8_t* enabled, Scalars* output,
                                  bool retain_orbital_gradient) {
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
  if (retain_orbital_gradient) result.orbital_gradient = output->orbital_gradient;
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

void merge_grid_contributions(cudaStream_t stream, std::size_t elements, double* potential,
                              double* totals, int* error, const double* nonlocal_potential,
                              const double* nonlocal_totals, const int* nonlocal_error) {
  const auto blocks = std::min<std::size_t>((elements + 127) / 128, 65535);
  merge_grid_contributions_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      elements, potential, totals, error, nonlocal_potential, nonlocal_totals, nonlocal_error);
}

void core_guess_frontier_tie(cudaStream_t stream, std::size_t dimension, int occupied_beta,
                             const double* energies, int* tied) {
  core_guess_frontier_tie_kernel<<<1, 1, 0, stream>>>(dimension, occupied_beta, energies, tied);
}

void break_core_guess_ties(cudaStream_t stream, std::size_t dimension, double* core_guess) {
  break_core_guess_ties_kernel<<<static_cast<unsigned>((dimension + 127) / 128), 128, 0, stream>>>(
      dimension, core_guess);
}

void orient_core_guess_orbitals(cudaStream_t stream, std::size_t dimension, double* coefficients) {
  orient_core_guess_orbitals_kernel<<<static_cast<unsigned>((dimension + 127) / 128), 128, 0,
                                      stream>>>(dimension, coefficients);
}

void cold_start_fock(cudaStream_t stream, std::size_t elements, const double* physical_fock,
                     double* previous_proposal, double* proposal, bool damp) {
  const auto blocks = std::min<std::size_t>((elements + 127) / 128, 65535);
  cold_start_fock_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      elements, physical_fock, previous_proposal, proposal, damp);
}

void stabilize_uks_proposal(cudaStream_t stream, std::size_t n, const double* overlap,
                            const double* occupied_projector, const std::uint8_t* enabled,
                            double* proposal_fock) {
  const auto blocks = std::min<std::size_t>((2 * n * n + 127) / 128, 65535);
  stabilize_uks_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      n * n, overlap, occupied_projector, enabled, proposal_fock);
}

__global__ void orbital_gradient_kernel(std::size_t matrix, const double* residual,
                                        const std::uint8_t* enabled, Scalars* output) {
  if (enabled != nullptr && *enabled == 0) return;
  __shared__ double partials[kDiagnosticThreads];
  double sum = 0.0;
  for (std::size_t index = threadIdx.x; index < matrix; index += blockDim.x)
    sum += residual[index] * residual[index];
  partials[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x != 0) return;
  double total = 0.0;
  for (unsigned index = 0; index < blockDim.x; ++index) total += partials[index];
  output->orbital_gradient = sqrt(0.5 * total);
}

void orbital_gradient(cudaStream_t stream, std::size_t n, const double* transformed_residual,
                      const std::uint8_t* enabled, Scalars* output) {
  orbital_gradient_kernel<<<1, kDiagnosticThreads, 0, stream>>>(n * n, transformed_residual,
                                                                enabled, output);
}

void diagnostics(cudaStream_t stream, std::size_t n, unsigned spins, const double* density,
                 const double* proposal, const double* residual, const double* hcore,
                 const double* overlap, const double* coulomb, const double* exchange,
                 double exchange_coefficient, const double* range_exchange,
                 double range_exchange_coefficient, const double* xc_totals, const int* xc_error,
                 const int* jk_error, const int* range_jk_error, const int* nonlocal_domain_error,
                 const int* nonlocal_pair_error, const int* solver_info,
                 const std::uint8_t* enabled, Scalars* output, bool retain_orbital_gradient) {
  diagnostic_kernel<<<1, kDiagnosticThreads, 0, stream>>>(
      n * n, spins, density, proposal, residual, hcore, overlap, coulomb, exchange,
      exchange_coefficient, range_exchange, range_exchange_coefficient, xc_totals, xc_error,
      jk_error, range_jk_error, nonlocal_domain_error, nonlocal_pair_error, solver_info, enabled,
      output, retain_orbital_gradient);
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
namespace {
using vv10_rebase::Error;
using vv10_rebase::Key;
using vv10_rebase::Transaction;

__device__ void rebase_finite_vote(Transaction* tx, int bad) {
  const int any = __syncthreads_or(bad);
  if (threadIdx.x == 0 && any && tx->ready) vv10_rebase::refuse(*tx, Error::Nonfinite);
  __syncthreads();
}

__global__ void rebase_refuse_kernel(Error reason, RebaseStorage s) {
  vv10_rebase::refuse(*s.transaction, reason);
  s.transaction->bank_count = 0;
}

__global__ void rebase_reset_kernel(unsigned history, RebaseStorage s) {
  if (threadIdx.x < history) s.tags[threadIdx.x] = {};
  if (threadIdx.x + 1 < history) s.bank_tags[threadIdx.x] = {};
  if (threadIdx.x == 0) {
    *s.transaction = {};
    s.transaction->enabled = 1;
  }
}

__global__ void rebase_density_kernel(std::size_t elements, unsigned history, const double* density,
                                      const std::uint8_t* enabled, const std::uint32_t* count,
                                      const std::uint32_t* head, Key key, RebaseStorage s) {
  auto* tx = s.transaction;
  if (!tx->enabled || !*enabled) return;
  if (!vv10_rebase::valid_window(history, *count, *head)) {
    if (threadIdx.x == 0) {
      vv10_rebase::refuse(*tx, Error::InvalidWindow);
      tx->enabled = 0;
    }
    return;
  }
  const unsigned slot = *head;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x)
    s.density[slot * elements + i] = density[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    s.tags[slot] = {key, 1};
    ++tx->work.density_writes;
  }
}

__global__ void rebase_snapshot_kernel(std::size_t elements, unsigned history, const double* fock,
                                       const double* residual, const double* old_potential,
                                       const std::uint32_t* count, const std::uint32_t* head,
                                       Key key, RebaseStorage s) {
  auto* tx = s.transaction;
  if (threadIdx.x == 0) vv10_rebase::prepare_snapshot(*tx, history, *count, *head, s.tags, key);
  __syncthreads();
  if (!tx->ready) return;
  int bad = 0;
  for (unsigned bank = 0; bank < tx->bank_count; ++bank) {
    const unsigned source = vv10_rebase::bank_source_slot(history, *count, *head, bank);
    for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
      const auto src = source * elements + i, dst = bank * elements + i;
      const double d = s.density[src], f = fock[src], r = residual[src];
      bad |= !isfinite(d) || !isfinite(f) || !isfinite(r);
      s.bank_density[dst] = d;
      s.bank_fock[dst] = f;
      s.bank_residual[dst] = r;
    }
    if (threadIdx.x == 0) s.bank_tags[bank] = s.tags[source];
  }
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    const double value = old_potential ? old_potential[i] : 0.0;
    bad |= !isfinite(value);
    s.potential[i] = value;
  }
  rebase_finite_vote(tx, bad);
  if (threadIdx.x == 0 && tx->ready) ++tx->work.snapshots;
}

__global__ void rebase_target_kernel(std::size_t elements, unsigned history,
                                     const double* potential, const std::uint32_t* count,
                                     const std::uint32_t* head, Key key, RebaseStorage s) {
  auto* tx = s.transaction;
  if (threadIdx.x == 0) vv10_rebase::prepare_target(*tx, history, *count, *head, s.tags, key);
  __syncthreads();
  if (!tx->ready) return;
  int bad = 0;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    const double old = s.potential[i], next = potential[i];
    const double delta = next - old;
    bad |= !isfinite(old) || !isfinite(next) || !isfinite(delta);
    s.potential[i] = delta;
  }
  rebase_finite_vote(tx, bad);
}

__global__ void rebase_finite_kernel(std::size_t elements, const double* values, RebaseStorage s) {
  if (!s.transaction->ready) return;
  int bad = 0;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) bad |= !isfinite(values[i]);
  rebase_finite_vote(s.transaction, bad);
}

__global__ void rebase_accumulate_kernel(std::size_t elements, unsigned bank, const double* product,
                                         bool subtract, bool update_fock, RebaseStorage s) {
  auto* tx = s.transaction;
  if (!tx->ready) return;
  int bad = 0;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    const auto index = bank * elements + i;
    const double term = product[i], old = s.bank_residual[index];
    double value;
    bad |= !vv10_rebase::accumulate_finite(old, term, subtract, value);
    s.bank_residual[index] = value;
    if (update_fock) {
      const double old_fock = s.bank_fock[index], delta = s.potential[i];
      double fock;
      bad |= !vv10_rebase::accumulate_finite(old_fock, delta, false, fock);
      s.bank_fock[index] = fock;
    }
  }
  rebase_finite_vote(tx, bad);
}

__global__ void rebase_restore_kernel(std::size_t elements, unsigned history, double* fock,
                                      double* residual, RebaseStorage s) {
  auto* tx = s.transaction;
  if (!tx->ready) return;
  if (threadIdx.x == 0) {
    for (unsigned bank = 0; bank < tx->bank_count; ++bank)
      if (!s.bank_tags[bank].valid || !vv10_rebase::same_key(s.bank_tags[bank].key, tx->bank_key)) {
        vv10_rebase::refuse(*tx, Error::AnchorMismatch);
        break;
      }
  }
  __syncthreads();
  if (!tx->ready) return;
  for (unsigned bank = 0; bank < tx->bank_count; ++bank) {
    const unsigned slot = vv10_rebase::restored_slot(history, tx->bank_count, bank);
    for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
      const auto src = bank * elements + i, dst = slot * elements + i;
      s.density[dst] = s.bank_density[src];
      fock[dst] = s.bank_fock[src];
      residual[dst] = s.bank_residual[src];
    }
    if (threadIdx.x == 0) s.tags[slot] = {tx->target_key, 1};
  }
}

__global__ void rebase_publish_kernel(unsigned history, std::uint32_t* count, std::uint32_t* head,
                                      RebaseStorage s) {
  auto& tx = *s.transaction;
  if (!tx.ready) {
    if (tx.error == Error::None) vv10_rebase::refuse(tx, Error::GramNonfinite);
    return;
  }
  if (*count != 1 || *head != 1 || tx.proposed_count != tx.bank_count + 1 ||
      tx.proposed_count > history || !s.tags[0].valid ||
      !vv10_rebase::same_key(s.tags[0].key, tx.target_key)) {
    vv10_rebase::refuse(tx, Error::TargetWindow);
    return;
  }
  // All matrix/Gram writes are earlier launches on this same stream.
  *count = tx.proposed_count;
  *head = tx.proposed_head;
  ++tx.work.successful_rebases;
  tx.work.entries_transferred += tx.bank_count;
  tx.bank_count = tx.ready = 0;
}
}  // namespace

void rebase_refuse(cudaStream_t stream, Error reason, RebaseStorage s) {
  rebase_refuse_kernel<<<1, 1, 0, stream>>>(reason, s);
}
void rebase_reset(cudaStream_t stream, unsigned history, RebaseStorage s) {
  rebase_reset_kernel<<<1, 64, 0, stream>>>(history, s);
}
void rebase_record_density(cudaStream_t stream, std::size_t elements, unsigned history,
                           const double* density, const std::uint8_t* enabled,
                           const std::uint32_t* count, const std::uint32_t* head, Key key,
                           RebaseStorage s) {
  rebase_density_kernel<<<1, 256, 0, stream>>>(elements, history, density, enabled, count, head,
                                               key, s);
}
void rebase_snapshot(cudaStream_t stream, std::size_t elements, unsigned history,
                     const double* fock, const double* residual, const double* potential,
                     const std::uint32_t* count, const std::uint32_t* head, Key key,
                     RebaseStorage s) {
  rebase_snapshot_kernel<<<1, 256, 0, stream>>>(elements, history, fock, residual, potential, count,
                                                head, key, s);
}
void rebase_prepare_target(cudaStream_t stream, std::size_t elements, unsigned history,
                           const double* potential, const std::uint32_t* count,
                           const std::uint32_t* head, Key key, RebaseStorage s) {
  rebase_target_kernel<<<1, 256, 0, stream>>>(elements, history, potential, count, head, key, s);
}
void rebase_check_finite(cudaStream_t stream, std::size_t elements, const double* values,
                         RebaseStorage s) {
  rebase_finite_kernel<<<1, 256, 0, stream>>>(elements, values, s);
}
void rebase_accumulate(cudaStream_t stream, std::size_t elements, unsigned bank,
                       const double* product, bool subtract, bool update_fock, RebaseStorage s) {
  rebase_accumulate_kernel<<<1, 256, 0, stream>>>(elements, bank, product, subtract, update_fock,
                                                  s);
}
void rebase_restore(cudaStream_t stream, std::size_t elements, unsigned history, double* fock,
                    double* residual, RebaseStorage s) {
  rebase_restore_kernel<<<1, 256, 0, stream>>>(elements, history, fock, residual, s);
}
void rebase_publish(cudaStream_t stream, unsigned history, std::uint32_t* count,
                    std::uint32_t* head, RebaseStorage s) {
  rebase_publish_kernel<<<1, 1, 0, stream>>>(history, count, head, s);
}

}  // namespace generativeqc::dft::cuda_ks_detail
