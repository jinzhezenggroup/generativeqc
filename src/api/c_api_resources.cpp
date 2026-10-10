#include <algorithm>
#include <cstdio>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "dft/grid.hpp"
#include "dft/scf_diagnostic.hpp"
#include "runtime/residency_boundaries.hpp"
#include "runtime/residency_observer.hpp"
#include "runtime/resource_ledger.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/density_fitting.hpp"
#include "scf/preliminary_guess.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#include "dft/cuda_xc.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda_direct_jk.hpp"
#endif

extern "C" {

/** Read immutable source tags from this actual library, with no CUDA or heap
 * work. Null is an unknown tag, not an inferred role/payload classification. */
const char* generativeqc_residency_boundary_name_v1(std::uint64_t category, std::uint64_t value) {
  return generativeqc::runtime::residency_boundary_name(category, value);
}

/** Bind a private observation-only callback on the current submitting thread.
 * The caller keeps its callback/code/context alive through unbind. No CUDA or
 * allocation is performed, and a second observer cannot replace the first. */
int generativeqc_residency_observer_bind_v1(generativeqc::runtime::ResidencyObserver callback,
                                            void* context) {
  return generativeqc::runtime::bind_residency_observer(callback, context);
}

/** Only the matching owner can detach; retain dispatch errors for the receipt.
 * Exceptions or recursive callbacks are observation failures, not solver errors. */
int generativeqc_residency_observer_unbind_v1(generativeqc::runtime::ResidencyObserver callback,
                                              void* context, std::uint64_t* errors) {
  return generativeqc::runtime::unbind_residency_observer(callback, context, errors);
}

/** Private prepared-request ledger. Creating it performs no CUDA operation. */
void* generativeqc_resource_ledger_create_v1(std::size_t bytes, int device) {
  if (device < 0) return nullptr;
  try {
    return new std::shared_ptr<generativeqc::runtime::DeviceResourceLedger>(
        std::make_shared<generativeqc::runtime::DeviceResourceLedger>(
            generativeqc::runtime::DeviceResourceLedger{bytes, device}));
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

void generativeqc_resource_ledger_destroy_v1(void* handle) {
  delete static_cast<std::shared_ptr<generativeqc::runtime::DeviceResourceLedger>*>(handle);
}

/** Bind only within an active, synchronous observation. Reset the peak to
 * retained live state so warm replay measurements include pre-existing caches.
 */
int generativeqc_resource_ledger_bind_v1(void* handle) {
  using namespace generativeqc::runtime;
  if (!cpu_resource_observation.active || active_device_resource_ledger || handle == nullptr)
    return 1;
  auto ledger = *static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle);
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  if (ledger->active) return 1;
  ledger->active = true;
  ledger->peak = ledger->live;
  ledger->allocations = 0;
  ledger->rejected = 0;
  ledger->requested_bytes = 0;
  ledger->requested_bytes_overflow = false;
  active_device_resource_ledger = std::move(ledger);
  return 0;
}

int generativeqc_resource_ledger_read_v1(void* handle, std::uint64_t* values) {
  if (handle == nullptr || values == nullptr) return 1;
  const auto& ledger =
      *static_cast<std::shared_ptr<generativeqc::runtime::DeviceResourceLedger>*>(handle);
  std::lock_guard<std::mutex> lock(generativeqc::runtime::device_resource_mutex);
  values[0] = ledger->live;
  values[1] = ledger->peak;
  values[2] = ledger->allocations;
  values[3] = ledger->rejected;
  return 0;
}

/** Preserve the four-field v1 ABI; v2 adds successful cumulative requested bytes.
 * All five counters share one locked snapshot. An overflow is unobservable,
 * not a wrapped zero-byte replay or a reason to fail numerical execution. */
int generativeqc_resource_ledger_read_v2(void* handle, std::uint64_t* values, std::size_t count) {
  if (handle == nullptr || values == nullptr || count != 5) return 1;
  const auto& ledger =
      *static_cast<std::shared_ptr<generativeqc::runtime::DeviceResourceLedger>*>(handle);
  std::lock_guard<std::mutex> lock(generativeqc::runtime::device_resource_mutex);
  if (ledger->requested_bytes_overflow) return 1;
  values[0] = ledger->live;
  values[1] = ledger->peak;
  values[2] = ledger->allocations;
  values[3] = ledger->rejected;
  values[4] = ledger->requested_bytes;
  return 0;
}

/** Snapshot retained owners and reserve journal slots before a replay begins.
 * A capture handle survives ledger-handle destruction and cached-buffer frees.
 * This is observation-only storage; it never selects a resource/solver route. */
void* generativeqc_resource_journal_create_v1(void* handle, std::size_t capacity) {
  using namespace generativeqc::runtime;
  if (handle == nullptr || capacity == 0 || capacity > (1u << 20)) return nullptr;
  const auto& ledger = *static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle);
  try {
    auto journal = std::make_shared<DeviceAllocationJournal>();
    journal->limit = capacity;
    journal->events.reserve(capacity);
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    if (ledger->active || ledger->journal) return nullptr;
    const auto owners =
        std::count_if(device_allocation_owners.begin(), device_allocation_owners.end(),
                      [&](const auto& entry) { return entry.second.ledger == ledger; });
    if (static_cast<std::size_t>(owners) > capacity) return nullptr;
    journal->initial.reserve(owners);
    for (const auto& entry : device_allocation_owners) {
      if (entry.second.ledger == ledger)
        journal->initial.push_back({0, entry.second.generation, entry.second.bytes});
    }
    std::sort(
        journal->initial.begin(), journal->initial.end(),
        [](const auto& first, const auto& second) { return first.generation < second.generation; });
    auto* capture = new std::shared_ptr<DeviceAllocationJournal>(journal);
    ledger->journal = std::move(journal);
    return capture;
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

/** Return initial-owner count, event count, dropped events and recording flag.
 * Records are triples (kind, allocation generation, requested bytes), initial
 * owners first. A null/zero record buffer queries the required size. A short
 * buffer returns 2 and leaves its records untouched; state is always atomic. */
int generativeqc_resource_journal_read_v1(void* handle, std::uint64_t* records, std::size_t count,
                                          std::uint64_t* state) {
  using namespace generativeqc::runtime;
  if (handle == nullptr || state == nullptr || (records == nullptr && count != 0)) return 1;
  const auto& journal = *static_cast<std::shared_ptr<DeviceAllocationJournal>*>(handle);
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  state[0] = journal->initial.size();
  state[1] = journal->events.size();
  state[2] = journal->dropped;
  state[3] = journal->recording;
  if (records == nullptr) return 0;
  if (count < 3 * (journal->initial.size() + journal->events.size())) return 2;
  std::size_t index = 0;
  const auto copy = [&](const auto& events) {
    for (const auto& event : events) {
      records[index++] = event.kind;
      records[index++] = event.generation;
      records[index++] = event.bytes;
    }
  };
  copy(journal->initial);
  copy(journal->events);
  return 0;
}

/** Stop recording without changing allocation ownership or CUDA visibility. */
void generativeqc_resource_journal_destroy_v1(void* handle) {
  using namespace generativeqc::runtime;
  if (handle == nullptr) return;
  auto* capture = static_cast<std::shared_ptr<DeviceAllocationJournal>*>(handle);
  {
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    (*capture)->recording = false;
  }
  delete capture;
}

/** Allocation-inventory contract implemented by resources_hf.py. Increment
 * when changing CPU allocation shapes/lifetimes beyond those documented
 * bounds. The current object-capacity allowance is validated on LP64 hosts.
 */
int generativeqc_cpu_resource_inventory_version_v1() { return sizeof(void*) == 8 ? 1 : 0; }

/** KS v1 host bounds use LP64 metadata and a 128-byte iteration-record
 * allowance. Keep this separate from HF's derivative-inclusive inventory. */
int generativeqc_ks_resource_inventory_version_v1() {
  return sizeof(void*) == 8 && sizeof(generativeqc::dft::ScfIteration) <= 128 ? 1 : 0;
}

/** Pure shape bridge to allocator-owned KS/XC and common #202 direct-J
 * layouts. The XC slot combines the borrowed-grid arena and its retained
 * MolecularGrid owner. It never constructs a grid, density or CUDA context. */
int generativeqc_resource_ks_cuda_v1(std::size_t nao, std::size_t atoms, std::size_t shells,
                                     std::size_t primitives, std::size_t points,
                                     std::size_t diis_history, std::size_t spins, std::size_t pbe,
                                     std::size_t tile_points, std::uint64_t* output,
                                     std::size_t count) {
  if (!output || count != 3 || diis_history > 64 || (spins != 1 && spins != 2) || pbe > 1) return 1;
#if GENERATIVEQC_HAS_CUDA
  try {
    const auto state = generativeqc::dft::cuda_ks_state_bytes(nao, spins, diis_history);
    const auto xc = generativeqc::dft::cuda_xc_layout_shape(
        atoms, primitives, nao, points, pbe != 0, spins == 2, tile_points, false,
        generativeqc::dft::CudaXcAoPrecision::Fp64, 1.0, 1.0, true);
    // The shared grid retains xyz, partition weights and atomic measures.
    // A nonborrowed XC arena owns only the first four arrays and cannot stand
    // in for this five-array lifetime, even when XC reads only points/weights.
    const auto xc_and_grid = generativeqc::runtime::size_add(
        xc.device_bytes, generativeqc::dft::cuda_resident_grid_bytes(points));
    const auto direct =
        generativeqc::scf::cuda_direct_coulomb_device_bytes(1, nao, atoms, shells, primitives);
    output[0] = state;
    output[1] = xc_and_grid;
    output[2] = direct;
    return 0;
  } catch (...) {
    return 1;
  }
#else
  return 2;
#endif
}

/** Shape-only opaque matrix-provider reservation, separate from explicit arenas.
 * This additive query keeps the KS v1 three-output numeric inventory unchanged.
 * It never creates a CUDA context or provider handle. */
int generativeqc_resource_ks_matrix_provider_cuda_v1(std::size_t nao, std::uint64_t* bytes) {
  if (!bytes || !nao || nao > static_cast<std::size_t>(std::numeric_limits<int>::max())) return 1;
#if GENERATIVEQC_HAS_CUDA
  *bytes = generativeqc::scf::cuda_execution::MatrixLibraryOwner::provider_allowance(
      static_cast<int>(nao));
  return 0;
#else
  return 2;
#endif
}

/** Separate additive bridge preserves the KS v1 three-output ABI. For the
 * device-fused route this is the quadrature-preparation peak: bounded scratch
 * plus the full-grid points/weights that remain resident after preparation.
 * The KS v1 XC slot deliberately remains a combined XC+grid persistent bound,
 * so the legacy three-output planner neither undercounts nor double-charges
 * the retained grid. */
int generativeqc_resource_quadrature_cuda_v1(std::size_t atoms, std::size_t points,
                                             std::uint64_t* bytes) {
  if (!bytes) return 1;
#if GENERATIVEQC_HAS_CUDA
  try {
    *bytes = generativeqc::dft::cuda_quadrature_bytes(atoms, points);
    return 0;
  } catch (...) {
    return 1;
  }
#else
  return 2;
#endif
}

/** Private execution-scope bridge. The selected CPU plan can explicitly cap
 * fleet concurrency as well as collect samples. No device query/allocation.
 */
int generativeqc_resource_tracking_begin_v1(unsigned cpu_worker_limit) {
  auto& observation = generativeqc::runtime::cpu_resource_observation;
  if (observation.active || cpu_worker_limit > 1) return 1;
  observation = {true, 0, 0, cpu_worker_limit};
  return 0;
}

/** Return the observed scope even after failed solves; zero samples means
 * unavailable data, never evidence of a zero-byte scientific execution.
 */
int generativeqc_resource_tracking_end_v1(std::uint64_t* peak, std::uint64_t* samples) {
  auto& observation = generativeqc::runtime::cpu_resource_observation;
  if (peak == nullptr || samples == nullptr || !observation.active) return 1;
  *peak = observation.peak_bytes;
  *samples = observation.samples;
  observation.active = false;
  observation.cpu_worker_limit = 0;
  if (generativeqc::runtime::active_device_resource_ledger) {
    std::lock_guard<std::mutex> lock(generativeqc::runtime::device_resource_mutex);
    generativeqc::runtime::active_device_resource_ledger->active = false;
  }
  generativeqc::runtime::active_device_resource_ledger.reset();
  return 0;
}

/** Read the optional CUDA arena samples before ending the synchronous scope. */
int generativeqc_resource_tracking_cuda_v1(std::uint64_t* peak, std::uint64_t* samples) {
  const auto& observation = generativeqc::runtime::cpu_resource_observation;
  if (peak == nullptr || samples == nullptr || !observation.active) return 1;
  *peak = observation.cuda_arena_peak_bytes;
  *samples = observation.cuda_arena_samples;
  return 0;
}

/** No runtime initialization, even in a CUDA-linked library. Status 2 means
 * the requested shape/provider is outside this allocation-inventory contract.
 */
int generativeqc_resource_small_hf_cuda_v1(std::size_t nbf, std::size_t direct_nbf,
                                           std::size_t atoms, std::size_t shells,
                                           std::size_t primitives, std::size_t diis_history,
                                           std::size_t spins, std::uint64_t* output) {
  if (output == nullptr) return 1;
#if GENERATIVEQC_HAS_CUDA
  std::size_t arena_bytes = 0;
  std::size_t plan_bytes = 0;
  if (!generativeqc::scf::small_hf_cuda_resource_layout(
          nbf, direct_nbf, atoms, shells, primitives, diis_history, spins, arena_bytes, plan_bytes))
    return 2;
  output[0] = arena_bytes;
  output[1] = plan_bytes;
  return 0;
#else
  return 2;
#endif
}

/** Topology-aware small-HF envelope. Unlike v1 this can account for the
 * quartet-direct force route while remaining a pure host-side shape query. */
int generativeqc_resource_small_hf_cuda_v2(
    std::size_t nbf, std::size_t direct_nbf, std::size_t atoms, std::size_t shells,
    const std::uint8_t* shell_angular, const std::size_t* shell_primitive_counts,
    std::size_t diis_history, std::size_t spins, int precision_mode, double energy_tolerance,
    double screening_tolerance, std::uint64_t* output, std::size_t count) {
  if (output == nullptr || count != 2 || shell_angular == nullptr ||
      shell_primitive_counts == nullptr) {
    return 1;
  }
#if GENERATIVEQC_HAS_CUDA
  std::size_t arena_bytes = 0;
  std::size_t plan_bytes = 0;
  if (!generativeqc::scf::small_hf_cuda_resource_layout_v2(
          nbf, direct_nbf, atoms, shell_angular, shell_primitive_counts, shells, diis_history,
          spins, precision_mode, energy_tolerance, screening_tolerance, arena_bytes, plan_bytes)) {
    return 2;
  }
  output[0] = arena_bytes;
  output[1] = plan_bytes;
  return 0;
#else
  return 2;
#endif
}

/** Shape-only adapter to the existing DF planner. No integrals, context or
 * workspace is allocated. Fixed source/other-provider bytes are supplied by
 * the composing planner, not independently spent again inside this budget.
 */
int generativeqc_resource_df_tiles_v3(std::size_t batch, std::size_t nbf, std::size_t naux,
                                      std::size_t occupied, std::size_t budget,
                                      std::size_t fixed_device_bytes, unsigned generated_source,
                                      std::size_t automatic_rhf_rank, std::uint64_t* output,
                                      std::size_t count, char* error, std::size_t error_size) {
  if (output == nullptr || (count != 6 && count != 7) || error == nullptr || error_size == 0 ||
      generated_source > 1)
    return 1;
  try {
    const auto plan = generativeqc::scf::plan_density_fitting_tiles(
        batch, nbf, naux, occupied, budget, fixed_device_bytes, generated_source != 0,
        automatic_rhf_rank);
    output[0] = plan.batch_tile;
    output[1] = plan.ao_pair_tile;
    output[2] = plan.auxiliary_tile;
    output[3] = plan.occupied_tile;
    output[4] = plan.peak_workspace_bytes;
    output[5] = plan.stores_full_three_center ? 1 : 0;
    if (count == 7) output[6] = plan.automatic_rhf_rank;
    error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}

/** Legacy shape queries have no reference/occupation authorization. */
int generativeqc_resource_df_tiles_v2(std::size_t batch, std::size_t nbf, std::size_t naux,
                                      std::size_t occupied, std::size_t budget,
                                      std::size_t fixed_device_bytes, unsigned generated_source,
                                      std::uint64_t* output, std::size_t count, char* error,
                                      std::size_t error_size) {
  return generativeqc_resource_df_tiles_v3(batch, nbf, naux, occupied, budget, fixed_device_bytes,
                                           generated_source, 0, output, count, error, error_size);
}

/** Preserve the original host-tensor capacity query ABI. */
int generativeqc_resource_df_tiles_v1(std::size_t batch, std::size_t nbf, std::size_t naux,
                                      std::size_t occupied, std::size_t budget,
                                      std::size_t fixed_device_bytes, std::uint64_t* output,
                                      std::size_t count, char* error, std::size_t error_size) {
  return generativeqc_resource_df_tiles_v2(batch, nbf, naux, occupied, budget, fixed_device_bytes,
                                           0, output, count, error, error_size);
}

/** Explicit physical-source packed representation, without changing the dense
 * query ABI or inferring storage from environment variables. The final four
 * fields report one immutable factor's bytes, total mutable scratch bytes,
 * complete-projection capacity and each bounded panel's capacity in doubles.
 * Raw A and transformed B are distinct allocations of the same factor size.
 * rank_capacity=0 reserves no complete U, while exact bounded K still exists.
 */
int generativeqc_resource_df_packed_tiles_v2(std::size_t batch, std::size_t nbf, std::size_t naux,
                                             std::size_t rank_capacity, std::size_t budget,
                                             std::size_t fixed_device_bytes,
                                             std::size_t automatic_rhf_rank, std::uint64_t* output,
                                             std::size_t count, char* error,
                                             std::size_t error_size) {
  if (!output || (count != 10 && count != 11) || !error || !error_size) return 1;
  try {
    const auto plan = generativeqc::scf::plan_packed_density_fitting_tiles(
        batch, nbf, naux, rank_capacity, budget, fixed_device_bytes, automatic_rhf_rank);
    const auto capacity = generativeqc::scf::df_packed_value_capacity(
        batch, nbf, naux, rank_capacity, plan.auxiliary_tile);
    const std::uint64_t values[]{
        plan.batch_tile,         plan.ao_pair_tile,         plan.auxiliary_tile,
        plan.occupied_tile,      plan.peak_workspace_bytes, 1,
        capacity.factor_bytes,   capacity.scratch_bytes,    capacity.projection_elements,
        capacity.panel_elements, plan.automatic_rhf_rank};
    std::copy_n(values, count, output);
    error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}

/** Preserve the packed shape ABI without guessing the method from rank capacity. */
int generativeqc_resource_df_packed_tiles_v1(std::size_t batch, std::size_t nbf, std::size_t naux,
                                             std::size_t rank_capacity, std::size_t budget,
                                             std::size_t fixed_device_bytes, std::uint64_t* output,
                                             std::size_t count, char* error,
                                             std::size_t error_size) {
  return generativeqc_resource_df_packed_tiles_v2(batch, nbf, naux, rank_capacity, budget,
                                                  fixed_device_bytes, 0, output, count, error,
                                                  error_size);
}

int generativeqc_resource_df_source_bytes_v1(std::size_t batch, std::size_t atoms,
                                             std::size_t shells, std::size_t cartesian_aos,
                                             std::size_t primitives, std::size_t transforms,
                                             std::uint64_t* output) {
  if (output == nullptr) return 1;
  try {
    *output = generativeqc::scf::density_fitting_source_metadata_bytes(
        batch, atoms, shells, cartesian_aos, primitives, transforms);
    return 0;
  } catch (const std::overflow_error&) {
    return 1;
  }
}

/** Private additive shape query for the exact native MINAO preparation bound.
 * Output is transactional; this does not construct a system, SCF or GPU state.
 * Old binaries simply lack the symbol and retain legacy Python planning. */
int generativeqc_resource_minao_numeric_capacity_v1(
    std::size_t target_aos, const std::int32_t* atomic_numbers, std::size_t atom_count,
    std::uint64_t* output) {
  if (!output) return 1;
  try {
    const auto value = generativeqc::scf::initial_guess::preliminary_minao_numeric_capacity(
        target_aos, atomic_numbers, atom_count);
    *output = value;
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}

/** Shape-only capacity of the shared device DIIS owner; never probes CUDA. */
int generativeqc_resource_df_diis_bytes_v1(std::size_t batch, std::size_t nbf, unsigned history,
                                           std::uint64_t* output) {
  if (!output || !batch || !nbf) return 1;
  const auto bytes = generativeqc::scf::density_fitting_scf_diis_device_bytes(batch, nbf, history);
  if (bytes == std::numeric_limits<std::size_t>::max()) return 1;
  *output = bytes;
  return 0;
}
}
