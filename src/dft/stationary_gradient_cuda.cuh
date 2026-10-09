#pragma once
// Bounded stationary runtime. Graph-emitted primitive, AO pullback and
// Becke entries precede this include; compiler-emitted contraction bodies follow it.
// This header owns only resource state, validation, transfers, launches and ABI.
#include <chrono>
#include <limits>
#include <vector>

#include "../tensor/cuda_runtime.cuh"
#include "grid_task_view.cuh"
#include "xc_point.hpp"

namespace generativeqc_stationary_cuda {
using namespace generativeqc_tensor;
constexpr size_t record_stride = 26, task_stride = 9;
struct Owner {
  Context context;
  generativeqc::runtime::OwnedCudaBuffer<unsigned char> phased_storage;
  size_t atoms{}, aos{}, primitives{}, points{}, task_capacity{}, spin_blocks{},
      max_page_primitive_work{}, bytes{}, geometry_lanes{}, geometry_threads{},
      geometry_peak_lanes{}, center_geometry_bytes{}, becke_threads_per_point{1},
      becke_shared_bytes{}, byte_budget{}, phased_bytes{};
  bool failed = true, topology_ready = false;
  bool profile = false, geometry_pending = false;
  bool becke_primitive_requested = false, becke_primitive = false,
       becke_primitive_configured = false;
  unsigned becke_primitive_mode{};
  double geometry_tolerance{};
  unsigned long long* becke_zero_seed_points{};
  bool becke_zero_seed_requested = true, becke_zero_seed_configured = false;
  bool becke_normalize_supported = false, becke_normalize_cooperative = false,
       becke_normalize_configured = false;
  cudaStream_t geometry_stream{};
  cudaEvent_t stage0{}, stage1{}, stage2{}, stage3{};
  cudaEvent_t becke_events[8]{};
  double becke_phase_ms[7]{};
  double synchronization_wait_ms{}, setup_transfer_ms{}, setup_validation_ms{};
  double primitive_h2d_ms{}, primitive_kernel_ms{}, primitive_reduction_ms{};
  double geometry_h2d_ms{}, geometry_kernel_ms{}, geometry_reduction_ms{}, final_d2h_wall_ms{};
  double *primitive_table{}, *ao_norms{}, *task_charges{}, *task_values{}, *centers{}, *weights{},
      *raw{}, *partial{}, *scratch{}, *sources{}, *density{}, *weighted_density{};
  generativeqc_grid_adjoint::CenterPair* center_pairs{};
  int64_t *ao_ranges{}, *tasks{}, *ao_atoms{}, *point_atoms{};
  uint64_t uploads{}, downloads{}, launches{}, primitive_count{}, point_count{}, pair_visits{},
      task_count{}, task_batches{};
  uint64_t h2d_calls{}, d2h_calls{}, synchronizations{}, geometry_batches{};
  uint64_t center_distance_evaluations{}, center_geometry_preparations{},
      becke_pair_state_evaluations{}, phased_batches{};
  uint64_t becke_primitive_batches{}, becke_primitive_reverse_pair_visits{};
  uint64_t phased_points{}, becke_profile_batches{};
  bool retains_becke_pair_state() const {
    return bool(phased_storage) ||
           (becke_threads_per_point > 1 && atoms <= stationary_becke_retained_max_atoms);
  }
  // Primitive metrics are cumulative across one reset/force execution. Admission
  // is page-local so arbitrarily many bounded pages may contribute to one force.
  void check_page_primitive_work(size_t work) const {
    if (work > max_page_primitive_work)
      throw std::invalid_argument("stationary primitive page work budget exceeded");
  }
  void count_primitive_work(size_t work) {
    if (work > std::numeric_limits<uint64_t>::max() - primitive_count)
      throw std::invalid_argument("stationary primitive work counter overflow");
    primitive_count += work;
  }
};
size_t phased_allocation(size_t atoms, size_t points) {
  static_assert(sizeof(size_t) == 8 && sizeof(double) == 8 && sizeof(uint2) == 8);
  const size_t pairs = atoms * (atoms - 1) / 2;
  return 8 * (4 * pairs * points + (12 * atoms + 2) * points + pairs);
}
PhasedBeckeInput phased_input(Owner& owner, size_t points) {
  const size_t pairs = owner.atoms * (owner.atoms - 1) / 2;
  auto* pair_storage = reinterpret_cast<double*>(owner.phased_storage.get());
  double* fields = pair_storage + 4 * pairs * owner.points;
  auto* zeros = reinterpret_cast<size_t*>(fields + 11 * owner.atoms * owner.points);
  double* maximum = reinterpret_cast<double*>(zeros + owner.atoms * owner.points);
  PhasedBeckeInput input{};
  input.work = {owner.atoms, points, pair_storage, fields, zeros, maximum};
  input.seeds = maximum + owner.points;
  input.indices = reinterpret_cast<uint2*>(input.seeds + owner.points);
  // Only the authenticated first-derivative route may elide zero cotangents.
  // A positive separation floor keeps prepared reciprocal partials finite;
  // smaller tolerances retain the ordinary error/overflow behavior unchanged.
  input.zero_seed_elision = owner.becke_primitive && owner.becke_primitive_mode == 2 &&
                            owner.becke_zero_seed_requested && owner.geometry_tolerance >= 1e-12;
  input.zero_seed_points = owner.becke_zero_seed_points;
  return input;
}
// Caps make all products below representable before any allocation or pointer
// dereference. D/W and AO metadata have no fixed-size AO array; 2048 admits
// full 96-atom def2-TZVPD without changing allocation formulas. Compiler-planned
// lanes bound O(lanes*natom) adjoint scratch; byte admission remains mandatory.
size_t allocation(size_t na, size_t n, size_t nprimitive, size_t np, size_t ntask, size_t ns,
                  size_t geometry_lanes, bool cache_center_geometry = false) {
  if (!na || na > 128 || !n || n > 2048 || !nprimitive || nprimitive > 16384 || !np || np > 4096 ||
      !ntask || ntask > 4096 || (ns != 1 && ns != 2) || ns != stationary_spin_blocks ||
      !geometry_lanes || geometry_lanes > np || geometry_lanes > stationary_geometry_max_lanes ||
      18 * geometry_lanes * na * sizeof(double) > stationary_geometry_max_scratch_bytes)
    throw std::invalid_argument("stationary CUDA shape exceeds bounded resource caps");
  // Three center coordinates, two planned nine-coordinate lane panels,
  // and one three-coordinate panel per compiler-owned gradient source.
  return 8 * (2 * nprimitive + 4 * n + 22 * ntask +
              (3 + 18 * geometry_lanes + 3 * stationary_source_count) * na + 3 * np +
              2 * ns * n * n) +
         (cache_center_geometry ? 48 * (na * (na - 1) / 2) : 0) + 256;
}
template <class F>
int guarded(Owner* owner, char* error, size_t size, F f) noexcept {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    if (owner) owner->failed = true;
    error_text(error, size, e.what());
    return 1;
  } catch (...) {
    if (owner) owner->failed = true;
    error_text(error, size, "unknown stationary CUDA failure");
    return 1;
  }
}
// A local collocation panel retains the global density and atom domains.
// Null IDs denote identity only for a full panel; an empty explicit selection
// needs no map dereference. GridPlan validates/uploads nonempty map contents.
bool valid_geometry_ao_map(const generativeqc::dft::GridTaskView& view, size_t aos) {
  return view.nao == aos && view.nactive <= aos &&
         (view.nactive == 0 || view.ao_ids != nullptr || view.nactive == aos);
}
void check(Owner& p) {
  if (p.failed) throw std::runtime_error("failed stationary owner; reset before reuse");
  if (!p.topology_ready) throw std::runtime_error("stationary topology is not prepared");
  p.context.check_device();
}
void profile_record(Owner& p, cudaEvent_t event, cudaStream_t stream) {
  if (p.profile) cuda_check(cudaEventRecord(event, stream));
}
void profile_elapsed(Owner& p, double& total, cudaEvent_t begin, cudaEvent_t end) {
  if (!p.profile) return;
  float elapsed = 0;
  cuda_check(cudaEventElapsedTime(&elapsed, begin, end));
  total += elapsed;
}
void finished(Owner& p, cudaStream_t stream) {
  int failure = 0;
  cuda_check(cudaGetLastError());
  cuda_check(
      cudaMemcpyAsync(&failure, p.context.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
  ++p.d2h_calls;
  auto sync_begin = std::chrono::steady_clock::time_point{};
  if (p.profile) sync_begin = std::chrono::steady_clock::now();
  cuda_check(cudaStreamSynchronize(stream));
  if (p.profile) {
    p.synchronization_wait_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_begin)
            .count();
  }
  ++p.synchronizations;
  p.downloads += sizeof(int);
  if (failure) throw std::runtime_error("nonfinite or invalid stationary CUDA source");
}
void drain_geometry(Owner& p) {
  if (!p.geometry_pending) return;
  auto stream = p.geometry_stream;
  p.geometry_pending = false;
  p.geometry_stream = nullptr;
  finished(p, stream);
}
template <class T>
void upload(Owner& p, T* out, const T* in, size_t n, cudaStream_t stream) {
  if (n && !in) throw std::invalid_argument("null stationary source");
  cuda_check(cudaMemcpyAsync(out, in, n * sizeof(T), cudaMemcpyHostToDevice, stream));
  ++p.h2d_calls;
  p.uploads += n * sizeof(T);
}
__global__ void task_kernel(const int64_t* tasks, const double* charges, size_t count,
                            const double* primitives, size_t nprimitive, const int64_t* ao_ranges,
                            const double* ao_norms, const int64_t* ao_atoms, const double* centers,
                            const double* density, const double* weighted_density, size_t nao,
                            size_t na, double* output, int* error);
__global__ void task_reduce(const double* input, const int64_t* tasks, size_t count,
                            const int64_t* ao_atoms, size_t na, double* output, int* error);
__global__ void nuclear_kernel(unsigned kind, int64_t a, int64_t b, double za, double zb,
                               const double* centers, size_t na, double* output, int* error);
__global__ void nuclear_all_kernel(unsigned kind, const double* charges, const double* centers,
                                   size_t na, double* output, int* error);
__global__ void validate_centers(const double* centers, size_t na, double tolerance,
                                 generativeqc_grid_adjoint::CenterPair* center_pairs, int* error);
__global__ void geometry_kernel(generativeqc::dft::GridTaskView view, const double* work,
                                const int64_t* ao_atoms, const int64_t* owners, size_t owner_offset,
                                size_t points_per_atom, const double* centers, size_t na,
                                const double* weights, const double* raw, const double* external,
                                size_t external_stride, size_t external_offset,
                                size_t geometry_lanes, double* partial, double* scratch,
                                const generativeqc_grid_adjoint::CenterPair* center_pairs,
                                int* error);
__global__ void geometry_point_kernel(generativeqc::dft::GridTaskView view, const int64_t* owners,
                                      size_t owner_offset, size_t points_per_atom, size_t na,
                                      const double* weights, const double* raw,
                                      const double* external, size_t external_stride,
                                      size_t external_offset, double* scratch, double* phase_seeds,
                                      int* error);
template <bool precomputed_point>
__global__ void geometry_cooperative_kernel(
    generativeqc::dft::GridTaskView view, const double* work, const int64_t* ao_atoms,
    const int64_t* owners, size_t owner_offset, size_t points_per_atom, const double* centers,
    size_t na, const double* weights, const double* raw, const double* external,
    size_t external_stride, size_t external_offset, size_t geometry_lanes, double* partial,
    double* scratch, const generativeqc_grid_adjoint::CenterPair* center_pairs, int* error,
    double* phase_seeds);
__global__ void geometry_reduce(const double* partial, size_t na, size_t geometry_lanes,
                                double* output, int* error);
__global__ void source_reduce(const double* input, size_t na, double* output, int* error);
void launch_geometry(Owner& owner, cudaStream_t stream, generativeqc::dft::GridTaskView view,
                     const double* work, const int64_t* ao_atoms, const int64_t* owners,
                     size_t owner_offset, size_t points_per_atom, const double* centers, size_t na,
                     const double* weights, const double* raw, const double* external,
                     size_t external_stride, size_t external_offset, size_t geometry_lanes,
                     double* partial, double* scratch,
                     const generativeqc_grid_adjoint::CenterPair* center_pairs, int* error) {
  PhasedBeckeInput phased{};
  if (owner.phased_storage) {
    if (geometry_lanes != view.npoint || !center_pairs || owner.becke_threads_per_point <= 1)
      throw std::invalid_argument("phased Becke lane/cache contract changed");
    phased = phased_input(owner, view.npoint);
    phased.normalized_adjoints = owner.becke_primitive && owner.becke_primitive_mode == 2;
    phased.points = view.points;
    phased.centers = centers;
    phased.owners = owners;
    phased.owner_offset = owner_offset;
    phased.points_per_atom = points_per_atom;
    phased.center_pairs = center_pairs;
    phased.partial = partial;
    phased.error = error;
  }
  // Only phased, one-point-per-lane geometry may lend its unused inline scratch
  // to the bulk point producer. Keep small atom counts and all bounded/nonphased
  // routes on the original evaluator; no new resident-storage requirement.
  static_assert(alignof(StationaryPointValue) <= alignof(double));
  const bool precomputed_point =
      owner.phased_storage &&
      na >= (sizeof(StationaryPointValue) + 3 * sizeof(double) - 1) / (3 * sizeof(double));
  if (precomputed_point) {
    // Preserve incoming launch status and stop before any consumer can observe
    // scratch from a failed producer. Peeking neither clears nor synchronizes.
    cuda_check(cudaPeekAtLastError());
    geometry_point_kernel<<<blocks(view.npoint, 128), 128, 0, stream>>>(
        view, owners, owner_offset, points_per_atom, na, weights, raw, external, external_stride,
        external_offset, scratch, phased.seeds, error);
    cuda_check(cudaPeekAtLastError());
    geometry_cooperative_kernel<true>
        <<<geometry_lanes, owner.becke_threads_per_point,
           owner.becke_shared_bytes - stationary_becke_control_bytes, stream>>>(
            view, work, ao_atoms, owners, owner_offset, points_per_atom, centers, na, weights, raw,
            external, external_stride, external_offset, geometry_lanes, partial, scratch,
            center_pairs, error, phased.seeds);
    cuda_check(cudaPeekAtLastError());
  } else if (owner.becke_threads_per_point > 1)
    geometry_cooperative_kernel<false>
        <<<geometry_lanes, owner.becke_threads_per_point,
           owner.becke_shared_bytes - stationary_becke_control_bytes, stream>>>(
            view, work, ao_atoms, owners, owner_offset, points_per_atom, centers, na, weights, raw,
            external, external_stride, external_offset, geometry_lanes, partial, scratch,
            center_pairs, error, phased.seeds);
  else
    geometry_kernel<<<blocks(geometry_lanes, owner.geometry_threads), owner.geometry_threads, 0,
                      stream>>>(view, work, ao_atoms, owners, owner_offset, points_per_atom,
                                centers, na, weights, raw, external, external_stride,
                                external_offset, geometry_lanes, partial, scratch, center_pairs,
                                error);
  if (owner.phased_storage) {
    const dim3 atom_blocks(blocks(view.npoint, 128), na);
    const dim3 pair_blocks(blocks(view.npoint, 128), na * (na - 1) / 2);
    // These events follow the AO/XC seed producer on its borrowed stream.
    // Disabled profiling records nothing and adds no device synchronization.
    profile_record(owner, owner.becke_events[0], stream);
    phased_becke_atom<0><<<atom_blocks, 128, 0, stream>>>(phased);
    profile_record(owner, owner.becke_events[1], stream);
    phased_becke_pair<false><<<pair_blocks, 128, 0, stream>>>(phased);
    profile_record(owner, owner.becke_events[2], stream);
    phased_becke_atom<1><<<atom_blocks, 128, 0, stream>>>(phased);
    profile_record(owner, owner.becke_events[3], stream);
    if constexpr (stationary_becke_cooperative_normalize) {
      if (owner.becke_normalize_cooperative)
        phased_becke_normalize_cooperative<<<blocks(view.npoint,
                                                    stationary_becke_normalize_point_lanes),
                                             dim3(stationary_becke_normalize_point_lanes,
                                                  128 / stationary_becke_normalize_point_lanes),
                                             0, stream>>>(phased);
      else
        phased_becke_normalize<<<blocks(view.npoint, 128), 128, 0, stream>>>(phased);
    } else {
      phased_becke_normalize<<<blocks(view.npoint, 128), 128, 0, stream>>>(phased);
    }
    profile_record(owner, owner.becke_events[4], stream);
    if (owner.becke_primitive) {
      if (owner.becke_primitive_mode == 2)
        phased_becke_pair<true, false, true><<<pair_blocks, 128, 0, stream>>>(phased);
      else
        phased_becke_pair<true, true><<<pair_blocks, 128, 0, stream>>>(phased);
      profile_record(owner, owner.becke_events[5], stream);
      if (owner.becke_primitive_mode == 2)
        phased_becke_atom<2><<<atom_blocks, 128, 0, stream>>>(phased);
      else
        phased_becke_atom<2, true><<<atom_blocks, 128, 0, stream>>>(phased);
      ++owner.becke_primitive_batches;
      owner.becke_primitive_reverse_pair_visits += view.npoint * na * (na - 1) / 2;
    } else {
      phased_becke_pair<true><<<pair_blocks, 128, 0, stream>>>(phased);
      profile_record(owner, owner.becke_events[5], stream);
      phased_becke_atom<2><<<atom_blocks, 128, 0, stream>>>(phased);
    }
    profile_record(owner, owner.becke_events[6], stream);
    phased_becke_atom<3><<<atom_blocks, 128, 0, stream>>>(phased);
    profile_record(owner, owner.becke_events[7], stream);
    owner.launches += 7;
    ++owner.phased_batches;
    owner.phased_points += view.npoint;
    if (owner.profile) {
      // Event reuse across deferred tiles is safe only after the last event.
      // This intentional per-tile fence is intrusive qualification, never a
      // clean endpoint sample; report its event/fence counts separately.
      cuda_check(cudaEventSynchronize(owner.becke_events[7]));
      ++owner.synchronizations;
      ++owner.becke_profile_batches;
      for (size_t phase = 0; phase < 7; ++phase)
        profile_elapsed(owner, owner.becke_phase_ms[phase], owner.becke_events[phase],
                        owner.becke_events[phase + 1]);
    }
  }
}
}  // namespace generativeqc_stationary_cuda

extern "C" {
int stationary_create(int device, int major, int minor, size_t na, size_t n, size_t nprimitive,
                      size_t np, size_t ntask, size_t ns, size_t max_primitive_work, size_t budget,
                      size_t geometry_lanes, size_t geometry_threads, void** output, char* error,
                      size_t size) {
  using namespace generativeqc_stationary_cuda;
  if (output) *output = nullptr;
  return guarded(nullptr, error, size, [&] {
    if (!output || !max_primitive_work)
      throw std::invalid_argument("invalid stationary owner output/work budget");
    size_t bytes = allocation(na, n, nprimitive, np, ntask, ns, geometry_lanes);
    if (bytes > budget) throw std::invalid_argument("stationary CUDA byte budget exceeded");
    // Reuse spare capacity only; point concurrency remains compiler-planned.
    const size_t pair_bytes = 48 * (na * (na - 1) / 2);
    size_t center_bytes = pair_bytes <= budget - bytes ? pair_bytes : 0;
    bytes += center_bytes;
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count));
    if (device < 0 || device >= device_count)
      throw std::invalid_argument("invalid stationary CUDA device ordinal");
    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, device));
    if (!geometry_threads || geometry_threads > stationary_geometry_max_threads ||
        geometry_threads > geometry_lanes ||
        geometry_threads > size_t(property.maxThreadsPerBlock) ||
        geometry_threads > size_t(property.maxThreadsDim[0]) ||
        (geometry_lanes + geometry_threads - 1) / geometry_threads >
            size_t(property.maxGridSize[0]))
      throw std::invalid_argument("invalid stationary geometry launch plan");
    auto p = std::make_unique<Owner>();
    try {
      p->context.prepare(device, major, minor, bytes, bytes - 256, 0, 0, 0, false);
    } catch (const DeviceAllocationError&) {
      if (!center_bytes) throw;
      // Release every partially constructed CUDA resource before retrying the
      // already-admitted direct arena once. Other device errors still fail.
      p.reset();
      (void)cudaGetLastError();
      bytes -= center_bytes;
      center_bytes = 0;
      p = std::make_unique<Owner>();
      p->context.prepare(device, major, minor, bytes, bytes - 256, 0, 0, 0, false);
    }
    p->atoms = na;
    p->aos = n;
    p->primitives = nprimitive;
    p->points = np;
    p->task_capacity = ntask;
    p->spin_blocks = ns;
    p->max_page_primitive_work = max_primitive_work;
    p->bytes = bytes;
    p->byte_budget = budget;
    p->geometry_lanes = geometry_lanes;
    p->geometry_threads = geometry_threads;
    p->center_geometry_bytes = center_bytes;
    auto* next = reinterpret_cast<double*>(p->context.arena);
    auto take = [&](size_t count) {
      auto* ptr = next;
      next += count;
      return ptr;
    };
    p->primitive_table = take(2 * nprimitive);
    p->ao_norms = take(n);
    p->task_charges = take(ntask);
    p->task_values = take(12 * ntask);
    p->centers = take(3 * na);
    if (center_bytes)
      p->center_pairs = reinterpret_cast<generativeqc_grid_adjoint::CenterPair*>(
          take(center_bytes / sizeof(double)));
    p->weights = take(np);
    p->raw = take(np);
    p->partial = take(geometry_lanes * 9 * na);
    p->scratch = take(geometry_lanes * 9 * na);
    p->sources = take(3 * stationary_source_count * na);
    p->ao_ranges = reinterpret_cast<int64_t*>(take(2 * n));
    p->tasks = reinterpret_cast<int64_t*>(take(task_stride * ntask));
    p->ao_atoms = reinterpret_cast<int64_t*>(take(n));
    p->point_atoms = reinterpret_cast<int64_t*>(take(np));
    p->density = take(ns * n * n);
    p->weighted_density = take(ns * n * n);
    // The charged 256-byte control panel reserves two error words followed by
    // this aligned counter. It is not part of the mathematical scratch alias.
    p->becke_zero_seed_points = reinterpret_cast<unsigned long long*>(
        reinterpret_cast<unsigned char*>(p->context.error) + 8);
    cuda_check(cudaMemsetAsync(p->becke_zero_seed_points, 0, sizeof(unsigned long long),
                               p->context.stream));
    *output = p.release();
  });
}
int stationary_configure_becke(void* pointer, size_t threads, size_t shared_bytes, char* error,
                               size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || p->topology_ready)
      throw std::invalid_argument("Becke schedule must be configured before topology");
    p->context.check_device();
    if (threads == 1 && shared_bytes == 0) {
      p->becke_threads_per_point = 1;
      p->becke_shared_bytes = 0;
      return;
    }
    const size_t rows = std::min(stationary_becke_pair_tile_rows, p->atoms - 1);
    const size_t state_count = p->atoms <= stationary_becke_retained_max_atoms
                                   ? p->atoms * (p->atoms - 1) / 2
                                   : rows * (2 * p->atoms - rows - 1) / 2;
    const size_t required =
        stationary_becke_control_bytes + sizeof(generativeqc_grid_adjoint::PointPair) * state_count;
    if (p->atoms < 2 || p->atoms > stationary_becke_max_atoms ||
        threads != stationary_becke_threads || shared_bytes != required)
      throw std::invalid_argument("invalid cooperative Becke resource plan");
    p->becke_threads_per_point = 1;
    p->becke_shared_bytes = 0;
    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, p->context.device));
    // A target catalog may be more permissive than the executing device. Keep
    // the admitted generic point-worker route if actual shared/launch caps fail.
    if (threads > size_t(property.maxThreadsPerBlock) ||
        threads > size_t(property.maxThreadsDim[0]) ||
        p->geometry_lanes > size_t(property.maxGridSize[0]) ||
        shared_bytes > size_t(property.sharedMemPerBlock))
      return;
    p->becke_threads_per_point = threads;
    p->becke_shared_bytes = shared_bytes;
  });
}
int stationary_configure_phased_becke_v1(void* pointer, size_t bytes, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* owner = static_cast<Owner*>(pointer);
  return guarded(owner, error, size, [&] {
    if (!owner || owner->topology_ready || owner->phased_storage)
      throw std::invalid_argument("phased Becke must be configured once before topology");
    owner->context.check_device();
    if (bytes != phased_allocation(owner->atoms, owner->points) ||
        bytes > owner->byte_budget - owner->bytes)
      throw std::invalid_argument("phased Becke concurrent byte budget mismatch");
    if (owner->atoms <= stationary_becke_retained_max_atoms ||
        owner->geometry_lanes != owner->points || !owner->center_pairs ||
        owner->becke_threads_per_point <= 1)
      return;
    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, owner->context.device));
    const size_t pairs = owner->atoms * (owner->atoms - 1) / 2;
    if (property.major != 12 || property.minor != 0 || property.maxThreadsPerBlock < 128 ||
        property.maxThreadsDim[0] < 128 || pairs > size_t(property.maxGridSize[1]) ||
        blocks(owner->points, 128) > size_t(property.maxGridSize[0]))
      return;
    try {
      owner->phased_storage.allocate(owner->context.device, bytes, owner->context.stream);
    } catch (const std::bad_alloc&) {
      // Optional retention must not revoke the already admitted bounded path.
      (void)cudaGetLastError();
      return;
    }
    auto input = phased_input(*owner, owner->points);
    phased_becke_indices<<<owner->atoms, 128, 0, owner->context.stream>>>(
        owner->atoms, const_cast<uint2*>(input.indices));
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(owner->context.stream));
    owner->phased_bytes = bytes;
    owner->bytes += bytes;
    if constexpr (stationary_becke_cooperative_normalize) {
      cudaDeviceProp property{};
      cudaFuncAttributes attributes{};
      cuda_check(cudaGetDeviceProperties(&property, owner->context.device));
      cuda_check(cudaFuncGetAttributes(&attributes, phased_becke_normalize_cooperative));
      owner->becke_normalize_supported =
          owner->atoms <= stationary_becke_normalize_max_atoms &&
          size_t(property.maxThreadsDim[0]) >= stationary_becke_normalize_point_lanes &&
          size_t(property.maxThreadsDim[1]) >= 128 / stationary_becke_normalize_point_lanes &&
          property.maxThreadsPerBlock >= 128 && attributes.maxThreadsPerBlock >= 128 &&
          attributes.sharedSizeBytes <= size_t(property.sharedMemPerBlock);
      owner->becke_normalize_cooperative = owner->becke_normalize_supported;
    }
    ++owner->launches;
    ++owner->synchronizations;
  });
}
int stationary_phased_becke_metrics_v1(void* pointer, uint64_t* output, size_t count) {
  auto* owner = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!owner || !output || count != 2) return 1;
  output[0] = owner->phased_bytes;
  output[1] = owner->phased_batches;
  return 0;
}
// Qualification may compare the two schedules, but never mutate a live
// topology/geometry owner. Both use the same reservation and canonical AD.
int stationary_configure_becke_normalize_v1(void* pointer, int enabled, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* owner = static_cast<Owner*>(pointer);
  return guarded(owner, error, size, [&] {
    if (!owner || owner->topology_ready || owner->becke_normalize_configured ||
        (enabled != 0 && enabled != 1))
      throw std::invalid_argument("Becke normalization must be configured once before topology");
    owner->context.check_device();
    owner->becke_normalize_configured = true;
    owner->becke_normalize_cooperative = enabled && owner->becke_normalize_supported;
  });
}
int stationary_becke_normalize_metrics_v1(void* pointer, uint64_t* output, size_t count) {
  auto* owner = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!owner || !output || count != 4) return 1;
  output[0] = owner->becke_normalize_supported;
  output[1] = owner->becke_normalize_cooperative;
  output[2] = owner->phased_batches;
  output[3] = owner->phased_points;
  return 0;
}
int stationary_configure_becke_primitive_v1(void* pointer, int enabled, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* owner = static_cast<Owner*>(pointer);
  return guarded(owner, error, size, [&] {
    if (!owner || owner->topology_ready || owner->becke_primitive_configured ||
        (enabled < 0 || enabled > 2))
      throw std::invalid_argument("Becke primitive must be configured once before topology");
    owner->context.check_device();
    owner->becke_primitive_configured = true;
    owner->becke_primitive_requested = enabled;
    owner->becke_primitive_mode = enabled;
    // A losing schedule is opt-in only. Reuse the already admitted reservation,
    // center lifetime and point lanes; never allocate or evict another owner.
    if (!enabled || !owner->phased_storage || !owner->center_pairs ||
        owner->atoms > stationary_becke_primitive_max_atoms)
      return;
    cudaFuncAttributes reverse{}, gather{};
    if (enabled == 2) {
      cuda_check(cudaFuncGetAttributes(&reverse, phased_becke_pair<true, false, true>));
      cuda_check(cudaFuncGetAttributes(&gather, phased_becke_atom<2>));
    } else {
      cuda_check(cudaFuncGetAttributes(&reverse, phased_becke_pair<true, true>));
      cuda_check(cudaFuncGetAttributes(&gather, phased_becke_atom<2, true>));
    }
    if (reverse.maxThreadsPerBlock < 128 || gather.maxThreadsPerBlock < 128) return;
    owner->becke_primitive = true;
  });
}
int stationary_configure_becke_normalized_adjoint_v1(void* pointer, char* error, size_t size) {
  return stationary_configure_becke_primitive_v1(pointer, 2, error, size);
}
int stationary_becke_primitive_metrics_v1(void* pointer, uint64_t* output, size_t count) {
  auto* owner = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!owner || !output || count != 4) return 1;
  output[0] = owner->becke_primitive_mode;
  output[1] = owner->becke_primitive ? owner->becke_primitive_mode : 0;
  output[2] = owner->becke_primitive_batches;
  output[3] = owner->becke_primitive_reverse_pair_visits;
  return 0;
}
int stationary_becke_phase_metrics_v1(void* pointer, uint64_t* output, size_t count) {
  auto* owner = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!owner || !output || count != 17) return 1;
  const uint64_t points = owner->phased_points;
  const uint64_t atom_entries = points * owner->atoms;
  const uint64_t pairs = points * owner->atoms * (owner->atoms - 1) / 2;
  const bool coefficients = owner->becke_primitive && owner->becke_primitive_mode == 1;
  const uint64_t reverse_words = coefficients ? 2 : 4;
  // Dense launched domains and logical distinct panel values, not hardware
  // transactions or completed work from a failed force. Conditional primal/log
  // reads and scalar transcendental evaluations are deliberately not inferred.
  const uint64_t values[]{owner->phased_batches,
                          points,
                          atom_entries,
                          pairs,
                          2 * pairs,
                          atom_entries,
                          pairs,
                          2 * pairs,
                          atom_entries,
                          7 * owner->phased_batches,
                          4 * 8 * pairs,
                          reverse_words * 8 * pairs,
                          2 * reverse_words * 8 * pairs,
                          coefficients ? 2 * 3 * 8 * pairs : 0,
                          owner->becke_profile_batches,
                          8 * owner->becke_profile_batches,
                          owner->becke_profile_batches};
  std::copy(values, values + 17, output);
  return 0;
}
int stationary_becke_phase_profile_v1(void* pointer, double* output, size_t count) {
  auto* owner = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!owner || !output || count != 7) return 1;
  std::copy(owner->becke_phase_ms, owner->becke_phase_ms + 7, output);
  return 0;
}
int stationary_becke_zero_seed_metrics_v1(void* pointer, uint64_t* output, size_t count) {
  using namespace generativeqc_stationary_cuda;
  auto* owner = static_cast<Owner*>(pointer);
  if (!owner || !output || count != 2) return 1;
  try {
    owner->context.check_device();
    drain_geometry(*owner);
    output[0] = owner->becke_primitive && owner->becke_primitive_mode == 2 &&
                owner->becke_zero_seed_requested && owner->geometry_tolerance >= 1e-12;
    output[1] = 0;
    if (owner->becke_primitive_mode != 2) return 0;
    cuda_check(cudaMemcpyAsync(output + 1, owner->becke_zero_seed_points, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, owner->context.stream));
    cuda_check(cudaStreamSynchronize(owner->context.stream));
    owner->downloads += sizeof(uint64_t);
    ++owner->d2h_calls;
    ++owner->synchronizations;
    return 0;
  } catch (...) {
    owner->failed = true;
    return 1;
  }
}
int stationary_topology(void* pointer, const double* primitives, const int64_t* ao_ranges,
                        const double* ao_norms, const int64_t* ao_atoms, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !primitives || !ao_ranges || !ao_norms || !ao_atoms || p->topology_ready)
      throw std::invalid_argument("invalid stationary topology");
    p->context.check_device();
    for (size_t i = 0; i < p->primitives; ++i)
      if (!std::isfinite(primitives[2 * i]) || !(primitives[2 * i] > 0) ||
          !std::isfinite(primitives[2 * i + 1]))
        throw std::invalid_argument("invalid stationary primitive topology");
    for (size_t i = 0; i < p->aos; ++i) {
      const auto begin = ao_ranges[2 * i], extent = ao_ranges[2 * i + 1];
      if (begin < 0 || extent <= 0 || begin > int64_t(p->primitives) ||
          extent > int64_t(p->primitives) - begin || !std::isfinite(ao_norms[i]) ||
          ao_atoms[i] < 0 || ao_atoms[i] >= int64_t(p->atoms))
        throw std::invalid_argument("invalid stationary AO topology");
    }
    auto stream = p->context.stream;
    profile_record(*p, p->stage0, stream);
    upload(*p, p->primitive_table, primitives, 2 * p->primitives, stream);
    upload(*p, p->ao_ranges, ao_ranges, 2 * p->aos, stream);
    upload(*p, p->ao_norms, ao_norms, p->aos, stream);
    upload(*p, p->ao_atoms, ao_atoms, p->aos, stream);
    profile_record(*p, p->stage1, stream);
    auto sync_begin = std::chrono::steady_clock::time_point{};
    if (p->profile) sync_begin = std::chrono::steady_clock::now();
    cuda_check(cudaStreamSynchronize(stream));
    if (p->profile) {
      p->synchronization_wait_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_begin)
              .count();
    }
    ++p->synchronizations;
    profile_elapsed(*p, p->setup_transfer_ms, p->stage0, p->stage1);
    p->topology_ready = true;
  });
}
int stationary_profile(void* pointer, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p) throw std::invalid_argument("null stationary owner");
    p->context.check_device();
    if (p->profile) return;
    cudaEvent_t events[12]{};
    try {
      for (auto& event : events) cuda_check(cudaEventCreate(&event));
    } catch (...) {
      for (auto event : events)
        if (event) cudaEventDestroy(event);
      throw;
    }
    p->stage0 = events[0];
    p->stage1 = events[1];
    p->stage2 = events[2];
    p->stage3 = events[3];
    std::copy(events + 4, events + 12, p->becke_events);
    p->profile = true;
  });
}
int stationary_configure_becke_zero_seed_v1(void* pointer, int enabled, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* owner = static_cast<Owner*>(pointer);
  return guarded(owner, error, size, [&] {
    if (!owner || owner->topology_ready || owner->becke_zero_seed_configured ||
        (enabled != 0 && enabled != 1))
      throw std::invalid_argument(
          "Becke zero-seed elision must be configured once before topology");
    owner->context.check_device();
    owner->becke_zero_seed_configured = true;
    owner->becke_zero_seed_requested = enabled;
  });
}
int stationary_reset(void* pointer, const double* centers, const double* density,
                     const double* weighted_density, double tolerance, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !p->topology_ready || !std::isfinite(tolerance) || tolerance < 0)
      throw std::invalid_argument("invalid reset");
    p->context.check_device();
    drain_geometry(*p);
    p->failed = false;
    p->geometry_peak_lanes = 0;
    p->geometry_tolerance = tolerance;
    auto stream = p->context.stream;
    profile_record(*p, p->stage0, stream);
    cuda_check(cudaMemsetAsync(p->context.error, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(p->sources, 0, 3 * stationary_source_count * p->atoms * 8, stream));
    upload(*p, p->centers, centers, 3 * p->atoms, stream);
    upload(*p, p->density, density, p->spin_blocks * p->aos * p->aos, stream);
    upload(*p, p->weighted_density, weighted_density, p->spin_blocks * p->aos * p->aos, stream);
    profile_record(*p, p->stage1, stream);
    validate_centers<<<1, 1, 0, stream>>>(p->centers, p->atoms, tolerance, p->center_pairs,
                                          p->context.error);
    profile_record(*p, p->stage2, stream);
    ++p->launches;
    p->pair_visits += p->atoms * (p->atoms - 1) / 2;
    p->center_distance_evaluations += p->atoms * (p->atoms - 1) / 2;
    if (p->center_pairs) ++p->center_geometry_preparations;
    finished(*p, stream);
    profile_elapsed(*p, p->setup_transfer_ms, p->stage0, p->stage1);
    profile_elapsed(*p, p->setup_validation_ms, p->stage1, p->stage2);
  });
}
int stationary_geometry_reset(void* pointer, const double* centers, double tolerance, char* error,
                              size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !p->topology_ready || !std::isfinite(tolerance) || tolerance < 0)
      throw std::invalid_argument("invalid geometry-only reset");
    p->context.check_device();
    drain_geometry(*p);
    p->failed = false;
    p->geometry_peak_lanes = 0;
    p->geometry_tolerance = tolerance;
    auto stream = p->context.stream;
    profile_record(*p, p->stage0, stream);
    cuda_check(cudaMemsetAsync(p->context.error, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(p->sources, 0, 3 * stationary_source_count * p->atoms * 8, stream));
    upload(*p, p->centers, centers, 3 * p->atoms, stream);
    profile_record(*p, p->stage1, stream);
    validate_centers<<<1, 1, 0, stream>>>(p->centers, p->atoms, tolerance, p->center_pairs,
                                          p->context.error);
    profile_record(*p, p->stage2, stream);
    ++p->launches;
    p->pair_visits += p->atoms * (p->atoms - 1) / 2;
    p->center_distance_evaluations += p->atoms * (p->atoms - 1) / 2;
    if (p->center_pairs) ++p->center_geometry_preparations;
    finished(*p, stream);
    profile_elapsed(*p, p->setup_transfer_ms, p->stage0, p->stage1);
    profile_elapsed(*p, p->setup_validation_ms, p->stage1, p->stage2);
  });
}
int stationary_tasks(void* pointer, const int64_t* tasks, const double* charges, size_t count,
                     char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !tasks || !charges || !count || count > p->task_capacity)
      throw std::invalid_argument("invalid stationary task page");
    check(*p);
    drain_geometry(*p);
    size_t primitive_work = 0;
    for (size_t i = 0; i < count; ++i) {
      const auto* task = tasks + task_stride * i;
      const auto source = task[1], rank = task[2], nucleus = task[3], work = task[8];
      if (!stationary_integral_source(source) || (rank != 2 && rank != 4) ||
          (nucleus >= 0 && (rank != 2 || nucleus >= int64_t(p->atoms))) || work <= 0)
        throw std::invalid_argument("invalid stationary task descriptor");
      for (size_t center = 0; center < size_t(rank); ++center)
        if (task[4 + center] < 0 || task[4 + center] >= int64_t(p->aos))
          throw std::invalid_argument("invalid stationary AO task index");
      if (size_t(work) > p->max_page_primitive_work ||
          primitive_work > p->max_page_primitive_work - size_t(work))
        throw std::invalid_argument("stationary primitive page work budget exceeded");
      primitive_work += size_t(work);
    }
    p->check_page_primitive_work(primitive_work);
    auto stream = p->context.stream;
    profile_record(*p, p->stage0, stream);
    upload(*p, p->tasks, tasks, task_stride * count, stream);
    upload(*p, p->task_charges, charges, count, stream);
    profile_record(*p, p->stage1, stream);
    task_kernel<<<blocks(count, 64), 64, 0, stream>>>(
        p->tasks, p->task_charges, count, p->primitive_table, p->primitives, p->ao_ranges,
        p->ao_norms, p->ao_atoms, p->centers, p->density, p->weighted_density, p->aos, p->atoms,
        p->task_values, p->context.error);
    profile_record(*p, p->stage2, stream);
    task_reduce<<<blocks(3 * stationary_source_count * p->atoms, 64), 64, 0, stream>>>(
        p->task_values, p->tasks, count, p->ao_atoms, p->atoms, p->sources, p->context.error);
    profile_record(*p, p->stage3, stream);
    p->launches += 2;
    p->count_primitive_work(primitive_work);
    p->task_count += count;
    ++p->task_batches;
    finished(*p, stream);
    profile_elapsed(*p, p->primitive_h2d_ms, p->stage0, p->stage1);
    profile_elapsed(*p, p->primitive_kernel_ms, p->stage1, p->stage2);
    profile_elapsed(*p, p->primitive_reduction_ms, p->stage2, p->stage3);
  });
}
int stationary_nuclear(void* pointer, unsigned kind, int64_t a, int64_t b, double za, double zb,
                       char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p) throw std::invalid_argument("invalid stationary owner");
    check(*p);
    drain_geometry(*p);
    p->check_page_primitive_work(1);
    auto stream = p->context.stream;
    profile_record(*p, p->stage0, stream);
    nuclear_kernel<<<1, 1, 0, stream>>>(kind, a, b, za, zb, p->centers, p->atoms, p->sources,
                                        p->context.error);
    profile_record(*p, p->stage1, stream);
    ++p->launches;
    p->count_primitive_work(1);
    finished(*p, stream);
    profile_elapsed(*p, p->primitive_kernel_ms, p->stage0, p->stage1);
  });
}
int stationary_nuclear_all(void* pointer, unsigned kind, const double* charges, size_t count,
                           char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !charges || count != p->atoms)
      throw std::invalid_argument("invalid stationary nuclear batch");
    check(*p);
    drain_geometry(*p);
    const auto pair_work = p->atoms * (p->atoms - 1) / 2;
    p->check_page_primitive_work(pair_work);
    auto stream = p->context.stream;
    profile_record(*p, p->stage0, stream);
    upload(*p, p->scratch, charges, p->atoms, stream);
    profile_record(*p, p->stage1, stream);
    nuclear_all_kernel<<<1, 1, 0, stream>>>(kind, p->scratch, p->centers, p->atoms, p->sources,
                                            p->context.error);
    profile_record(*p, p->stage2, stream);
    ++p->launches;
    p->count_primitive_work(pair_work);
    finished(*p, stream);
    profile_elapsed(*p, p->primitive_h2d_ms, p->stage0, p->stage1);
    profile_elapsed(*p, p->primitive_kernel_ms, p->stage1, p->stage2);
  });
}
int stationary_geometry_external(void* pointer, const generativeqc::dft::GridTaskView* view,
                                 const double* work, const int64_t* owners, const double* weights,
                                 const double* raw, const double* external, char* error,
                                 size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points)
      throw std::invalid_argument("invalid geometry task lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    drain_geometry(*p);
    auto stream = view->stream;
    // This optional bounded tile is accounted separately by the nonlocal
    // caller (6*npoint FP64 values). It lives through the borrowed stream.
    generativeqc::runtime::OwnedCudaBuffer<double> seeds;
    if (external) {
      for (size_t i = 0; i < 6 * view->npoint; ++i)
        if (!std::isfinite(external[i]))
          throw std::invalid_argument("nonfinite nonlocal geometry seed");
      seeds.allocate(p->context.device, 6 * view->npoint, stream);
      upload(*p, seeds.get(), external, 6 * view->npoint, stream);
    }
    // CudaGrid synchronized its producer before lending this view. Finish on
    // the SAME borrowed stream before the lease ends; retain no task pointers.
    try {
      profile_record(*p, p->stage0, stream);
      upload(*p, p->point_atoms, owners, view->npoint, stream);
      upload(*p, p->weights, weights, view->npoint, stream);
      upload(*p, p->raw, raw, view->npoint, stream);
      profile_record(*p, p->stage1, stream);
      launch_geometry(*p, stream, *view, work, p->ao_atoms, p->point_atoms, 0, 0, p->centers,
                      p->atoms, p->weights, p->raw, seeds.get(), view->npoint, 0, geometry_lanes,
                      p->partial, p->scratch, p->center_pairs, p->context.error);
      profile_record(*p, p->stage2, stream);
      geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
          p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
          p->context.error);
      profile_record(*p, p->stage3, stream);
      p->launches += 2;
      p->point_count += view->npoint;
      p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
      p->becke_pair_state_evaluations +=
          view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
      if (!p->center_pairs)
        p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
      ++p->geometry_batches;
      finished(*p, stream);
      profile_elapsed(*p, p->geometry_h2d_ms, p->stage0, p->stage1);
      profile_elapsed(*p, p->geometry_kernel_ms, p->stage1, p->stage2);
      profile_elapsed(*p, p->geometry_reduction_ms, p->stage2, p->stage3);
    } catch (...) {
      // Even an upload/launch failure must drain the borrowed stream before
      // our arena can be freed or the grid owner can reuse its leased buffers.
      const auto sync_begin = std::chrono::steady_clock::now();
      if (cudaStreamSynchronize(stream) == cudaSuccess) {
        if (p->profile) {
          p->synchronization_wait_ms += std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - sync_begin)
                                            .count();
        }
        ++p->synchronizations;
      }
      throw;
    }
  });
}
int stationary_geometry_external_device(void* pointer, const generativeqc::dft::GridTaskView* view,
                                        const double* work, const int64_t* owners,
                                        const double* weights, const double* raw,
                                        const double* external_device, size_t external_stride,
                                        size_t external_offset, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points || !external_device || external_stride < external_offset ||
        view->npoint > external_stride - external_offset)
      throw std::invalid_argument("invalid resident nonlocal geometry seed lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    drain_geometry(*p);
    auto stream = view->stream;
    // external_device is caller-owned device memory. The caller guarantees
    // readiness on this borrowed stream and retains the allocation until this
    // synchronous gate returns. The kernel validates all six seed fields.
    try {
      profile_record(*p, p->stage0, stream);
      upload(*p, p->point_atoms, owners, view->npoint, stream);
      upload(*p, p->weights, weights, view->npoint, stream);
      upload(*p, p->raw, raw, view->npoint, stream);
      profile_record(*p, p->stage1, stream);
      launch_geometry(*p, stream, *view, work, p->ao_atoms, p->point_atoms, 0, 0, p->centers,
                      p->atoms, p->weights, p->raw, external_device, external_stride,
                      external_offset, geometry_lanes, p->partial, p->scratch, p->center_pairs,
                      p->context.error);
      profile_record(*p, p->stage2, stream);
      geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
          p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
          p->context.error);
      profile_record(*p, p->stage3, stream);
      p->launches += 2;
      p->point_count += view->npoint;
      p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
      p->becke_pair_state_evaluations +=
          view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
      if (!p->center_pairs)
        p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
      ++p->geometry_batches;
      finished(*p, stream);
      profile_elapsed(*p, p->geometry_h2d_ms, p->stage0, p->stage1);
      profile_elapsed(*p, p->geometry_kernel_ms, p->stage1, p->stage2);
      profile_elapsed(*p, p->geometry_reduction_ms, p->stage2, p->stage3);
    } catch (...) {
      const auto sync_begin = std::chrono::steady_clock::now();
      if (cudaStreamSynchronize(stream) == cudaSuccess) {
        if (p->profile) {
          p->synchronization_wait_ms += std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - sync_begin)
                                            .count();
        }
        ++p->synchronizations;
      }
      throw;
    }
  });
}
int stationary_geometry_external_device_enqueue(
    void* pointer, const generativeqc::dft::GridTaskView* view, const double* work,
    const int64_t* owners, const double* weights, const double* raw, const double* external_device,
    size_t external_stride, size_t external_offset, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points || !external_device || external_stride < external_offset ||
        view->npoint > external_stride - external_offset)
      throw std::invalid_argument("invalid deferred resident nonlocal geometry seed lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    auto stream = view->stream;
    if (p->geometry_pending && p->geometry_stream != stream)
      throw std::invalid_argument("stationary deferred geometry stream changed before drain");
    if (!p->geometry_pending) {
      p->geometry_stream = stream;
      p->geometry_pending = true;
    }
    // The caller retains the full-grid seed owner until drain_geometry().
    // Stream order protects both the borrowed GridTaskView and device seeds.
    upload(*p, p->point_atoms, owners, view->npoint, stream);
    upload(*p, p->weights, weights, view->npoint, stream);
    upload(*p, p->raw, raw, view->npoint, stream);
    launch_geometry(*p, stream, *view, work, p->ao_atoms, p->point_atoms, 0, 0, p->centers,
                    p->atoms, p->weights, p->raw, external_device, external_stride, external_offset,
                    geometry_lanes, p->partial, p->scratch, p->center_pairs, p->context.error);
    geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
        p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
        p->context.error);
    cuda_check(cudaGetLastError());
    p->launches += 2;
    p->point_count += view->npoint;
    p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
    p->becke_pair_state_evaluations +=
        view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
    if (!p->center_pairs)
      p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
    ++p->geometry_batches;
  });
}

int stationary_geometry_external_device_molecular_enqueue(
    void* pointer, const generativeqc::dft::GridTaskView* view, const double* work,
    size_t owner_offset, size_t points_per_atom, const double* weights, const double* raw,
    const double* external_device, size_t external_stride, size_t external_offset, char* error,
    size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points || !external_device || !points_per_atom ||
        points_per_atom > SIZE_MAX / p->atoms || owner_offset > p->atoms * points_per_atom ||
        view->npoint > p->atoms * points_per_atom - owner_offset ||
        external_stride < external_offset || view->npoint > external_stride - external_offset)
      throw std::invalid_argument("invalid implicit-owner nonlocal geometry lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    auto stream = view->stream;
    if (p->geometry_pending && p->geometry_stream != stream)
      throw std::invalid_argument("stationary deferred geometry stream changed before drain");
    if (!p->geometry_pending) {
      p->geometry_stream = stream;
      p->geometry_pending = true;
    }
    upload(*p, p->weights, weights, view->npoint, stream);
    upload(*p, p->raw, raw, view->npoint, stream);
    launch_geometry(*p, stream, *view, work, p->ao_atoms, nullptr, owner_offset, points_per_atom,
                    p->centers, p->atoms, p->weights, p->raw, external_device, external_stride,
                    external_offset, geometry_lanes, p->partial, p->scratch, p->center_pairs,
                    p->context.error);
    geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
        p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
        p->context.error);
    cuda_check(cudaGetLastError());
    p->launches += 2;
    p->point_count += view->npoint;
    p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
    p->becke_pair_state_evaluations +=
        view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
    if (!p->center_pairs)
      p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
    ++p->geometry_batches;
  });
}

int stationary_geometry_external_device_molecular_resident_weights_enqueue(
    void* pointer, const generativeqc::dft::GridTaskView* view, const double* work,
    size_t owner_offset, size_t points_per_atom, const double* device_weights,
    const double* device_raw, const double* external_device, size_t external_stride,
    size_t external_offset, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points || !device_weights || !device_raw || !external_device ||
        !points_per_atom || points_per_atom > SIZE_MAX / p->atoms ||
        owner_offset > p->atoms * points_per_atom ||
        view->npoint > p->atoms * points_per_atom - owner_offset ||
        external_stride < external_offset || view->npoint > external_stride - external_offset)
      throw std::invalid_argument("invalid resident-weight nonlocal geometry lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    auto stream = view->stream;
    if (p->geometry_pending && p->geometry_stream != stream)
      throw std::invalid_argument("stationary deferred geometry stream changed before drain");
    if (!p->geometry_pending) {
      p->geometry_stream = stream;
      p->geometry_pending = true;
    }
    launch_geometry(*p, stream, *view, work, p->ao_atoms, nullptr, owner_offset, points_per_atom,
                    p->centers, p->atoms, device_weights, device_raw, external_device,
                    external_stride, external_offset, geometry_lanes, p->partial, p->scratch,
                    p->center_pairs, p->context.error);
    geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
        p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
        p->context.error);
    cuda_check(cudaGetLastError());
    p->launches += 2;
    p->point_count += view->npoint;
    p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
    p->becke_pair_state_evaluations +=
        view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
    if (!p->center_pairs)
      p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
    ++p->geometry_batches;
  });
}

int stationary_geometry_molecular_resident_weights_enqueue(
    void* pointer, const generativeqc::dft::GridTaskView* view, const double* work,
    size_t owner_offset, size_t points_per_atom, const double* device_weights,
    const double* device_raw, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points || !device_weights || !device_raw || !points_per_atom ||
        points_per_atom > SIZE_MAX / p->atoms || owner_offset > p->atoms * points_per_atom ||
        view->npoint > p->atoms * points_per_atom - owner_offset)
      throw std::invalid_argument("invalid resident-weight geometry task lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    auto stream = view->stream;
    if (p->geometry_pending && p->geometry_stream != stream)
      throw std::invalid_argument("stationary deferred geometry stream changed before drain");
    if (!p->geometry_pending) {
      p->geometry_stream = stream;
      p->geometry_pending = true;
    }
    launch_geometry(*p, stream, *view, work, p->ao_atoms, nullptr, owner_offset, points_per_atom,
                    p->centers, p->atoms, device_weights, device_raw, nullptr, 0, 0, geometry_lanes,
                    p->partial, p->scratch, p->center_pairs, p->context.error);
    geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
        p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
        p->context.error);
    cuda_check(cudaGetLastError());
    p->launches += 2;
    p->point_count += view->npoint;
    p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
    p->becke_pair_state_evaluations +=
        view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
    if (!p->center_pairs)
      p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
    ++p->geometry_batches;
  });
}

int stationary_geometry_molecular_enqueue(void* pointer,
                                          const generativeqc::dft::GridTaskView* view,
                                          const double* work, size_t owner_offset,
                                          size_t points_per_atom, const double* weights,
                                          const double* raw, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points || !points_per_atom || points_per_atom > SIZE_MAX / p->atoms ||
        owner_offset > p->atoms * points_per_atom ||
        view->npoint > p->atoms * points_per_atom - owner_offset)
      throw std::invalid_argument("invalid implicit-owner geometry task lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    auto stream = view->stream;
    if (p->geometry_pending && p->geometry_stream != stream)
      throw std::invalid_argument("stationary deferred geometry stream changed before drain");
    if (!p->geometry_pending) {
      p->geometry_stream = stream;
      p->geometry_pending = true;
    }
    upload(*p, p->weights, weights, view->npoint, stream);
    upload(*p, p->raw, raw, view->npoint, stream);
    launch_geometry(*p, stream, *view, work, p->ao_atoms, nullptr, owner_offset, points_per_atom,
                    p->centers, p->atoms, p->weights, p->raw, nullptr, 0, 0, geometry_lanes,
                    p->partial, p->scratch, p->center_pairs, p->context.error);
    geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
        p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
        p->context.error);
    cuda_check(cudaGetLastError());
    p->launches += 2;
    p->point_count += view->npoint;
    p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
    p->becke_pair_state_evaluations +=
        view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
    if (!p->center_pairs)
      p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
    ++p->geometry_batches;
  });
}

int stationary_geometry_enqueue(void* pointer, const generativeqc::dft::GridTaskView* view,
                                const double* work, const int64_t* owners, const double* weights,
                                const double* raw, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || !valid_geometry_ao_map(*view, p->aos) ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao || !view->points)
      throw std::invalid_argument("invalid deferred geometry task lease");
    check(*p);
    if (!view->npoint) return;
    const size_t geometry_lanes = std::min(p->geometry_lanes, view->npoint);
    p->geometry_peak_lanes = std::max(p->geometry_peak_lanes, geometry_lanes);
    auto stream = view->stream;
    if (p->geometry_pending && p->geometry_stream != stream)
      throw std::invalid_argument("stationary deferred geometry stream changed before drain");
    if (!p->geometry_pending) {
      p->geometry_stream = stream;
      p->geometry_pending = true;
    }
    // The borrowed task pointers are consumed only by work enqueued here.
    // Later grid tiles reuse the same buffers on the same stream, so CUDA
    // stream order protects their lifetime without a per-tile host fence.
    upload(*p, p->point_atoms, owners, view->npoint, stream);
    upload(*p, p->weights, weights, view->npoint, stream);
    upload(*p, p->raw, raw, view->npoint, stream);
    launch_geometry(*p, stream, *view, work, p->ao_atoms, p->point_atoms, 0, 0, p->centers,
                    p->atoms, p->weights, p->raw, nullptr, 0, 0, geometry_lanes, p->partial,
                    p->scratch, p->center_pairs, p->context.error);
    geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
        p->partial, p->atoms, geometry_lanes, p->sources + 3 * stationary_xc_source * p->atoms,
        p->context.error);
    cuda_check(cudaGetLastError());
    p->launches += 2;
    p->point_count += view->npoint;
    p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
    p->becke_pair_state_evaluations +=
        view->npoint * p->atoms * (p->atoms - 1) / (p->retains_becke_pair_state() ? 2 : 1);
    if (!p->center_pairs)
      p->center_distance_evaluations += view->npoint * p->atoms * (p->atoms - 1);
    ++p->geometry_batches;
  });
}
int stationary_geometry(void* pointer, const generativeqc::dft::GridTaskView* view,
                        const double* work, const int64_t* owners, const double* weights,
                        const double* raw, char* error, size_t size) {
  return stationary_geometry_external(pointer, view, work, owners, weights, raw, nullptr, error,
                                      size);
}
int stationary_geometry_drain(void* pointer, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p) throw std::invalid_argument("invalid stationary owner");
    check(*p);
    drain_geometry(*p);
  });
}
int stationary_finish(void* pointer, double* output, size_t count, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !output || count != 3 * stationary_source_count * p->atoms)
      throw std::invalid_argument("invalid source output");
    check(*p);
    drain_geometry(*p);
    finished(*p, p->context.stream);
    // Host output is touched only after every device source passed its gate.
    std::vector<double> candidate(count);
    auto d2h_begin = std::chrono::steady_clock::time_point{};
    if (p->profile) d2h_begin = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(candidate.data(), p->sources, count * 8, cudaMemcpyDeviceToHost));
    if (p->profile) {
      p->final_d2h_wall_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d2h_begin)
              .count();
    }
    ++p->d2h_calls;
    p->downloads += count * 8;
    for (double v : candidate)
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite gradient");
    std::copy(candidate.begin(), candidate.end(), output);
  });
}
int stationary_finish_span(void* pointer, size_t source_begin, size_t source_count, double* output,
                           size_t count, char* error, size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !output || !source_count || source_begin >= stationary_source_count ||
        source_count > stationary_source_count - source_begin ||
        count != 3 * source_count * p->atoms)
      throw std::invalid_argument("invalid source span output");
    check(*p);
    drain_geometry(*p);
    finished(*p, p->context.stream);
    std::vector<double> candidate(count);
    auto d2h_begin = std::chrono::steady_clock::time_point{};
    if (p->profile) d2h_begin = std::chrono::steady_clock::now();
    const auto offset = 3 * source_begin * p->atoms;
    cuda_check(
        cudaMemcpy(candidate.data(), p->sources + offset, count * 8, cudaMemcpyDeviceToHost));
    if (p->profile) {
      p->final_d2h_wall_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d2h_begin)
              .count();
    }
    ++p->d2h_calls;
    p->downloads += count * 8;
    for (double v : candidate)
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite gradient source span");
    std::copy(candidate.begin(), candidate.end(), output);
  });
}
int stationary_finish_reduced(void* pointer, double* output, size_t count, char* error,
                              size_t size) {
  using namespace generativeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !output || count != 3 * p->atoms)
      throw std::invalid_argument("invalid reduced output");
    if (!stationary_native_reduction_supported)
      throw std::invalid_argument("stationary source inventory requires external reduction");
    check(*p);
    drain_geometry(*p);
    auto stream = p->context.stream;
    source_reduce<<<blocks(3 * p->atoms, 64), 64, 0, stream>>>(p->sources, p->atoms, p->partial,
                                                               p->context.error);
    ++p->launches;
    finished(*p, stream);
    std::vector<double> candidate(count);
    auto d2h_begin = std::chrono::steady_clock::time_point{};
    if (p->profile) d2h_begin = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(candidate.data(), p->partial, count * 8, cudaMemcpyDeviceToHost));
    if (p->profile) {
      p->final_d2h_wall_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d2h_begin)
              .count();
    }
    ++p->d2h_calls;
    p->downloads += count * 8;
    for (double v : candidate)
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite reduced gradient");
    std::copy(candidate.begin(), candidate.end(), output);
  });
}
int stationary_profile_metrics(void* pointer, double* output, size_t count) {
  auto* p = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!p || !output || count != 10) return 1;
  const double values[]{p->synchronization_wait_ms, p->setup_transfer_ms,
                        p->setup_validation_ms,     p->primitive_h2d_ms,
                        p->primitive_kernel_ms,     p->primitive_reduction_ms,
                        p->geometry_h2d_ms,         p->geometry_kernel_ms,
                        p->geometry_reduction_ms,   p->final_d2h_wall_ms};
  std::copy(values, values + 10, output);
  return 0;
}
int stationary_metrics(void* pointer, uint64_t* output, size_t count) {
  auto* p = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!p || !output || count != 24) return 1;
  const uint64_t values[]{p->bytes,
                          p->uploads,
                          p->downloads,
                          p->launches,
                          p->primitive_count,
                          p->point_count,
                          p->pair_visits,
                          reinterpret_cast<uintptr_t>(p->context.stream),
                          p->task_count,
                          p->task_batches,
                          p->h2d_calls,
                          p->d2h_calls,
                          p->synchronizations,
                          p->geometry_batches,
                          p->geometry_lanes,
                          p->geometry_threads,
                          18 * p->geometry_lanes * p->atoms * sizeof(double),
                          p->geometry_peak_lanes,
                          p->center_geometry_bytes,
                          p->center_distance_evaluations,
                          p->center_geometry_preparations,
                          p->becke_threads_per_point,
                          p->becke_shared_bytes,
                          p->becke_pair_state_evaluations};
  std::copy(values, values + 24, output);
  return 0;
}
void stationary_destroy(void* pointer) {
  auto* p = static_cast<generativeqc_stationary_cuda::Owner*>(pointer);
  if (!p) return;
  int previous = 0;
  const bool have_device = cudaGetDevice(&previous) == cudaSuccess;
  if (cudaSetDevice(p->context.device) == cudaSuccess) {
    if (p->geometry_pending && p->geometry_stream) cudaStreamSynchronize(p->geometry_stream);
    p->geometry_pending = false;
    p->geometry_stream = nullptr;
    if (p->stage0) cudaEventDestroy(p->stage0);
    if (p->stage1) cudaEventDestroy(p->stage1);
    if (p->stage2) cudaEventDestroy(p->stage2);
    if (p->stage3) cudaEventDestroy(p->stage3);
    for (auto event : p->becke_events)
      if (event) cudaEventDestroy(event);
  }
  if (have_device) cudaSetDevice(previous);
  delete p;
}
}
