#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "dft/dispersion/d3_response.hpp"
#include "dft/dispersion/d3_runtime.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "dft/dispersion/d4_second_order.hpp"
#include "dft/dispersion/gcp_cuda.hpp"
#include "dft/hessian_aot.hpp"
#include "integrals/coulomb_derivative_envelope.hpp"
#include "response/cuda_ks_cpks_reference.hpp"
#include "response/wb97mv_stationary_assembly.cuh"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_wb97mv_stationary_maps;
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
struct HessianCapacityExceeded final : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void check(cudaError_t error, const char* phase = "stationary Hessian CUDA operation") {
  if (error == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(phase) + ": " + cudaGetErrorString(error));
}
std::size_t product(std::size_t first, std::size_t second) {
  require(!second || first <= SIZE_MAX / second, "stationary Hessian extent overflow");
  return first * second;
}
struct DrainOnFailure {
  cudaStream_t stream;
  bool active{true};
  ~DrainOnFailure() {
    // Failed submissions may leave earlier work borrowing caller/staging data.
    // The null handle is CUDA's valid default stream and must also drain.
    if (active) (void)cudaStreamSynchronize(stream);
  }
};
struct Layout {
  std::size_t bytes{};
  std::size_t reserve(std::size_t count, std::size_t size = sizeof(double)) {
    const auto offset = bytes, extra = product(count, size);
    require(extra <= SIZE_MAX - 255 && bytes <= SIZE_MAX - extra - 255,
            "stationary Hessian arena overflow");
    bytes = (bytes + extra + 255) / 256 * 256;
    return offset;
  }
};
struct Range {
  const void* pointer{};
  std::size_t bytes{};
};
bool disjoint(Range first, Range second) {
  const auto left = reinterpret_cast<std::uintptr_t>(first.pointer),
             right = reinterpret_cast<std::uintptr_t>(second.pointer);
  return left <= right ? right - left >= first.bytes : left - right >= second.bytes;
}
__global__ void finite(const double* input, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(input[index])) atomicExch(status, 1);
}
__global__ void symmetric(const double* input, std::size_t count, double absolute, double relative,
                          int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count * count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = input[index], other = input[index % count * count + index / count];
    if (!isfinite(value) || !isfinite(other) ||
        fabs(value - other) > absolute + relative * fmax(fabs(value), fabs(other)))
      atomicExch(status, 1);
  }
}
__global__ void unit(double* output, std::size_t count, std::size_t coordinate) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[index] = index == coordinate ? 1.0 : 0.0;
}
__global__ void add_fitted_range(double* total, const double* range, std::size_t count) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    total[index] += range[index];
}

__global__ void d4_correction_direction(int atoms, const std::int32_t* numbers, const double* xyz,
                                        const double* direction, double total_charge,
                                        dispersion::D4Parameters parameters,
                                        dispersion::D4Tables tables, dispersion::EEQTables charge,
                                        double* scratch, double* gradient, double* action,
                                        int* status) {
  const auto result = dispersion::d4_eeq_direction(
      atoms, numbers, xyz, direction, total_charge, parameters, tables, charge, scratch,
      dispersion::d4_eeq_direction_workspace_elements(atoms), gradient, action);
  *status = static_cast<int>(result);
}

__global__ void add_correction_direction(std::size_t coordinates, const double* d4,
                                         const double* gcp, const int* statuses, double* result,
                                         int* failure) {
  const auto coordinate = blockIdx.x * blockDim.x + threadIdx.x;
  if (coordinate >= coordinates) return;
  if (statuses[0] || (gcp && statuses[1 + coordinate / 3])) {
    atomicExch(failure, 1);
    return;
  }
  const double candidate = result[coordinate] + d4[coordinate] + (gcp ? gcp[coordinate] : 0.0);
  if (!isfinite(candidate))
    atomicExch(failure, 1);
  else
    result[coordinate] = candidate;
}

/** Separate launches provide global ordering without a cooperative grid or
 * atomics on FP64 outputs. Each thread owns one atom of response workspace. */
template <unsigned Phase>
__global__ void d3_direction(dispersion::D3CudaDirectionSource source,
                             dispersion::D3ModelParameters model, const double* geometry,
                             const double* motion, double* workspace, double* action,
                             double* candidate, int* status) {
  const auto atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= source.atoms || atomicAdd(status, 0)) return;
  namespace d3 = dispersion::d3_response;
  const auto work = d3::workspace(workspace, source.atoms);
  const auto cutoff = d3::cn_cutoff(model);
  auto result = dispersion::D3Status::success;
  if constexpr (Phase == 0) {
    result = d3::weights_for_atom(source.atoms, atom, source.numbers, geometry, motion, cutoff,
                                  source.tables, work);
  } else if constexpr (Phase == 1) {
    result = d3::local_response(source.atoms, atom, source.numbers, geometry, motion, model,
                                source.tables, work, action + 3 * atom);
  } else {
    result = d3::cn_pullback(source.atoms, atom, source.numbers, geometry, motion, cutoff,
                             source.tables, work, action + 3 * atom);
    if (result == dispersion::D3Status::success)
      for (int axis = 0; axis < 3; ++axis) {
        const auto component = 3 * atom + axis;
        const auto total = candidate[component] + action[component];
        if (!isfinite(total))
          result = dispersion::D3Status::numerical_failure;
        else
          candidate[component] = total;
      }
  }
  if (result != dispersion::D3Status::success) atomicExch(status, 1);
}

__global__ void frozen_exact_response(const double* derivatives, const double* frozen,
                                      const double* delta_density, std::size_t panels,
                                      std::size_t coordinates, std::size_t column, double* output,
                                      int* status, bool screened) {
  for (std::size_t left = blockIdx.x * blockDim.x + threadIdx.x; left < coordinates;
       left += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value = frozen[column * coordinates + left];
    for (std::size_t index = 0; index < panels; ++index) {
      const auto response = delta_density[index];
      if (screened &&
          !(fabs(response) <= integrals::CoulombDerivativeOmissions::max_response_element))
        atomicExch(status, 1);
      value += derivatives[left * panels + index] * response;
    }
    if (!isfinite(value)) atomicExch(status, 1);
    output[left] = value;
  }
}
__global__ void frozen_core_response(const double* derivatives, const double* delta_weights,
                                     std::size_t matrix, std::size_t coordinates,
                                     std::size_t column, double* output, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto* frozen = derivatives + coordinates * (2 * matrix) + column * coordinates;
  for (std::size_t left = blockIdx.x * blockDim.x + threadIdx.x; left < coordinates;
       left += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value = frozen[left];
    for (std::size_t index = 0; index < 2 * matrix; ++index)
      value += derivatives[left * (2 * matrix) + index] * delta_weights[index];
    if (!isfinite(value)) atomicExch(status, 1);
    output[left] = value;
  }
}
__global__ void pack_pairs(const double* basis, const double* charges, const double* direction,
                           const std::size_t* atoms, std::size_t pairs, double* first_charge,
                           double* second_charge, double* first_position, double* second_position,
                           double* first_motion, double* second_motion) {
  for (std::size_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pairs;
       pair += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto first = atoms[pair], second = atoms[pairs + pair];
    first_charge[pair] = charges[first];
    second_charge[pair] = charges[second];
    for (unsigned axis = 0; axis < 3; ++axis) {
      first_position[axis * pairs + pair] = basis[3 * first + axis];
      second_position[axis * pairs + pair] = basis[3 * second + axis];
      first_motion[axis * pairs + pair] = direction[3 * first + axis];
      second_motion[axis * pairs + pair] = direction[3 * second + axis];
    }
  }
}
/** Gather each physical incidence exactly once; generated maps own its sign
 * and reduction. Triangular indexing gives O(atoms^2), not O(atoms*pairs). */
__global__ void gather_atom(const double* first, const double* second, std::size_t atoms,
                            std::size_t atom, std::size_t pairs, double* incidences) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 3 * (atoms - 1);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto axis = index / (atoms - 1), neighbor = index % (atoms - 1),
               other = neighbor < atom ? neighbor : neighbor + 1, low = atom < other ? atom : other,
               high = atom < other ? other : atom,
               pair = low * (2 * atoms - low - 1) / 2 + high - low - 1;
    incidences[index] = (atom < other ? first : second)[axis * pairs + pair];
  }
}
struct DenseAssemblyAudit {
  std::size_t completed;
  int failed;
  response::CudaCpksDenseReceipt cpks;
};

__global__ void store_column(const double* vector, std::size_t count, std::size_t column,
                             const int* column_status, double* matrix, DenseAssemblyAudit* audit) {
  const auto flags = *column_status;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // One writer per column on the owner stream, before its status is reused.
    if (flags)
      audit->failed |= flags;
    else
      ++audit->completed;
  }
  if (flags) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    matrix[index * count + column] = vector[index];
}
}  // namespace

struct CudaKsWb97mvHessian::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsWb97mvHessianOptions options;
  CudaKsGridReference grid_reference, vv10_reference;
  response::CudaKsCpksReference cpks_reference;
  response::CudaCpksOperatorCache cpks_operator;
  CudaKsResidentFockBinding fock;
  CudaKsResidentOrbitalFrameBinding frame;
  CudaKsWb97mvResponse induced;
  CudaGridGeometryResponse geometry, vv10_geometry;
  CudaWb97mvGridResponse grid, vv10_grid;
  CudaKsOneElectronResponse one_electron;
  CudaKsWb97mvNuclearJk exact;
  struct FittedSource {
    CudaKsWb97mvDfResponse response;
    CudaKsWb97mvDfNuclear nuclear;
    std::size_t raw{}, metric{}, values{}, weights{}, energy{}, fock{}, direction{};
  } full_df, range_df;
  CudaKsWb97mvEcpNuclear ecp;
  methods::detail::KsHessianCorrections corrections;
  dispersion::D3CudaDirectionSource d3_source{};
  dispersion::D4Tables d4_tables{};
  dispersion::EEQTables eeq_tables{};
  std::size_t correction_geometry{}, correction_action{}, d3_scratch{}, d4_numbers{}, d4_elements{},
      d4_references{}, d4_c6{}, eeq_parameters{}, d4_scratch{}, d4_gradient{}, gcp_action{},
      d4_status{};
  CudaKsWb97mvHessianWork receipt;
  const CudaKsWb97mvHessianMode mode;
  bool has_df{}, has_ecp{}, has_exact{};
  cudaStream_t stream{};
  void* arena{};
  void* operator_matrix{};
  double* frozen_exact{};
  double* frozen_core{};
  bool frozen_exact_ready{};
  bool frozen_core_ready{};
  std::size_t bytes{}, remaining{}, matrix{}, panels{}, coordinates{}, pairs{}, range_vector{};
  std::size_t map_offset{}, map_bytes{}, cpks_offset{}, cpks_bytes{}, geometry_offset{},
      geometry_bytes{}, grid_offset{}, grid_bytes{}, status_offset{}, dense_audit_offset{},
      vv10_geometry_offset{}, vv10_geometry_bytes{}, vv10_grid_offset{}, vv10_grid_bytes{},
      vv10_direction{};
  std::size_t zero{}, weights{}, delta_density{}, delta_weights{}, totals{}, delta_totals{}, core{},
      overlap{}, core_direction{}, overlap_direction{}, exact_energy{}, exact_fock{},
      exact_direction{}, xc_direction{}, rhs{}, ecp_values{}, vectors{}, scratch_gradient{},
      direction{}, left{}, candidate{}, complete{}, charges{}, pair_atoms{}, pair_fields{},
      pair_inputs{}, pair_first{}, pair_second{}, incidences{};

  Impl(methods::PreparedBatch& owner, std::size_t index, CudaKsFinalStateToken token,
       CudaKsWb97mvHessianOptions policy, CudaKsWb97mvHessianMode preparation_mode)
      : batch(owner),
        item(index),
        expected(std::move(token)),
        options(policy),
        grid_reference(batch, item, expected),
        vv10_reference(batch, item, expected, CudaKsGridRole::separate_nonlocal),
        cpks_reference(batch, item, expected),
        mode(preparation_mode),
        remaining(policy.max_device_bytes) {}
  ~Impl() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (operator_matrix) (void)runtime::resource_cuda_free(operator_matrix);
    if (frozen_exact) (void)runtime::resource_cuda_free(frozen_exact);
    if (frozen_core) (void)runtime::resource_cuda_free(frozen_core);
    if (arena) (void)runtime::resource_cuda_free(arena);
  }
  template <class Value = double>
  Value* at(std::size_t offset) {
    return reinterpret_cast<Value*>(static_cast<char*>(arena) + offset);
  }
  double* vector(std::size_t source) { return at(vectors) + source * coordinates; }
  maps::Extents extents() const {
    return {3, coordinates, frame.nao, coordinates / 3 - 1, pairs, frame.spins};
  }
  void charge(std::size_t size) {
    if (size > remaining)
      throw HessianCapacityExceeded(
          "stationary Hessian aggregate device budget exceeded (requested " + std::to_string(size) +
          " bytes, remaining " + std::to_string(remaining) + " of " +
          std::to_string(options.max_device_bytes) + " bytes)");
    remaining -= size;
    receipt.device_bytes += size;
  }
  cudaError_t authenticate() noexcept {
    try {
      CudaKsResidentFockBinding current;
      CudaKsResidentOrbitalFrameBinding current_frame;
      std::string detail;
      if (methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, current, detail) !=
              GENERATIVEQC_STATUS_SUCCESS ||
          current != fock || !cuda_ks_hessian_model_matches(expected, current) ||
          methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, current_frame,
                                                           detail) != GENERATIVEQC_STATUS_SUCCESS ||
          current_frame != frame)
        return cudaErrorInvalidResourceHandle;
      const auto lease = grid_reference.reference();
      if (!lease.authenticate) return cudaErrorInvalidResourceHandle;
      const auto error = lease.authenticate(lease.owner);
      if (error != cudaSuccess || !expected.identity.model.nonlocal_grid) return error;
      const auto vv10 = vv10_reference.reference();
      return vv10.authenticate ? vv10.authenticate(vv10.owner) : cudaErrorInvalidResourceHandle;
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
  cudaError_t gate() {
    int status{};
    auto error = cudaMemcpyAsync(&status, at<int>(status_offset), sizeof(status),
                                 cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    return error != cudaSuccess ? error : status ? cudaErrorInvalidValue : cudaSuccess;
  }

  cudaError_t correction_direction() {
    if (corrections.d3) {
      const auto policy = corrections.d3->parameters();
      const auto launch = [&]<unsigned Phase>() {
        d3_direction<Phase><<<maps::blocks(d3_source.atoms), 128, 0, stream>>>(
            d3_source, policy, at(correction_geometry), at(direction), at(d3_scratch),
            at(correction_action), at(candidate), at<int>(status_offset));
        return cudaGetLastError();
      };
      auto error = launch.template operator()<0>();
      if (error == cudaSuccess) error = launch.template operator()<1>();
      if (error == cudaSuccess) error = launch.template operator()<2>();
      if (error == cudaSuccess) ++receipt.correction_directions;
      return error;
    }
    if (!corrections.d4) return cudaSuccess;
    const auto atoms = static_cast<int>(coordinates / 3);
    const auto* centers = at(correction_geometry);
    d4_correction_direction<<<1, 1, 0, stream>>>(
        atoms, at<std::int32_t>(d4_numbers), centers, at(direction),
        static_cast<double>(fock.primary->system().charge), corrections.d4->parameters(), d4_tables,
        eeq_tables, at(d4_scratch), at(d4_gradient), at(correction_action), at<int>(d4_status));
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    double* gcp = nullptr;
    if (corrections.gcp) {
      gcp = at(gcp_action);
      error = corrections.gcp->launch_hessian_device(centers, at(direction), gcp,
                                                     at<int>(d4_status) + 1, stream);
      if (error != cudaSuccess) return error;
    }
    add_correction_direction<<<maps::blocks(coordinates), 128, 0, stream>>>(
        coordinates, at(correction_action), gcp, at<int>(d4_status), at(candidate),
        at<int>(status_offset));
    error = cudaGetLastError();
    if (error == cudaSuccess) ++receipt.correction_directions;
    return error;
  }
  cudaError_t admit(const double* input, double* output, bool full, cudaStream_t requested) {
    if (requested != stream) return cudaErrorInvalidResourceHandle;
    cudaStreamCaptureStatus capture{};
    auto error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    if ((error = authenticate()) != cudaSuccess) return error;
    const auto lease = grid_reference.reference();
    const auto geometry_lease = grid_reference.geometry_reference();
    const Range destination{
        output, product(full ? product(coordinates, coordinates) : coordinates, sizeof(double))};
    const Range source{input, input ? coordinates * sizeof(double) : 0};
    const auto device = [&](Range range) {
      if (!range.pointer || reinterpret_cast<std::uintptr_t>(range.pointer) % alignof(double))
        return cudaErrorInvalidValue;
      cudaPointerAttributes attributes{};
      const auto result = cudaPointerGetAttributes(&attributes, range.pointer);
      if (result != cudaSuccess) return result;
      return attributes.type == cudaMemoryTypeDevice && attributes.device == lease.device_id
                 ? cudaSuccess
                 : cudaErrorInvalidValue;
    };
    if ((error = device(destination)) != cudaSuccess) return error;
    if (!full && (error = device(source)) != cudaSuccess) return error;
    if (!disjoint(destination, {arena, bytes}) ||
        (frozen_core && !disjoint(destination, {frozen_core, receipt.frozen_core_bytes})) ||
        (frozen_exact && !disjoint(destination, {frozen_exact, receipt.frozen_exact_bytes})) ||
        (!full &&
         (!disjoint(source, {arena, bytes}) || !disjoint(source, destination) ||
          (frozen_core && !disjoint(source, {frozen_core, receipt.frozen_core_bytes})) ||
          (frozen_exact && !disjoint(source, {frozen_exact, receipt.frozen_exact_bytes})))))
      return cudaErrorInvalidValue;
    const auto shape = grid_reference.shape();
    const Range immutable[]{
        {lease.basis, (3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao) * sizeof(double)},
        {lease.points, 3 * shape.points * sizeof(double)},
        {lease.weights, shape.points * sizeof(double)},
        {geometry_lease.atomic_weights, shape.points * sizeof(double)},
        {geometry_lease.atomic_radii, shape.atoms * sizeof(double)},
        {geometry_lease.atom_offsets, (shape.atoms + 1) * sizeof(std::uint64_t)},
        {lease.density, panels * sizeof(double)},
        {frame.coefficients, panels * sizeof(double)},
        {frame.orbital_energies, frame.spins * frame.nao * sizeof(double)},
        {frame.overlap, matrix * sizeof(double)},
        {fock.hcore, matrix * sizeof(double)},
        {fock.fock, panels * sizeof(double)}};
    for (const auto range : immutable)
      if (!disjoint(destination, range)) return cudaErrorInvalidValue;
    if (expected.identity.model.nonlocal_grid) {
      const auto vv10 = vv10_reference.reference();
      const auto extent = vv10_reference.shape();
      const auto quad = vv10_reference.geometry_reference();
      const Range secondary[]{
          {vv10.basis,
           (3 * extent.atoms + 2 * extent.primitives + 16 * extent.nao) * sizeof(double)},
          {vv10.points, 3 * extent.points * sizeof(double)},
          {vv10.weights, extent.points * sizeof(double)},
          {quad.atomic_weights, extent.points * sizeof(double)},
          {quad.atomic_radii, extent.atoms * sizeof(double)},
          {quad.atom_offsets, (extent.atoms + 1) * sizeof(std::uint64_t)}};
      for (const auto range : secondary)
        if (!disjoint(destination, range)) return cudaErrorInvalidValue;
    }
    for (const auto* provider : {has_df ? fock.primary : nullptr, fock.fitted_range}) {
      if (!provider) continue;
      const auto* fitted_plan = provider->cuda_fitted_source();
      const auto source_df = scf::cuda_density_fitting_readonly_source(fitted_plan);
      const auto metric_df = scf::cuda_density_fitting_readonly_metric(fitted_plan);
      if (!source_df || !metric_df) return cudaErrorInvalidResourceHandle;
      if (!disjoint(destination,
                    {source_df.whitening, source_df.naux * source_df.naux * sizeof(double)}) ||
          !disjoint(destination, {metric_df.eigenvalues, source_df.naux * sizeof(double)}) ||
          !disjoint(destination,
                    {metric_df.eigenvectors, source_df.naux * source_df.naux * sizeof(double)}))
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }
  cudaError_t sum_weights(const double* density, const double* weight, std::size_t offset) {
    return maps::launch_weights(extents(), density, weight, at(offset), at(offset) + matrix,
                                at(map_offset), map_bytes, at<int>(status_offset), stream);
  }
  cudaError_t fitted_action(FittedSource& source, const double* density_direction,
                            const char*& failure_check) {
    const auto [raw, metric] = std::pair{source.raw, source.metric};
    const auto* values = at(source.values);
    auto* weight = at(source.weights);
    const CudaWb97mvDfOutput output{
        at(source.energy), at(source.fock),       at(source.direction),     weight,
        weight + raw,      weight + raw + metric, weight + 2 * raw + metric};
    return source.response.apply(density_direction, values + raw + metric,
                                 values + 2 * raw + metric, output, stream, &failure_check);
  }
  integrals::CudaOneElectronOutput core_output() {
    return {at(core), at(overlap), at(core_direction), at(overlap_direction), at(scratch_gradient),
            vector(0)};
  }
  CudaWb97mvJkOutput exact_output() {
    return {at(exact_energy), at(exact_fock), at(exact_direction), at(scratch_gradient), vector(3)};
  }
  cudaError_t nuclear() {
    if (!pairs) return cudaMemsetAsync(vector(6), 0, coordinates * sizeof(double), stream);
    const auto shape = grid_reference.shape();
    auto* fields = at(pair_fields);
    auto* packed = at(pair_inputs);
    pack_pairs<<<maps::blocks(pairs), 128, 0, stream>>>(
        grid_reference.reference().basis, at(charges), at(direction), at<std::size_t>(pair_atoms),
        pairs, fields, fields + pairs, fields + 2 * pairs, fields + 5 * pairs, fields + 8 * pairs,
        fields + 11 * pairs);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    // The scalar-array ABI is charge, dr[3], r[3], all field-major.
    error = maps::launch_nuclear_inputs(extents(), fields, fields + 8 * pairs, fields + 2 * pairs,
                                        fields + pairs, fields + 11 * pairs, fields + 5 * pairs,
                                        packed, packed + 4 * pairs, packed + pairs, at(map_offset),
                                        map_bytes, at<int>(status_offset), stream);
    if (error != cudaSuccess) return error;
    error = hessian_aot::launch_array("nuclear_pair_hvp", packed, pairs, at(pair_first),
                                      at<int>(status_offset), stream);
    if (error != cudaSuccess) return error;
    error = maps::launch_nuclear_second(extents(), at(pair_first), at(pair_second), at(map_offset),
                                        map_bytes, at<int>(status_offset), stream);
    if (error != cudaSuccess) return error;
    for (std::size_t atom = 0; atom < shape.atoms; ++atom) {
      gather_atom<<<maps::blocks(3 * (shape.atoms - 1)), 128, 0, stream>>>(
          at(pair_first), at(pair_second), shape.atoms, atom, pairs, at(incidences));
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
      error = maps::launch_nuclear_atom(extents(), at(incidences), vector(6) + 3 * atom,
                                        at(map_offset), map_bytes, at<int>(status_offset), stream);
      if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
  }
  cudaError_t prepare_frozen_sources() {
    frozen_exact_ready = false;
    frozen_core_ready = false;
    if (!frozen_exact && !frozen_core) return cudaSuccess;
    auto error = authenticate();
    auto* status = at<int>(status_offset);
    if (error == cudaSuccess) error = cudaMemsetAsync(status, 0, sizeof(*status), stream);
    if (error == cudaSuccess && frozen_core)
      error = cudaMemsetAsync(at(delta_totals), 0, 2 * matrix * sizeof(double), stream);
    const integrals::CudaOneElectronWeights weights{at(totals), at(delta_totals),
                                                    at(totals) + matrix, at(delta_totals) + matrix};
    bool streamed_exact = false;
    if (frozen_exact && error == cudaSuccess) {
      const auto dense =
          exact.enqueue_private_frozen_dense(at(direction), at(zero), frozen_exact, status, stream);
      if (dense == cudaErrorNotSupported && exact.work().omitted_frozen_source_groups)
        return cudaErrorInvalidResourceHandle;
      if (dense != cudaErrorNotSupported) {
        error = dense;
        streamed_exact = error == cudaSuccess;
        if (streamed_exact) receipt.frozen_exact_columns += coordinates;
      }
    }
    for (std::size_t column = 0; column < coordinates && error == cudaSuccess; ++column) {
      if (!frozen_core && (!frozen_exact || streamed_exact)) break;
      unit<<<maps::blocks(coordinates), 128, 0, stream>>>(at(direction), coordinates, column);
      error = cudaGetLastError();
      if (error == cudaSuccess && frozen_core) {
        // One evaluation per unit direction supplies H_R and S_R to both the
        // CPKS RHS and every subsequent density response contraction.
        auto output = core_output();
        output.core_direction = frozen_core + column * (2 * matrix);
        output.overlap_direction = output.core_direction + matrix;
        output.hvp = frozen_core + coordinates * (2 * matrix) + column * coordinates;
        error =
            one_electron.enqueue_private(at(direction), weights, output, status, stream, column);
        if (error == cudaSuccess) ++receipt.frozen_core_columns;
      }
      if (!frozen_exact || streamed_exact) continue;
      if (error == cudaSuccess)
        error = exact.enqueue_private_frozen_column(column, at(direction), at(zero), exact_output(),
                                                    status, stream);
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(frozen_exact + column * panels, at(exact_direction),
                                panels * sizeof(double), cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess)
        error =
            cudaMemcpyAsync(frozen_exact + coordinates * panels + column * coordinates, vector(3),
                            coordinates * sizeof(double), cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess) ++receipt.frozen_exact_columns;
    }
    if (error == cudaSuccess) error = gate();
    if (error == cudaSuccess) error = authenticate();
    frozen_exact_ready = frozen_exact && error == cudaSuccess;
    frozen_core_ready = frozen_core && error == cudaSuccess;
    return error;
  }
  cudaError_t compute(CudaKsWb97mvHessianResult& result, bool repeated_rhs = false,
                      std::size_t column = SIZE_MAX,
                      response::CudaCpksDenseReplay* cpks_replay = nullptr) {
    ++receipt.attempted_directions;
    result.failure_check = "stationary Hessian direction validation";
    auto error = authenticate();
    if (error != cudaSuccess) return error;
    if ((error = cudaMemsetAsync(at<int>(status_offset), 0, sizeof(int), stream)) != cudaSuccess)
      return error;
    // Dense directions are unit vectors generated on this stream. Only a
    // caller-supplied HVP direction needs a separate finite-input gate.
    if (!repeated_rhs) {
      finite<<<maps::blocks(coordinates), 128, 0, stream>>>(at(direction), coordinates,
                                                            at<int>(status_offset));
      if ((error = cudaGetLastError()) != cudaSuccess || (error = gate()) != cudaSuccess)
        return error;
    }
    if ((error = cudaMemsetAsync(at(delta_totals), 0, 2 * matrix * sizeof(double), stream)) !=
        cudaSuccess)
      return error;
    const integrals::CudaOneElectronWeights frozen{at(totals), at(delta_totals),
                                                   at(totals) + matrix, at(delta_totals) + matrix};
    result.failure_check = "frozen one-electron Hessian action";
    const bool cached_core = repeated_rhs && frozen_core_ready && column < coordinates;
    const auto* rhs_core = at(core_direction);
    const auto* rhs_overlap = at(overlap_direction);
    if (cached_core) {
      rhs_core = frozen_core + column * (2 * matrix);
      rhs_overlap = rhs_core + matrix;
    } else if ((error = one_electron.enqueue_private(at(direction), frozen, core_output(),
                                                     at<int>(status_offset), stream)) !=
               cudaSuccess) {
      return error;
    }
    result.failure_check = "frozen Direct J/K Hessian action";
    const bool cached_exact = repeated_rhs && frozen_exact_ready && column < coordinates;
    if (cached_exact)
      error = cudaMemcpyAsync(at(exact_direction), frozen_exact + column * panels,
                              panels * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    else if (has_exact)
      error = exact.apply(at(direction), at(zero), exact_output(), stream);
    if (error != cudaSuccess) return error;
    for (auto* source : {&full_df, &range_df}) {
      if (!source->raw) continue;
      auto* values = at(source->values);
      const auto raw = source->raw, metric = source->metric;
      const integrals::CudaDfNuclearValues output{values, values + raw, values + raw + metric,
                                                  values + 2 * raw + metric};
      result.failure_check = "fitted nuclear values";
      if ((error = source->nuclear.values(at(direction), output, stream)) != cudaSuccess ||
          (error = fitted_action(*source, at(zero), result.failure_check)) != cudaSuccess)
        return error;
    }
    if (range_df.raw) {
      add_fitted_range<<<maps::blocks(panels), 128, 0, stream>>>(at(full_df.direction),
                                                                 at(range_df.direction), panels);
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
    }
    if (has_ecp) {
      result.failure_check = "ECP Hessian values";
      auto* values = at(ecp_values);
      if ((error = ecp.values(at(direction),
                              {values, values + matrix, values + 2 * matrix, values + 3 * matrix},
                              stream)) != cudaSuccess)
        return error;
    }
    // Prepare the right geometry once. The full Cartesian column can keep the
    // mixed weights live across CPKS without a second diagonal Becke sweep.
    result.failure_check = "moving-grid CPKS right-hand side";
    const auto shape = grid_reference.shape();
    const auto ao_panel = 4 * shape.points * shape.nao;
    const auto vv10_shape = vv10_reference.shape();
    const auto vv10_panel = 4 * vv10_shape.points * vv10_shape.nao;
    const bool separate = expected.identity.model.nonlocal_grid.has_value();
    bool cartesian_mixed =
        repeated_rhs && column < coordinates && !separate && grid.admits_cartesian_mixed();
    CudaGridGeometryCartesianView cartesian_column;
    CudaGridGeometryDirectionView right_motion;
    if (cartesian_mixed && grid.admits_cartesian_mixed(coordinates)) {
      error = geometry.enqueue_private_cartesian_column(
          at(direction), column, cartesian_column, at<int>(status_offset), stream, &right_motion);
      if (error != cudaSuccess && error != cudaErrorNotSupported) return error;
    }
    CudaGridGeometryMixedView motion, vv10_motion;
    if (!cartesian_column.weights) {
      error = repeated_rhs && column < coordinates
                  ? geometry.enqueue_private_cartesian_diagonal_view(at(direction), column, motion,
                                                                     at<int>(status_offset), stream)
                  : geometry.enqueue_private_mixed_view(at(direction), at(direction), motion,
                                                        at<int>(status_offset), stream);
      if (error != cudaSuccess) return error;
      right_motion = {motion.ao + ao_panel, motion.weights + shape.points,
                      motion.point_directions + 3 * shape.points};
    }
    if (separate && (error = vv10_geometry.enqueue_private_mixed_view(
                         at(direction), at(direction), vv10_motion, at<int>(status_offset),
                         stream)) != cudaSuccess)
      return error;
    if ((error = grid.enqueue_private_nuclear_direction(
             right_motion.ao, right_motion.weights, right_motion.point_directions, at(zero),
             at(xc_direction), at<int>(status_offset), stream)) != cudaSuccess)
      return error;
    if (separate) {
      if ((error = vv10_grid.enqueue_private_nuclear_direction(
               vv10_motion.ao + vv10_panel, vv10_motion.weights + vv10_shape.points,
               vv10_motion.point_directions + 3 * vv10_shape.points, at(zero), at(vv10_direction),
               at<int>(status_offset), stream)) != cudaSuccess)
        return error;
      add_fitted_range<<<maps::blocks(panels), 128, 0, stream>>>(at(xc_direction),
                                                                 at(vv10_direction), panels);
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
    }
    // The generated RHS finite check can replace the public failure flag with
    // 1. Keep the upstream reason in the adjacent private diagnostic slot.
    if ((error = cudaMemcpyAsync(at<int>(status_offset) + 1, at<int>(status_offset), sizeof(int),
                                 cudaMemcpyDeviceToDevice, stream)) != cudaSuccess)
      return error;
    error = maps::launch_rhs(extents(), rhs_core, at(ecp_values) + 2 * matrix,
                             at(ecp_values) + 3 * matrix, at(exact_direction),
                             at(full_df.direction), at(xc_direction), at(rhs), at(map_offset),
                             map_bytes, at<int>(status_offset), stream);
    if (error != cudaSuccess) return error;
    result.failure_check = "stationary CPKS solve and physical response audit";
    error = response::solve_cuda_cpks(
        cpks_reference.shape(), options.cpks, cpks_reference.reference(), at(rhs), rhs_overlap,
        CudaKsWb97mvResponse::induced_fock, &induced, at(delta_density), at(delta_weights),
        at(cpks_offset), cpks_bytes, stream, result.response,
        repeated_rhs && operator_matrix ? &cpks_operator : nullptr, at<int>(status_offset),
        cpks_replay);
    if (error != cudaSuccess) {
      if (const char* check = induced.failure_check()) result.failure_check = check;
      return error;
    }
    receipt.cpks_proposal_iterations += result.response.proposal_iterations;
    receipt.cpks_proposal_seeds += result.response.proposal_seeded;
    if (!repeated_rhs || !operator_matrix ||
        options.cpks.resident_precision == response::CudaCpksResidentPrecision::fp32_proposal)
      receipt.cpks_strict_iterations += result.response.gmres.iterations;
    if (!result.response.converged() &&
        !(cpks_replay && result.response.status == response::CudaCpksStatus::running)) {
      // A synchronous solve can stop before store_column copies the sticky
      // geometry status into the dense audit. Report that upstream failure
      // instead of treating an invalid right-hand side as nonconvergence.
      if (result.response.status == response::CudaCpksStatus::numerical_failure) {
        int transaction_flags[2]{};
        error = cudaMemcpyAsync(transaction_flags, at<int>(status_offset),
                                sizeof(transaction_flags), cudaMemcpyDeviceToHost, stream);
        const auto synchronized = cudaStreamSynchronize(stream);
        if (error != cudaSuccess || synchronized != cudaSuccess)
          return error != cudaSuccess ? error : synchronized;
        const auto transaction_status = transaction_flags[0] | transaction_flags[1];
        if (transaction_status) {
          if (transaction_status & cuda_grid_cartesian_repair_exhausted)
            result.failure_check = "Cartesian moving-grid repair work exhausted";
          else if (transaction_status & cuda_grid_geometry_invalid)
            result.failure_check = "Cartesian moving-grid geometry failed its device audit";
          else if (transaction_status & cuda_grid_nuclear_xc_invalid)
            result.failure_check = "moving-grid XC right-hand side failed its device audit";
          else
            result.failure_check = "Hessian right-hand side failed its device transaction audit";
          return cudaErrorInvalidValue;
        }
      }
      result.status = CudaKsWb97mvHessianStatus::response_not_converged;
      return cudaSuccess;
    }
    result.failure_check = "response one-electron Hessian action";
    // The weight reduction and private contractions share the sticky assembly
    // audit. The final gate covers their failures before any result is exposed.
    if ((error = sum_weights(at(delta_density), at(delta_weights), delta_totals)) != cudaSuccess)
      return error;
    if (cached_core) {
      // H and S are linear in their stationary weights: the only missing
      // terms are H_left:dD and S_left:(-dW), in full public AO ordering.
      frozen_core_response<<<maps::blocks(coordinates), 128, 0, stream>>>(
          frozen_core, at(delta_totals), matrix, coordinates, column, vector(0),
          at<int>(status_offset));
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
      ++receipt.frozen_core_contractions;
    } else if ((error = one_electron.enqueue_private(at(direction), frozen, core_output(),
                                                     at<int>(status_offset), stream)) !=
               cudaSuccess) {
      return error;
    }
    result.failure_check = "response Direct J/K Hessian action";
    if (cached_exact) {
      // Exact J/K is quadratic in D: d(gradient)/dD : dD = F_R(D) : dD.
      // Retain contracted AO derivatives, not four-index value/gradient jets.
      frozen_exact_response<<<maps::blocks(coordinates), 128, 0, stream>>>(
          frozen_exact, frozen_exact + coordinates * panels, at(delta_density), panels, coordinates,
          column, vector(3), at<int>(status_offset),
          exact.work().omitted_frozen_source_groups != 0);
      error = cudaGetLastError();
      if (error == cudaSuccess) ++receipt.frozen_exact_contractions;
    } else if (has_exact) {
      error = exact.apply(at(direction), at(delta_density), exact_output(), stream);
    }
    if (error != cudaSuccess) return error;
    for (auto* source : {&full_df, &range_df}) {
      if (!source->raw) continue;
      if ((error = fitted_action(*source, at(delta_density), result.failure_check)) != cudaSuccess)
        return error;
      const auto* weight = at(source->weights);
      const auto raw = source->raw, metric = source->metric;
      const integrals::CudaDfNuclearWeights weights{weight, weight + raw, weight + raw + metric,
                                                    weight + 2 * raw + metric};
      auto* output = source == &full_df ? vector(4) : at(range_vector);
      if ((error = source->nuclear.contract(at(direction), weights, {at(scratch_gradient), output},
                                            stream)) != cudaSuccess) {
        result.failure_check = "fitted nuclear cotangent contraction";
        return error;
      }
    }
    if (range_df.raw) {
      add_fitted_range<<<maps::blocks(coordinates), 128, 0, stream>>>(vector(4), at(range_vector),
                                                                      coordinates);
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
    }
    result.failure_check = "ECP Hessian contraction";
    if (has_ecp && (error = ecp.contract(at(direction), at(delta_density),
                                         {at(scratch_gradient), at(scratch_gradient) + coordinates,
                                          vector(1), vector(2)},
                                         stream)) != cudaSuccess)
      return error;
    result.failure_check = "moving-grid response Hessian contraction";
    error = cartesian_column.weights
                ? grid.enqueue_private_induced_density_covector(at(delta_density),
                                                                at<int>(status_offset), stream)
                : grid.enqueue_private_nuclear_covector(
                      right_motion.ao, right_motion.weights, right_motion.point_directions,
                      at(delta_density), at<int>(status_offset), stream);
    if (error != cudaSuccess) return error;
    if (separate && (error = vv10_grid.enqueue_private_nuclear_covector(
                         vv10_motion.ao + vv10_panel, vv10_motion.weights + vv10_shape.points,
                         vv10_motion.point_directions + 3 * vv10_shape.points, at(delta_density),
                         at<int>(status_offset), stream)) != cudaSuccess)
      return error;
    if (cartesian_column.weights &&
        (error = grid.enqueue_private_cartesian_mixed(
             cartesian_column, vector(5), at<int>(status_offset), stream)) != cudaSuccess)
      return error;
    for (std::size_t coordinate = 0; !cartesian_column.weights && coordinate < coordinates;
         ++coordinate) {
      unit<<<maps::blocks(coordinates), 128, 0, stream>>>(at(left), coordinates, coordinate);
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
      if (cartesian_mixed) {
        CudaGridGeometryCartesianView cartesian;
        error = geometry.enqueue_private_cartesian_view(at(left), at(direction), coordinate, column,
                                                        cartesian, at<int>(status_offset), stream);
        if (error == cudaErrorNotSupported) {
          cartesian_mixed = false;
        } else {
          if (error != cudaSuccess || (error = grid.enqueue_private_cartesian_mixed(
                                           cartesian, vector(5) + coordinate,
                                           at<int>(status_offset), stream)) != cudaSuccess)
            return error;
          continue;
        }
      }
      if ((error = geometry.enqueue_private_mixed_view(
               at(left), at(direction), motion, at<int>(status_offset), stream)) != cudaSuccess)
        return error;
      if ((error = grid.enqueue_private_nuclear_mixed(
               motion.ao, motion.ao + 2 * ao_panel, motion.weights,
               motion.weights + 2 * shape.points, motion.point_directions, vector(5) + coordinate,
               at<int>(status_offset), stream)) != cudaSuccess)
        return error;
      if (separate) {
        if ((error = vv10_geometry.enqueue_private_mixed_view(at(left), at(direction), vv10_motion,
                                                              at<int>(status_offset), stream)) !=
                cudaSuccess ||
            (error = vv10_grid.enqueue_private_nuclear_mixed(
                 vv10_motion.ao, vv10_motion.ao + 2 * vv10_panel, vv10_motion.weights,
                 vv10_motion.weights + 2 * vv10_shape.points, vv10_motion.point_directions,
                 at(vv10_direction), at<int>(status_offset), stream)) != cudaSuccess)
          return error;
        add_fitted_range<<<1, 1, 0, stream>>>(vector(5) + coordinate, at(vv10_direction), 1);
        if ((error = cudaGetLastError()) != cudaSuccess) return error;
      }
    }
    result.failure_check = "stationary Hessian assembly and correction";
    if ((error = nuclear()) != cudaSuccess) return error;
    error = maps::launch_hvp(extents(), vector(0), vector(1), vector(2), vector(3), vector(4),
                             vector(5), vector(6), at(candidate), at(map_offset), map_bytes,
                             at<int>(status_offset), stream);
    if (error != cudaSuccess || (error = correction_direction()) != cudaSuccess ||
        (!repeated_rhs && (error = gate()) != cudaSuccess) ||
        (error = authenticate()) != cudaSuccess)
      return error;
    if (!repeated_rhs) {
      ++receipt.completed_directions;
      ++result.completed_directions;
    }
    result.failure_check = nullptr;
    result.status = CudaKsWb97mvHessianStatus::converged;
    return cudaSuccess;
  }
};

CudaKsWb97mvHessian::CudaKsWb97mvHessian() = default;
CudaKsWb97mvHessian::~CudaKsWb97mvHessian() = default;
generativeqc_status CudaKsWb97mvHessian::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                 const CudaKsFinalStateToken& expected,
                                                 CudaKsWb97mvHessianMode mode,
                                                 const CudaKsWb97mvHessianOptions& options,
                                                 std::string& detail) {
  impl_.reset();
  if (!expected.strict_physical_fock) {
    detail = "native Hessian requires a fresh strict-FP64 physical SCF Fock";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (expected.relaxed_stationarity) {
    detail = "native Hessian cannot use an SCF state with relaxed application stationarity";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    require(options.max_device_bytes && options.max_assembly_terms &&
                std::isfinite(options.symmetry_absolute_tolerance) &&
                std::isfinite(options.symmetry_relative_tolerance) &&
                options.symmetry_absolute_tolerance >= 0 &&
                options.symmetry_relative_tolerance >= 0,
            "invalid stationary Hessian options");
    auto owner = std::make_unique<Impl>(batch, item, expected, options, mode);
    auto status = owner->grid_reference.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = owner->cpks_reference.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status =
        methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, owner->fock, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(cuda_ks_hessian_model_matches(expected, owner->fock, &owner->options.grid),
            "stationary Hessian requires a matched native grid and Fock model");
    status = methods::detail::borrow_dft_hessian_corrections(batch, item, expected,
                                                             owner->corrections, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, owner->frame,
                                                              detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto lease = owner->grid_reference.reference();
    const auto shape = owner->grid_reference.shape();
    auto geometry_shape = owner->grid_reference.geometry_shape();
    const bool separate = expected.identity.model.nonlocal_grid.has_value();
    if (separate) {
      status = owner->vv10_reference.prepare(detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    }
    const auto vv10_shape = owner->vv10_reference.shape();
    auto vv10_geometry_shape = owner->vv10_reference.geometry_shape();
    // Keep the existing read-heavy spatial cache for the qualified small
    // endpoints. Wide, pure s bases need the smaller geometry lease.
    geometry_shape.radial_s = shape.nao > 64 && !separate;
    for (const auto& shell : owner->fock.primary->system().shells) {
      const auto angular = static_cast<std::size_t>(shell.angular_momentum);
      geometry_shape.radial_s =
          geometry_shape.radial_s && angular == 0 &&
          std::all_of(shell.primitives.begin(), shell.primitives.end(), [](const auto& primitive) {
            return primitive.exponent >= 0x1p-32 && primitive.exponent <= 0x1p16 &&
                   std::abs(primitive.coefficient) <= 0x1p16;
          });
      const bool spherical =
          owner->fock.primary->system().basis_representation == core::kSphericalBasis;
      const auto components = spherical ? 2 * angular + 1 : (angular + 1) * (angular + 2) / 2;
      const auto terms = spherical && angular > 1 ? 3 : 1;
      const auto incidences = product(product(components, shell.primitives.size()), terms);
      require(incidences <= SIZE_MAX - geometry_shape.ao_primitive_terms,
              "moving-grid AO primitive incidence overflow");
      geometry_shape.ao_primitive_terms += incidences;
    }
    if (separate) {
      require(vv10_geometry_shape.atoms == shape.atoms && vv10_shape.nao == shape.nao &&
                  vv10_shape.primitives == shape.primitives,
              "stationary Hessian requires a native independent VV10 quadrature");
      vv10_geometry_shape.ao_primitive_terms = geometry_shape.ao_primitive_terms;
    }
    require(geometry_shape.atoms && shape.nao == owner->frame.nao &&
                shape.spins == owner->frame.spins && lease.density == owner->frame.density &&
                lease.device_id == owner->frame.device_id &&
                reinterpret_cast<void*>(lease.stream) == owner->frame.stream,
            "stationary Hessian requires matching native quadrature and canonical frame");
    owner->stream = lease.stream;
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(owner->stream, &capture));
    require(capture == cudaStreamCaptureStatusNone,
            "stationary Hessian preparation cannot capture");
    owner->coordinates = product(3, shape.atoms);
    owner->matrix = product(shape.nao, shape.nao);
    owner->panels = product(owner->matrix, shape.spins);
    owner->pairs = product(shape.atoms, shape.atoms - 1) / 2;
    require(owner->pairs <= options.max_nuclear_pairs, "nuclear pair work exceeds explicit bound");
    if (owner->pairs) {
      const auto* program = hessian_aot::find("nuclear_pair_hvp");
      require(program && program->kind == hessian_aot::Kind::array && program->inputs.size() == 7 &&
                  program->outputs == 3,
              "linked nuclear pair HVP closure is missing or incompatible");
    }
    owner->has_df = owner->fock.primary->strategy().spec.coulomb.approximation ==
                    scf::FockApproximation::DensityFitted;
    owner->has_ecp = !owner->fock.primary->system().ecp_terms.empty();
    // A fully fitted model has no exact nuclear quartet
    // source. Its zero-initialized exact assembly slots remain untouched.
    owner->has_exact = !owner->has_df || (owner->fock.correction && !owner->fock.fitted_range);
    const auto size_source = [&](const scf::PreparedFockPlan& provider, auto& fitted) {
      const auto source = scf::cuda_density_fitting_readonly_source(provider.cuda_fitted_source());
      require(source && source.nao == shape.nao,
              "stationary Hessian requires native raw DF sources");
      fitted.raw = product(owner->matrix, source.naux);
      fitted.metric = product(source.naux, source.naux);
      require(fitted.raw <= SIZE_MAX - fitted.metric, "DF Hessian source extent overflow");
    };
    if (owner->has_df) size_source(*owner->fock.primary, owner->full_df);
    if (owner->fock.fitted_range) size_source(*owner->fock.fitted_range, owner->range_df);
    const auto extents = owner->extents();
    const auto map = [&](auto workspace, auto work, std::size_t repetitions = 1) {
      std::size_t size{}, terms{};
      check(workspace(extents, size));
      require(work(extents, terms), "stationary assembly work overflow");
      terms = product(terms, repetitions);
      require(terms <= options.max_assembly_terms - owner->receipt.assembly_terms_per_direction,
              "stationary assembly exceeds explicit term bound");
      owner->receipt.assembly_terms_per_direction += terms;
      owner->map_bytes = std::max(owner->map_bytes, size);
    };
    map(maps::workspace_weights, maps::work_weights, 2);
    map(maps::workspace_rhs, maps::work_rhs);
    map(maps::workspace_hvp, maps::work_hvp);
    if (owner->range_df.raw) {
      const auto remaining =
          options.max_assembly_terms - owner->receipt.assembly_terms_per_direction;
      require(owner->panels <= remaining && owner->coordinates <= remaining - owner->panels,
              "fitted LR merge exceeds the stationary assembly term bound");
      owner->receipt.assembly_terms_per_direction += owner->panels + owner->coordinates;
    }
    if (owner->pairs) {
      map(maps::workspace_nuclear_inputs, maps::work_nuclear_inputs);
      map(maps::workspace_nuclear_second, maps::work_nuclear_second);
      map(maps::workspace_nuclear_atom, maps::work_nuclear_atom, shape.atoms);
    }
    if (owner->corrections.d3) {
      const auto& model = owner->corrections.d3->parameters();
      owner->d3_source = owner->corrections.d3->cuda_direction_source();
      require(dispersion::d3_hessian_supported(model) && owner->d3_source.numbers &&
                  owner->d3_source.atoms == shape.atoms &&
                  owner->d3_source.device_id == lease.device_id,
              "stationary D3 requires one matching native molecular correction owner");
      // Charge a conservative bound on reference visits, even when a real-space
      // cutoff excludes every edge. Linear memory does not bound ATM work.
      const auto directed_pairs = product(shape.atoms, shape.atoms - 1);
      auto terms = product(directed_pairs, 52);
      const auto available =
          options.max_assembly_terms - owner->receipt.assembly_terms_per_direction;
      require(terms <= available, "D3 two-body response exceeds explicit work bound");
      if (model.atm_enabled && shape.atoms >= 3) {
        const auto incident_triples = product(directed_pairs, shape.atoms - 2) / 2;
        const auto triple_terms = product(incident_triples, 3 * 49 + 3);
        require(triple_terms <= available - terms, "D3 ATM response exceeds explicit work bound");
        terms += triple_terms;
      }
      owner->receipt.assembly_terms_per_direction += terms;
    }
    // First reserve the complete mandatory baseline. Optional grid replay and
    // parallel one-electron schedules may spend only the surplus after every
    // mandatory consumer, including the moving-grid candidate and exact sources.
    owner->options.grid.nuclear_response = true;
    owner->options.grid.replay_nuclear_maps = false;
    owner->options.grid.max_nuclear_adjoint_bytes = 0;
    owner->options.grid.max_point_linearization_bytes = 0;
    auto grid_admission = cuda_wb97mv_grid_workspace(shape, owner->options.grid, owner->grid_bytes);
    if (grid_admission == cudaErrorInvalidValue && !separate && shape.spins == 1 &&
        shape.nao > 64 && shape.nao <= 96 && !owner->options.grid.includes_nonlocal() &&
        mode == CudaKsWb97mvHessianMode::dense_and_action && options.grid.replay_nuclear_maps) {
      // The required adjoints are charged to the original shared grid arena.
      // They replace the over-work affine fallback, not other consumers' budgets.
      owner->options.grid.max_nuclear_adjoint_bytes = 512ULL << 20;
      grid_admission = cuda_wb97mv_grid_workspace(shape, owner->options.grid, owner->grid_bytes);
    }
    check(grid_admission, "stationary Hessian XC/VV10 workspace admission");
    auto vv10_grid_options = owner->options.grid;
    if (separate) {
      vv10_grid_options.component = CudaWb97mvGridComponent::vv10_only;
      vv10_grid_options.max_workspace_bytes -= owner->grid_bytes;
      check(cuda_wb97mv_grid_workspace(vv10_shape, vv10_grid_options, owner->vv10_grid_bytes),
            "stationary Hessian independent VV10 workspace admission");
    }
    owner->options.geometry.partition_iterations =
        owner->grid_reference.geometry_options().partition_iterations;
    owner->options.geometry.coincident_tolerance =
        owner->grid_reference.geometry_options().coincident_tolerance;
    auto vv10_geometry_options = options.geometry;
    if (separate) {
      const auto policy = owner->vv10_reference.geometry_options();
      vv10_geometry_options.partition_iterations = policy.partition_iterations;
      vv10_geometry_options.coincident_tolerance = policy.coincident_tolerance;
      std::size_t minimum{};
      check(cuda_grid_geometry_workspace(vv10_geometry_shape, vv10_geometry_options,
                                         owner->vv10_geometry_bytes, &minimum),
            "stationary Hessian independent VV10 geometry workspace admission");
      owner->options.geometry.max_workspace_bytes -= minimum;
      // If full residency fails, keep a replay tile rather than spending the
      // remaining shared capacity before the second geometry plan can use it.
      owner->options.geometry.retain_partial_jets = false;
    }
    check(cuda_grid_geometry_workspace(geometry_shape, owner->options.geometry,
                                       owner->geometry_bytes),
          "stationary Hessian moving-grid workspace admission");
    if (separate) {
      vv10_geometry_options.max_workspace_bytes -= owner->geometry_bytes;
      check(cuda_grid_geometry_workspace(vv10_geometry_shape, vv10_geometry_options,
                                         owner->vv10_geometry_bytes),
            "stationary Hessian independent VV10 moving-grid workspace admission");
    }
    check(response::cuda_cpks_workspace(owner->cpks_reference.shape(), owner->options.cpks,
                                        owner->cpks_bytes),
          "stationary Hessian CPKS workspace admission");
    Layout layout;
    owner->map_offset = layout.reserve(owner->map_bytes, 1);
    owner->status_offset = layout.reserve(2, sizeof(int));
    const auto matrix = owner->matrix, panels = owner->panels, coordinates = owner->coordinates;
    for (auto* offset :
         {&owner->zero, &owner->weights, &owner->delta_density, &owner->delta_weights,
          &owner->exact_fock, &owner->exact_direction, &owner->xc_direction, &owner->rhs})
      *offset = layout.reserve(panels);
    for (auto* offset : {&owner->totals, &owner->delta_totals})
      *offset = layout.reserve(product(2, matrix));
    for (auto* offset :
         {&owner->core, &owner->overlap, &owner->core_direction, &owner->overlap_direction})
      *offset = layout.reserve(matrix);
    owner->exact_energy = layout.reserve(1);
    for (auto* source : {&owner->full_df, &owner->range_df}) {
      if (!source->raw && source == &owner->range_df) continue;
      source->direction = layout.reserve(owner->panels);
      source->fock = layout.reserve(source->raw ? owner->panels : 0);
      source->energy = layout.reserve(source->raw ? 1 : 0);
      source->values = layout.reserve(product(2, source->raw + source->metric));
      source->weights = layout.reserve(product(2, source->raw + source->metric));
    }
    owner->range_vector = layout.reserve(owner->range_df.raw ? coordinates : 0);
    owner->ecp_values = layout.reserve(product(4, matrix));
    owner->vectors = layout.reserve(product(7, coordinates));
    owner->scratch_gradient = layout.reserve(product(2, coordinates));
    for (auto* offset : {&owner->direction, &owner->left, &owner->candidate})
      *offset = layout.reserve(coordinates);
    if (mode == CudaKsWb97mvHessianMode::dense_and_action) {
      owner->complete = layout.reserve(product(coordinates, coordinates));
      owner->dense_audit_offset = layout.reserve(1, sizeof(DenseAssemblyAudit));
    }
    owner->charges = layout.reserve(shape.atoms);
    owner->pair_atoms = layout.reserve(product(2, owner->pairs), sizeof(std::size_t));
    owner->pair_fields = layout.reserve(product(14, owner->pairs));
    owner->pair_inputs = layout.reserve(product(7, owner->pairs));
    owner->pair_first = layout.reserve(product(3, owner->pairs));
    owner->pair_second = layout.reserve(product(3, owner->pairs));
    owner->incidences = layout.reserve(product(3, shape.atoms - 1));
    if (owner->corrections.d3 || owner->corrections.d4) {
      owner->correction_geometry = layout.reserve(coordinates);
      owner->correction_action = layout.reserve(coordinates);
    }
    if (owner->corrections.d3)
      owner->d3_scratch = layout.reserve(dispersion::d3_hessian_workspace_elements(shape.atoms));
    if (owner->corrections.d4) {
      const auto d4_workspace =
          shape.atoms <= dispersion::kD4MaximumAtoms
              ? dispersion::d4_eeq_direction_workspace_elements(static_cast<int>(shape.atoms))
              : 0;
      require(d4_workspace, "unqualified D4 Hessian correction shape");
      owner->d4_numbers = layout.reserve(shape.atoms, sizeof(std::int32_t));
      owner->d4_elements =
          layout.reserve(dispersion::kD4TableElementCount, sizeof(dispersion::data::D4ElementData));
      owner->d4_references = layout.reserve(dispersion::kD4TableReferenceCount,
                                            sizeof(dispersion::data::D4ReferenceData));
      owner->d4_c6 = layout.reserve(dispersion::kD4PackedReferenceC6Count);
      owner->eeq_parameters = layout.reserve(dispersion::kD4TableElementCount,
                                             sizeof(dispersion::D4EEQChargeElementData));
      owner->d4_scratch = layout.reserve(d4_workspace);
      owner->d4_gradient = layout.reserve(coordinates);
      // Standard D4 needs only its single status; gCP reports one per atom.
      if (owner->corrections.gcp) owner->gcp_action = layout.reserve(coordinates);
      owner->d4_status =
          layout.reserve(1 + (owner->corrections.gcp ? shape.atoms : 0), sizeof(int));
    }
    owner->geometry_offset = layout.reserve(owner->geometry_bytes, 1);
    if (separate) {
      owner->vv10_geometry_offset = layout.reserve(owner->vv10_geometry_bytes, 1);
      owner->vv10_grid_offset = layout.reserve(owner->vv10_grid_bytes, 1);
      owner->vv10_direction = layout.reserve(panels);
    }
    owner->cpks_offset = layout.reserve(owner->cpks_bytes, 1);
    // Keep the optional grid tail last so its aligned reservation can grow
    // without moving any mandatory output or response workspace.
    owner->grid_offset = layout.reserve(owner->grid_bytes, 1);
    owner->bytes = layout.bytes;
    owner->charge(owner->bytes);
    auto core_options = options.one_electron;
    core_options.parallel_streams = 0;
    const auto core_bytes = cuda_ks_one_electron_workspace(
        owner->fock.primary->system(), {shape.atoms, shape.primitives, shape.nao}, core_options);
    owner->charge(core_bytes);
    auto induced_options = options.induced;
    induced_options.defer_resident_integrals = true;
    induced_options.max_device_bytes = std::min(induced_options.max_device_bytes, owner->remaining);
    status = owner->induced.prepare(batch, item, expected, induced_options, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      detail = "stationary Hessian induced Fock: " + detail;
      return status;
    }
    owner->charge(owner->induced.work().device_bytes);
    for (auto* source : {&owner->full_df, &owner->range_df}) {
      if (!source->raw) continue;
      const bool range = source == &owner->range_df;
      auto raw_options = options.raw_df;
      raw_options.range_omega = range ? owner->fock.correction->spec.exchange.omega : 0;
      raw_options.retain_directional_source = mode == CudaKsWb97mvHessianMode::dense_and_action;
      raw_options.max_workspace_bytes = std::min(raw_options.max_workspace_bytes, owner->remaining);
      status = source->nuclear.prepare(batch, item, expected, raw_options, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) {
        detail = std::string(range ? "stationary Hessian raw LR DF" : "stationary Hessian raw DF") +
                 " nuclear source: " + detail;
        return status;
      }
      owner->charge(source->nuclear.work().workspace_bytes);
      auto fitted_options = options.fitted;
      fitted_options.coulomb = !range;
      fitted_options.max_workspace_bytes =
          std::min(fitted_options.max_workspace_bytes, owner->remaining);
      status = source->response.prepare(batch, item, expected, fitted_options, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) {
        detail = (range ? "stationary Hessian LR fitted response: "
                        : "stationary Hessian fitted response: ") +
                 detail;
        return status;
      }
      owner->charge(source->response.work().workspace_bytes);
    }
    std::size_t ecp_mandatory_bytes{};
    if (owner->has_ecp) {
      auto ecp_options = options.ecp;
      ecp_options.max_workspace_bytes = std::min(ecp_options.max_workspace_bytes, owner->remaining);
      ecp_mandatory_bytes = CudaKsWb97mvEcpNuclear::mandatory_workspace(
          owner->fock.primary->system(), shape, ecp_options);
      owner->charge(ecp_mandatory_bytes);
    }
    // Reserve every mandatory consumer before optional integral residency.
    auto exact_options = options.exact;
    // A dense sweep can reuse contracted frozen Fock derivatives instead of
    // revisiting the same right direction's primitive jets.
    const auto frozen_bytes = product(product(coordinates, panels + coordinates), sizeof(double));
    exact_options.prefer_fresh_direction_batches =
        mode == CudaKsWb97mvHessianMode::dense_and_action &&
        frozen_bytes <= options.max_frozen_exact_bytes;
    const auto& exchange = owner->fock.primary->strategy().spec.exchange;
    exact_options.full_exchange_fraction = 0;
    if (owner->fock.correction) {
      exact_options.part = owner->has_df ? CudaWb97mvJkPart::correction : CudaWb97mvJkPart::exact;
    } else if (exchange.present) {
      exact_options.part = CudaWb97mvJkPart::full_hybrid;
      exact_options.full_exchange_fraction = -exchange.coefficient * (shape.spins == 1 ? 2 : 1);
    } else {
      exact_options.part = CudaWb97mvJkPart::coulomb;
    }
    exact_options.max_workspace_bytes =
        std::min(exact_options.max_workspace_bytes, owner->remaining);
    if (owner->has_exact) {
      CudaWb97mvJkAdmissionFailure exact_failure{};
      status =
          owner->exact.prepare(batch, item, expected, exact_options, detail, false, &exact_failure);
      if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
          exact_failure == CudaWb97mvJkAdmissionFailure::canonical_source_limit &&
          exact_options.prefer_fresh_direction_batches && frozen_bytes < owner->remaining &&
          shape.spins == 1 && !owner->has_df && !owner->fock.correction) {
        // A derivative-only schedule is not a fallback for arbitrary HVPs.
        // Reserve its required table before admitting the screened source.
        check(runtime::resource_cuda_malloc(&owner->frozen_exact, frozen_bytes));
        owner->charge(frozen_bytes);
        owner->receipt.frozen_exact_bytes = frozen_bytes;
        exact_options.max_workspace_bytes =
            std::min(exact_options.max_workspace_bytes, owner->remaining);
        status = owner->exact.prepare(batch, item, expected, exact_options, detail, true);
      }
      if (status != GENERATIVEQC_STATUS_SUCCESS) {
        detail = "stationary Hessian exact nuclear J/K: " + detail;
        return status;
      }
      owner->charge(owner->exact.work().workspace_bytes);
    }
    const auto grid_reserved = owner->bytes - owner->grid_offset;
    owner->options.grid.replay_nuclear_maps = options.grid.replay_nuclear_maps;
    if (mode == CudaKsWb97mvHessianMode::dense_and_action && options.grid.replay_nuclear_maps) {
      owner->options.grid.max_nuclear_adjoint_bytes =
          std::max<std::size_t>(owner->options.grid.max_nuclear_adjoint_bytes, 64ULL << 20);
      owner->options.grid.max_point_linearization_bytes = 16ULL << 20;
    }
    // All arena slots are 256-byte aligned. Do not let an optional payload
    // consume the alignment bytes needed by its enclosing reservation.
    owner->options.grid.max_workspace_bytes =
        std::min(options.grid.max_workspace_bytes - owner->vv10_grid_bytes,
                 (grid_reserved + owner->remaining) / 256 * 256);
    check(cuda_wb97mv_grid_workspace(shape, owner->options.grid, owner->grid_bytes),
          "stationary Hessian optional XC/VV10 workspace admission");
    layout.bytes = owner->grid_offset;
    owner->grid_offset = layout.reserve(owner->grid_bytes, 1);
    owner->charge(layout.bytes - owner->bytes);
    owner->bytes = layout.bytes;
    core_options = options.one_electron;
    // A dense sweep reuses the same packed batch for every coordinate. Bound
    // its task descriptors by the admitted job census, and let payload bytes
    // and primitive work decide whether the batch actually fits.
    if (mode == CudaKsWb97mvHessianMode::dense_and_action &&
        core_options.max_parallel_tasks == integrals::CudaOneElectronOptions{}.max_parallel_tasks)
      core_options.max_parallel_tasks = core_options.max_jobs;
    core_options.max_workspace_bytes =
        std::min(core_options.max_workspace_bytes, core_bytes + owner->remaining);
    status = owner->one_electron.prepare(batch, item, expected, core_options, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(owner->one_electron.work().device_bytes >= core_bytes,
            "one-electron preparation differs from its mandatory reservation");
    owner->charge(owner->one_electron.work().device_bytes - core_bytes);
    check(runtime::resource_cuda_malloc(&owner->arena, owner->bytes));
    std::vector<double> charges;
    for (const auto& atom : owner->fock.primary->system().atoms)
      charges.push_back(atom.ionic_charge());
    std::vector<std::size_t> pair_atoms(product(2, owner->pairs));
    std::size_t pair{};
    for (std::size_t first = 0; first < shape.atoms; ++first)
      for (std::size_t second = first + 1; second < shape.atoms; ++second, ++pair) {
        pair_atoms[pair] = first;
        pair_atoms[owner->pairs + pair] = second;
      }
    DrainOnFailure upload_drain{owner->stream};
    check(cudaMemsetAsync(owner->arena, 0, owner->bytes, owner->stream));
    check(cudaMemcpyAsync(owner->at(owner->charges), charges.data(),
                          charges.size() * sizeof(double), cudaMemcpyHostToDevice, owner->stream));
    if (owner->pairs)
      check(cudaMemcpyAsync(owner->at<std::size_t>(owner->pair_atoms), pair_atoms.data(),
                            pair_atoms.size() * sizeof(std::size_t), cudaMemcpyHostToDevice,
                            owner->stream));
    if (owner->corrections.d3 || owner->corrections.d4) {
      std::vector<std::int32_t> numbers;
      std::vector<double> xyz;
      for (const auto& atom : owner->fock.primary->system().atoms) {
        if (owner->corrections.d4) numbers.push_back(atom.atomic_number);
        xyz.insert(xyz.end(), atom.position.begin(), atom.position.end());
      }
      // All pageable inputs remain live until this producer-stream upload drains.
      DrainOnFailure correction_upload{owner->stream};
      const auto upload = [&]<typename Value>(std::size_t offset, const Value* data,
                                              std::size_t items) {
        check(cudaMemcpyAsync(owner->at(offset), data, items * sizeof(Value),
                              cudaMemcpyHostToDevice, owner->stream));
        return owner->at<Value>(offset);
      };
      upload(owner->correction_geometry, xyz.data(), xyz.size());
      if (owner->corrections.d4) {
        const auto model = dispersion::eeq_d4_host_tables(owner->corrections.d4->profile());
        const auto charge = dispersion::eeq2019_host_tables();
        upload(owner->d4_numbers, numbers.data(), numbers.size());
        owner->d4_tables = model;
        owner->d4_tables.elements = upload(owner->d4_elements, model.elements, model.element_count);
        owner->d4_tables.references =
            upload(owner->d4_references, model.references, model.reference_count);
        owner->d4_tables.reference_c6 =
            upload(owner->d4_c6, model.reference_c6, model.reference_c6_count);
        owner->eeq_tables = {
            owner->d4_tables.elements,
            upload(owner->eeq_parameters, charge.charge_elements, charge.element_count),
            charge.element_count};
      }
      check(cudaStreamSynchronize(owner->stream));
      correction_upload.active = false;
    }
    check(cudaStreamSynchronize(owner->stream));
    upload_drain.active = false;
    check(owner->geometry.prepare(geometry_shape, owner->options.geometry,
                                  owner->grid_reference.geometry_reference(),
                                  owner->at(owner->geometry_offset), owner->geometry_bytes));
    check(owner->grid.prepare(shape, owner->options.grid, lease, owner->at(owner->grid_offset),
                              owner->grid_bytes));
    if (separate) {
      check(owner->vv10_geometry.prepare(
          vv10_geometry_shape, vv10_geometry_options, owner->vv10_reference.geometry_reference(),
          owner->at(owner->vv10_geometry_offset), owner->vv10_geometry_bytes));
      check(owner->vv10_grid.prepare(vv10_shape, vv10_grid_options,
                                     owner->vv10_reference.reference(),
                                     owner->at(owner->vv10_grid_offset), owner->vv10_grid_bytes));
    }
    if (owner->has_ecp) {
      // The incumbent pair was charged before exact J/K. Only the device
      // budget left after preparing the required consumers can fund replay.
      auto ecp_options = options.ecp;
      ecp_options.max_workspace_bytes =
          std::min(ecp_options.max_workspace_bytes, ecp_mandatory_bytes + owner->remaining);
      status = owner->ecp.prepare(batch, item, expected, ecp_options, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      require(owner->ecp.device_bytes() >= ecp_mandatory_bytes,
              "ECP preparation differs from its mandatory reservation");
      owner->charge(owner->ecp.device_bytes() - ecp_mandatory_bytes);
    }
    // Only the dense column loop consumes these caches. Do not size or charge
    // them on the directional path, including when providers compete for slack.
    if (mode == CudaKsWb97mvHessianMode::dense_and_action) {
      std::size_t operator_bytes{};
      check(response::cuda_cpks_operator_workspace(owner->cpks_reference.shape(),
                                                   owner->options.cpks, operator_bytes));
      const bool eligible =
          owner->options.cpks.resident_precision == response::CudaCpksResidentPrecision::fp64 ||
          owner->options.cpks.gmres.max_iterations > 1;
      // Building the dense operator costs one physical Fock action per
      // rotation. Keep the established small-cache regime, but require enough
      // RHSs to amortize larger caches. A small molecule in a large basis can
      // otherwise pay for many more columns than it has useful repeated work.
      constexpr std::size_t small_operator_bytes = 128ULL << 10;
      const auto cpks_shape = owner->cpks_reference.shape();
      constexpr std::size_t ordinary_operator_bytes = 512ULL << 10;
      // The explicit wide lease can amortize a fitted semilocal operator
      // across all columns without evicting the mandatory fitted sources.
      const bool wide_fitted = options.max_resident_cpks_bytes >= (64ULL << 20) &&
                               !owner->fock.primary->strategy().spec.exchange.present;
      const bool extended =
          operator_bytes > ordinary_operator_bytes && cpks_shape.spins == 1 &&
          (!owner->has_df || wide_fitted) && !owner->has_ecp && !owner->fock.correction &&
          !owner->expected.identity.model.nonlocal_grid &&
          owner->options.cpks.resident_precision == response::CudaCpksResidentPrecision::fp64;
      const auto operator_limit =
          extended ? options.max_resident_cpks_bytes
                   : std::min(options.max_resident_cpks_bytes, ordinary_operator_bytes);
      // Dense semilocal Direct solves in the larger lease can revisit the
      // physical Fock operator more than 25 times per RHS. Still bound the
      // one-time build to ten rotations per RHS in this additional regime.
      constexpr std::size_t previous_extended_bytes = 2ULL << 20;
      const std::size_t build_actions_per_rhs =
          extended ? (operator_bytes > previous_extended_bytes ? 10 : 8) : 6;
      std::size_t rotations{};
      for (unsigned spin = 0; spin < cpks_shape.spins; ++spin)
        rotations += cpks_shape.occupied[spin] * (cpks_shape.nao - cpks_shape.occupied[spin]);
      const bool amortized =
          operator_bytes <= small_operator_bytes ||
          (owner->options.cpks.gmres.max_iterations > build_actions_per_rhs &&
           (rotations == 0 || (rotations - 1) / build_actions_per_rhs < coordinates));
      if (eligible && amortized && operator_bytes && operator_bytes <= operator_limit &&
          operator_bytes <= owner->remaining &&
          operator_bytes <= owner->options.cpks.max_workspace_bytes - owner->cpks_bytes) {
        const auto allocation =
            runtime::resource_cuda_malloc(&owner->operator_matrix, operator_bytes);
        if (allocation == cudaErrorMemoryAllocation) {
          (void)cudaGetLastError();
          owner->operator_matrix = nullptr;
        } else {
          check(allocation);
          owner->charge(operator_bytes);
          owner->receipt.cpks_operator_bytes = operator_bytes;
          owner->cpks_operator.reset(owner->operator_matrix, operator_bytes,
                                     owner->options.cpks.resident_precision);
        }
      }
      if (owner->has_exact && !owner->frozen_exact &&
          frozen_bytes <= options.max_frozen_exact_bytes && frozen_bytes <= owner->remaining) {
        const auto allocation = runtime::resource_cuda_malloc(&owner->frozen_exact, frozen_bytes);
        if (allocation == cudaErrorMemoryAllocation) {
          (void)cudaGetLastError();
          owner->frozen_exact = nullptr;
        } else {
          check(allocation);
          owner->charge(frozen_bytes);
          owner->receipt.frozen_exact_bytes = frozen_bytes;
        }
      }
      const auto core_cache_bytes =
          product(product(owner->coordinates, product(2, owner->matrix) + owner->coordinates),
                  sizeof(double));
      if (core_cache_bytes <= options.max_frozen_core_bytes &&
          core_cache_bytes <= owner->remaining) {
        const auto allocation =
            runtime::resource_cuda_malloc(&owner->frozen_core, core_cache_bytes);
        if (allocation == cudaErrorMemoryAllocation) {
          (void)cudaGetLastError();
          owner->frozen_core = nullptr;
        } else {
          check(allocation);
          owner->charge(core_cache_bytes);
          owner->receipt.frozen_core_bytes = core_cache_bytes;
        }
      }
    }
    const auto induced_bytes = owner->induced.work().device_bytes;
    status = owner->induced.retain_integrals(
        owner->remaining, mode == CudaKsWb97mvHessianMode::dense_and_action, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (mode == CudaKsWb97mvHessianMode::dense_and_action && options.grid.replay_nuclear_maps &&
        owner->remaining > owner->induced.work().device_bytes - induced_bytes) {
      const auto spare = owner->remaining - (owner->induced.work().device_bytes - induced_bytes);
      const auto point_lease = options.max_resident_cpks_bytes >= (64ULL << 20) ? 32ULL : 16ULL;
      check(
          owner->induced.retain_fixed_grid_points(std::min<std::size_t>(point_lease << 20, spare)),
          "stationary Hessian optional fixed XC residency");
    }
    if (owner->operator_matrix &&
        owner->options.cpks.resident_precision == response::CudaCpksResidentPrecision::fp64) {
      const auto spare = owner->remaining - (owner->induced.work().device_bytes - induced_bytes);
      check(owner->induced.retain_transition_builder(owner->cpks_reference.shape(),
                                                     std::min<std::size_t>(12ULL << 20, spare)),
            "stationary Hessian optional CPKS XC projection");
      owner->cpks_operator.set_build_decomposition(CudaKsWb97mvResponse::seed_local_xc,
                                                   CudaKsWb97mvResponse::induced_nonlocal_fock);
    }
    owner->charge(owner->induced.work().device_bytes - induced_bytes);
    check(response::cuda_cpks_reference_weights(
        owner->cpks_reference.shape(), owner->options.cpks, owner->cpks_reference.reference(),
        owner->at(owner->weights), owner->at(owner->cpks_offset), owner->cpks_bytes,
        owner->stream));
    check(owner->sum_weights(lease.density, owner->at(owner->weights), owner->totals));
    check(owner->gate());
    check(owner->authenticate());
    owner->receipt.nuclear_pairs = owner->pairs;
    impl_ = std::move(owner);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const HessianCapacityExceeded& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc&) {
    detail = "native stationary Hessian allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
cudaError_t CudaKsWb97mvHessian::apply(const double* direction, double* output, cudaStream_t stream,
                                       CudaKsWb97mvHessianResult& result) noexcept {
  result = {};
  if (!impl_) return cudaErrorInvalidResourceHandle;
  try {
    auto& owner = *impl_;
    auto error = owner.admit(direction, output, false, stream);
    if (error == cudaSuccess) error = owner.geometry.admit_actions(owner.coordinates + 1);
    if (error == cudaSuccess && owner.expected.identity.model.nonlocal_grid)
      error = owner.vv10_geometry.admit_actions(owner.coordinates + 1);
    if (error != cudaSuccess) return error;
    DrainOnFailure drain{stream};
    error = cudaMemcpyAsync(owner.at(owner.direction), direction,
                            owner.coordinates * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess || (error = owner.compute(result)) != cudaSuccess ||
        !result.converged())
      return error;
    if ((error = owner.induced.collect_work()) != cudaSuccess) {
      result.status = CudaKsWb97mvHessianStatus::failed;
      result.failure_check = "induced Fock work receipt";
      return error;
    }
    error = cudaMemcpyAsync(output, owner.at(owner.candidate), owner.coordinates * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) {
      drain.active = false;
      ++owner.receipt.published_hvps;
    } else {
      result.status = CudaKsWb97mvHessianStatus::failed;
    }
    return error;
  } catch (...) {
    result.status = CudaKsWb97mvHessianStatus::failed;
    return cudaErrorUnknown;
  }
}
cudaError_t CudaKsWb97mvHessian::hessian(double* output, cudaStream_t stream,
                                         CudaKsWb97mvHessianResult& result) noexcept {
  result = {};
  if (!impl_ || impl_->mode != CudaKsWb97mvHessianMode::dense_and_action)
    return cudaErrorInvalidResourceHandle;
  try {
    auto& owner = *impl_;
    auto error = owner.admit(nullptr, output, true, stream);
    if (error != cudaSuccess) return error;
    const auto actions = product(owner.coordinates, owner.coordinates + 1);
    if ((error = owner.geometry.admit_actions(actions)) != cudaSuccess) {
      result.failure_check = "dense Hessian exceeds the moving-grid AO replay work budget";
      return error;
    }
    if (owner.expected.identity.model.nonlocal_grid &&
        (error = owner.vv10_geometry.admit_actions(actions)) != cudaSuccess) {
      result.failure_check = "dense Hessian exceeds the nonlocal-grid AO replay work budget";
      return error;
    }
    auto* audit = owner.at<DenseAssemblyAudit>(owner.dense_audit_offset);
    response::CudaCpksDenseReplay cpks_replay{&audit->cpks, owner.at<int>(owner.status_offset), {}};
    const bool deferred_cpks =
        owner.options.cpks.resident_precision == response::CudaCpksResidentPrecision::fp64 &&
        !owner.has_df && !owner.has_ecp && !owner.fock.correction &&
        !owner.fock.primary->strategy().spec.exchange.present &&
        !owner.expected.identity.model.nonlocal_grid && owner.frame.spins == 1 &&
        owner.fock.primary->system().basis_representation == core::kCartesianBasis;
    if (deferred_cpks && owner.coordinates > SIZE_MAX / 2 / owner.options.cpks.gmres.max_iterations)
      return cudaErrorInvalidValue;
    // Drain before destroying the graph or releasing its captured arena.
    DrainOnFailure drain{stream};
    result.failure_check = "stationary Hessian frozen-source preparation";
    if ((error = owner.prepare_frozen_sources()) != cudaSuccess) return error;
    if ((error = cudaMemsetAsync(audit, 0, sizeof(*audit), stream)) != cudaSuccess) return error;
    const auto collect_columns = [&](cudaError_t pending) {
      DenseAssemblyAudit columns{};
      auto read = cudaMemcpyAsync(&columns, audit, sizeof(columns), cudaMemcpyDeviceToHost, stream);
      if (read == cudaSuccess) read = cudaStreamSynchronize(stream);
      if (read != cudaSuccess) return pending == cudaSuccess ? read : pending;
      result.completed_directions = columns.completed;
      owner.receipt.completed_directions += columns.completed;
      if (deferred_cpks && columns.cpks.last.code) {
        owner.cpks_operator.collect_dense(columns.cpks);
        if (!owner.operator_matrix) owner.receipt.cpks_strict_iterations += columns.cpks.iterations;
        const auto& last = columns.cpks.last;
        auto& gmres = result.response.gmres;
        gmres.status = static_cast<response::CudaGmresStatus>(last.code);
        gmres.rhs_norm = last.rhs_norm;
        gmres.residual_norm = last.residual_norm;
        gmres.tolerance = last.tolerance;
        gmres.iterations = last.iterations;
        gmres.restarts = last.restarts;
        gmres.operator_actions = 2 * last.iterations;
        gmres.preconditioner_actions = last.iterations;
        if (columns.cpks.failure != response::CudaCpksStatus::running) {
          result.response.status = columns.cpks.failure;
          result.status = CudaKsWb97mvHessianStatus::response_not_converged;
          result.failure_check = "stationary CPKS solve and physical response audit";
          return pending;
        }
        result.response.status = response::CudaCpksStatus::converged;
      }
      if (!columns.failed) return pending;
      result.status = CudaKsWb97mvHessianStatus::failed;
      if (columns.failed & cuda_grid_cartesian_repair_exhausted)
        result.failure_check = "Cartesian moving-grid repair work exhausted";
      else if (columns.completed != owner.coordinates)
        result.failure_check = "Hessian column failed its device transaction audit";
      else
        result.failure_check = "nonfinite or nonsymmetric assembled Hessian";
      return pending == cudaSuccess ? cudaErrorInvalidValue : pending;
    };
    for (std::size_t coordinate = 0; coordinate < owner.coordinates; ++coordinate) {
      result.status = CudaKsWb97mvHessianStatus::failed;
      unit<<<maps::blocks(owner.coordinates), 128, 0, stream>>>(owner.at(owner.direction),
                                                                owner.coordinates, coordinate);
      if ((error = cudaGetLastError()) != cudaSuccess ||
          (error = owner.compute(result, true, coordinate,
                                 deferred_cpks ? &cpks_replay : nullptr)) != cudaSuccess ||
          !result.converged())
        return collect_columns(error);
      result.status = CudaKsWb97mvHessianStatus::failed;
      store_column<<<maps::blocks(owner.coordinates), 128, 0, stream>>>(
          owner.at(owner.candidate), owner.coordinates, coordinate,
          owner.at<int>(owner.status_offset), owner.at(owner.complete), audit);
      if ((error = cudaGetLastError()) != cudaSuccess) return collect_columns(error);
    }
    result.status = CudaKsWb97mvHessianStatus::failed;
    symmetric<<<maps::blocks(owner.coordinates * owner.coordinates), 128, 0, stream>>>(
        owner.at(owner.complete), owner.coordinates, owner.options.symmetry_absolute_tolerance,
        owner.options.symmetry_relative_tolerance, &audit->failed);
    if ((error = collect_columns(cudaGetLastError())) != cudaSuccess) return error;
    if (result.status == CudaKsWb97mvHessianStatus::response_not_converged) return cudaSuccess;
    if (result.completed_directions != owner.coordinates) return cudaErrorInvalidValue;
    if ((error = owner.authenticate()) != cudaSuccess) return error;
    if ((error = owner.induced.collect_work()) != cudaSuccess) {
      result.failure_check = "induced Fock work receipt";
      return error;
    }
    error = cudaMemcpyAsync(output, owner.at(owner.complete),
                            owner.coordinates * owner.coordinates * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) {
      drain.active = false;
      ++owner.receipt.published_hessians;
      result.status = CudaKsWb97mvHessianStatus::converged;
    }
    return error;
  } catch (...) {
    result.status = CudaKsWb97mvHessianStatus::failed;
    return cudaErrorUnknown;
  }
}
CudaKsWb97mvHessianWork CudaKsWb97mvHessian::work() const noexcept {
  if (!impl_) return {};
  auto receipt = impl_->receipt;
  receipt.induced = impl_->induced.work();
  receipt.cpks_operator_columns = impl_->cpks_operator.columns();
  receipt.cpks_cached_actions = impl_->cpks_operator.actions();
  receipt.cpks_physical_checks = impl_->cpks_operator.physical_checks();
  receipt.geometry = impl_->geometry.work();
  receipt.grid = impl_->grid.work();
  receipt.vv10_geometry = impl_->vv10_geometry.work();
  receipt.separate_vv10 = impl_->vv10_grid.work();
  receipt.one_electron = impl_->one_electron.work();
  receipt.exact = impl_->exact.work();
  if (impl_->has_df) {
    receipt.fitted = impl_->full_df.response.work();
    receipt.raw_df = impl_->full_df.nuclear.work();
  }
  if (impl_->range_df.raw) {
    receipt.fitted_range = impl_->range_df.response.work();
    receipt.raw_df_range = impl_->range_df.nuclear.work();
  }
  if (impl_->has_ecp) receipt.ecp = impl_->ecp.work();
  return receipt;
}
}  // namespace generativeqc::dft
