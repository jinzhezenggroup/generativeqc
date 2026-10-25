#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "dft/hessian_aot.hpp"
#include "response/cuda_ks_cpks_reference.hpp"
#include "response/wb97mv_stationary_assembly.cuh"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_wb97mv_stationary_maps;
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void check(cudaError_t error, const char* phase = "stationary Hessian CUDA operation") {
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
__global__ void frozen_exact_response(const double* derivatives, const double* frozen,
                                      const double* delta_density, std::size_t panels,
                                      std::size_t coordinates, std::size_t column, double* output,
                                      int* status) {
  for (std::size_t left = blockIdx.x * blockDim.x + threadIdx.x; left < coordinates;
       left += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value = frozen[column * coordinates + left];
    for (std::size_t index = 0; index < panels; ++index)
      value += derivatives[left * panels + index] * delta_density[index];
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
__global__ void store_column(const double* vector, std::size_t count, std::size_t column,
                             double* matrix) {
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
  CudaKsGridReference grid_reference;
  response::CudaKsCpksReference cpks_reference;
  response::CudaCpksOperatorCache cpks_operator;
  CudaKsResidentFockBinding fock;
  CudaKsResidentOrbitalFrameBinding frame;
  CudaKsWb97mvResponse induced;
  CudaGridGeometryResponse geometry;
  CudaWb97mvGridResponse grid;
  CudaKsOneElectronResponse one_electron;
  CudaKsWb97mvNuclearJk exact;
  CudaKsWb97mvDfResponse fitted;
  CudaKsWb97mvDfNuclear raw_df;
  CudaKsWb97mvEcpNuclear ecp;
  CudaKsWb97mvHessianWork receipt;
  bool has_df{}, has_ecp{};
  cudaStream_t stream{};
  void* arena{};
  double* operator_matrix{};
  double* frozen_exact{};
  bool frozen_exact_ready{};
  std::size_t bytes{}, remaining{}, matrix{}, panels{}, coordinates{}, pairs{}, raw{}, metric{};
  std::size_t map_offset{}, map_bytes{}, cpks_offset{}, cpks_bytes{}, geometry_offset{},
      geometry_bytes{}, grid_offset{}, grid_bytes{}, status_offset{};
  std::size_t zero{}, weights{}, delta_density{}, delta_weights{}, totals{}, delta_totals{}, core{},
      overlap{}, core_direction{}, overlap_direction{}, exact_energy{}, exact_fock{},
      exact_direction{}, fitted_energy{}, fitted_fock{}, fitted_direction{}, xc_direction{}, rhs{},
      raw_values{}, fitted_weights{}, ecp_values{}, vectors{}, scratch_gradient{}, geometry_ao{},
      geometry_weights{}, direction{}, left{}, candidate{}, complete{}, charges{}, pair_atoms{},
      pair_fields{}, pair_inputs{}, pair_first{}, pair_second{}, incidences{};

  Impl(methods::PreparedBatch& owner, std::size_t index, CudaKsFinalStateToken token,
       CudaKsWb97mvHessianOptions policy)
      : batch(owner),
        item(index),
        expected(std::move(token)),
        options(policy),
        grid_reference(batch, item, expected),
        cpks_reference(batch, item, expected),
        remaining(policy.max_device_bytes) {}
  ~Impl() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (operator_matrix) (void)cudaFree(operator_matrix);
    if (frozen_exact) (void)cudaFree(frozen_exact);
    if (arena) (void)cudaFree(arena);
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
    require(size <= remaining, "stationary Hessian aggregate device budget exceeded");
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
          current != fock || !cuda_ks_wb97mv_model_matches(expected, current) ||
          methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, current_frame,
                                                           detail) != GENERATIVEQC_STATUS_SUCCESS ||
          current_frame != frame)
        return cudaErrorInvalidResourceHandle;
      const auto lease = grid_reference.reference();
      return lease.authenticate ? lease.authenticate(lease.owner) : cudaErrorInvalidResourceHandle;
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
        (frozen_exact && !disjoint(destination, {frozen_exact, receipt.frozen_exact_bytes})) ||
        (!full &&
         (!disjoint(source, {arena, bytes}) || !disjoint(source, destination) ||
          (frozen_exact && !disjoint(source, {frozen_exact, receipt.frozen_exact_bytes})))))
      return cudaErrorInvalidValue;
    const auto shape = grid_reference.shape();
    const Range immutable[]{
        {lease.basis, (3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao) * sizeof(double)},
        {lease.points, 3 * shape.points * sizeof(double)},
        {lease.weights, shape.points * sizeof(double)},
        {geometry_lease.atomic_weights, shape.points * sizeof(double)},
        {lease.density, panels * sizeof(double)},
        {frame.coefficients, panels * sizeof(double)},
        {frame.orbital_energies, frame.spins * frame.nao * sizeof(double)},
        {frame.overlap, matrix * sizeof(double)},
        {fock.hcore, matrix * sizeof(double)},
        {fock.fock, panels * sizeof(double)}};
    for (const auto range : immutable)
      if (!disjoint(destination, range)) return cudaErrorInvalidValue;
    if (has_df) {
      const auto source_df =
          scf::cuda_density_fitting_readonly_source(fock.primary->cuda_fitted_source());
      const auto metric_df =
          scf::cuda_density_fitting_readonly_metric(fock.primary->cuda_fitted_source());
      if (!source_df || !metric_df) return cudaErrorInvalidResourceHandle;
      if (!disjoint(destination, {source_df.whitening, metric * sizeof(double)}) ||
          !disjoint(destination, {metric_df.eigenvalues, source_df.naux * sizeof(double)}))
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }
  cudaError_t sum_weights(const double* density, const double* weight, std::size_t offset) {
    return maps::launch_weights(extents(), density, weight, at(offset), at(offset) + matrix,
                                at(map_offset), map_bytes, at<int>(status_offset), stream);
  }
  cudaError_t fitted_action(const double* density_direction) {
    const auto* values = at(raw_values);
    auto* weight = at(fitted_weights);
    const CudaWb97mvDfOutput output{
        at(fitted_energy), at(fitted_fock),       at(fitted_direction),     weight,
        weight + raw,      weight + raw + metric, weight + 2 * raw + metric};
    return fitted.apply(density_direction, values + raw + metric, values + 2 * raw + metric, output,
                        stream);
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
  cudaError_t prepare_frozen_exact() {
    frozen_exact_ready = false;
    if (!frozen_exact) return cudaSuccess;
    auto error = authenticate();
    for (std::size_t column = 0; column < coordinates && error == cudaSuccess; ++column) {
      unit<<<maps::blocks(coordinates), 128, 0, stream>>>(at(direction), coordinates, column);
      error = cudaGetLastError();
      if (error == cudaSuccess)
        error = exact.apply(at(direction), at(zero), exact_output(), stream);
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(frozen_exact + column * panels, at(exact_direction),
                                panels * sizeof(double), cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess)
        error =
            cudaMemcpyAsync(frozen_exact + coordinates * panels + column * coordinates, vector(3),
                            coordinates * sizeof(double), cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess) ++receipt.frozen_exact_columns;
    }
    const auto completed = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) error = completed;
    if (error == cudaSuccess) error = authenticate();
    frozen_exact_ready = error == cudaSuccess;
    return error;
  }
  cudaError_t compute(CudaKsWb97mvHessianResult& result, bool repeated_rhs = false,
                      std::size_t column = SIZE_MAX) {
    ++receipt.attempted_directions;
    auto error = authenticate();
    if (error != cudaSuccess) return error;
    if ((error = cudaMemsetAsync(at<int>(status_offset), 0, sizeof(int), stream)) != cudaSuccess)
      return error;
    finite<<<maps::blocks(coordinates), 128, 0, stream>>>(at(direction), coordinates,
                                                          at<int>(status_offset));
    if ((error = cudaGetLastError()) != cudaSuccess || (error = gate()) != cudaSuccess)
      return error;
    if ((error = cudaMemsetAsync(at(delta_totals), 0, 2 * matrix * sizeof(double), stream)) !=
        cudaSuccess)
      return error;
    const integrals::CudaOneElectronWeights frozen{at(totals), at(delta_totals),
                                                   at(totals) + matrix, at(delta_totals) + matrix};
    if ((error = one_electron.apply(at(direction), frozen, core_output(), stream)) != cudaSuccess)
      return error;
    const bool cached_exact = repeated_rhs && frozen_exact_ready && column < coordinates;
    if (cached_exact)
      error = cudaMemcpyAsync(at(exact_direction), frozen_exact + column * panels,
                              panels * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    else
      error = exact.apply(at(direction), at(zero), exact_output(), stream);
    if (error != cudaSuccess) return error;
    if (has_df) {
      auto* values = at(raw_values);
      const integrals::CudaDfNuclearValues output{values, values + raw, values + raw + metric,
                                                  values + 2 * raw + metric};
      if ((error = raw_df.values(at(direction), output, stream)) != cudaSuccess ||
          (error = fitted_action(at(zero))) != cudaSuccess)
        return error;
    }
    if (has_ecp) {
      auto* values = at(ecp_values);
      if ((error = ecp.values(at(direction),
                              {values, values + matrix, values + 2 * matrix, values + 3 * matrix},
                              stream)) != cudaSuccess)
        return error;
    }
    // Right geometry is prepared once for the RHS, and again with solved dD.
    // Mixed contractions below reuse AO jets and the cached bilateral VV10 JVP.
    if ((error = geometry.mixed(at(direction), at(direction), at(geometry_ao), at(geometry_weights),
                                stream)) != cudaSuccess)
      return error;
    const double* point_motion{};
    if ((error = geometry.point_directions(point_motion)) != cudaSuccess) return error;
    const auto shape = grid_reference.shape();
    const auto ao_panel = 4 * shape.points * shape.nao;
    if ((error = grid.prepare_nuclear_direction(
             at(geometry_ao) + ao_panel, at(geometry_weights) + shape.points,
             point_motion + 3 * shape.points, at(zero), at(xc_direction), stream)) != cudaSuccess)
      return error;
    error = maps::launch_rhs(extents(), at(core_direction), at(ecp_values) + 2 * matrix,
                             at(ecp_values) + 3 * matrix, at(exact_direction), at(fitted_direction),
                             at(xc_direction), at(rhs), at(map_offset), map_bytes,
                             at<int>(status_offset), stream);
    if (error != cudaSuccess || (error = gate()) != cudaSuccess) return error;
    error = response::solve_cuda_cpks(
        cpks_reference.shape(), options.cpks, cpks_reference.reference(), at(rhs),
        at(overlap_direction), CudaKsWb97mvResponse::induced_fock, &induced, at(delta_density),
        at(delta_weights), at(cpks_offset), cpks_bytes, stream, result.response,
        repeated_rhs && operator_matrix ? &cpks_operator : nullptr);
    if (error != cudaSuccess) return error;
    if (!result.response.converged()) {
      result.status = CudaKsWb97mvHessianStatus::response_not_converged;
      return cudaSuccess;
    }
    if ((error = sum_weights(at(delta_density), at(delta_weights), delta_totals)) != cudaSuccess ||
        (error = gate()) != cudaSuccess)
      return error;
    if ((error = one_electron.apply(at(direction), frozen, core_output(), stream)) != cudaSuccess)
      return error;
    if (cached_exact) {
      // Exact J/K is quadratic in D: d(gradient)/dD : dD = F_R(D) : dD.
      // Retain contracted AO derivatives, not four-index value/gradient jets.
      frozen_exact_response<<<maps::blocks(coordinates), 128, 0, stream>>>(
          frozen_exact, frozen_exact + coordinates * panels, at(delta_density), panels, coordinates,
          column, vector(3), at<int>(status_offset));
      error = cudaGetLastError();
      if (error == cudaSuccess) ++receipt.frozen_exact_contractions;
    } else {
      error = exact.apply(at(direction), at(delta_density), exact_output(), stream);
    }
    if (error != cudaSuccess) return error;
    if (has_df) {
      if ((error = fitted_action(at(delta_density))) != cudaSuccess) return error;
      const auto* weight = at(fitted_weights);
      const integrals::CudaDfNuclearWeights source{weight, weight + raw, weight + raw + metric,
                                                   weight + 2 * raw + metric};
      if ((error = raw_df.contract(at(direction), source, {at(scratch_gradient), vector(4)},
                                   stream)) != cudaSuccess)
        return error;
    }
    if (has_ecp && (error = ecp.contract(at(direction), at(delta_density),
                                         {at(scratch_gradient), at(scratch_gradient) + coordinates,
                                          vector(1), vector(2)},
                                         stream)) != cudaSuccess)
      return error;
    if ((error = grid.prepare_nuclear_direction(
             at(geometry_ao) + ao_panel, at(geometry_weights) + shape.points,
             point_motion + 3 * shape.points, at(delta_density), at(xc_direction), stream)) !=
        cudaSuccess)
      return error;
    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      unit<<<maps::blocks(coordinates), 128, 0, stream>>>(at(left), coordinates, coordinate);
      if ((error = cudaGetLastError()) != cudaSuccess ||
          (error = geometry.mixed(at(left), at(direction), at(geometry_ao), at(geometry_weights),
                                  stream)) != cudaSuccess ||
          (error = geometry.point_directions(point_motion)) != cudaSuccess)
        return error;
      if ((error = grid.nuclear_mixed(at(geometry_ao), at(geometry_ao) + 2 * ao_panel,
                                      at(geometry_weights), at(geometry_weights) + 2 * shape.points,
                                      point_motion, vector(5) + coordinate, stream)) != cudaSuccess)
        return error;
    }
    if ((error = nuclear()) != cudaSuccess) return error;
    error = maps::launch_hvp(extents(), vector(0), vector(1), vector(2), vector(3), vector(4),
                             vector(5), vector(6), at(candidate), at(map_offset), map_bytes,
                             at<int>(status_offset), stream);
    if (error != cudaSuccess || (error = gate()) != cudaSuccess ||
        (error = authenticate()) != cudaSuccess)
      return error;
    ++receipt.completed_directions;
    ++result.completed_directions;
    result.status = CudaKsWb97mvHessianStatus::converged;
    return cudaSuccess;
  }
};

CudaKsWb97mvHessian::CudaKsWb97mvHessian() = default;
CudaKsWb97mvHessian::~CudaKsWb97mvHessian() = default;
generativeqc_status CudaKsWb97mvHessian::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                 const CudaKsFinalStateToken& expected,
                                                 const CudaKsWb97mvHessianOptions& options,
                                                 std::string& detail) {
  impl_.reset();
  if (expected.identity.model.nonlocal_grid) {
    detail = "independent XC/VV10 grids do not yet support Hessians";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    require(options.max_device_bytes && options.max_assembly_terms &&
                std::isfinite(options.symmetry_absolute_tolerance) &&
                std::isfinite(options.symmetry_relative_tolerance) &&
                options.symmetry_absolute_tolerance >= 0 &&
                options.symmetry_relative_tolerance >= 0,
            "invalid stationary Hessian options");
    auto owner = std::make_unique<Impl>(batch, item, expected, options);
    auto status = owner->grid_reference.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = owner->cpks_reference.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status =
        methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, owner->fock, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(cuda_ks_wb97mv_model_matches(expected, owner->fock),
            "stationary Hessian requires the complete native WB97M-V model");
    status = methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, owner->frame,
                                                              detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto lease = owner->grid_reference.reference();
    const auto shape = owner->grid_reference.shape();
    auto geometry_shape = owner->grid_reference.geometry_shape();
    for (const auto& shell : owner->fock.primary->system().shells) {
      const auto angular = static_cast<std::size_t>(shell.angular_momentum);
      const bool spherical =
          owner->fock.primary->system().basis_representation == core::kSphericalBasis;
      const auto components = spherical ? 2 * angular + 1 : (angular + 1) * (angular + 2) / 2;
      const auto terms = spherical && angular > 1 ? 3 : 1;
      const auto incidences = product(product(components, shell.primitives.size()), terms);
      require(incidences <= SIZE_MAX - geometry_shape.ao_primitive_terms,
              "moving-grid AO primitive incidence overflow");
      geometry_shape.ao_primitive_terms += incidences;
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
    if (owner->has_df) {
      const auto source =
          scf::cuda_density_fitting_readonly_source(owner->fock.primary->cuda_fitted_source());
      require(source && source.nao == shape.nao,
              "stationary Hessian requires native raw DF sources");
      owner->raw = product(owner->matrix, source.naux);
      owner->metric = product(source.naux, source.naux);
    }
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
    if (owner->pairs) {
      map(maps::workspace_nuclear_inputs, maps::work_nuclear_inputs);
      map(maps::workspace_nuclear_second, maps::work_nuclear_second);
      map(maps::workspace_nuclear_atom, maps::work_nuclear_atom, shape.atoms);
    }
    // First reserve the complete mandatory baseline. Optional grid replay and
    // parallel one-electron schedules may spend only the surplus after every
    // mandatory consumer, including the geometry AO output and exact sources.
    owner->options.grid.nuclear_response = true;
    owner->options.grid.replay_nuclear_maps = false;
    check(cuda_wb97mv_grid_workspace(shape, owner->options.grid, owner->grid_bytes),
          "stationary Hessian XC/VV10 workspace admission");
    owner->options.geometry.partition_iterations =
        owner->grid_reference.geometry_options().partition_iterations;
    owner->options.geometry.coincident_tolerance =
        owner->grid_reference.geometry_options().coincident_tolerance;
    check(cuda_grid_geometry_workspace(geometry_shape, owner->options.geometry,
                                       owner->geometry_bytes),
          "stationary Hessian moving-grid workspace admission");
    check(response::cuda_cpks_workspace(owner->cpks_reference.shape(), owner->options.cpks,
                                        owner->cpks_bytes),
          "stationary Hessian CPKS workspace admission");
    Layout layout;
    owner->map_offset = layout.reserve(owner->map_bytes, 1);
    owner->status_offset = layout.reserve(1, sizeof(int));
    const auto matrix = owner->matrix, panels = owner->panels, coordinates = owner->coordinates;
    for (auto* offset :
         {&owner->zero, &owner->weights, &owner->delta_density, &owner->delta_weights,
          &owner->exact_fock, &owner->exact_direction, &owner->fitted_fock,
          &owner->fitted_direction, &owner->xc_direction, &owner->rhs})
      *offset = layout.reserve(panels);
    for (auto* offset : {&owner->totals, &owner->delta_totals})
      *offset = layout.reserve(product(2, matrix));
    for (auto* offset :
         {&owner->core, &owner->overlap, &owner->core_direction, &owner->overlap_direction})
      *offset = layout.reserve(matrix);
    for (auto* offset : {&owner->exact_energy, &owner->fitted_energy}) *offset = layout.reserve(1);
    owner->raw_values = layout.reserve(product(2, owner->raw + owner->metric));
    owner->fitted_weights = layout.reserve(product(2, owner->raw + owner->metric));
    owner->ecp_values = layout.reserve(product(4, matrix));
    owner->vectors = layout.reserve(product(7, coordinates));
    owner->scratch_gradient = layout.reserve(product(2, coordinates));
    owner->geometry_ao = layout.reserve(product(12, product(shape.points, shape.nao)));
    owner->geometry_weights = layout.reserve(product(3, shape.points));
    for (auto* offset : {&owner->direction, &owner->left, &owner->candidate})
      *offset = layout.reserve(coordinates);
    owner->complete = layout.reserve(product(coordinates, coordinates));
    owner->charges = layout.reserve(shape.atoms);
    owner->pair_atoms = layout.reserve(product(2, owner->pairs), sizeof(std::size_t));
    owner->pair_fields = layout.reserve(product(14, owner->pairs));
    owner->pair_inputs = layout.reserve(product(7, owner->pairs));
    owner->pair_first = layout.reserve(product(3, owner->pairs));
    owner->pair_second = layout.reserve(product(3, owner->pairs));
    owner->incidences = layout.reserve(product(3, shape.atoms - 1));
    owner->geometry_offset = layout.reserve(owner->geometry_bytes, 1);
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
    if (owner->has_df) {
      auto raw_options = options.raw_df;
      raw_options.max_workspace_bytes = std::min(raw_options.max_workspace_bytes, owner->remaining);
      status = owner->raw_df.prepare(batch, item, expected, raw_options, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      owner->charge(owner->raw_df.work().workspace_bytes);
      auto fitted_options = options.fitted;
      fitted_options.max_workspace_bytes =
          std::min(fitted_options.max_workspace_bytes, owner->remaining);
      status = owner->fitted.prepare(batch, item, expected, fitted_options, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      owner->charge(owner->fitted.work().workspace_bytes);
    }
    if (owner->has_ecp) {
      auto ecp_options = options.ecp;
      ecp_options.max_workspace_bytes = std::min(ecp_options.max_workspace_bytes, owner->remaining);
      status = owner->ecp.prepare(batch, item, expected, ecp_options, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      owner->charge(owner->ecp.device_bytes());
    }
    // Reserve every mandatory consumer before optional integral residency.
    auto exact_options = options.exact;
    exact_options.part = owner->has_df ? CudaWb97mvJkPart::correction : CudaWb97mvJkPart::exact;
    exact_options.max_workspace_bytes =
        std::min(exact_options.max_workspace_bytes, owner->remaining);
    status = owner->exact.prepare(batch, item, expected, exact_options, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      detail = "stationary Hessian exact nuclear J/K: " + detail;
      return status;
    }
    owner->charge(owner->exact.work().workspace_bytes);
    const auto grid_reserved = owner->bytes - owner->grid_offset;
    owner->options.grid.replay_nuclear_maps = options.grid.replay_nuclear_maps;
    // All arena slots are 256-byte aligned. Do not let an optional payload
    // consume the alignment bytes needed by its enclosing reservation.
    owner->options.grid.max_workspace_bytes =
        std::min(options.grid.max_workspace_bytes, (grid_reserved + owner->remaining) / 256 * 256);
    check(cuda_wb97mv_grid_workspace(shape, owner->options.grid, owner->grid_bytes),
          "stationary Hessian optional XC/VV10 workspace admission");
    layout.bytes = owner->grid_offset;
    owner->grid_offset = layout.reserve(owner->grid_bytes, 1);
    owner->charge(layout.bytes - owner->bytes);
    owner->bytes = layout.bytes;
    core_options = options.one_electron;
    core_options.max_workspace_bytes =
        std::min(core_options.max_workspace_bytes, core_bytes + owner->remaining);
    status = owner->one_electron.prepare(batch, item, expected, core_options, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    require(owner->one_electron.work().device_bytes >= core_bytes,
            "one-electron preparation differs from its mandatory reservation");
    owner->charge(owner->one_electron.work().device_bytes - core_bytes);
    check(cudaMalloc(&owner->arena, owner->bytes));
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
    check(cudaStreamSynchronize(owner->stream));
    upload_drain.active = false;
    check(owner->geometry.prepare(geometry_shape, owner->options.geometry,
                                  owner->grid_reference.geometry_reference(),
                                  owner->at(owner->geometry_offset), owner->geometry_bytes));
    check(owner->grid.prepare(shape, owner->options.grid, lease, owner->at(owner->grid_offset),
                              owner->grid_bytes));
    std::size_t operator_bytes{};
    check(response::cuda_cpks_operator_workspace(owner->cpks_reference.shape(), owner->options.cpks,
                                                 operator_bytes));
    if (operator_bytes && operator_bytes <= options.max_resident_cpks_bytes &&
        operator_bytes <= owner->remaining &&
        operator_bytes <= owner->options.cpks.max_workspace_bytes - owner->cpks_bytes) {
      const auto allocation =
          cudaMalloc(reinterpret_cast<void**>(&owner->operator_matrix), operator_bytes);
      if (allocation == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        owner->operator_matrix = nullptr;
      } else {
        check(allocation);
        owner->charge(operator_bytes);
        owner->receipt.cpks_operator_bytes = operator_bytes;
        owner->cpks_operator.reset(owner->operator_matrix, operator_bytes);
      }
    }
    const auto induced_bytes = owner->induced.work().device_bytes;
    const auto frozen_bytes =
        product(product(owner->coordinates, owner->panels + owner->coordinates), sizeof(double));
    if (frozen_bytes <= options.max_frozen_exact_bytes && frozen_bytes <= owner->remaining) {
      const auto allocation =
          cudaMalloc(reinterpret_cast<void**>(&owner->frozen_exact), frozen_bytes);
      if (allocation == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        owner->frozen_exact = nullptr;
      } else {
        check(allocation);
        owner->charge(frozen_bytes);
        owner->receipt.frozen_exact_bytes = frozen_bytes;
      }
    }
    status = owner->induced.retain_integrals(owner->remaining, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
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
    if (error != cudaSuccess) return error;
    DrainOnFailure drain{stream};
    error = cudaMemcpyAsync(owner.at(owner.direction), direction,
                            owner.coordinates * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess || (error = owner.compute(result)) != cudaSuccess ||
        !result.converged())
      return error;
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
  if (!impl_) return cudaErrorInvalidResourceHandle;
  try {
    auto& owner = *impl_;
    auto error = owner.admit(nullptr, output, true, stream);
    if (error != cudaSuccess) return error;
    DrainOnFailure drain{stream};
    if ((error = owner.prepare_frozen_exact()) != cudaSuccess) return error;
    for (std::size_t coordinate = 0; coordinate < owner.coordinates; ++coordinate) {
      result.status = CudaKsWb97mvHessianStatus::failed;
      unit<<<maps::blocks(owner.coordinates), 128, 0, stream>>>(owner.at(owner.direction),
                                                                owner.coordinates, coordinate);
      if ((error = cudaGetLastError()) != cudaSuccess ||
          (error = owner.compute(result, true, coordinate)) != cudaSuccess || !result.converged())
        return error;
      result.status = CudaKsWb97mvHessianStatus::failed;
      store_column<<<maps::blocks(owner.coordinates), 128, 0, stream>>>(
          owner.at(owner.candidate), owner.coordinates, coordinate, owner.at(owner.complete));
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
    }
    result.status = CudaKsWb97mvHessianStatus::failed;
    symmetric<<<maps::blocks(owner.coordinates * owner.coordinates), 128, 0, stream>>>(
        owner.at(owner.complete), owner.coordinates, owner.options.symmetry_absolute_tolerance,
        owner.options.symmetry_relative_tolerance, owner.at<int>(owner.status_offset));
    if ((error = cudaGetLastError()) != cudaSuccess || (error = owner.gate()) != cudaSuccess ||
        (error = owner.authenticate()) != cudaSuccess)
      return error;
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
  receipt.one_electron = impl_->one_electron.work();
  receipt.exact = impl_->exact.work();
  if (impl_->has_df) {
    receipt.fitted = impl_->fitted.work();
    receipt.raw_df = impl_->raw_df.work();
  }
  if (impl_->has_ecp) receipt.ecp = impl_->ecp.work();
  return receipt;
}
}  // namespace generativeqc::dft
