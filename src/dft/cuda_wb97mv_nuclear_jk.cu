#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "dft/cuda_wb97mv_nuclear_jk.hpp"
#include "dft/hessian_aot.hpp"
#include "response/nuclear_integral_response.cuh"
#include "response/wb97mv_nuclear_jk.cuh"

namespace generativeqc::dft {
namespace {
namespace aot = hessian_aot;
namespace maps = generativeqc_wb97mv_nuclear_jk_ad;
using Shell = integrals::CudaOneElectronShell;

struct DeviceGroup {
  Shell shells[4];
  long long atoms[4];
  std::size_t components{}, quartets{};
  maps::Kernel kernel;
};
struct DeviceTile {
  unsigned components[16]{};
  unsigned count{};
};
struct Tile {
  const aot::Program* program{};
  DeviceTile device;
};
struct Group {
  DeviceGroup device;
  std::size_t radials{}, closure{};
};
struct Layout {
  std::size_t bytes{}, matrix{}, panels{}, basis_elements{}, groups{}, tiles{}, shells{}, records{},
      raw{}, cache{}, projected{}, candidates{}, status{}, max_components{}, max_quartets{},
      radial_records{}, candidate_count{};
  std::uint64_t components{}, projections{}, reductions{};
};
struct Schedule {
  Layout plan;
  std::vector<Group> groups;
  std::vector<Tile> tiles;
  std::array<std::vector<std::size_t>, 1280> closure;
};

bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
bool add(std::size_t& value, std::size_t term) {
  if (term > SIZE_MAX - value) return false;
  value += term;
  return true;
}
bool append(Layout& plan, std::size_t count, std::size_t width, std::size_t& offset) {
  std::size_t bytes{};
  if (!multiply(count, width, bytes) || bytes > SIZE_MAX - 255 ||
      plan.bytes > SIZE_MAX - bytes - 255)
    return false;
  offset = plan.bytes;
  plan.bytes = (plan.bytes + bytes + 255) / 256 * 256;
  return true;
}
unsigned cartesian_count(unsigned angular) { return (angular + 1) * (angular + 2) / 2; }
unsigned blocks(std::size_t count) { return std::min<std::size_t>((count - 1) / 128 + 1, 65535); }

cudaError_t schedule(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                     const CudaWb97mvJkOptions& options, Schedule& result) {
  auto& plan = result.plan;
  if (!shape.atoms || !shape.primitives || !shape.nao || shells.empty() ||
      (shape.spins != 1 && shape.spins != 2) || !options.radial_tile || options.radial_tile > 64 ||
      shape.atoms > static_cast<std::size_t>(std::numeric_limits<long long>::max()) / 3 ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 || !multiply(shape.nao, shape.nao, plan.matrix) ||
      !multiply(plan.matrix, shape.spins, plan.panels) ||
      !multiply(shape.atoms, 3, plan.basis_elements))
    return cudaErrorInvalidValue;
  std::size_t count{};
  if (!multiply(shape.primitives, 2, count) || !add(plan.basis_elements, count) ||
      !multiply(shape.nao, 16, count) || !add(plan.basis_elements, count) ||
      plan.basis_elements > SIZE_MAX / sizeof(double))
    return cudaErrorInvalidValue;
  std::size_t primitive_begin{}, ao_begin{};
  for (const auto& shell : shells) {
    if (shell.atom >= shape.atoms || shell.angular > 3 || !shell.primitive_count ||
        shell.primitive_begin != primitive_begin || shell.ao_begin != ao_begin ||
        (shell.ao_count != cartesian_count(shell.angular) &&
         shell.ao_count != 2 * shell.angular + 1) ||
        !add(primitive_begin, shell.primitive_count) || !add(ao_begin, shell.ao_count) ||
        primitive_begin > shape.primitives || ao_begin > shape.nao)
      return cudaErrorInvalidValue;
  }
  if (primitive_begin != shape.primitives || ao_begin != shape.nao) return cudaErrorInvalidValue;
  std::vector<maps::Kernel> kernels;
  switch (options.part) {
    case CudaWb97mvJkPart::exact:
      kernels = {maps::Kernel::k_primary, maps::Kernel::k_correction};
      break;
    case CudaWb97mvJkPart::primary:
      kernels = {maps::Kernel::k_primary};
      break;
    case CudaWb97mvJkPart::correction:
      kernels = {maps::Kernel::k_correction};
      break;
    case CudaWb97mvJkPart::coulomb:
      kernels = {maps::Kernel::k_coulomb};
      break;
    case CudaWb97mvJkPart::short_range:
      kernels = {maps::Kernel::k_short};
      break;
    case CudaWb97mvJkPart::long_range:
      kernels = {maps::Kernel::k_long};
      break;
    default:
      return cudaErrorInvalidValue;
  }
  count = kernels.size();
  for (unsigned slot = 0; slot < 4; ++slot)
    if (!multiply(count, shells.size(), count)) return cudaErrorInvalidValue;
  if (count > options.max_groups) return cudaErrorInvalidValue;
  result.groups.reserve(count);
  // Complete ordered closure, not symmetry reconstruction. Each package is
  // scanned once per selected angular tuple and method part, never per radial.
  for (const auto& first : shells)
    for (const auto& second : shells)
      for (const auto& third : shells)
        for (const auto& fourth : shells)
          for (const auto kernel : kernels) {
            Group group{{{first, second, third, fourth}, {}, 1, 1, kernel}, 1, 0};
            const auto selected = static_cast<std::size_t>(kernel);
            auto key = selected;
            for (unsigned slot = 0; slot < 4; ++slot) {
              const auto& shell = group.device.shells[slot];
              group.device.atoms[slot] = shell.atom;
              key = key * 4 + shell.angular;
              if (!multiply(group.radials, shell.primitive_count, group.radials) ||
                  !multiply(group.device.components, cartesian_count(shell.angular),
                            group.device.components) ||
                  !multiply(group.device.quartets, shell.ao_count, group.device.quartets))
                return cudaErrorInvalidValue;
            }
            group.closure = key;
            auto& closure = result.closure[key];
            if (closure.empty()) {
              std::array<bool, 10000> covered{};
              for (const auto& program : aot::programs()) {
                const auto& signature = program.signature;
                if (program.kind != aot::Kind::integral || signature.family != "eri" ||
                    signature.angular.size() != 4 || signature.radial != maps::radial[selected])
                  continue;
                bool matches = true;
                for (unsigned slot = 0; slot < 4; ++slot)
                  matches &= signature.angular[slot] == group.device.shells[slot].angular;
                if (!matches) continue;
                if (signature.omega != maps::omega[selected] || !program.integral ||
                    program.centers != 4 || program.record_stride != 17 || program.outputs != 25 ||
                    program.components.empty() || program.components.size() > 16)
                  return cudaErrorNotSupported;
                Tile tile{&program, {}};
                tile.device.count = program.components.size();
                for (unsigned index = 0; index < tile.device.count; ++index) {
                  const auto component = program.components[index];
                  if (component >= group.device.components || covered[component])
                    return cudaErrorNotSupported;
                  covered[component] = true;
                  tile.device.components[index] = component;
                }
                closure.push_back(result.tiles.size());
                result.tiles.push_back(tile);
              }
              if (!std::all_of(covered.begin(), covered.begin() + group.device.components,
                               [](bool value) { return value; }))
                return cudaErrorNotSupported;
            }
            const auto chunks = (group.radials - 1) / options.radial_tile + 1;
            if (!multiply(group.radials, group.device.components, count) ||
                !add(plan.components, count) || !multiply(group.device.quartets, 81, count) ||
                !add(plan.projections, count) ||
                !multiply(group.device.quartets, 25 + 8 * shape.spins, count) ||
                !add(plan.reductions, count) || !multiply(group.device.components, 25, count) ||
                !multiply(count, chunks, count) || !add(plan.reductions, count) ||
                !multiply(group.radials, closure.size(), count) ||
                !add(plan.radial_records, count) ||
                plan.components > options.max_primitive_component_evaluations ||
                plan.projections > options.max_projection_terms ||
                plan.reductions > options.max_reduction_terms)
              return cudaErrorInvalidValue;
            plan.max_components = std::max(plan.max_components, group.device.components);
            plan.max_quartets = std::max(plan.max_quartets, group.device.quartets);
            result.groups.push_back(group);
          }
  plan.candidate_count = 1;
  if (!multiply(plan.panels, 2, count) || !add(plan.candidate_count, count) ||
      !multiply(shape.atoms, 6, count) || !add(plan.candidate_count, count) ||
      !append(plan, result.groups.size(), sizeof(DeviceGroup), plan.groups) ||
      !append(plan, result.tiles.size(), sizeof(DeviceTile), plan.tiles) ||
      !append(plan, shells.size(), sizeof(Shell), plan.shells) ||
      !append(plan, options.radial_tile * 17, sizeof(double), plan.records) ||
      !append(plan, 16 * 25, sizeof(double), plan.raw) ||
      !append(plan, plan.max_components * 25, sizeof(double), plan.cache) ||
      !append(plan, plan.max_quartets * 41, sizeof(double), plan.projected) ||
      !append(plan, plan.candidate_count, sizeof(double), plan.candidates) ||
      !append(plan, 1, sizeof(int), plan.status) || plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  if (!first || !second) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first),
             right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
cudaError_t device_buffer(const void* pointer, int device) {
  if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double))
    return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  const auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
}
cudaError_t drain(cudaError_t error, cudaStream_t stream) {
  const auto completed = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completed : error;
}
cudaError_t gate(int* status, cudaStream_t stream) {
  int failed{};
  auto error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error == cudaSuccess && failed ? cudaErrorInvalidValue : error;
}
__global__ void finite_values(const double* values, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}
__global__ void symmetric(const double* values, std::size_t nao, std::size_t spins, double absolute,
                          double relative, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < spins * nao * nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto within = index % (nao * nao), spin = index / (nao * nao);
    const auto value = values[index],
               other = values[spin * nao * nao + (within % nao) * nao + within / nao];
    if (!isfinite(value) || !isfinite(other) ||
        fabs(value - other) > absolute + relative * fmax(fabs(value), fabs(other)))
      atomicExch(status, 1);
  }
}
__global__ void validate_basis(CudaWb97mvJkShape shape, const Shell* shells, std::size_t count,
                               const double* basis, int* status) {
  const auto* primitives = basis + 3 * shape.atoms;
  const auto* aos = primitives + 2 * shape.primitives;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < max(count, shape.primitives);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (index < shape.primitives && !(primitives[2 * index] > 0)) atomicExch(status, 1);
    if (index >= count) continue;
    const auto shell = shells[index];
    for (std::size_t orbital = 0; orbital < shell.ao_count; ++orbital) {
      const auto* record = aos + 16 * (shell.ao_begin + orbital);
      if (record[0] != shell.atom || record[1] != shell.primitive_begin ||
          record[2] != shell.primitive_count || record[3] < 1 || record[3] > 3 ||
          floor(record[3]) != record[3]) {
        atomicExch(status, 1);
        continue;
      }
      for (unsigned term = 0; term < static_cast<unsigned>(record[3]); ++term) {
        double degree{};
        for (unsigned axis = 0; axis < 3; ++axis) {
          const auto power = record[4 + 4 * term + axis];
          if (!isfinite(power) || power < 0 || power > 3 || floor(power) != power)
            atomicExch(status, 1);
          degree += power;
        }
        if (degree != shell.angular) atomicExch(status, 1);
      }
    }
  }
}
__global__ void build_records(CudaWb97mvJkShape shape, const DeviceGroup* group,
                              const double* basis, std::size_t begin, std::size_t count,
                              double* records, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto* primitives = basis + 3 * shape.atoms;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    auto radial = begin + index;
    auto* record = records + 17 * index;
    double scale = 1;
    for (unsigned slot = 4; slot-- > 0;) {
      const auto shell = group->shells[slot];
      const auto primitive = shell.primitive_begin + radial % shell.primitive_count;
      radial /= shell.primitive_count;
      record[slot] = primitives[2 * primitive];
      scale *= primitives[2 * primitive + 1];
      for (unsigned axis = 0; axis < 3; ++axis)
        record[4 + 3 * slot + axis] = basis[3 * shell.atom + axis];
    }
    if (!isfinite(scale)) {
      atomicExch(status, 1);
      scale = 0;
    }
    record[16] = scale;
  }
}
__global__ void accumulate(const DeviceGroup* group, const DeviceTile* tile, const double* raw,
                           double* cache, int* status) {
  if (atomicAdd(status, 0)) return;
  for (unsigned index = threadIdx.x; index < tile->count * 25; index += blockDim.x) {
    const auto component = tile->components[index / 25], field = index % 25;
    cache[field * group->components + component] += raw[index];
  }
}
__device__ std::size_t cartesian_index(const double* term, unsigned angular) {
  const auto remainder = angular - static_cast<unsigned>(term[0]);
  return remainder * (remainder + 1) / 2 + static_cast<unsigned>(term[2]);
}
__device__ void quartet_coordinates(std::size_t linear, const DeviceGroup* group,
                                    std::size_t* local) {
  for (unsigned slot = 4; slot-- > 0;) {
    local[slot] = linear % group->shells[slot].ao_count;
    linear /= group->shells[slot].ao_count;
  }
}
__device__ std::size_t quartet_index(const std::size_t* local, const DeviceGroup* group) {
  std::size_t result{};
  for (unsigned slot = 0; slot < 4; ++slot)
    result = result * group->shells[slot].ao_count + local[slot];
  return result;
}
__global__ void project(CudaWb97mvJkShape shape, const DeviceGroup* group, const double* basis,
                        const double* direction, const double* density, const double* delta_density,
                        const double* cache, double* projected, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t quartet = blockIdx.x * blockDim.x + threadIdx.x; quartet < group->quartets;
       quartet += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    std::size_t local[4], ao[4];
    quartet_coordinates(quartet, group, local);
    const double* records[4];
    for (unsigned slot = 0; slot < 4; ++slot) {
      ao[slot] = group->shells[slot].ao_begin + local[slot];
      records[slot] = basis + 3 * shape.atoms + 2 * shape.primitives + 16 * ao[slot];
    }
    double integral[25]{};
    // Public AO polynomials gather the cached raw fields. Component factors
    // appear here exactly once, including real-spherical expansion coefficients.
    for (unsigned first = 0; first < static_cast<unsigned>(records[0][3]); ++first)
      for (unsigned second = 0; second < static_cast<unsigned>(records[1][3]); ++second)
        for (unsigned third = 0; third < static_cast<unsigned>(records[2][3]); ++third)
          for (unsigned fourth = 0; fourth < static_cast<unsigned>(records[3][3]); ++fourth) {
            const unsigned terms[] = {first, second, third, fourth};
            std::size_t component{};
            double coefficient = 1;
            for (unsigned slot = 0; slot < 4; ++slot) {
              const auto angular = static_cast<unsigned>(group->shells[slot].angular);
              const auto* term = records[slot] + 4 + 4 * terms[slot];
              component =
                  component * ((angular + 1) * (angular + 2) / 2) + cartesian_index(term, angular);
              coefficient *= term[3];
            }
            for (unsigned field = 0; field < 25; ++field)
              integral[field] += coefficient * cache[field * group->components + component];
          }
    double motion{};
    for (unsigned coordinate = 0; coordinate < 12; ++coordinate)
      motion +=
          integral[1 + coordinate] * direction[3 * group->atoms[coordinate / 3] + coordinate % 3];
    maps::Result result;
    if (!maps::evaluate(group->kernel, shape.spins, density, delta_density, shape.nao, ao,
                        integral[0], motion, result))
      atomicExch(status, 1);
    auto* output = projected + 41 * quartet;
    output[0] = result.energy;
    for (unsigned incidence = 0; incidence < 8; ++incidence) {
      output[1 + incidence] = result.fock[incidence];
      output[9 + incidence] = result.motion[incidence];
    }
    for (unsigned coordinate = 0; coordinate < 12; ++coordinate) {
      double contraction[4];
      if (!generativeqc_nuclear_integral_ad::contract(integral[0], integral[1 + coordinate],
                                                      integral[13 + coordinate], result.weight,
                                                      result.delta_weight, 1, 0, contraction))
        atomicExch(status, 1);
      output[17 + coordinate] = contraction[2];
      output[29 + coordinate] = contraction[3];
    }
  }
}
__global__ void reduce_fock(CudaWb97mvJkShape shape, const DeviceGroup* group, unsigned incidence,
                            const double* projected, double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const unsigned pairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  const auto first = pairs[incidence][0], second = pairs[incidence][1];
  unsigned rest[2], used{};
  for (unsigned slot = 0; slot < 4; ++slot)
    if (slot != first && slot != second) rest[used++] = slot;
  const auto width = group->shells[second].ao_count;
  const auto pairs_count = group->shells[first].ao_count * width;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < shape.spins * pairs_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto spin = index / pairs_count, pair = index % pairs_count;
    std::size_t local[4]{};
    local[first] = pair / width;
    local[second] = pair % width;
    double value{}, motion{};
    for (local[rest[0]] = 0; local[rest[0]] < group->shells[rest[0]].ao_count; ++local[rest[0]])
      for (local[rest[1]] = 0; local[rest[1]] < group->shells[rest[1]].ao_count; ++local[rest[1]]) {
        const auto* output = projected + 41 * quartet_index(local, group);
        value += output[1 + 2 * incidence + spin];
        motion += output[9 + 2 * incidence + spin];
      }
    const auto target = spin * shape.nao * shape.nao +
                        (group->shells[first].ao_begin + local[first]) * shape.nao +
                        group->shells[second].ao_begin + local[second];
    candidate[1 + target] += value;
    candidate[1 + shape.spins * shape.nao * shape.nao + target] += motion;
  }
}
__global__ void reduce_vectors(CudaWb97mvJkShape shape, const DeviceGroup* group,
                               const double* projected, double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto coordinate = threadIdx.x;
  if (coordinate >= 12) return;
  const auto center = coordinate / 3, axis = coordinate % 3;
  // Only the first mathematical occurrence owns a physical atom-axis. Retain
  // all coincident centers inside its reduction, without floating atomics.
  for (unsigned previous = 0; previous < center; ++previous)
    if (group->atoms[previous] == group->atoms[center]) return;
  double gradient{}, hvp{};
  for (std::size_t quartet = 0; quartet < group->quartets; ++quartet)
    for (unsigned slot = center; slot < 4; ++slot)
      if (group->atoms[slot] == group->atoms[center]) {
        gradient += projected[41 * quartet + 17 + 3 * slot + axis];
        hvp += projected[41 * quartet + 29 + 3 * slot + axis];
      }
  const auto base = 1 + 2 * shape.spins * shape.nao * shape.nao;
  candidate[base + 3 * group->atoms[center] + axis] += gradient;
  candidate[base + 3 * shape.atoms + 3 * group->atoms[center] + axis] += hvp;
}
__global__ void reduce_energy(const DeviceGroup* group, const double* projected, double* candidate,
                              int* status) {
  if (atomicAdd(status, 0)) return;
  double value{};
  for (std::size_t quartet = 0; quartet < group->quartets; ++quartet)
    value += projected[41 * quartet];
  candidate[0] += value;
}
}  // namespace

struct CudaWb97mvNuclearJk::Impl {
  CudaWb97mvJkShape shape;
  CudaWb97mvJkOptions options;
  CudaWb97mvJkReference reference;
  CudaWb97mvJkWork work;
  Schedule schedule;
  void* workspace{};
  bool ready{};
  cudaError_t authenticate(cudaStream_t stream) noexcept {
    if (!ready || stream != reference.stream) return cudaErrorInvalidResourceHandle;
    int device{-1};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != reference.device_id) return cudaErrorInvalidResourceHandle;
    cudaStreamCaptureStatus capture{};
    error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    error = reference.authenticate(reference.owner);
    if (error != cudaSuccess) ready = false;
    return error;
  }
};

cudaError_t cuda_wb97mv_jk_workspace(const CudaWb97mvJkShape& shape, std::span<const Shell> shells,
                                     const CudaWb97mvJkOptions& options,
                                     std::size_t& bytes) noexcept {
  bytes = 0;
  try {
    Schedule selected;
    const auto error = schedule(shape, shells, options, selected);
    if (error == cudaSuccess) bytes = selected.plan.bytes;
    return error;
  } catch (...) {
    return cudaErrorMemoryAllocation;
  }
}
CudaWb97mvNuclearJk::CudaWb97mvNuclearJk() : impl_(std::make_unique<Impl>()) {}
CudaWb97mvNuclearJk::~CudaWb97mvNuclearJk() = default;
const CudaWb97mvJkWork& CudaWb97mvNuclearJk::work() const noexcept { return impl_->work; }

cudaError_t CudaWb97mvNuclearJk::prepare(const CudaWb97mvJkShape& shape,
                                         std::span<const Shell> shells,
                                         const CudaWb97mvJkOptions& options,
                                         const CudaWb97mvJkReference& reference, void* workspace,
                                         std::size_t workspace_bytes) noexcept {
  impl_->ready = false;
  int device{-1};
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  if (device != reference.device_id) return cudaErrorInvalidResourceHandle;
  cudaStreamCaptureStatus capture{};
  if ((error = cudaStreamIsCapturing(reference.stream, &capture)) != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  try {
    Schedule selected;
    if ((error = schedule(shape, shells, options, selected)) != cudaSuccess) return error;
    const auto& plan = selected.plan;
    if (workspace_bytes < plan.bytes || !reference.stream || !reference.authenticate ||
        !reference.owner ||
        !disjoint(workspace, plan.bytes, reference.basis, plan.basis_elements * sizeof(double)))
      return cudaErrorInvalidValue;
    for (const auto* pointer : {reference.basis, static_cast<const double*>(workspace)})
      if ((error = device_buffer(pointer, reference.device_id)) != cudaSuccess) return error;
    auto& state = *impl_;
    state.shape = shape;
    state.options = options;
    state.reference = reference;
    state.schedule = std::move(selected);
    state.workspace = workspace;
    state.ready = true;
    if ((error = state.authenticate(reference.stream)) != cudaSuccess) {
      state.ready = false;
      return error;
    }
    const auto& layout = state.schedule.plan;
    const auto stream = reference.stream;
    auto* base = static_cast<char*>(workspace);
    auto* status = reinterpret_cast<int*>(base + layout.status);
    auto* device_groups = reinterpret_cast<DeviceGroup*>(base + layout.groups);
    auto* device_tiles = reinterpret_cast<DeviceTile*>(base + layout.tiles);
    auto* device_shells = reinterpret_cast<Shell*>(base + layout.shells);
    error = cudaMemsetAsync(status, 0, sizeof(int), stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(device_shells, shells.data(), shells.size_bytes(),
                              cudaMemcpyHostToDevice, stream);
    for (std::size_t index = 0; index < state.schedule.groups.size() && error == cudaSuccess;
         ++index)
      error = cudaMemcpyAsync(device_groups + index, &state.schedule.groups[index].device,
                              sizeof(DeviceGroup), cudaMemcpyHostToDevice, stream);
    for (std::size_t index = 0; index < state.schedule.tiles.size() && error == cudaSuccess;
         ++index)
      error = cudaMemcpyAsync(device_tiles + index, &state.schedule.tiles[index].device,
                              sizeof(DeviceTile), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) {
      finite_values<<<blocks(layout.basis_elements), 128, 0, stream>>>(
          reference.basis, layout.basis_elements, status);
      error = cudaGetLastError();
    }
    // Complete the finite gate before interpreting any packed floating metadata.
    error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    if (error == cudaSuccess) {
      validate_basis<<<blocks(std::max(shells.size(), shape.primitives)), 128, 0, stream>>>(
          shape, device_shells, shells.size(), reference.basis, status);
      error = cudaGetLastError();
      error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
    }
    if (error == cudaSuccess) error = state.authenticate(stream);
    state.ready = error == cudaSuccess;
    if (state.ready)
      state.work = {layout.bytes,      state.schedule.groups.size(), layout.radial_records,
                    layout.components, layout.projections,           layout.reductions};
    return error;
  } catch (...) {
    impl_->ready = false;
    return cudaErrorMemoryAllocation;
  }
}

cudaError_t CudaWb97mvNuclearJk::apply(const double* direction, const double* density,
                                       const double* delta_density,
                                       const CudaWb97mvJkOutput& output,
                                       cudaStream_t stream) noexcept {
  auto& state = *impl_;
  if (state.work.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
  ++state.work.attempted_actions;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto& plan = state.schedule.plan;
  const auto& shape = state.shape;
  const auto& reference = state.reference;
  const double* inputs[] = {reference.basis, direction, density, delta_density};
  const std::size_t input_counts[] = {plan.basis_elements, 3 * shape.atoms, plan.panels,
                                      plan.panels};
  double* outputs[] = {output.energy, output.fock, output.fock_direction, output.gradient,
                       output.hvp};
  const std::size_t output_counts[] = {1, plan.panels, plan.panels, 3 * shape.atoms,
                                       3 * shape.atoms};
  for (unsigned index = 0; index < 4; ++index) {
    if (!disjoint(state.workspace, plan.bytes, inputs[index], input_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  for (unsigned index = 0; index < 5; ++index) {
    if (!disjoint(state.workspace, plan.bytes, outputs[index],
                  output_counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    for (unsigned input = 0; input < 4; ++input)
      if (!disjoint(outputs[index], output_counts[index] * sizeof(double), inputs[input],
                    input_counts[input] * sizeof(double)))
        return cudaErrorInvalidValue;
    for (unsigned previous = 0; previous < index; ++previous)
      if (!disjoint(outputs[index], output_counts[index] * sizeof(double), outputs[previous],
                    output_counts[previous] * sizeof(double)))
        return cudaErrorInvalidValue;
    if ((error = device_buffer(outputs[index], reference.device_id)) != cudaSuccess) return error;
  }
  if (state.work.component_evaluations > UINT64_MAX - plan.components ||
      state.work.projection_terms_bound > UINT64_MAX - plan.projections ||
      state.work.reduction_terms > UINT64_MAX - plan.reductions)
    return cudaErrorInvalidValue;
  auto* base = static_cast<char*>(state.workspace);
  auto* status = reinterpret_cast<int*>(base + plan.status);
  auto* records = reinterpret_cast<double*>(base + plan.records);
  auto* raw = reinterpret_cast<double*>(base + plan.raw);
  auto* cache = reinterpret_cast<double*>(base + plan.cache);
  auto* projected = reinterpret_cast<double*>(base + plan.projected);
  auto* candidate = reinterpret_cast<double*>(base + plan.candidates);
  const auto* groups = reinterpret_cast<DeviceGroup*>(base + plan.groups);
  const auto* tiles = reinterpret_cast<DeviceTile*>(base + plan.tiles);
  error = cudaMemsetAsync(status, 0, sizeof(int), stream);
  if (error == cudaSuccess) {
    finite_values<<<blocks(3 * shape.atoms), 128, 0, stream>>>(direction, 3 * shape.atoms, status);
    error = cudaGetLastError();
  }
  for (const auto* panel : {density, delta_density})
    if (error == cudaSuccess) {
      symmetric<<<blocks(plan.panels), 128, 0, stream>>>(
          panel, shape.nao, shape.spins, state.options.symmetry_absolute_tolerance,
          state.options.symmetry_relative_tolerance, status);
      error = cudaGetLastError();
    }
  error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(candidate, 0, plan.candidate_count * sizeof(double), stream);
  for (std::size_t index = 0; index < state.schedule.groups.size() && error == cudaSuccess;
       ++index) {
    const auto& group = state.schedule.groups[index];
    error = cudaMemsetAsync(cache, 0, group.device.components * 25 * sizeof(double), stream);
    for (const auto tile_index : state.schedule.closure[group.closure]) {
      const auto& tile = state.schedule.tiles[tile_index];
      for (std::size_t begin = 0; begin < group.radials && error == cudaSuccess;) {
        const auto count = std::min(state.options.radial_tile, group.radials - begin);
        build_records<<<1, 128, 0, stream>>>(shape, groups + index, reference.basis, begin, count,
                                             records, status);
        error = cudaGetLastError();
        if (error == cudaSuccess)
          error = static_cast<cudaError_t>(tile.program->integral(
              records, count, groups[index].atoms, direction, raw, status, stream));
        if (error == cudaSuccess) {
          state.work.component_evaluations += count * tile.device.count;
          accumulate<<<1, 128, 0, stream>>>(groups + index, tiles + tile_index, raw, cache, status);
          error = cudaGetLastError();
          state.work.reduction_terms += 25 * tile.device.count;
        }
        begin += count;
      }
    }
    if (error == cudaSuccess) {
      project<<<blocks(group.device.quartets), 128, 0, stream>>>(
          shape, groups + index, reference.basis, direction, density, delta_density, cache,
          projected, status);
      error = cudaGetLastError();
      state.work.projection_terms_bound += 81 * group.device.quartets;
    }
    for (unsigned incidence = 0; incidence < 4 && error == cudaSuccess; ++incidence) {
      reduce_fock<<<blocks(shape.spins * 100), 128, 0, stream>>>(shape, groups + index, incidence,
                                                                 projected, candidate, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      reduce_vectors<<<1, 32, 0, stream>>>(shape, groups + index, projected, candidate, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      reduce_energy<<<1, 1, 0, stream>>>(groups + index, projected, candidate, status);
      error = cudaGetLastError();
      state.work.reduction_terms += group.device.quartets * (25 + 8 * shape.spins);
    }
  }
  for (unsigned panel = 0; panel < 2 && error == cudaSuccess; ++panel) {
    symmetric<<<blocks(plan.panels), 128, 0, stream>>>(
        candidate + 1 + panel * plan.panels, shape.nao, shape.spins,
        state.options.symmetry_absolute_tolerance, state.options.symmetry_relative_tolerance,
        status);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    finite_values<<<blocks(plan.candidate_count), 128, 0, stream>>>(candidate, plan.candidate_count,
                                                                    status);
    error = cudaGetLastError();
  }
  error = error == cudaSuccess ? gate(status, stream) : drain(error, stream);
  if (error == cudaSuccess) error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  std::size_t offset{};
  for (unsigned index = 0; index < 5 && error == cudaSuccess; ++index) {
    error =
        cudaMemcpyAsync(outputs[index], candidate + offset, output_counts[index] * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
    offset += output_counts[index];
  }
  error = drain(error, stream);
  if (error == cudaSuccess) ++state.work.completed_actions;
  return error;
}
}  // namespace generativeqc::dft
