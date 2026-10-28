// SPDX-License-Identifier: LGPL-3.0-or-later
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "dft/dispersion/gcp_cuda.hpp"
#include "dft/dispersion/gcp_r2scan3c.hpp"
#include "generativeqc/generativeqc.hpp"
#include "runtime/cuda_resources.cuh"

namespace generativeqc::dft::dispersion {
namespace {
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw Error(GENERATIVEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}

// Each lane owns an atom and visits the other atoms in a fixed order. Both
// directed pair legs are evaluated, avoiding floating-point atomics or O(N^2)
// scratch. The half-energy reduction counts each unordered pair exactly once.
__global__ void pairs(int n, const std::int32_t* z, const double* xyz,
                      const gcp_data::Element* elements, const double* radii,
                      GCPParameters parameters, bool derivative, double* energies,
                      double* gradients, int* statuses) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double energy = 0, gradient[3]{};
  auto status = GCPStatus::success;
  for (int j = 0; j < n; ++j) {
    if (i == j) continue;
    // Canonical pair orientation preserves identical rounding for the two
    // legs, including mixed Slater shell classes.
    const int high_atom = i > j ? i : j, low_atom = i > j ? j : i;
    double v[3], r2 = 0;
    for (int axis = 0; axis < 3; ++axis) {
      v[axis] = xyz[3 * high_atom + axis] - xyz[3 * low_atom + axis];
      r2 += v[axis] * v[axis];
    }
    if (!(r2 > 0) || !gcp_detail::finite(r2)) {
      status = GCPStatus::coincident_atoms;
      break;
    }
    const int za = z[high_atom], zb = z[low_atom];
    const int hi = za > zb ? za : zb, lo = za > zb ? zb : za;
    const double r = sqrt(r2);
    double e = 0, dedr = 0;
    status = gcp_detail::pair(elements[za - 1], elements[zb - 1],
                              radii[hi * (hi - 1) / 2 + lo - 1] * gcp_detail::kAngstromToBohr, r,
                              parameters, &e, derivative ? &dedr : nullptr);
    if (status != GCPStatus::success) break;
    energy += 0.5 * e;
    if (derivative)
      for (int axis = 0; axis < 3; ++axis)
        gradient[axis] += (i == high_atom ? 1.0 : -1.0) * dedr * v[axis] / r;
  }
  if (!gcp_detail::finite(energy)) status = GCPStatus::numerical_failure;
  energies[i] = energy;
  for (int axis = 0; axis < 3; ++axis) {
    if (!gcp_detail::finite(gradient[axis])) status = GCPStatus::numerical_failure;
    if (derivative) gradients[3 * i + axis] = gradient[axis];
  }
  statuses[i] = static_cast<int>(status);
}

__global__ void reduce(int n, double* energies, int* statuses) {
  double energy = 0;
  int status = 0;
  for (int i = 0; i < n; ++i) {
    energy += energies[i];
    if (statuses[i]) status = statuses[i];
  }
  if (!gcp_detail::finite(energy)) status = static_cast<int>(GCPStatus::numerical_failure);
  energies[0] = energy;
  statuses[0] = status;
}
}  // namespace

struct GcpCudaPlan::Impl {
  int device, atoms;
  // Buffers die before their stream, including exceptional setup/execution.
  runtime::OwnedCudaStream stream;
  runtime::OwnedCudaBuffer<std::int32_t> numbers;
  runtime::OwnedCudaBuffer<int> statuses;
  runtime::OwnedCudaBuffer<double> coordinates, energies, gradients, radii;
  runtime::OwnedCudaBuffer<gcp_data::Element> elements;
  Impl(int selected, int count) : device(selected), atoms(count), stream(selected) {}
};

std::size_t GcpCudaPlan::numeric_bytes(std::size_t atoms) {
  // Coordinate indices are signed ints; all molecular storage is dynamic.
  if (!atoms || atoms > std::numeric_limits<int>::max() / 3 ||
      atoms > (std::numeric_limits<std::size_t>::max() - sizeof(gcp_data::kElements) -
               sizeof(gcp_data::kVdwAngstrom)) /
                  (2 * sizeof(int) + 7 * sizeof(double)))
    throw std::invalid_argument("CUDA gCP shape exceeds its index or byte capacity");
  return atoms * (2 * sizeof(int) + 7 * sizeof(double)) + sizeof(gcp_data::kElements) +
         sizeof(gcp_data::kVdwAngstrom);
}

GcpCudaPlan::GcpCudaPlan(int device, std::span<const std::int32_t> numbers,
                         std::size_t maximum_bytes) {
  if (numeric_bytes(numbers.size()) > maximum_bytes) throw std::bad_alloc();
  if (std::any_of(numbers.begin(), numbers.end(), [](int z) { return z < 1 || z > 18; }))
    throw std::invalid_argument("CUDA r2SCAN-3c gCP supports H-Ar");
  auto owner = std::make_unique<Impl>(device, static_cast<int>(numbers.size()));
  runtime::CudaDeviceScope guard(device, check);
  const auto stream = owner->stream.get();
  owner->numbers.allocate(device, numbers.size(), stream);
  owner->statuses.allocate(device, numbers.size(), stream);
  owner->coordinates.allocate(device, 3 * numbers.size(), stream);
  owner->energies.allocate(device, numbers.size(), stream);
  owner->gradients.allocate(device, 3 * numbers.size(), stream);
  owner->radii.allocate(device, gcp_data::kVdwAngstrom.size(), stream);
  owner->elements.allocate(device, gcp_data::kElements.size(), stream);
  check(cudaMemcpyAsync(owner->numbers.get(), numbers.data(), numbers.size_bytes(),
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(owner->radii.get(), gcp_data::kVdwAngstrom.data(),
                        sizeof(gcp_data::kVdwAngstrom), cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(owner->elements.get(), gcp_data::kElements.data(),
                        sizeof(gcp_data::kElements), cudaMemcpyHostToDevice, stream));
  check(cudaStreamSynchronize(stream));
  impl_ = std::move(owner);
}
GcpCudaPlan::~GcpCudaPlan() = default;

void GcpCudaPlan::execute(std::span<const double> xyz, double& energy, std::span<double> gradient) {
  auto& owner = *impl_;
  if (xyz.size() != 3u * owner.atoms || (!gradient.empty() && gradient.size() != xyz.size()) ||
      !std::all_of(xyz.begin(), xyz.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("invalid CUDA gCP geometry/output shape");
  runtime::CudaDeviceScope guard(owner.device, check);
  const auto stream = owner.stream.get();
  check(cudaMemcpyAsync(owner.coordinates.get(), xyz.data(), xyz.size_bytes(),
                        cudaMemcpyHostToDevice, stream));
  pairs<<<(owner.atoms + 127) / 128, 128, 0, stream>>>(
      owner.atoms, owner.numbers.get(), owner.coordinates.get(), owner.elements.get(),
      owner.radii.get(), r2scan3c_gcp_parameters(), !gradient.empty(), owner.energies.get(),
      owner.gradients.get(), owner.statuses.get());
  check(cudaGetLastError());
  reduce<<<1, 1, 0, stream>>>(owner.atoms, owner.energies.get(), owner.statuses.get());
  check(cudaGetLastError());
  int status = 0;
  double candidate = 0;
  // Status is checked before publishing any numerical output. Every invocation
  // rewrites all device inputs/results, so a failed replay cannot leak a receipt.
  check(cudaMemcpyAsync(&status, owner.statuses.get(), sizeof(status), cudaMemcpyDeviceToHost,
                        stream));
  check(cudaMemcpyAsync(&candidate, owner.energies.get(), sizeof(candidate), cudaMemcpyDeviceToHost,
                        stream));
  check(cudaStreamSynchronize(stream));
  if (status) throw Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "CUDA r2SCAN-3c gCP pair failed");
  if (!gradient.empty()) {
    check(cudaMemcpyAsync(gradient.data(), owner.gradients.get(), gradient.size_bytes(),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
  }
  energy = candidate;
}
}  // namespace generativeqc::dft::dispersion
