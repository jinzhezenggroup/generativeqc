#define GENERATIVEQC_TEST_HOOKS

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "generated_basis_transform_cuda.cuh"
#include "molecule/basis.hpp"
#include "tensor/cuda_contraction_selection.cuh"

namespace lower = generativeqc::scf::basis_transform_lowering;
namespace tensor = generativeqc::tensor;

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceBuffer {
  double* pointer{};
  explicit DeviceBuffer(std::size_t count) {
    check(cudaMallocManaged(&pointer, count * sizeof(double)));
  }
  ~DeviceBuffer() {
    if (pointer) (void)cudaFree(pointer);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void fold_hcore(double* fock, const double* hcore, std::size_t count) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) fock[i] += hcore[i];
}

double compare(const double* actual, const std::vector<long double>& reference,
               std::string_view label) {
  double max_error = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const auto expected = static_cast<double>(reference[i]);
    const auto error = std::abs(actual[i] - expected);
    if (!std::isfinite(actual[i]) || error > 2e-11 * (1 + std::abs(expected)))
      throw std::runtime_error(std::string(label) + " differs from independent oracle");
    max_error = std::max(max_error, error);
  }
  return max_error;
}

void one_case(std::size_t public_nbf, std::size_t direct_nbf, bool generated,
              bool real_basis = false) {
  lower::require_packed_stage(public_nbf, direct_nbf, 1, 1, nullptr, nullptr);
  const auto pn = public_nbf, dn = direct_nbf;
  DeviceBuffer c(pn * dn), d(pn * pn), f(dn * dn), h(pn * pn);
  DeviceBuffer dr(dn * pn), dd(dn * dn), fl(dn * pn), fr(pn * pn);
  for (std::size_t i = 0; i < pn * dn; ++i) c.pointer[i] = 0;
  if (real_basis) {
    const auto angular = pn == 5 ? 2U : 3U;
    const auto cartesian = generativeqc::molecule::cartesian_components(angular);
    const auto expansions =
        generativeqc::molecule::ao_expansions(angular, GENERATIVEQC_BASIS_SPHERICAL);
    if (expansions.size() != pn || cartesian.size() != dn)
      throw std::runtime_error("real d/f basis dimensions differ from SCF topology");
    for (std::size_t public_ao = 0; public_ao < pn; ++public_ao)
      for (const auto& term : expansions[public_ao]) {
        const auto found = std::find(cartesian.begin(), cartesian.end(), term.component);
        if (found == cartesian.end()) throw std::runtime_error("missing Cartesian component");
        const auto direct_ao = std::size_t(found - cartesian.begin());
        c.pointer[direct_ao * pn + public_ao] = term.coefficient;
      }
  } else {
    for (std::size_t i = 0; i < pn * dn; ++i)
      c.pointer[i] = std::sin(0.13 * double(i + 1)) + 0.03 * double(i % 7);
  }
  for (std::size_t i = 0; i < pn * pn; ++i) {
    d.pointer[i] = std::cos(0.17 * double(i + 2)) + 0.01 * double(i % 5);
    h.pointer[i] = std::sin(0.11 * double(i + 5));
  }
  for (std::size_t i = 0; i < dn * dn; ++i)
    f.pointer[i] = std::cos(0.07 * double(i + 3)) - 0.02 * double(i % 9);

  std::vector<long double> density_right(dn * pn), density_direct(dn * dn);
  std::vector<long double> fock_left(dn * pn), fock_right(pn * pn);
  std::vector<long double> fock_public(pn * pn);
  for (std::size_t a = 0; a < dn; ++a)
    for (std::size_t q = 0; q < pn; ++q)
      for (std::size_t p = 0; p < pn; ++p)
        density_right[a * pn + q] +=
            static_cast<long double>(c.pointer[a * pn + p]) * d.pointer[p * pn + q];
  for (std::size_t a = 0; a < dn; ++a)
    for (std::size_t b = 0; b < dn; ++b)
      for (std::size_t p = 0; p < pn; ++p)
        density_direct[a * dn + b] += density_right[a * pn + p] * c.pointer[b * pn + p];
  for (std::size_t a = 0; a < dn; ++a)
    for (std::size_t p = 0; p < pn; ++p)
      for (std::size_t b = 0; b < dn; ++b)
        fock_left[a * pn + p] +=
            static_cast<long double>(f.pointer[a * dn + b]) * c.pointer[b * pn + p];
  for (std::size_t p = 0; p < pn; ++p)
    for (std::size_t q = 0; q < pn; ++q) {
      for (std::size_t a = 0; a < dn; ++a)
        fock_right[p * pn + q] +=
            static_cast<long double>(c.pointer[a * pn + p]) * fock_left[a * pn + q];
      fock_public[p * pn + q] = fock_right[p * pn + q] + h.pointer[p * pn + q];
    }

  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int* error{};
  check(cudaMallocManaged(&error, sizeof(int)));
  check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  tensor::contraction_libraries_unavailable_for_test = generated;
  constexpr std::size_t budget =
      (96ULL << 20) + tensor::PreparedBoundedContraction::host_reservation;
  {
    tensor::PreparedBoundedContraction density_right_plan(
        lower::basis_density_right_request, lower::basis_density_right_candidates,
        lower::basis_density_right_target, lower::basis_density_right_compilation,
        lower::density_right(pn, dn), stream, budget);
    tensor::PreparedBoundedContraction density_direct_plan(
        lower::basis_density_direct_request, lower::basis_density_direct_candidates,
        lower::basis_density_direct_target, lower::basis_density_direct_compilation,
        lower::density_direct(pn, dn), stream, budget);
    tensor::PreparedBoundedContraction fock_left_plan(
        lower::basis_fock_left_request, lower::basis_fock_left_candidates,
        lower::basis_fock_left_target, lower::basis_fock_left_compilation, lower::fock_left(pn, dn),
        stream, budget);
    tensor::PreparedBoundedContraction fock_right_plan(
        lower::basis_fock_right_request, lower::basis_fock_right_candidates,
        lower::basis_fock_right_target, lower::basis_fock_right_compilation,
        lower::fock_right(pn, dn), stream, budget);
    const std::array plans{&density_right_plan, &density_direct_plan, &fock_left_plan,
                           &fock_right_plan};
    for (const auto* plan : plans) {
      const auto provider = plan->candidate().provider;
      if (provider != (generated ? "generated.cuda" : "cublas"))
        throw std::runtime_error("unexpected shared provider selection");
    }
    auto invalid = lower::density_right(pn, dn);
    invalid.operands[0].strides[1]++;
    try {
      invalid.validate();
      throw std::runtime_error("invalid stride accepted");
    } catch (const std::invalid_argument&) {
    }
    try {
      density_right_plan.execute(lower::fock_left(pn, dn), stream, c.pointer, d.pointer, dr.pointer,
                                 error);
      throw std::runtime_error("cross-stage identity accepted");
    } catch (const std::invalid_argument&) {
    }
    try {
      density_right_plan.execute(lower::density_right(pn, dn), stream, c.pointer, d.pointer,
                                 c.pointer, error);
      throw std::runtime_error("output/input alias accepted");
    } catch (const std::invalid_argument&) {
    }
    density_right_plan.execute(lower::density_right(pn, dn), stream, c.pointer, d.pointer,
                               dr.pointer, error);
    density_direct_plan.execute(lower::density_direct(pn, dn), stream, dr.pointer, c.pointer,
                                dd.pointer, error);
    fock_left_plan.execute(lower::fock_left(pn, dn), stream, f.pointer, c.pointer, fl.pointer,
                           error);
    fock_right_plan.execute(lower::fock_right(pn, dn), stream, c.pointer, fl.pointer, fr.pointer,
                            error);
    check(cudaStreamSynchronize(stream));
    if (*error) throw std::runtime_error("shared contraction reported numerical failure");
    const auto density_right_error = compare(dr.pointer, density_right, "density right");
    const auto density_direct_error = compare(dd.pointer, density_direct, "density direct");
    const auto fock_left_error = compare(fl.pointer, fock_left, "fock left");
    const auto fock_right_error = compare(fr.pointer, fock_right, "fock right");
    fold_hcore<<<unsigned((pn * pn + 127) / 128), 128, 0, stream>>>(fr.pointer, h.pointer, pn * pn);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    const auto fock_public_error = compare(fr.pointer, fock_public, "fock public with hcore");
    std::cout << "public=" << pn << " direct=" << dn
              << " basis=" << (real_basis ? (pn == 5 ? "d" : "f") : "asymmetric")
              << " provider=" << density_right_plan.candidate().provider
              << " provider_version=" << density_right_plan.provider_version()
              << " max_abs_errors[d_right,d_direct,f_left,f_right,f_public]=" << density_right_error
              << ',' << density_direct_error << ',' << fock_left_error << ',' << fock_right_error
              << ',' << fock_public_error << " logical_summands="
              << density_right_plan.summands() + density_direct_plan.summands() +
                     fock_left_plan.summands() + fock_right_plan.summands()
              << '\n';
  }
  check(cudaFree(error));
  check(cudaStreamDestroy(stream));
}

int main() try {
  int marker = 0;
  const std::array<std::array<std::size_t, 4>, 2> invalid_groups{
      std::array<std::size_t, 4>{5, 7, 2, 1}, std::array<std::size_t, 4>{5, 7, 1, 2}};
  for (const auto invalid : invalid_groups) {
    try {
      lower::require_packed_stage(invalid[0], invalid[1], invalid[2], invalid[3], nullptr, nullptr);
    } catch (const std::invalid_argument&) {
      ++marker;
    }
  }
  try {
    lower::require_packed_stage(5, 7, 1, 1, &marker, nullptr);
  } catch (const std::invalid_argument&) {
    ++marker;
  }
  try {
    lower::require_packed_stage(5, 7, 1, 1, nullptr, &marker);
  } catch (const std::invalid_argument&) {
    ++marker;
  }
  const std::array<std::array<std::int64_t, 2>, 5> invalid_dimensions{
      std::array<std::int64_t, 2>{0, 7}, {-1, 7}, {5, 0}, {5, -1}, {INT64_MAX, 7}};
  for (const auto invalid : invalid_dimensions) {
    try {
      lower::require_packed_stage(invalid[0], invalid[1], 1, 1, nullptr, nullptr);
    } catch (const std::invalid_argument&) {
      ++marker;
    }
  }
  if (marker != 9) throw std::runtime_error("unsupported stage domain was admitted");
  try {
    lower::require_packed_stage(INT32_MAX, INT32_MAX, 1, 1, nullptr, nullptr);
    throw std::runtime_error("overflow stage dimensions were admitted");
  } catch (const std::length_error&) {
  }
  const std::array<std::array<std::size_t, 2>, 3> asymmetric_dimensions{
      std::array<std::size_t, 2>{5, 7}, {7, 5}, {12, 17}};
  for (const bool generated : {false, true})
    for (const auto dims : asymmetric_dimensions) one_case(dims[0], dims[1], generated);
  for (const bool generated : {false, true}) {
    one_case(5, 6, generated, true);
    one_case(7, 10, generated, true);
  }
  std::cout << "basis transform stage qualification PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
