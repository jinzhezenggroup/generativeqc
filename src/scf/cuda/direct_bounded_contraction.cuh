#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_force_quartet.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

// Retained direct bounded contraction contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace generativeqc::scf::cuda_execution {

/** Runtime angular dispatch used only by the generic large-topology path. */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_fock_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock, std::size_t subtile,
    unsigned lane) {
#define GENERATIVEQC_BOUNDED_FOCK_CASE(order)                                                 \
  case order:                                                                                 \
    contract_fock_direct_quartet_subtile<Unrestricted, order>(                                \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, fock, \
        nullptr, subtile, lane);                                                              \
    break
  switch (angular_order) {
    GENERATIVEQC_BOUNDED_FOCK_CASE(0);
    GENERATIVEQC_BOUNDED_FOCK_CASE(1);
    GENERATIVEQC_BOUNDED_FOCK_CASE(2);
    GENERATIVEQC_BOUNDED_FOCK_CASE(3);
    GENERATIVEQC_BOUNDED_FOCK_CASE(4);
    GENERATIVEQC_BOUNDED_FOCK_CASE(5);
    GENERATIVEQC_BOUNDED_FOCK_CASE(6);
    GENERATIVEQC_BOUNDED_FOCK_CASE(7);
    GENERATIVEQC_BOUNDED_FOCK_CASE(8);
    GENERATIVEQC_BOUNDED_FOCK_CASE(9);
    GENERATIVEQC_BOUNDED_FOCK_CASE(10);
    GENERATIVEQC_BOUNDED_FOCK_CASE(11);
    GENERATIVEQC_BOUNDED_FOCK_CASE(12);
    default:
      break;
  }
#undef GENERATIVEQC_BOUNDED_FOCK_CASE
}

template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_force_subtile_scaled(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces, double coulomb_coefficient,
    double exchange_coefficient, std::size_t subtile, unsigned lane) {
#define GENERATIVEQC_BOUNDED_FORCE_CASE(order)                                                  \
  case order:                                                                                   \
    contract_two_electron_force_quartet_subtile_scaled<Unrestricted, order>(                    \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, forces, \
        0U, coulomb_coefficient, exchange_coefficient, subtile, lane);                          \
    break
  // Total order 0/1 is consumed by the generated ssss/psss exact-shell
  // tasks before generic bounded dispatch. Do not reinstantiate retired math.
  switch (angular_order) {
    GENERATIVEQC_BOUNDED_FORCE_CASE(2);
    GENERATIVEQC_BOUNDED_FORCE_CASE(4);
    GENERATIVEQC_BOUNDED_FORCE_CASE(5);
    GENERATIVEQC_BOUNDED_FORCE_CASE(6);
    GENERATIVEQC_BOUNDED_FORCE_CASE(7);
    GENERATIVEQC_BOUNDED_FORCE_CASE(8);
    GENERATIVEQC_BOUNDED_FORCE_CASE(9);
    GENERATIVEQC_BOUNDED_FORCE_CASE(10);
    GENERATIVEQC_BOUNDED_FORCE_CASE(11);
    GENERATIVEQC_BOUNDED_FORCE_CASE(12);
    default:
      break;
  }
#undef GENERATIVEQC_BOUNDED_FORCE_CASE
}

template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_force_subtile_range_scaled(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces, double exchange_coefficient,
    generativeqc::integrals::CoulombRange range, double omega, std::size_t subtile, unsigned lane) {
#define GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(order)                                            \
  case order:                                                                                   \
    contract_two_electron_force_quartet_subtile_range_scaled<Unrestricted, order>(              \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, forces, \
        exchange_coefficient, range, omega, subtile, lane);                                     \
    break
  switch (angular_order) {
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(0);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(1);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(2);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(3);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(4);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(5);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(6);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(7);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(8);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(9);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(10);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(11);
    GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE(12);
    default:
      break;
  }
#undef GENERATIVEQC_BOUNDED_RANGE_FORCE_CASE
}

template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_rsh_force_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* source_forces,
    double coulomb_coefficient, double short_exchange_coefficient, double long_exchange_coefficient,
    double omega, std::size_t subtile, unsigned lane) {
#define GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(order)                                                 \
  case order:                                                                                      \
    contract_two_electron_force_quartet_subtile_rsh_scaled<Unrestricted, order>(                   \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active,            \
        source_forces, coulomb_coefficient, short_exchange_coefficient, long_exchange_coefficient, \
        omega, subtile, lane);                                                                     \
    break
  switch (angular_order) {
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(0);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(1);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(2);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(3);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(4);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(5);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(6);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(7);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(8);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(9);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(10);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(11);
    GENERATIVEQC_BOUNDED_RSH_FORCE_CASE(12);
    default:
      break;
  }
#undef GENERATIVEQC_BOUNDED_RSH_FORCE_CASE
}

template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_force_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces, std::size_t subtile,
    unsigned lane) {
  constexpr double exchange_coefficient = Unrestricted ? -1.0 : -0.5;
  contract_bounded_direct_force_subtile_scaled<Unrestricted>(
      batch, angular_order, queue_count, task, screening_tolerance, schwarz_bounds, density, active,
      forces, 1.0, exchange_coefficient, subtile, lane);
}

}  // namespace generativeqc::scf::cuda_execution
