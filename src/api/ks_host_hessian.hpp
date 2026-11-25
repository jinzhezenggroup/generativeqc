#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "generativeqc/ks_hessian.h"

namespace generativeqc::core {
struct System;
}

namespace generativeqc::api {
/** Authenticate the prepared system with the same gate as derivative execution. */
void validate_host_hessian_system(const core::System& system, std::string_view identity);

inline generativeqc_ks_host_hessian_options_v1 default_host_hessian_options() {
  generativeqc_ks_host_hessian_options_v1 options{};
  generativeqc_ks_hessian_cpu_default_options_v1(&options);
  return options;
}

inline generativeqc_ks_host_hessian_receipt_v1 empty_host_hessian_receipt() {
  generativeqc_ks_host_hessian_receipt_v1 receipt{};
  receipt.struct_size = sizeof(receipt);
  receipt.abi_version = GENERATIVEQC_ABI_VERSION;
  return receipt;
}
}  // namespace generativeqc::api

/** Canonical orbitals for occupied-subspace tracking, not a second electronic
 * response provider. Only C crosses the wire; no AO, Fock or grid is exported.
 * The reservation includes the detached verified frame, staged C and caller C,
 * but excludes the producer's verification/eigensolver scratch and Python copies.
 */
struct generativeqc_ks_host_orbitals_receipt_v1 {
  std::size_t struct_size{sizeof(generativeqc_ks_host_orbitals_receipt_v1)};
  std::uint32_t abi_version{GENERATIVEQC_ABI_VERSION};
  std::uint64_t basis{}, spins{}, alpha_occupied{}, beta_occupied{};
  std::uint64_t output_elements{}, numeric_reservation_bytes{};
};

extern "C" {
/** Same expected-geometry/charge/spin and owner gates as the Hessian bridge.
 * Only options' descriptor, expected geometry and numeric_bytes are consumed.
 * Null output/count=0 queries the bounded shape without exporting a frame.
 * Otherwise C is spin-major then row-major [AO,orbital]. Refusal publishes neither
 * C nor the receipt. This performs no SCF, JIT, grid reconstruction or response.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_orbitals_cpu_v1(
    generativeqc_batch* batch, std::size_t index,
    const generativeqc_ks_host_hessian_options_v1* options, double* output,
    std::size_t output_count, generativeqc_ks_host_orbitals_receipt_v1* receipt);
}
