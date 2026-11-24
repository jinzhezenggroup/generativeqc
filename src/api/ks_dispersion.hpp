#pragma once

#include "generativeqc/generativeqc.h"

/** Private detached copies of a complete D4+gCP transaction and its gCP term.
 * Each optional gradient buffer has exactly 3*atom_count entries. No query
 * executes missing derivatives; energy-only receipts reject gradient requests. */
struct generativeqc_ks_composite_correction_result_v1 {
  uint32_t struct_size, abi_version;
  double energy, gcp_energy;
  double* gradient;
  uint64_t gradient_count;
  double* gcp_gradient;
  uint64_t gcp_gradient_count;
};

/** Private name-to-MethodIR metadata for the Python constructor. Borrowed
 * strings live as long as the selected native library. */
struct generativeqc_ks_catalog_binding_v1 {
  uint32_t struct_size, abi_version;
  const char* identifier;
  const char* identity;
  uint32_t spin;
};

extern "C" {
/** Query the same selector table used by SDK preparation and the CLI, without
 * opening a context. A successful miss clears all fields after the ABI tags. */
GENERATIVEQC_API generativeqc_status
generativeqc_ks_catalog_lookup_v1(const char* name, size_t name_size, generativeqc_backend backend,
                                  generativeqc_ks_catalog_binding_v1* output);
/** Backend-neutral method registry transport for compiled public named programs.
 * A successful miss writes zero; invalid inputs leave output
 * unchanged. A carrier identifies a registry row, not contextual admission. */
GENERATIVEQC_API generativeqc_status generativeqc_ks_catalog_registry_carrier_v1(const char* name,
                                                                                 size_t name_size,
                                                                                 int32_t* carrier);
/** Private exact-identity discovery. NULL name means the library has no complete
 * matching program, not permission to substitute a similar functional. */
GENERATIVEQC_API generativeqc_status generativeqc_ks_catalog_name_v1(const char* identity,
                                                                     generativeqc_backend backend,
                                                                     const char** name);
/** Return the shared CPU DF energy inventory selector for an exact MethodIR
 * digest. A successful miss publishes NULL and never chooses Direct J/K. */
GENERATIVEQC_API generativeqc_status generativeqc_ks_cpu_df_energy_program_v1(const char* identity,
                                                                              const char** name);
/** Read completed correction outputs from the last successful item transaction.
 * Optional buffers copy only existing work; a missing gradient is NOT_IMPLEMENTED.
 * D3 uses two_body_energy for its complete correction, including ATM if enabled. */
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_dispersion_result_v1(
    const generativeqc_batch*, uint32_t, generativeqc_d4_batch_item_result_descriptor*);
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_composite_correction_result_v1(
    const generativeqc_batch*, uint32_t, generativeqc_ks_composite_correction_result_v1*);
GENERATIVEQC_API const char* generativeqc_ks_batch_dispersion_variant_v1(const generativeqc_batch*,
                                                                         uint32_t);
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_d3_diagnostic_v1(
    const generativeqc_batch*, uint32_t, generativeqc_d3_runtime_diagnostic*);
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_d4_diagnostic_v1(
    const generativeqc_batch*, uint32_t, generativeqc_d4_runtime_diagnostic*);
}
