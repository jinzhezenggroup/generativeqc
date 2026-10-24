#pragma once

#include <cstddef>
#include <functional>

#include "generativeqc/fock.h"

namespace generativeqc::core {
struct System;
}
namespace generativeqc::scf {
struct FockBuildSpec;
}

namespace generativeqc::api {

/** Borrowed method-owned density response. The input is a resident symmetric
 * AO direction; the returned resident potential must be ready for the supplied
 * stream. The method owner validates its scientific generation and owns all XC
 * storage. No matrix may be staged through the host by this callback. */
using CudaDensityResponseAction = std::function<const double*(const double*, void*)>;

/** Validate the borrowed provider even for empty/zero or refused requests. */
generativeqc_status validate_cuda_cpks_source(const generativeqc_fock_plan* plan,
                                              const core::System& system,
                                              const scf::FockBuildSpec& spec, int device);

/** Reuse the RHF resident orbital/vector engine with authenticated full-range
 * Fock coefficients and an additional method-owned density action. This does
 * not extend the public RHF reconstruction contract to Kohn-Sham response. */
generativeqc_status create_cuda_cpks_response(
    generativeqc_fock_plan* plan, const core::System& system, const scf::FockBuildSpec& spec,
    int device, const double* coefficients, std::size_t coefficient_count, const double* energies,
    std::size_t energy_count, unsigned occupied, unsigned vector_slots,
    std::size_t device_budget_bytes, const CudaDensityResponseAction& density_response,
    generativeqc_rhf_response_resident** output);

/** Retained provider bytes charged alongside XC and resident orbital storage. */
std::size_t cuda_cpks_response_provider_bytes(const generativeqc_fock_plan* plan);

}  // namespace generativeqc::api
