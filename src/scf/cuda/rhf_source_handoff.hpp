#pragma once

#include "core/types.hpp"
#include "scf/rhf_source_handoff.hpp"

namespace generativeqc::scf {
struct CudaRhfBucketPlan;

/** Called only by the single-system RHF adapter after physical publication.
 * orbital is the exact system used for that fresh bucket execution. Moving
 * its snapshot avoids constructing another normalized basis owner. */
CudaRhfSourceHandoff detach_rhf_cuda_source(const CudaRhfBucketPlan& plan, core::System&& orbital,
                                            std::size_t reference_peak_bytes,
                                            std::size_t numeric_budget);
}  // namespace generativeqc::scf
