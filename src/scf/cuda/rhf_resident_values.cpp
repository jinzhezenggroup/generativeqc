#include "scf/cuda/rhf_resident_values.hpp"

#include <cstdlib>

#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda/reference_eri_policy.hpp"
#include "scf/cuda/runtime_support.hpp"

namespace generativeqc::scf::cuda_execution {

RhfResidentValues::~RhfResidentValues() {
  if (stream_) (void)cudaStreamSynchronize(stream_);
  graphs.reset();
}

generativeqc_status RhfResidentValues::prepare(int device, const std::vector<core::System>& systems,
                                               unsigned maximum_iterations, std::size_t direct_nbf,
                                               std::size_t required, std::size_t budget,
                                               bool cold_reference, bool generic_fock) {
  const auto policy = reference_resident_bucket_policy();
  return source_.prepare(device, systems, maximum_iterations, direct_nbf, required, budget,
                         std::getenv("GENERATIVEQC_RHF_RESIDENT_VALUES"),
                         "cuda_rhf_resident_values_prepare", cold_reference, generic_fock, 0.0,
                         policy.ceiling, true, DirectJkSourceAutoPolicy::generic_f,
                         policy.valid ? policy.mask : 1U << 28);
}

generativeqc_status RhfResidentValues::enqueue(const double* density, const double* hcore,
                                               double* fock) {
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.derivative_order = 0;
  std::string detail;
  const auto status =
      source_.enqueue(spec, density, source_.coulomb(), source_.exchange(), nullptr, 0.0, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  const auto elements = source_.matrix_elements();
  cuda_df::launch_assemble_rhf_fock_kernel((elements + 255) / 256, 256, 0, stream_, elements, hcore,
                                           source_.coulomb(), source_.exchange(), fock);
  return cuda_status(cudaPeekAtLastError());
}

}  // namespace generativeqc::scf::cuda_execution
