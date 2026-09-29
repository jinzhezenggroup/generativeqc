#include "scf/cuda_weighted_eri.hpp"

namespace generativeqc::scf {

generativeqc_status contract_cuda_weighted_eri_primitives(
    int, const CudaWeightedEriPrimitive*, std::size_t, std::size_t, std::size_t, bool,
    std::vector<CudaWeightedEriResult>& output, CudaWeightedEriDiagnostic& diagnostic,
    std::string& detail) {
  output.clear();
  diagnostic = {};
  detail = "CUDA external-weight ERI contraction is unavailable in this build";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace generativeqc::scf
