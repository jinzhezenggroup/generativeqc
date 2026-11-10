#pragma once
#include <string>

#include "generativeqc/generativeqc.h"
#include "integrals/ecp.hpp"

namespace generativeqc::integrals {
// Raw CUDA grids retain their independently qualified domain. Checked production
// calls select fixed grids internally and do not consume raw grid arguments.
inline generativeqc_status validate_ecp_cuda_grid(unsigned radial, unsigned polar,
                                                  std::string& detail) {
  if (radial < 16 || radial > 512 || polar < 8 || polar > 96) {
    detail = "CUDA ECP quadrature requires 16..512 radial and 8..96 polar points";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}
// Host export for independent validation and existing host-visible consumers.
generativeqc_status ecp_integrals_cuda(int device, const core::System& system, unsigned radial,
                                       unsigned polar, bool derivatives, EcpData& output,
                                       std::string& detail, bool convergence = false);
// Production device consumer: add values to hcore and/or -D:dV to forces.
// Pointers and stream are borrowed; no host integral or density staging.
generativeqc_status add_ecp_cuda(int device, const core::System& system, void* stream,
                                 double* hcore, const double* density, double* forces,
                                 std::string& detail, double derivative_scale = 1.0);
}  // namespace generativeqc::integrals
