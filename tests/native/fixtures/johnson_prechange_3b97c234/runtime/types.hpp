// Test-only status ABI excerpt from 3b97c234eb18f5e6f354842b458a4e56c1619f95.
// All numerical owner code remains byte-for-byte frozen.
#pragma once
#include <stdint.h>

typedef int32_t generativeqc_xtb_status_t;
enum generativeqc_xtb_status_value {
  GENERATIVEQC_XTB_STATUS_SUCCESS = 0,
  GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT = 1,
  GENERATIVEQC_XTB_STATUS_BACKEND_UNAVAILABLE = 2,
  GENERATIVEQC_XTB_STATUS_NOT_SUPPORTED = 3,
  GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED = 4,
  GENERATIVEQC_XTB_STATUS_NOT_IMPLEMENTED = 5,
  GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR = 6,
  /* Per-system SCC reached max_scc_iterations without satisfying both tolerances. */
  GENERATIVEQC_XTB_STATUS_SCC_NOT_CONVERGED = 7,
  /* Per-system generalized eigensolver failed or produced an unusable eigensystem. */
  GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED = 8
};
