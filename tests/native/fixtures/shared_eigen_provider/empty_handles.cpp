// These host probes never prepare a solver. Link the real owner and fail if an
// empty owner's construction/destruction unexpectedly invokes a vendor API.
#include <cusolverDn.h>

#include <cstdlib>

namespace {
[[noreturn]] void unexpected_vendor_call() { std::abort(); }
}  // namespace

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t*) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t, cudaStream_t) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t*) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t*) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t, double) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t, int) { unexpected_vendor_call(); }
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t, int) { unexpected_vendor_call(); }
