/** Exercise the actual shared owner and xTB status adapter with a host double.
 * Fault injection concerns thread-local device ownership, not CUDA arithmetic.
 */
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "runtime/cuda_descriptor_validation.hpp"
#include "runtime/cuda_device_scope.hpp"
#include "runtime/nvidia_host_api.h"

namespace {
int current_device = 0;
int query_count = 0;
int selection_count = 0;
int consumed_errors = 0;
cudaError_t query_error = cudaSuccess;
cudaError_t selection_error = cudaSuccess;
cudaError_t last_error = cudaSuccess;

struct BoundaryError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void boundary_check(cudaError_t status) {
  if (status != cudaSuccess) throw BoundaryError("boundary error");
}

void shared_scope(const std::string& scenario) {
  using generativeqc::runtime::CudaDeviceScope;
  static_assert(!std::is_copy_constructible_v<CudaDeviceScope>);
  static_assert(!std::is_move_constructible_v<CudaDeviceScope>);
  if (scenario == "nested") {
    {
      CudaDeviceScope outer(1);
      assert(current_device == 1);
      {
        CudaDeviceScope inner(2);
        assert(current_device == 2);
        assert(inner.previous_device() == 1);
        assert(inner.selected_device() == 2);
      }
      assert(current_device == 1);
    }
    assert(current_device == 0);
  } else if (scenario == "mixed") {
    {
      CudaDeviceScope outer(1);
      std::string error;
      {
        generativeqc::xtb::detail::ScopedCudaDevice inner(2, error);
        assert(inner.ok() && current_device == 2);
        {
          CudaDeviceScope resource(3);
          assert(current_device == 3);
        }
        assert(current_device == 2);
      }
      assert(current_device == 1);
    }
    assert(current_device == 0);
  } else if (scenario == "same-device") {
    {
      CudaDeviceScope resource(0);
      assert(resource.status() == cudaSuccess);
      assert(selection_count == 1);
    }
    const int before = selection_count;
    CudaDeviceScope boundary(0, std::nothrow);
    assert(boundary.status() == cudaSuccess);
    assert(boundary.restore() == cudaSuccess);
    assert(selection_count == before);
  } else if (scenario == "unwind") {
    try {
      CudaDeviceScope device(1);
      throw BoundaryError("consumer failure");
    } catch (const BoundaryError&) {
      assert(current_device == 0);
    }
  } else if (scenario == "query-failure" || scenario == "selection-failure") {
    if (scenario == "query-failure")
      query_error = cudaErrorInitializationError;
    else
      selection_error = cudaErrorInitializationError;
    bool caught = false;
    try {
      CudaDeviceScope device(1, boundary_check);
    } catch (const BoundaryError&) {
      caught = true;
    }
    assert(caught && current_device == 0);
    assert(selection_count == (scenario == "query-failure" ? 0 : 1));
  } else if (scenario == "allocation-failure") {
    selection_error = cudaErrorMemoryAllocation;
    bool caught = false;
    try {
      CudaDeviceScope device(1);
    } catch (const std::bad_alloc&) {
      caught = true;
    }
    assert(caught && current_device == 0);
  } else if (scenario == "restore-retry") {
    CudaDeviceScope device(1, std::nothrow);
    selection_error = cudaErrorInitializationError;
    assert(device.restore() == cudaErrorInitializationError);
    assert(current_device == 1);
    assert(device.restore() == cudaSuccess);
    assert(current_device == 0);
    const int before = selection_count;
    assert(device.restore() == cudaSuccess);
    assert(selection_count == before);
  } else if (scenario == "destructor-retry") {
    {
      CudaDeviceScope device(1, std::nothrow);
      selection_error = cudaErrorInitializationError;
      assert(device.restore() == cudaErrorInitializationError);
      assert(current_device == 1);
    }
    assert(current_device == 0);
  } else if (scenario == "cleanup-failure") {
    {
      CudaDeviceScope device(1);
      selection_error = cudaErrorInitializationError;
    }
    assert(current_device == 1);
    assert(consumed_errors == 1 && last_error == cudaSuccess);
  } else {
    throw std::logic_error("unknown shared scenario");
  }
}

void xtb_scope(const std::string& scenario) {
  using generativeqc::xtb::detail::ScopedCudaDevice;
  std::string error = "stale diagnostic";
  if (scenario == "negative") {
    ScopedCudaDevice device(-1, error);
    assert(!device.ok() && device.status() == GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT);
    assert(query_count == 0 && selection_count == 0);
    assert(error == "CUDA device id must be nonnegative");
  } else if (scenario == "same-device") {
    ScopedCudaDevice device(0, error);
    assert(device.ok() && error.empty());
    error = "earlier failure";
    assert(device.restore(error) == GENERATIVEQC_XTB_STATUS_SUCCESS);
    assert(error == "earlier failure" && selection_count == 0);
  } else if (scenario == "query-failure" || scenario == "query-invalid-device" ||
             scenario == "selection-failure" || scenario == "invalid-device") {
    if (scenario == "query-failure")
      query_error = cudaErrorInitializationError;
    else if (scenario == "query-invalid-device")
      query_error = cudaErrorInvalidDevice;
    else if (scenario == "invalid-device")
      selection_error = cudaErrorInvalidDevice;
    else
      selection_error = cudaErrorInitializationError;
    ScopedCudaDevice device(1, error);
    const auto expected = scenario == "invalid-device"
                              ? GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT
                              : GENERATIVEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    assert(!device.ok() && device.status() == expected);
    const char* operation = scenario.starts_with("query-") ? "cudaGetDevice" : "cudaSetDevice";
    assert(error.starts_with(std::string(operation) + " failed:"));
    assert(current_device == 0 && consumed_errors == 1 && last_error == cudaSuccess);
  } else if (scenario == "restore-retry" || scenario == "destructor-retry") {
    {
      ScopedCudaDevice device(1, error);
      assert(device.ok() && error.empty());
      assert(device.previous_device() == 0 && device.selected_device() == 1);
      selection_error = cudaErrorInitializationError;
      assert(device.restore(error) == GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR);
      assert(error.starts_with("cudaSetDevice while restoring the caller device failed:"));
      assert(current_device == 1 && consumed_errors == 1 && last_error == cudaSuccess);
      if (scenario == "restore-retry") {
        error = "earlier failure";
        assert(device.restore(error) == GENERATIVEQC_XTB_STATUS_SUCCESS);
        assert(error == "earlier failure" && current_device == 0);
        const int before = selection_count;
        assert(device.restore(error) == GENERATIVEQC_XTB_STATUS_SUCCESS);
        assert(selection_count == before);
      }
    }
    assert(current_device == 0);
  } else if (scenario == "cleanup-failure") {
    {
      ScopedCudaDevice device(1, error);
      assert(device.ok());
      selection_error = cudaErrorInitializationError;
    }
    assert(current_device == 1 && consumed_errors == 1 && last_error == cudaSuccess);
  } else {
    throw std::logic_error("unknown xTB scenario");
  }
}
}  // namespace

extern "C" cudaError_t cudaGetDevice(int* device) {
  ++query_count;
  if (query_error != cudaSuccess) return last_error = query_error;
  *device = current_device;
  return cudaSuccess;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  ++selection_count;
  const cudaError_t result = selection_error;
  selection_error = cudaSuccess;
  if (result != cudaSuccess) return last_error = result;
  current_device = device;
  return cudaSuccess;
}

extern "C" cudaError_t cudaGetLastError() {
  ++consumed_errors;
  const cudaError_t result = last_error;
  last_error = cudaSuccess;
  return result;
}

extern "C" const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }

/* Metadata operations are outside this control-only fixture's contract.
 * Abort rather than let an accidental call masquerade as GPU validation.
 */
extern "C" cudaError_t cudaPointerGetAttributes(cudaPointerAttributes*, const void*) {
  std::abort();
}
extern "C" cudaError_t cudaHostGetFlags(unsigned int*, void*) { std::abort(); }
extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*) {
  std::abort();
}
extern "C" cudaError_t cudaStreamGetDevice(cudaStream_t, int*) { std::abort(); }
extern "C" CUresult cuGetErrorString(CUresult, const char**) { std::abort(); }
extern "C" CUresult cuMemGetAddressRange_v2(CUdeviceptr*, size_t*, CUdeviceptr) { std::abort(); }

int main(int argc, char** argv) {
  assert(argc == 3);
  if (std::string(argv[1]) == "shared")
    shared_scope(argv[2]);
  else
    xtb_scope(argv[2]);
}
