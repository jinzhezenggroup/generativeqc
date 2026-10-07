"""Host-check retained DF source teardown using the production owners/deleter."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _eigen_handle_test_support import empty_eigen_owner_units

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, marker: str) -> str:
    begin = source.index(marker)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


STUBS = r"""
#include "solver/cuda/symmetric_eigen_handles.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown };
int current_device = 1, owner_device = 0, allocations = 0, queries = 0, selections = 0;
int synchronizations = 0, source_destroys = 0, stream_destroys = 0;
int binding_destroys = 0, expected_binding_destroys = 0, binding_layout = 0;
bool stream_alive = false, blas_alive = false, query_fails = false;
struct PreparedBinding {
  ~PreparedBinding() {
    assert(current_device == owner_device && stream_alive && blas_alive);
    ++binding_destroys;
  }
};
cudaError_t cudaGetDevice(int* value) {
  ++queries;
  if (query_fails) return cudaErrorUnknown;
  *value = current_device;
  return cudaSuccess;
}
cudaError_t cudaSetDevice(int value) {
  ++selections;
  current_device = value;
  return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "mock CUDA failure"; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  assert(stream && stream_alive && current_device == owner_device);
  ++synchronizations;
  return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  assert(stream && stream_alive && current_device == owner_device && !allocations);
  assert(source_destroys == 1);
  assert(binding_destroys == expected_binding_destroys && !blas_alive);
  stream_alive = false;
  ++stream_destroys;
  return cudaSuccess;
}
void cublasDestroy(void*) {
  assert(current_device == owner_device && stream_alive && blas_alive);
  assert(binding_destroys == expected_binding_destroys);
  blas_alive = false;
}
namespace generativeqc::runtime {
template <class T> struct TensorView { T* data; std::size_t size; };
std::size_t size_mul(std::size_t a, std::size_t b, const char*) { return a * b; }
template <class T> cudaError_t resource_cuda_malloc(T** data, std::size_t bytes) {
  assert(current_device == owner_device && stream_alive && !allocations);
  *data = static_cast<T*>(std::malloc(bytes));
  assert(*data);
  ++allocations;
  return cudaSuccess;
}
cudaError_t resource_cuda_free(void* data) {
  if (data) {
    assert(current_device == owner_device && stream_alive && synchronizations == 1);
    assert(allocations == 1 && source_destroys == 0);
    --allocations;
    std::free(data);
  }
  return cudaSuccess;
}
"""

DESTROY_STUBS = r"""
void destroy_persistent_scf_state(void*) {}
void destroy_ordinary_eigensystem(void*) {}
void destroy_final_validation(void*) {}
void destroy_cuda_density_fitting_integral_source(void* source) {
  if (!source) return;
  assert(current_device == owner_device && stream_alive && !allocations);
  ++source_destroys;
}
"""

DRIVER = r"""
using namespace generativeqc;
std::shared_ptr<cc::DFSourceState> create_state(bool coefficients = true) {
  runtime::CudaDeviceScope scope(owner_device);
  auto state = std::make_shared<cc::DFSourceState>();
  state->plan.reset(new scf::CudaDensityFittingJkPlan{});
  state->plan->device_id = owner_device;
  state->plan->stream = reinterpret_cast<void*>(1);
  state->plan->integral_source = reinterpret_cast<void*>(2);
  state->plan->blas = reinterpret_cast<void*>(3);
  stream_alive = blas_alive = true;
  if (binding_layout == 0) {
    state->plan->charge_contraction = std::make_unique<PreparedBinding>();
    state->plan->coulomb_contraction = std::make_unique<PreparedBinding>();
    expected_binding_destroys = 2;
  } else {
    state->plan->metric_project = std::make_unique<PreparedBinding>();
    expected_binding_destroys = 1;
    // Partial preparation must safely destroy the first binding after failure.
    if (binding_layout != 3) {
      state->plan->metric_rotate = std::make_unique<PreparedBinding>();
      const std::size_t panels = binding_layout == 1 ? 1 : 2;
      for (std::size_t i = 0; i != panels; ++i) {
        state->plan->metric_charge[i] = std::make_unique<PreparedBinding>();
        state->plan->metric_potential[i] = std::make_unique<PreparedBinding>();
      }
      expected_binding_destroys = 2 + 2 * panels;
    }
  }
  if (coefficients) state->coefficients.allocate(owner_device, 4, state->plan->stream);
  return state;
}
// Match the public response signature: the parameter can be the final shared
// reference, and its destruction occurs after the function-local device scope.
void consume_last(std::shared_ptr<cc::DFSourceState> source, int failure) {
  std::lock_guard lock(source->response_mutex);
  if (failure == 2) throw std::invalid_argument("validation before device scope");
  runtime::CudaDeviceScope scope(source->plan->device_id);
  if (failure == 1) throw std::runtime_error("callback after device scope");
}
int main(int argc, char** argv) {
  assert(argc == 3);
  const int mode = std::atoi(argv[1]);
  binding_layout = std::atoi(argv[2]);
  assert(binding_layout >= 0 && binding_layout <= 3);
  static_assert(noexcept(cc::PlanDelete{}(nullptr)));
  static_assert(std::is_nothrow_destructible_v<cc::DFSourceState>);
  for (int caller_device : {0, 1}) {
    owner_device = 1 - caller_device;
    current_device = caller_device;
    allocations = queries = selections = synchronizations = source_destroys = stream_destroys = 0;
    binding_destroys = expected_binding_destroys = 0;
    stream_alive = blas_alive = query_fails = false;
    if (mode == 6) {
      cc::PlanDelete{}(nullptr);
      assert(!queries && !selections && current_device == caller_device);
      continue;
    }
    if (mode == 5) {
      // Before retention, PlanDelete ran inside the forward device scope.
      runtime::CudaDeviceScope scope(owner_device);
      auto state = create_state();
    } else {
      auto state = create_state(mode != 7 && mode != 8);
      assert(current_device == caller_device);
      if (mode == 1) {
        auto last = state;
        state.reset();
        assert(stream_alive && allocations == 1 && current_device == caller_device);
        last.reset();
      } else if (mode >= 2 && mode <= 4) {
        try {
          consume_last(std::move(state), mode - 2);
          assert(mode == 2);
        } catch (const std::runtime_error&) { assert(mode == 3); }
          catch (const std::invalid_argument&) { assert(mode == 4); }
        assert(!state);
      } else if (mode == 8) {
        // A failed query must not restore an invented device zero. Exercise
        // PlanDelete alone: an empty coefficient owner makes no CUDA queries.
        query_fails = true;
        const int before = selections;
        state.reset();
        query_fails = false;
        assert(selections == before + 1 && current_device == owner_device);
      } else {
        state.reset();
      }
    }
    assert(!allocations && !stream_alive && source_destroys == 1 && stream_destroys == 1);
    assert(binding_destroys == expected_binding_destroys && !blas_alive);
    assert(synchronizations == (mode == 7 || mode == 8 ? 0 : 1));
    if (mode != 8 && current_device != caller_device) {
      std::cerr << "retained DF source teardown changed caller device " << caller_device
                << " to owner device " << current_device << '\n';
      return 1;
    }
  }
}
"""


@pytest.fixture(scope="module")
def state_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires a host C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/cc/df_source_cuda.cu").read_text()
    resources = (ROOT / "src/runtime/cuda_resources.cuh").read_text()
    lifetime = (ROOT / "src/scf/cuda/df_plan_lifetime.cpp").read_text()
    plan = (ROOT / "src/scf/cuda/df_plan.cpp").read_text()
    release = _definition(lifetime, "void release(")
    # Keep the actual release body as well as the CC owner/deleter. Only unused
    # plan data and CUDA APIs are mocked; do not model their device transitions.
    fields = sorted(set(re.findall(r"plan\.(\w+)", release)) - {"device_id"})
    owned_fields = {
        "charge_contraction",
        "coulomb_contraction",
        "metric_project",
        "metric_rotate",
    }
    array_fields = {"metric_charge", "metric_potential"}
    assert (owned_fields | array_fields).issubset(fields)
    unit = STUBS + _definition(resources, "inline void cuda_resource_check(")
    for marker in (
        "class CudaDeviceScope",
        "template <class T>\nstruct BorrowedCudaBuffer",
        "template <class T>\nclass OwnedCudaBuffer",
    ):
        unit += "\n" + _definition(resources, marker) + ";\n"
    unit += "}\nnamespace generativeqc::scf {\nstruct CudaDensityFittingJkPlan {\n"
    unit += "int device_id{-1};\n" + "".join(
        f"std::unique_ptr<PreparedBinding> {field};\n"
        if field in owned_fields
        else f"std::array<std::unique_ptr<PreparedBinding>, 2> {field};\n"
        if field in array_fields
        else "::generativeqc::solver::cuda::PreparedSymmetricEigenHandles eigen_handles;\n"
        if field == "eigen_handles"
        else f"void* {field}{{}};\n"
        for field in fields
    )
    unit += "};\n" + DESTROY_STUBS + release + "\n"
    unit += _definition(plan, "void destroy_cuda_density_fitting_jk_plan(")
    unit += "\n}\nnamespace generativeqc::cc {\n"
    unit += _definition(source, "struct PlanDelete") + ";\n"
    unit += _definition(source, "class DFSourceState") + ";\n}\n" + DRIVER
    directory = tmp_path_factory.mktemp("df-source-state-lifetime")
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t = void*;\n"
    )
    eigen_units = empty_eigen_owner_units(directory)
    cpp, executable = directory / "state.cpp", directory / "state"
    cpp.write_text(unit)
    objects = []
    for source_file in (cpp, *eigen_units):
        obj = directory / (source_file.stem + ".o")
        result = subprocess.run(
            [
                cache,
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I" + str(directory),
                "-I" + str(ROOT / "src"),
                "-c",
                str(source_file),
                "-o",
                str(obj),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(obj))
    result = subprocess.run(
        [compiler, *objects, "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "mode",
    range(9),
    ids=(
        "last-reference",
        "shared-reference",
        "by-value-return",
        "by-value-callback-failure",
        "by-value-validation-failure",
        "forward-scoped-baseline",
        "null-plan",
        "partial-publication",
        "failed-device-query",
    ),
)
@pytest.mark.parametrize(
    "bindings", range(4), ids=("resident", "streamed-full", "streamed-tail", "partial")
)
def test_retained_source_restores_caller_device(
    state_probe: Path, mode: int, bindings: int
) -> None:
    result = subprocess.run(
        [str(state_probe), str(mode), str(bindings)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
