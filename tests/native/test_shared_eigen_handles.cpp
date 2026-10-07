#include <cusolverDn.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "scf/cuda/eigensolver.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "solver/cuda/symmetric_eigen_handles.hpp"

namespace shared = generativeqc::solver::cuda;
using Handles = shared::PreparedSymmetricEigenHandles;
static_assert(sizeof(generativeqc::scf::cuda_execution::OrdinaryStreamEigensolver) <=
              generativeqc::scf::kOrdinaryEigensolverBindingHostBytes);

std::uint32_t private_handle_create(Handles& owner);
std::uint32_t private_handle_bind(Handles& owner, void* stream);
shared::SymmetricEigenResources private_handle_view(const Handles& owner);
void* private_gfn2_create(void* stream);
std::int32_t private_gfn2_ensure(void*, std::string&);
bool private_gfn2_published(void*);
bool private_gfn2_empty(void*);
void private_gfn2_destroy(void*);
void private_scf_lifetime(const std::string&, void*);

namespace {
std::vector<std::string> calls;
std::set<void*> live_solvers, live_parameters, live_jacobi;
std::set<void*> live_blas;
int current_device{11};
bool enforce_device{};
std::uintptr_t next_pointer = 0x1000;
int stage{}, fail_stage{};
cusolverStatus_t fail_status = CUSOLVER_STATUS_ALLOC_FAILED;
double tolerance{};
int sweeps{}, sorting{};
void* bound_stream{};
void* const stream = reinterpret_cast<void*>(0x4321);

cusolverStatus_t record(const char* name) {
  calls.emplace_back(name);
  return ++stage == fail_stage ? fail_status : CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t acquire(const char* name, T** output, std::set<void*>& live) {
  assert(output);
  const auto status = record(name);
  *output = nullptr;
  if (status == CUSOLVER_STATUS_SUCCESS) {
    *output = reinterpret_cast<T*>(next_pointer++);
    assert(live.insert(*output).second);
  }
  return status;
}

cusolverStatus_t release(const char* name, void* pointer, std::set<void*>& live) {
  assert(!enforce_device || current_device == 3);
  calls.emplace_back(name);
  assert(pointer && live.erase(pointer) == 1);
  return CUSOLVER_STATUS_SUCCESS;
}

void expect_calls(const std::vector<std::string>& expected) {
  if (calls == expected) return;
  std::cerr << "actual:";
  for (const auto& call : calls) std::cerr << ' ' << call;
  std::cerr << "\nexpected:";
  for (const auto& call : expected) std::cerr << ' ' << call;
  std::cerr << '\n';
  std::abort();
}

void empty_view(const Handles& owner) {
  const auto view = owner.view();
  assert(!view.solver && !view.parameters && !view.jacobi);
  assert(!view.device_workspace && !view.device_workspace_bytes);
  assert(!view.host_workspace && !view.host_workspace_bytes);
}

void same_view(const shared::SymmetricEigenResources& a, const shared::SymmetricEigenResources& b) {
  assert(a.solver == b.solver && a.parameters == b.parameters && a.jacobi == b.jacobi);
  assert(!a.device_workspace && !a.host_workspace);
  assert(!a.device_workspace_bytes && !a.host_workspace_bytes);
}

std::uint32_t prepare(Handles& owner, const std::string& mode) {
  auto status = owner.create();
  if (status) return status;
  if (mode == "gfn2") {
    status = owner.create_parameters();
    if (status) return status;
    status = owner.configure_jacobi(std::numeric_limits<double>::epsilon(), 100, 1);
    if (status) return status;
    return owner.bind_stream(stream);
  }
  status = owner.bind_stream(stream);
  if (status) return status;
  if (mode == "scf-jacobi") return owner.configure_jacobi(1.0e-13, 100, 1);
  assert(mode == "scf-generic");
  return owner.create_parameters();
}

std::vector<std::string> preparation_calls(const std::string& mode) {
  if (mode == "gfn2")
    return {"create", "parameters", "jacobi", "tolerance", "sweeps", "sort", "stream"};
  if (mode == "scf-jacobi") return {"create", "stream", "jacobi", "tolerance", "sweeps", "sort"};
  assert(mode == "scf-generic");
  return {"create", "stream", "parameters"};
}

void append_destruction(std::vector<std::string>& expected,
                        const shared::SymmetricEigenResources& view) {
  if (view.jacobi) expected.emplace_back("destroy-jacobi");
  if (view.parameters) expected.emplace_back("destroy-parameters");
  if (view.solver) expected.emplace_back("destroy");
}

void success(const std::string& mode) {
  auto expected = preparation_calls(mode);
  {
    Handles owner;
    empty_view(owner);
    assert(prepare(owner, mode) == 0);
    const auto view = owner.view();
    assert(view.solver && bound_stream == stream);
    assert(bool(view.parameters) == (mode != "scf-jacobi"));
    assert(bool(view.jacobi) == (mode != "scf-generic"));
    same_view(view, owner.view());
    expect_calls(expected);
    if (view.jacobi) {
      assert(tolerance == (mode == "gfn2" ? std::numeric_limits<double>::epsilon() : 1.0e-13));
      assert(sweeps == 100 && sorting == 1);
    }
    append_destruction(expected, view);
  }
  expect_calls(expected);
}

void failure(int failed, int raw_status) {
  fail_stage = failed;
  fail_status = static_cast<cusolverStatus_t>(raw_status);
  Handles owner;
  assert(prepare(owner, "gfn2") == static_cast<std::uint32_t>(raw_status));
  auto expected = preparation_calls("gfn2");
  expected.resize(failed);
  expect_calls(expected);  // No later setup call executes after failure.
  const auto failed_view = owner.view();
  assert(bool(failed_view.solver) == (failed > 1));
  assert(bool(failed_view.parameters) == (failed > 2));
  assert(bool(failed_view.jacobi) == (failed > 3));
  append_destruction(expected, failed_view);
  owner.reset();
  empty_view(owner);
  owner.reset();
  expect_calls(expected);
  assert(live_solvers.empty() && live_parameters.empty() && live_jacobi.empty());

  // The same owner can be retried after a partial setup was retired.
  calls.clear();
  fail_stage = stage = 0;
  assert(prepare(owner, "gfn2") == 0);
  expected = preparation_calls("gfn2");
  append_destruction(expected, owner.view());
  owner.reset();
  expect_calls(expected);
}

void moves() {
  static_assert(!std::is_copy_constructible_v<Handles>);
  static_assert(!std::is_copy_assignable_v<Handles>);
  static_assert(std::is_nothrow_move_constructible_v<Handles>);
  static_assert(std::is_nothrow_move_assignable_v<Handles>);
  static_assert(std::is_nothrow_destructible_v<Handles>);
  Handles source;
  assert(prepare(source, "gfn2") == 0);
  const auto borrowed = source.view();
  calls.clear();
  Handles moved(std::move(source));
  empty_view(source);
  same_view(borrowed, moved.view());
  source.reset();
  assert(calls.empty());
  auto* const self = &moved;
  moved = std::move(*self);
  assert(calls.empty());
  same_view(borrowed, moved.view());

  Handles destination;
  assert(prepare(destination, "scf-generic") == 0);
  const auto replaced = destination.view();
  calls.clear();
  destination = std::move(moved);
  std::vector<std::string> expected;
  append_destruction(expected, replaced);
  expect_calls(expected);
  empty_view(moved);
  same_view(borrowed, destination.view());
  moved.reset();
  source.reset();
  expect_calls(expected);
  append_destruction(expected, borrowed);
  destination.reset();
  destination.reset();
  expect_calls(expected);
}

void duplicates() {
  Handles owner;
  constexpr auto invalid = static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  assert(owner.bind_stream(stream) == invalid);
  assert(owner.create_parameters() == invalid);
  assert(owner.configure_jacobi(1.0e-13, 100, 1) == invalid);
  assert(calls.empty());
  assert(prepare(owner, "gfn2") == 0);
  const auto before = owner.view();
  calls.clear();
  assert(owner.create() == invalid);
  assert(owner.create_parameters() == invalid);
  assert(owner.configure_jacobi(1.0e-13, 3, 0) == invalid);
  assert(calls.empty());
  same_view(before, owner.view());
  // Duplicate configuration cannot silently change scientific tolerances.
  assert(tolerance == std::numeric_limits<double>::epsilon() && sweeps == 100 && sorting == 1);
}

void private_abi() {
  Handles owner;
  fail_stage = 1;
  fail_status = CUSOLVER_STATUS_NOT_SUPPORTED;
  assert(private_handle_create(owner) == 9);
  empty_view(owner);
  fail_stage = 0;
  assert(private_handle_create(owner) == 0);
  assert(owner.create_parameters() == 0);
  assert(owner.configure_jacobi(std::numeric_limits<double>::epsilon(), 100, 1) == 0);
  assert(private_handle_bind(owner, stream) == 0);
  same_view(owner.view(), private_handle_view(owner));
  assert(bound_stream == stream);
  fail_stage = stage + 1;
  fail_status = CUSOLVER_STATUS_EXECUTION_FAILED;
  assert(private_handle_bind(owner, stream) == 6);
  same_view(owner.view(), private_handle_view(owner));
}

void gfn2_lifetime(int failed) {
  enforce_device = true;
  auto* owner = private_gfn2_create(stream);
  fail_stage = failed;
  std::string error;
  const auto status = private_gfn2_ensure(owner, error);
  const std::vector<std::string> initialization{
      "set-device-3", "gfn2-parameters", "create", "parameters", "jacobi", "tolerance", "sweeps",
      "sort",         "blas-create",     "stream", "blas-stream"};
  if (failed) {
    assert(status == 2 && private_gfn2_empty(owner));
    std::vector<std::string> expected(initialization.begin(),
                                      initialization.begin() + (failed == 8 ? 11 : 2 + failed));
    if (failed > 7) expected.emplace_back("destroy-blas");
    if (failed > 3) expected.emplace_back("destroy-jacobi");
    if (failed > 2) expected.emplace_back("destroy-parameters");
    if (failed > 1) expected.emplace_back("destroy");
    expect_calls(expected);
    assert(error == (failed == 1   ? "cusolverDnCreate failed"
                     : failed == 2 ? "cusolverDnCreateParams failed"
                     : failed <= 6 ? "failed to configure the CUDA small-matrix Jacobi eigensolver"
                     : failed == 7
                         ? "cublasCreate failed"
                         : "failed to bind CUDA linear-algebra handles to the context stream"));
    assert(live_solvers.empty() && live_parameters.empty() && live_jacobi.empty() &&
           live_blas.empty());
    calls.clear();
    fail_stage = stage = 0;
    assert(private_gfn2_ensure(owner, error) == 0);
  } else {
    assert(status == 0);
  }
  expect_calls(initialization);
  assert(private_gfn2_published(owner));
  assert(tolerance == std::numeric_limits<double>::epsilon() && sweeps == 100 && sorting == 1);
  calls.clear();
  assert(private_gfn2_ensure(owner, error) == 0);
  expect_calls({"set-device-3", "gfn2-parameters"});  // Prepared context reuses every handle.

  calls.clear();
  current_device = 11;
  private_gfn2_destroy(owner);
  expect_calls({"get-device", "set-device-3", "sync", "prepared", "destroy-blas", "destroy-jacobi",
                "destroy-parameters", "destroy", "set-device-11"});
  assert(current_device == 11);  // Member destruction after restoration is a no-op.
}

void scf_lifetime(const std::string& mode) {
  enforce_device = true;
  private_scf_lifetime(mode, stream);
  if (mode == "ordinary") {
    expect_calls({"get-device", "set-device-3", "sync", "free-512", "destroy-parameters", "destroy",
                  "set-device-11"});
    assert(current_device == 11);
  } else {
    expect_calls({"set-device-3", mode == "rhf-jacobi" ? "destroy-jacobi" : "destroy-parameters",
                  "destroy", "destroy-blas", "free-async-256", "free-async-1280", "free-async-512",
                  "free-async-768", "free-async-1024", "sync", "destroy-stream"});
    assert(current_device == 3);  // RHF deliberately preserves its existing selection contract.
  }
}
}  // namespace

cudaError_t cudaGetDevice(int* device) {
  calls.emplace_back("get-device");
  *device = current_device;
  return cudaSuccess;
}
cudaError_t cudaSetDevice(int device) {
  calls.emplace_back("set-device-" + std::to_string(device));
  current_device = device;
  return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t input) {
  assert(current_device == 3 && input == stream);
  calls.emplace_back("sync");
  return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t input) {
  assert(current_device == 3 && input == stream);
  calls.emplace_back("destroy-stream");
  return cudaSuccess;
}
std::uint32_t trace_blas_create(void** handle) { return acquire("blas-create", handle, live_blas); }
std::uint32_t trace_blas_bind(void* handle, void* input) {
  assert(live_blas.count(handle) == 1 && input == stream);
  return record("blas-stream");
}
std::uint32_t trace_blas_destroy(void* handle) {
  return release("destroy-blas", handle, live_blas);
}
void trace_external(const char* name) { calls.emplace_back(name); }
void begin_teardown_trace() {
  calls.clear();
  current_device = 11;
}
cudaError_t trace_release(void* pointer, bool async) {
  assert(current_device == 3);
  calls.emplace_back(std::string(async ? "free-async-" : "free-") +
                     std::to_string(reinterpret_cast<std::uintptr_t>(pointer)));
  return cudaSuccess;
}

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
  return acquire("create", handle, live_solvers);
}
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* parameters) {
  return acquire("parameters", parameters, live_parameters);
}
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* jacobi) {
  return acquire("jacobi", jacobi, live_jacobi);
}
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t input) {
  assert(live_solvers.count(handle) == 1);
  bound_stream = input;
  return record("stream");
}
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t jacobi, double value) {
  assert(live_jacobi.count(jacobi) == 1);
  tolerance = value;
  return record("tolerance");
}
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t jacobi, int value) {
  assert(live_jacobi.count(jacobi) == 1);
  sweeps = value;
  return record("sweeps");
}
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t jacobi, int value) {
  assert(live_jacobi.count(jacobi) == 1);
  sorting = value;
  return record("sort");
}
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t jacobi) {
  return release("destroy-jacobi", jacobi, live_jacobi);
}
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t parameters) {
  return release("destroy-parameters", parameters, live_parameters);
}
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
  return release("destroy", handle, live_solvers);
}

int main(int argc, char** argv) {
  assert(argc >= 2);
  const std::string operation = argv[1];
  if (operation == "success") {
    assert(argc == 3);
    success(argv[2]);
  } else if (operation == "failure") {
    assert(argc == 4);
    failure(std::atoi(argv[2]), std::atoi(argv[3]));
  } else if (operation == "move") {
    moves();
  } else if (operation == "duplicate") {
    duplicates();
  } else if (operation == "private-abi") {
    private_abi();
  } else if (operation == "gfn2-lifetime") {
    assert(argc == 3);
    gfn2_lifetime(std::atoi(argv[2]));
  } else if (operation == "scf-lifetime") {
    assert(argc == 3);
    scf_lifetime(argv[2]);
  } else {
    return 2;
  }
  assert(live_solvers.empty() && live_parameters.empty() && live_jacobi.empty() &&
         live_blas.empty());
}
