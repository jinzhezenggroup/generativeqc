"""Execute the production fixed-D control flow with host CUDA/provider doubles.

These checks exercise leases, failures, buffer routing and timing boundaries.
They do not qualify GPU numerics or CUDA event performance.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.fixture(scope="module")
def profile_program(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    methods = "\n".join(
        _definition(source, signature).replace("CudaKsPlan::", "")
        for signature in (
            "void CudaKsPlan::invalidate_final_state()",
            "generativeqc_status CudaKsPlan::profile_fixed_density_components(",
        )
    )
    guard = _definition(
        (ROOT / "src/runtime/resource_cuda.cuh").read_text(),
        "template <class Cleanup>\nclass ResourceScopeExit",
    )
    harness = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.hpp"
using cudaStream_t = int;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0, cudaMemcpyDeviceToHost = 1;
static int mode{}, drains{}, submissions{}, records{}, synchronizations{}, reads{};
static bool timing{};
static std::vector<std::function<void()>> pending;
static std::vector<char> stages;
static std::function<void()> change_token;
void flush() { auto work = std::move(pending); pending.clear(); for (auto& f : work) f(); }
int cudaSetDevice(int device) { assert(device == 2); return 0; }
int cudaStreamSynchronize(int stream) { assert(stream == 7); ++drains; flush(); return 0; }
int cudaStreamIsCapturing(int, int* capture) { *capture = mode == 6; return 0; }
int cudaMemcpy(void* destination, const void* source, std::size_t bytes, int) {
  assert(!timing && pending.empty()); ++reads; std::memcpy(destination, source, bytes); return 0;
}
void check(int status, const std::string& detail = {}) {
  if (status) throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, detail);
}
namespace generativeqc::runtime {
GUARD;
class OwnedCudaEvent {
 public:
  explicit OwnedCudaEvent(int device) {
    assert(device == 2);
    if (mode == 20) throw std::bad_alloc();
  }
  void record(int stream) {
    assert(stream == 7);
    ++records;
    if (mode == 7 && records == 1) throw std::runtime_error("begin record failed");
    if (mode == 13 && records == 2) throw std::runtime_error("end record failed");
    timing = records % 2;
  }
  void synchronize() {
    ++synchronizations;
    if (mode == 14) throw std::runtime_error("event synchronize failed");
    flush();
  }
  float elapsed_since(const OwnedCudaEvent&) {
    assert(pending.empty());
    if (mode == 15) throw std::runtime_error("elapsed failed");
    return 2.0f;
  }
};
}
namespace runtime = generativeqc::runtime;
namespace scf {
struct Term { bool present{true}; double coefficient{1.0}; };
struct Spec { Term coulomb, exchange{true, -0.125}; };
struct Strategy { Spec spec; };
struct Direct {};
struct Fitted {};
struct Provider {
  Strategy model;
  Direct direct;
  Fitted fitted;
  const Strategy& strategy() const { return model; }
  Direct* cuda_direct_source() { return mode == 4 ? nullptr : &direct; }
  Fitted* cuda_fitted_source() { return mode == 4 ? &fitted : nullptr; }
};
struct JkTermSelection { bool coulomb, exchange; };
enum class FockMatrixLayout { RowMajor };
int enqueue(char stage, const double* density, const double* beta, std::size_t matrix,
            double* j, double* k, double* kb, int* error) {
  assert(timing && density && matrix == 4);
  assert(!beta || beta == density + matrix);
  assert(bool(kb) == (beta && k));
  assert(!kb || kb == k + matrix);
  assert(j != density && k != density && kb != density);
  assert((stage == 'J') == bool(j));
  assert((stage != 'J') == bool(k));
  ++submissions; stages.push_back(stage);
  pending.emplace_back([=] {
    if (j) *j = 10;
    if (k) *k = 20;
    if (kb) *kb = 30;
    if (error) *error = (mode == 10 && stage == 'R');
  });
  if (mode == 8 && stage == 'J') throw std::bad_alloc();
  return mode == 9 && stage == 'K' ? GENERATIVEQC_STATUS_CUDA_ERROR : 0;
}
int enqueue_cuda_direct_jk_device(Direct*, Spec spec, const double* density,
                                 const double* beta, std::size_t matrix, double* j,
                                 double* k, double* kb, int* error, std::string&) {
  assert(spec.coulomb.coefficient == 1.0 && spec.exchange.coefficient == -0.125);
  assert(spec.coulomb.present != spec.exchange.present);
  return enqueue(spec.coulomb.present ? 'J' : 'K', density, beta, matrix, j, k, kb, error);
}
int execute_cuda_density_fitting_uhf_jk_device(Fitted*, const double* density,
    const double* beta, double* j, double* k, double* kb, std::string&,
    JkTermSelection terms, FockMatrixLayout) {
  assert(terms.coulomb != terms.exchange);
  return enqueue(terms.coulomb ? 'J' : 'K', density, beta, 4, j, k, kb, nullptr);
}
int execute_cuda_density_fitting_rhf_jk_device(Fitted* p, const double* density,
    double* j, double* k, std::string& detail, JkTermSelection terms, FockMatrixLayout layout) {
  return execute_cuda_density_fitting_uhf_jk_device(p, density, nullptr, j, k, nullptr,
                                                   detail, terms, layout);
}
int enqueue_prepared_cuda_exchange_correction(Provider&, const Spec& spec,
    const double* density, const double* beta, std::size_t matrix, double* k,
    double* kb, int* error, std::string&) {
  assert(spec.exchange.coefficient == -0.375);
  return enqueue('R', density, beta, matrix, nullptr, k, kb, error);
}
}
struct CudaKsFinalStateToken {
  int version{1}, identity{5};
  bool operator==(const CudaKsFinalStateToken&) const = default;
};
struct CudaKsFixedDensityProfile {
  std::array<double, 4> milliseconds{};
  std::uint32_t present_mask{};
};
struct CudaXcView { int* error{}; };
enum class CudaXcDensityPrecision { Fp64 };
struct Xc {
  int error{};
  double potential{99};
  CudaXcView enqueue_replay_body(const double* density, std::size_t elements,
                               CudaXcDensityPrecision) {
    assert(timing && density && (elements == 4 || elements == 8));
    ++submissions; stages.push_back('X');
    pending.emplace_back([this] { potential = 42; error = mode == 12; });
    if (mode == 11) throw std::runtime_error("XC failed after mutation");
    if (mode == 18) throw std::bad_alloc();
    if (mode == 19) throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, "XC failed");
    if (mode == 16) change_token();
    return {&error};
  }
};
struct Impl {
  bool final_state_ready{true}, final_frame_ready{true}, final_stationary_weights_ready{true};
  bool final_fitted_projection_ready{true};
  unsigned final_generation{5};
  int identity{5}, device{2}, stream{7}, primary_error{}, range_error{};
  unsigned spins{1};
  std::size_t matrix{4}, elements{4};
  double density[8]{1,2,3,4,5,6,7,8}, j[4]{}, exchange[8]{}, range_exchange[8]{};
  int* jk_error{&primary_error};
  int* range_jk_error{&range_error};
  bool has_range_correction{true};
  scf::Spec correction;
  scf::Spec* range_correction{&correction};
  scf::Provider provider;
  std::unique_ptr<Xc> xc{std::make_unique<Xc>()};
  void* nonlocal_correlation{};
  CudaKsFinalStateToken token() const {
    if (!final_state_ready || !final_generation) throw std::invalid_argument("revoked");
    return {1, identity};
  }
  void current_device() const { check(cudaSetDevice(device)); }
  Impl() { correction.exchange.coefficient = -0.375; }
};
struct Owner {
  std::unique_ptr<Impl> impl_{std::make_unique<Impl>()};
METHODS
};
int main(int argc, char** argv) {
  assert(argc == 3); mode = std::atoi(argv[1]);
  Owner owner; auto& p = *owner.impl_;
  p.spins = std::atoi(argv[2]); p.elements = 4 * p.spins;
  if (mode == 2) { p.provider.model.spec.exchange.present = false; p.has_range_correction = false; }
  if (mode == 3) p.nonlocal_correlation = &p;
  if (mode == 4) p.has_range_correction = false;
  if (mode == 17) p.xc.reset();
  change_token = [&] { ++p.identity; };
  CudaKsFinalStateToken expected = p.token();
  if (mode == 5) ++expected.identity;
  CudaKsFixedDensityProfile profile{{99,99,99,99}, 15};
  std::string detail;
  auto run = [&] { return owner.profile_fixed_density_components(expected, profile, detail); };
  const auto status = run();
  const bool preflight_failure = mode == 5 || mode == 6 || mode == 7 || mode == 20;
  const bool success = mode < 5 || mode == 17;
  assert((status == GENERATIVEQC_STATUS_SUCCESS) == success);
  for (int i = 0; i != 8; ++i) assert(p.density[i] == i + 1);
  assert(pending.empty());
  if (success) {
    const unsigned mask = mode == 2 ? 9 : mode == 3 || mode == 17 ? 7 : mode == 4 ? 11 : 15;
    assert(profile.present_mask == mask && p.token() == expected && drains == 0);
    for (unsigned i = 0; i != 4; ++i) assert(profile.milliseconds[i] == ((mask & (1U << i)) ? 2 : 0));
    assert(stages == (mode == 2 ? std::vector<char>{'J','X'} : mode == 4 ?
      std::vector<char>{'J','K','X'} : mode == 3 || mode == 17 ? std::vector<char>{'J','K','R'} :
      std::vector<char>{'J','K','R','X'}));
    const auto count = stages.size();
    assert(records == int(2 * count) && synchronizations == int(count));
    assert(run() == GENERATIVEQC_STATUS_SUCCESS && p.token() == expected);
    assert(stages.size() == 2 * count && drains == 0);
  } else {
    assert(profile.present_mask == 0);
    for (auto value : profile.milliseconds) assert(value == 0);
    assert(p.final_state_ready == preflight_failure);
    assert(p.final_frame_ready == preflight_failure);
    assert(p.final_stationary_weights_ready == preflight_failure);
    assert(p.final_fitted_projection_ready == preflight_failure);
    assert(p.final_generation == (preflight_failure ? 5U : 0U));
    assert(drains == (preflight_failure ? 0 : 1));
    if (preflight_failure) assert(submissions == 0);
    else {
      const auto count = submissions;
      assert(run() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && submissions == count);
    }
  }
}
"""
    directory = tmp_path_factory.mktemp("fixed-density-profile")
    cpp = directory / "profile.cpp"
    binary = directory / "profile"
    cpp.write_text(harness.replace("GUARD", guard).replace("METHODS", methods))
    cache = shutil.which("ccache")
    compiled = subprocess.run(
        [
            *([cache] if cache else []),
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "include"),
            str(cpp),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    return binary


@pytest.mark.parametrize("mode", range(21))
@pytest.mark.parametrize("spins", [1, 2])
def test_fixed_density_profile_lease_and_routing(
    profile_program: Path, mode: int, spins: int
) -> None:
    completed = subprocess.run(
        [str(profile_program), str(mode), str(spins)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("mask", [0, 1, 7, 9, 11, 15])
def test_python_profile_preserves_null_and_converts_event_units(mask: int) -> None:
    import ctypes as ct
    from types import SimpleNamespace

    from generativeqc._ks_snapshot import NativeKsSnapshot

    checks = []

    def binding(
        _batch: object, _snapshot: object, values: object, count: int, present: object
    ) -> int:
        assert count == 4
        for index in range(4):
            values[index] = 2.0 * index
        ct.cast(present, ct.POINTER(ct.c_uint32))[0] = mask
        return 0

    snapshot = SimpleNamespace(
        backend="cuda",
        check_current=lambda: checks.append(True),
        _library=SimpleNamespace(
            generativeqc_ks_snapshot_cuda_fixed_density_profile_v1=binding
        ),
        _batch=SimpleNamespace(_batch=1, _context=2),
        _handle=3,
    )
    result = NativeKsSnapshot.cuda_fixed_density_profile(snapshot)
    assert len(checks) == 2
    assert tuple(result.values()) == tuple(
        0.002 * index if mask & (1 << index) else None for index in range(4)
    )
    with pytest.raises(TypeError):
        result["scf_fock_j"] = 4


def test_fixed_density_benchmark_repeats_one_snapshot_and_preserves_missing_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generativeqc._ks_snapshot as snapshots

    from benchmarks.dft_force_matrix import _fixed_density_scf_profile

    calls = []
    batch = object()
    samples = [
        {
            "scf_fock_j": 0.003,
            "scf_full_range_k": 0.010,
            "scf_long_range_k": None,
            "semilocal_ao_grid_xc": None,
        },
        {
            "scf_fock_j": 0.001,
            "scf_full_range_k": 0.040,
            "scf_long_range_k": None,
            "semilocal_ao_grid_xc": None,
        },
        {
            "scf_fock_j": 0.002,
            "scf_full_range_k": 0.020,
            "scf_long_range_k": None,
            "semilocal_ao_grid_xc": None,
        },
    ]

    class Snapshot:
        def __init__(self, owner: object, index: int) -> None:
            assert owner is batch and index == 0
            calls.append("create")

        def cuda_fixed_density_profile(self) -> dict[str, float | None]:
            calls.append("sample")
            return samples[calls.count("sample") - 1]

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(snapshots, "NativeKsSnapshot", Snapshot)
    result = _fixed_density_scf_profile(
        batch, repeats=3, exchange_operators=("full-range",), nonlocal_correlation=True
    )
    assert calls == ["create", "sample", "sample", "sample", "close"]
    assert result["status"] == "measured"
    assert result["fixed_density"] is True and result["scf_replayed"] is False
    profile = result["profile"]
    assert profile["repeats"] == 3 and profile["samples_seconds"] == samples
    assert profile["profiled_ms"]["scf_fock_j"] == 2
    assert profile["profiled_ms"]["scf_full_range_k"] == 20
    assert profile["profiled_ms"]["scf_long_range_k"] is None
    assert profile["profiled_ms"]["semilocal_ao_grid_xc"] is None
    assert "semilocal_ao_grid_xc" in profile["missing_expected_components"]
