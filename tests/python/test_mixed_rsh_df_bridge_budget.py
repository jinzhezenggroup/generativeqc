"""Host-only source execution for the mixed bridge's additional-device budget."""

from __future__ import annotations

import ast
import ctypes as ct
import subprocess
import typing
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mixed_bridge_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    begin = source.index("  generativeqc_status cuda_mixed_rsh_df_integral_gradient(")
    end = source.index("\n  generativeqc_status cuda_integral_gradient(", begin)
    directory = tmp_path_factory.mktemp("mixed-rsh-df-admission")
    unit, binary = directory / "probe.cpp", directory / "probe"
    unit.write_text(_PREFIX + source[begin:end] + _SUFFIX)
    return native_cxx.build_executable(
        [unit],
        binary,
        compile_args=(
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
    )


def test_mixed_bridge_reserves_lazy_lr_on_cold_and_reused_calls(
    mixed_bridge_probe: Path,
) -> None:
    result = subprocess.run(
        [str(mixed_bridge_probe)],
        cwd=mixed_bridge_probe.parent,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _snapshot_functions() -> dict[str, typing.Any]:
    module = ast.parse((ROOT / "python/generativeqc/_ks_snapshot.py").read_text())
    helper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_integral_source_resources"
    )
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeKsSnapshot"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "cuda_integral_derivatives"
    )
    namespace = {
        "typing": typing,
        "ct": ct,
        "np": np,
        "MappingProxyType": MappingProxyType,
        "immutable": lambda value: value,
        "_native": SimpleNamespace(
            STATUS_NOT_IMPLEMENTED=2, check=lambda *args, **kwargs: None
        ),
    }
    exec(  # noqa: S102 - Execute only functions parsed from the local repository.
        compile(
            ast.Module(body=[helper, method], type_ignores=[]),
            "snapshot-source",
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("mixed", [False, True])
def test_wire_scope_names_additional_lr_without_double_count(mixed: bool) -> None:
    namespace = _snapshot_functions()

    def evaluate(
        batch: object,
        handle: object,
        values: object,
        count: int,
        budget: int,
        usage: object,
        slots: int,
    ) -> int:
        assert slots == 9 and count == 15 and budget == 5000
        np.ctypeslib.as_array(ct.cast(values, ct.POINTER(ct.c_double)), shape=(count,))[
            :
        ] = 0
        np.ctypeslib.as_array(ct.cast(usage, ct.POINTER(ct.c_uint64)), shape=(slots,))[
            :
        ] = [100, 0, 5000, 240, 0, 24, 0, 0, 0]
        return 0

    owner = SimpleNamespace(
        backend="cuda",
        density_fitted=mixed,
        check_current=lambda: None,
        _library=SimpleNamespace(
            generativeqc_ks_snapshot_cuda_integral_gradient_v1=evaluate
        ),
        _batch=SimpleNamespace(_batch=None, _context=None),
        _handle=None,
    )
    output, work = namespace["cuda_integral_derivatives"](
        owner, 1, 5000, range_exchange=True
    )
    assert output.shape == (5, 1, 3) and work["retained_device_bytes"] == 100
    if mixed:
        assert work["mixed_rsh_additional_device_peak_bytes"] == 5000
        assert "one_electron_device_peak_bytes" not in work
        assert work["compact_source_publication_host_peak_bytes"] == 240
        assert work["density_fitted_response_resources_included"] == 0
        assert work["additional_device_peak_includes_lazy_lr"] == 1
    else:
        assert work["one_electron_device_peak_bytes"] == 5000
        assert "mixed_rsh_additional_device_peak_bytes" not in work
        assert "density_fitted_response_resources_included" not in work


def test_composite_reuses_resource_decoder_and_marks_partial_df_bound() -> None:
    source = (ROOT / "python/generativeqc/_stationary_composite_cuda.py").read_text()
    assert "_integral_source_resources(" in source
    assert '"additional_bounds_exclude_df_response"' in source


_PREFIX = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/mixed_rsh_df_sources.hpp"
#define GENERATIVEQC_HAS_CUDA 1
using namespace generativeqc;
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaEventDisableTiming = 1;
static std::vector<std::string> trace;
static int fail_call = 0, device_bytes = 4096, constructed = 0;
static std::size_t seen_host = 0, seen_device = 0;
static int cudaSetDevice(int) {
  trace.push_back("device");
  return fail_call == 1 ? 1 : 0;
}
static const char* cudaGetErrorString(int) { return "injected error"; }
static int cudaEventCreateWithFlags(cudaEvent_t* x, int) {
  trace.push_back("create");
  *x = (void*)3;
  return fail_call == 2 ? 1 : 0;
}
static int cudaEventRecord(cudaEvent_t, cudaStream_t) {
  trace.push_back("record");
  return fail_call == 3 ? 1 : 0;
}
static int cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, int) {
  trace.push_back("wait");
  return fail_call == 4 ? 1 : 0;
}
static int cudaEventDestroy(cudaEvent_t) {
  trace.push_back("destroy");
  return 0;
}
struct System {
  std::vector<int> atoms{1};
  std::vector<int> ecp_terms;
};
namespace generativeqc::scf {
namespace reference {
using Matrix = std::vector<double>;
}
enum class FockSpin { Restricted, Unrestricted };
enum class FockBackend { Cuda, Cpu };
enum class FockOperator { FullRange, LongRange };
enum class FockApproximation { DensityFitted, Exact };
struct Term {
  bool present = true;
  double coefficient = 1;
  FockOperator op = FockOperator::FullRange;
  double omega = 0;
  FockApproximation approximation = FockApproximation::DensityFitted;
  bool operator==(const Term&) const = default;
};
struct Spec {
  FockSpin spin = FockSpin::Restricted;
  unsigned derivative_order = 0;
  Term coulomb;
  Term exchange;
  bool operator==(const Spec&) const = default;
};
struct ResolvedFockBuild {
  Spec spec;
  FockBackend backend = FockBackend::Cuda;
  double screening_tolerance = 1e-12;
  bool operator==(const ResolvedFockBuild&) const = default;
};
struct PreparedFockPlan {
  ResolvedFockBuild model;
  std::size_t budget = 0;
  PreparedFockPlan() = default;
  PreparedFockPlan(const System&, void*, ResolvedFockBuild m, int, std::size_t b, unsigned)
      : model(m), budget(b) {
    ++constructed;
    trace.push_back("allocate");
    if (fail_call == 7) throw std::bad_alloc();
  }
  const ResolvedFockBuild& strategy() const { return model; }
  bool matches_system(const System&) const { return true; }
  struct Diagnostic {
    std::size_t nbf = 2, device_bytes = 100;
  };
  Diagnostic diagnostic() const { return {2, budget ? budget : 100}; }
};
struct Binding {
  int device_id = 0;
  std::size_t nbf = 2;
  cudaStream_t stream = (void*)2;
  std::size_t retained_device_bytes;
  explicit operator bool() const { return true; }
};
Binding prepared_cuda_direct_derivative_binding(const PreparedFockPlan&) {
  return {0, 2, (void*)2, (std::size_t)device_bytes};
}
generativeqc_status execute_prepared_cuda_direct_long_range_derivatives_device(
    const PreparedFockPlan&, const double*, const double*, std::size_t, std::vector<double>& out,
    std::string&) {
  trace.push_back("lr");
  out.assign(3, 2);
  return fail_call == 5 ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE : GENERATIVEQC_STATUS_SUCCESS;
}
}
namespace generativeqc::dft {
struct Model {
  unsigned spins = 1;
  int device = 0;
  bool operator==(const Model&) const = default;
};
struct Identity {
  Model model;
  bool operator==(const Identity&) const = default;
};
struct CudaKsFinalStateToken {
  Identity identity;
  int generation = 1;
  bool operator==(const CudaKsFinalStateToken&) const = default;
};
struct VerifiedKsFinalState {
  std::vector<scf::reference::Matrix> density{{1, 0, 0, 1}}, weighted_density{{1, 0, 0, 1}};
};
struct CudaKsResidentDensityBinding {
  int device_id = 0;
  std::size_t matrix_elements = 4;
  unsigned spins = 1;
  void* stream = (void*)1;
  const double* alpha = (double*)1;
  const double* beta = nullptr;
  explicit operator bool() const { return true; }
};
struct MockCuda {
  generativeqc_status resident_final_density(const CudaKsFinalStateToken& t,
                                             CudaKsResidentDensityBinding& b, std::string&) {
    trace.push_back("resident");
    b.spins = t.identity.model.spins;
    b.beta = b.spins == 2 ? (double*)2 : nullptr;
    return t.generation == 1 ? GENERATIVEQC_STATUS_SUCCESS : GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
};
}
std::size_t ks_provider_bytes(const System&, generativeqc_backend, unsigned) {
  return device_bytes;
}
class Harness {
 public:
  System system_;
  scf::PreparedFockPlan fock_;
  std::optional<scf::ResolvedFockBuild> range_strategy_;
  std::unique_ptr<scf::PreparedFockPlan> range_correction_, range_derivative_correction_;
  std::unique_ptr<dft::MockCuda> cuda_ = std::make_unique<dft::MockCuda>();
  generativeqc_backend backend_ = GENERATIVEQC_BACKEND_CUDA;
  struct Options {
    generativeqc_density_fitting_mode density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
    generativeqc_precision_mode precision_mode = GENERATIVEQC_PRECISION_FP64;
  } options_;
  struct Execution {
    bool range_exchange = true;
    double range_omega = .3, short_range_exchange = .15, long_range_exchange = 1.;
  } execution_plan_;
  Harness() {
    fock_.model.spec.exchange.coefficient = -.075;
    range_strategy_ = fock_.model;
    auto& c = range_strategy_->spec;
    c.coulomb.present = false;
    c.exchange.coefficient = -.425;
    c.exchange.op = scf::FockOperator::LongRange;
    c.exchange.omega = .3;
    c.exchange.approximation = scf::FockApproximation::Exact;
    range_correction_ = std::make_unique<scf::PreparedFockPlan>();
    range_correction_->model = *range_strategy_;
  }
  generativeqc_status read_final_state(const dft::CudaKsFinalStateToken&, bool,
                                       dft::VerifiedKsFinalState&, std::string&) {
    trace.push_back("read");
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  generativeqc_status final_state_token(dft::CudaKsFinalStateToken& t, std::string&) {
    trace.push_back("token");
    if (fail_call == 6) t.generation = 2;
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  generativeqc_status density_fitted_integral_gradient(
      const dft::CudaKsFinalStateToken&, const std::vector<scf::reference::Matrix>&,
      const std::vector<scf::reference::Matrix>&, std::vector<double>& out, std::size_t budget,
      std::array<std::uint64_t, 9>& work, std::string&, bool,
      std::optional<std::size_t> device_budget = std::nullopt) {
    trace.push_back("df");
    seen_host = budget;
    seen_device = device_budget.value_or(budget);
    out.assign(12, 1);
    work[0] = 100;
    work[2] = seen_device;
    work[3] = 192;
    return GENERATIVEQC_STATUS_SUCCESS;
  }
"""

_SUFFIX = r"""
}
;
static void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
int main() {
  const dft::CudaKsFinalStateToken token{};
  std::vector<double> out;
  std::array<std::uint64_t, 9> work{};
  std::string detail;
  std::vector<scf::reference::Matrix> dw{{1, 0, 0, 1}};
  for (auto budget : {240U, 4095U, 4096U}) {
    Harness h;
    constructed = 0;
    trace.clear();
    auto status = h.cuda_mixed_rsh_df_integral_gradient(token, out, budget, work, detail, &dw, &dw);
    require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY, "lazy LR allocation bypassed admission");
    require(constructed == 0 && out.empty(),
            "rejection allocated a derivative owner or published data");
  }
  Harness h;
  constructed = 0;
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto status = h.cuda_mixed_rsh_df_integral_gradient(token, out, 4336, work, detail, &dw, &dw);
    require(status == GENERATIVEQC_STATUS_SUCCESS && out.size() == 15, "admitted bridge failed");
    require(constructed == 1, "lazy derivative owner was not reused");
    require(seen_host == 4336 && seen_device == 240, "host and device allowances were conflated");
    require(work[0] == 100 && work[2] == 4336, "additional LR peak omitted or double-counted");
  }
  auto status = h.cuda_mixed_rsh_df_integral_gradient(token, out, 240, work, detail, &dw, &dw);
  require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && out.empty(),
          "reuse escaped a tighter cap");
  for (int injected = 0; injected <= 7; ++injected) {
    Harness owner;
    fail_call = injected;
    trace.clear();
    auto result =
        owner.cuda_mixed_rsh_df_integral_gradient(token, out, 5000, work, detail, &dw, &dw);
    require((injected == 0) == (result == GENERATIVEQC_STATUS_SUCCESS),
            "wrong injected-error status");
    if (injected) require(out.empty(), "failure published source rows");
    if (injected == 3 || injected == 4)
      require(std::count(trace.begin(), trace.end(), "destroy") == 1,
              "failed event dependency leaked its event");
    if (injected == 0) {
      const auto record = std::find(trace.begin(), trace.end(), "record");
      const auto wait = std::find(trace.begin(), trace.end(), "wait");
      const auto lr = std::find(trace.begin(), trace.end(), "lr");
      require(record < wait && wait < lr, "LR consumer ran before its producer dependency");
    }
  }
  std::cout
      << "mixed bridge admission, reuse, independent host allowance, peak and error paths passed\n";
}
"""
