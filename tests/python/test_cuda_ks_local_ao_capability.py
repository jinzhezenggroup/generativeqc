"""Host regression for component-wise CUDA KS local-AO admission."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.dft.xc_contraction_cuda import XcMatrixSchedule, _emit_tiled
from generativeqc_compiler.dft.xc_density_lowering import emit_density_binding

ROOT = Path(__file__).resolve().parents[2]


def _guard() -> str:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("    const char* ao_selection = std::getenv(")
    end = source.index("    constexpr std::size_t ao_map_host_budget", start)
    return source[start:end]


def test_local_ao_guard_depends_on_layout_capability_not_method_or_schedule(
    tmp_path: Path,
) -> None:
    guard = _guard()
    assert "cuda_xc_execution_capabilities(xc_layout).local_ao_selection" in guard
    assert "precision_schedule" not in guard
    assert "SemilocalFamily" not in guard

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    unit = tmp_path / "guard.cpp"
    executable = tmp_path / "guard"
    unit.write_text(
        r"""
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

struct Layout { bool selectable{}; };
struct Capabilities { bool local_ao_selection{}, mixed_density_contraction{}; };
Capabilities cuda_xc_execution_capabilities(const Layout& layout) {
  return {layout.selectable, false};
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  if (std::string(argv[1]) == "unset")
    unsetenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
  else
    setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", argv[1], 1);
  Layout xc_layout{std::string(argv[2]) == "capable"};
  const bool host_unfused = std::string(argv[3]) == "host";
  try {
"""
        + guard
        + r"""
    std::cout << (select_ao ? "local" : "dense");
  } catch (const std::invalid_argument&) {
    std::cout << "rejected";
  }
}
"""
    )
    built = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(unit),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert built.returncode == 0, built.stderr

    cases = {
        ("unset", "blocked", "host"): "dense",
        ("unset", "capable", "device"): "local",
        ("0", "capable", "device"): "dense",
        ("0", "blocked", "host"): "dense",
        ("1", "capable", "device"): "local",
        ("1", "blocked", "device"): "rejected",
        ("1", "capable", "host"): "rejected",
        ("yes", "capable", "device"): "rejected",
    }
    for args, expected in cases.items():
        result = subprocess.run(
            [str(executable), *args],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),
        )
        assert result.stdout == expected


def test_prepared_density_preserves_local_ao_component_precision(
    tmp_path: Path,
) -> None:
    """Run the real KS admission/accounting block with bounded provider facts."""
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    panel_header = (ROOT / "src/tensor/cuda_panel_product.hpp").read_text()
    reservation_start = panel_header.index(
        "  static constexpr std::size_t host_reservation"
    )
    reservation_end = panel_header.index(";", reservation_start) + 1
    prepare_start = source.index(
        "        const auto admitted_precision = resolve_cuda_ks_iteration_precision("
    )
    prepare_end = source.index("        prepared_ao_work =", prepare_start)
    start = source.index(
        "      const auto iteration_precision = resolve_cuda_ks_iteration_precision("
    )
    end = source.index("      // Provider selection stays inside", start)
    block = source[start:end]
    assert (
        "cuda_xc_execution_capabilities(xc_layout).mixed_density_contraction" in block
    )
    assert (
        "iteration_precision.uses_lower_precision(cuda_ks_precision_region::kCoulombJ)"
        in block
    )
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    unit, executable = tmp_path / "prepared.cpp", tmp_path / "prepared"
    unit.write_text(
        r"""
#include <cassert>
#include "dft/cuda_ks.hpp"
#include "dft/cuda_ks_precision.hpp"
#include "runtime/bounded_workspace.hpp"
using namespace generativeqc;
using namespace generativeqc::dft;
namespace generativeqc::tensor {
struct PreparedPanelProduct {
"""
        + panel_header[reservation_start:reservation_end]
        + r"""
};
struct PanelProductDiagnostic {
  std::size_t matrix_bytes{}, provider_allowance{}, host_bytes{};
};
}
std::size_t sum(std::size_t left, std::size_t right) {
  return runtime::size_add(left, right, "test resource overflow");
}
struct Layout { bool mixed_density_contraction; };
Layout cuda_xc_execution_capabilities(const Layout& layout) { return layout; }
struct Xc {
  struct Binding { struct Precision { runtime::PrecisionDirective arithmetic; } precision; };
  Binding admitted{}, strict{{runtime::strict_fp64_precision()}};
  bool capable;
  bool qualified{};
  std::size_t reserved{};
  tensor::PanelProductDiagnostic diagnostic{};
  void prepare_density(runtime::PrecisionDirective directive, std::uint64_t replays,
                       std::size_t budget) {
    assert(replays == 50);
    // A mapped layout must receive the narrowed strict directive at setup.
    assert(capable || directive.is_strict_fp64());
    admitted.precision.arithmetic = directive;
    reserved = budget;
    const bool selected = qualified && budget >= 160;
    diagnostic = {selected ? 64U : 0U, selected ? 96U : 0U,
                  tensor::PreparedPanelProduct::host_reservation};
  }
  const tensor::PanelProductDiagnostic* density_provider_diagnostic() const {
    return reserved ? &diagnostic : nullptr;
  }
  const Binding& density_binding(runtime::PrecisionPhase phase) const {
    return phase == runtime::PrecisionPhase::Admitted ? admitted : strict;
  }
};
int main() {
  for (bool capable : {false, true}) for (bool automatic : {false, true})
    for (bool nonlocal : {false, true})
      for (auto qualification : {CudaXcCapability::Unavailable,
                                 CudaXcCapability::QualificationRequired,
                                 CudaXcCapability::Qualified})
      for (std::size_t host : {std::size_t{0},
                               tensor::PreparedPanelProduct::host_reservation - 1,
                               tensor::PreparedPanelProduct::host_reservation,
                               tensor::PreparedPanelProduct::host_reservation + 1})
      for (std::size_t device : {std::size_t{0}, std::size_t{1}, std::size_t{160}})
      for (bool qualified : {false, true}) {
    CudaXcFastPathCapabilities formal;
    formal.mixed_density_precision = qualification;
    const auto precision_schedule = resolve_cuda_ks_precision_schedule(
        automatic ? GENERATIVEQC_PRECISION_AUTO : GENERATIVEQC_PRECISION_FP64,
        formal, false, nonlocal);
    const bool mixed_density = automatic && capable && !nonlocal &&
                               qualification == CudaXcCapability::Qualified;
    Layout xc_layout{capable};
    Xc owner{{}, {{runtime::strict_fp64_precision()}}, capable};
    owner.qualified = qualified;
    auto* xc = &owner;
    struct { std::uint64_t max_iterations = 50; } options;
    const CudaXcPreparationBudget xc_budget{device, host};
    CudaKsResources resource;
    resource.xc_device_bytes = 1000;
    resource.provider_device_bytes = 2000;
    resource.retained_host_numeric_bytes = 3000;
"""
        + source[prepare_start:prepare_end]
        + r"""
    assert(owner.admitted.precision.arithmetic.is_strict_fp64() == !mixed_density);
    const bool admitted = host >= tensor::PreparedPanelProduct::host_reservation;
    assert(owner.reserved == (admitted ? device : 0));
    const bool selected = admitted && device >= 160 && qualified;
    assert(resource.xc_device_bytes == 1000 + (selected ? 64 : 0));
    assert(resource.provider_device_bytes == 2000 + (selected ? 96 : 0));
    assert(resource.retained_host_numeric_bytes ==
           3000 + (admitted && device ? tensor::PreparedPanelProduct::host_reservation : 0));
    for (bool strict_refinement : {false, true}) {
      bool pending_mixed_coulomb{}, pending_mixed_density{};
"""
        + block
        + r"""
      assert(pending_mixed_coulomb == (automatic && !strict_refinement));
      assert(pending_mixed_density == (mixed_density && !strict_refinement));
    }
  }
}
"""
    )
    compile_owner(compiler, tmp_path, [unit], executable)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_density_preparation_requires_physical_support_and_formal_qualification(
    tmp_path: Path,
) -> None:
    """Run real preparation/selection against both directions of gate disagreement."""
    source = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    start = source.index("void CudaXcPlan::prepare_density(")
    end = source.index("bool CudaXcPlan::select_local_ao(", start)
    preparation = source[start:end]
    local_start = source.index(
        "std::vector<CudaXcDensityLauncher> local_density_launchers("
    )
    local_end = source.index("}  // namespace", local_start)
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    unit = tmp_path / "density_guard.cpp"
    executable = tmp_path / "density_guard"
    emitted = _emit_tiled(XcMatrixSchedule(16))
    shape_admission = emitted[
        emitted.index("inline bool tiled_xc_admitted(") : emitted.index(
            "template <bool Mixed, bool Tiled>"
        )
    ]
    unit.write_text(
        r"""
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>
#include "dft/xc_capabilities.hpp"
#include "runtime/lowering_binding.hpp"
using namespace generativeqc::dft;
using namespace generativeqc::runtime;
using I = std::int64_t;
using CudaXcDensityLauncher = void (*)(int,const double*,const double*,I,I,I,I,double*,int*,const std::size_t*,I);
struct CudaXcDensityBinding {
  CudaXcDensityLauncher launch{};
  NativeLoweringCandidate candidate;
  NativeLoweringPrecision precision;
  bool retained_incumbent{};
};
struct Layout {
  bool local_ao{}, physical_mixed{}, response{};
  CudaXcFastPathCapabilities fast_paths{};
  std::size_t nao = 17, npoint = 33, tile_points = 16, spins = 2, work_jets = 4;
};
using CudaXcLayout = Layout;
struct Execution { bool mixed_density_contraction{}; };
Execution cuda_xc_execution_capabilities(const Layout& layout) {
  return {!layout.local_ao && layout.physical_mixed};
}
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
int capture_status{};
int cudaStreamIsCapturing(int, cudaStreamCaptureStatus* status) {
  *status = capture_status;
  return 0;
}
void check(int status) { assert(status == 0); }
namespace tensor {
struct PreparedPanelProduct {
  struct Diagnostic { NativeLoweringCandidate candidate; };
  bool enabled() const { return false; }
  const Diagnostic& diagnostic() const { return diagnostic_; }
  Diagnostic diagnostic_{};
};
}
namespace cuda_xc_detail {
std::unique_ptr<tensor::PreparedPanelProduct> prepare_density_provider(
    const Layout&, int, std::size_t) {
  return {};
}
template<bool Mixed,bool Tiled>
void launch_density_product(int,const double*,const double*,I,I,I,I,double*,int*,const std::size_t*,I) {}
"""
        + shape_admission
        + emit_density_binding(16, emitted)
        + "\n}\n"
        + source[local_start:local_end]
        + r"""
struct CudaXcPlan {
  Layout layout_;
  bool evaluation_started_{};
  int stream_{};
  std::array<CudaXcDensityBinding, 2> strict_density_, admitted_density_;
  std::unique_ptr<tensor::PreparedPanelProduct> density_provider_;
  std::unique_ptr<CudaXcDensityBinding> provider_density_binding_;
  std::vector<CudaXcDensityLauncher> local_density_launchers_;
  std::vector<std::size_t> ao_offsets_{0, 17, 17, 21};
  void check_device() const {}
  void prepare_density(PrecisionDirective, std::uint64_t expected_replays = 1,
                       std::size_t provider_budget = 0);
  const tensor::PreparedPanelProduct* density_execution_provider(PrecisionPhase) const;
  const CudaXcDensityBinding& density_binding(PrecisionPhase) const;
};
"""
        + preparation
        + r"""
template<class Action> bool rejects(Action action) {
  try { action(); } catch (const std::invalid_argument&) { return true; }
  return false;
}
int main() {
  const auto strict = strict_fp64_precision();
  const auto mixed = fp32_compute_fp64_accumulation("dft.cuda.auto/density-contraction-v1");
  unsigned cases = 0;
  for (bool local : {false, true}) for (bool physical : {false, true})
    for (auto qualification : {CudaXcCapability::Unavailable,
                               CudaXcCapability::QualificationRequired,
                               CudaXcCapability::Qualified}) {
      CudaXcPlan plan;
      plan.layout_.local_ao = local;
      plan.layout_.physical_mixed = physical;
      plan.layout_.fast_paths.mixed_density_precision = qualification;
      plan.prepare_density(strict, 50);
      const auto prior = plan.admitted_density_;
      const auto prior_local = plan.local_density_launchers_;
      if (local) {
        assert(prior_local.size() == 3);
        assert((prior_local[0] == &cuda_xc_detail::launch_density_product<false,true>));
        // The empty middle map and one-point final tile use strict scalar bodies.
        assert((prior_local[1] == &cuda_xc_detail::launch_density_product<false,false>));
        assert((prior_local[2] == &cuda_xc_detail::launch_density_product<false,false>));
      } else {
        assert(prior_local.empty());
      }
      const bool allowed = !local && physical && qualification == CudaXcCapability::Qualified;
      assert(rejects([&] { plan.prepare_density(mixed, 50); }) == !allowed);
      for (std::size_t slot = 0; slot != 2; ++slot) {
        const auto& binding = plan.admitted_density_[slot];
        assert(binding.precision.arithmetic.is_strict_fp64() == !allowed);
        assert(binding.retained_incumbent);
        const auto expected = allowed
            ? (slot ? &cuda_xc_detail::launch_density_product<true,false>
                    : &cuda_xc_detail::launch_density_product<true,true>)
            : (slot ? &cuda_xc_detail::launch_density_product<false,false>
                    : &cuda_xc_detail::launch_density_product<false,true>);
        assert(binding.launch == expected);
        const auto& record = cuda_xc_detail::density_lowering_candidates[(allowed ? 2 : 0) + !slot];
        assert(binding.candidate.identity == record.identity);
        assert(binding.candidate.request_identity == record.request_identity);
        assert(binding.candidate.precision_identity == record.precision_identity);
        assert(binding.precision.identity == record.precision_identity);
        assert(plan.strict_density_[slot].precision.arithmetic.is_strict_fp64());
        if (!allowed) assert(binding.launch == prior[slot].launch);
      }
      assert(plan.local_density_launchers_ == prior_local);
      assert(plan.density_binding(PrecisionPhase::StrictAudit).precision.arithmetic.is_strict_fp64());
      assert(rejects([&] { plan.density_binding(static_cast<PrecisionPhase>(99)); }));
      const auto before_capture = plan.admitted_density_;
      capture_status = 1;
      assert(rejects([&] { plan.prepare_density(strict, 50); }));
      capture_status = cudaStreamCaptureStatusNone;
      for (std::size_t slot = 0; slot != 2; ++slot)
        assert(plan.admitted_density_[slot].launch == before_capture[slot].launch);
      // No malformed dtype, math mode or qualification can publish a partial table.
      const auto retained = plan.admitted_density_;
      for (auto bad : {PrecisionDirective{PrecisionDtype::Fp32,PrecisionDtype::Fp32,PrecisionDtype::Fp32,"bad"},
                       fp32_compute_fp64_accumulation("unqualified"),
                       PrecisionDirective{PrecisionDtype::Fp64,PrecisionDtype::Fp64,PrecisionDtype::Fp64,"","fast"}}) {
        assert(rejects([&] { plan.prepare_density(bad, 50); }));
        for (std::size_t slot = 0; slot != 2; ++slot)
          assert(plan.admitted_density_[slot].launch == retained[slot].launch);
      }
      // Once evaluation starts, neither arithmetic may rebind or allocate maps.
      plan.evaluation_started_ = true;
      assert(rejects([&] { plan.prepare_density(strict, 50); }));
      assert(rejects([&] { plan.prepare_density(mixed, 50); }));
      assert(plan.local_density_launchers_ == prior_local);
      ++cases;
    }
  assert(cases == 12);
}
"""
    )
    compile_owner(compiler, tmp_path, [unit], executable)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_selected_physical_layout_propagates_back_to_the_ks_owner() -> None:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("        if (admit_ao) {")
    end = source.index("        prepared_ao_work = xc->ao_selection_work();", start)
    block = source[start:end]
    assert "xc->select_local_ao(1e-16, ao_map_host_budget)" in block
    assert "xc_layout = xc->layout();" in block
