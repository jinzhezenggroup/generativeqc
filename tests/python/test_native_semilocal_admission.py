"""Preserve native KS admission while deriving curated families from metadata."""

from __future__ import annotations

import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition
from generativeqc.ks import (
    KsOptions,
    _native_semilocal_code,
    ks_coefficients,
    ks_range_exchange_parameters,
    resolve_ks_options,
)
from generativeqc_compiler.dft.grid import GridSpec
from generativeqc_compiler.method import (
    MethodSpec,
    original_nonlocal_correlation,
    resolve_method,
)

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ((("GGA_X_PBE", Fraction(1)),), (1.0, 0.0, 0.0)),
        ((("GGA_C_PBE", Fraction(1)),), (0.0, 1.0, 0.0)),
        (
            (
                ("GGA_X_PBE", Fraction(1)),
                ("GGA_X_PBE", Fraction(-1)),
                ("GGA_C_PBE", Fraction(1)),
            ),
            (0.0, 1.0, 0.0),
        ),
    ],
)
def test_partial_pbe_graphs_use_their_imported_component_program(
    spin: str,
    components: tuple[tuple[str, Fraction], ...],
    expected: tuple[float, float, float],
) -> None:
    graph = resolve_method(MethodSpec("scaled-fragment", components), spin=spin)
    selector = "pbe-uks" if spin == "polarized" else "pbe-rks"
    options = resolve_ks_options(
        selector, KsOptions(composition=graph, grid=GridSpec())
    )
    assert options.method_ir.identity == graph.identity
    from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code

    # Partial PBE used to resolve to curated scales in Python but failed native
    # component matching. The atomic program owns its sole surviving term;
    # outer X/C scales therefore remain unity and no missing partner is added.
    component = "GGA_X_PBE" if expected[0] else "GGA_C_PBE"
    assert options.coefficients == (1.0, 1.0, 0.0)
    assert _native_semilocal_code(graph) == automatic_functional_code(component)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_pbe_point_program_does_not_own_exchange_omega(spin: str) -> None:
    graph = resolve_method(
        MethodSpec(
            "independent-range-exchange",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
            short_range_exchange=Fraction(1, 5),
            long_range_exchange=Fraction(4, 5),
            range_omega=Fraction(3, 10),
        ),
        spin=spin,
    )
    selector = "pbe-uks" if spin == "polarized" else "pbe-rks"
    options = resolve_ks_options(
        selector, KsOptions(composition=graph, grid=GridSpec())
    )
    assert _native_semilocal_code(options.method_ir) == 1
    assert options.coefficients[:2] == (1.0, 1.0)
    assert ks_range_exchange_parameters(options.method_ir) == (0.2, 0.8, 0.3)


@pytest.mark.parametrize(
    "components",
    [
        (("LDA_X", Fraction(1)), ("LDA_C_PW", Fraction(1))),
        (("MGGA_X_R2SCAN", Fraction(1)), ("MGGA_C_R2SCAN", Fraction(1))),
    ],
)
def test_pure_semilocal_family_does_not_admit_new_nonlocal_contributions(
    components: tuple[tuple[str, Fraction], ...],
) -> None:
    graph = resolve_method(
        MethodSpec(
            "unsupported-nonlocal-composition",
            components,
            nonlocal_correlation=original_nonlocal_correlation("vv10"),
        )
    )
    with pytest.raises(NotImplementedError, match="unsupported native KS"):
        ks_coefficients(graph)


def test_cuda_rsh_admission_is_resolved_provider_capability(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    root = Path(__file__).resolve().parents[2]
    # Admission moved, while prepared CUDA provider eligibility stays with the owner.
    source = (
        (root / "src/methods/dft_method_preparation.cpp").read_text()
        + "\n"
        + (root / "src/methods/dft_method.cpp").read_text()
    )
    options_begin = source.index("scf::ScfOptions dft_options(")
    options_end = source.index("/** Copy every pointee", options_begin)
    options = source[options_begin:options_end]
    assert (
        "CUDA range-separated KS is qualified only for complete WB97M-V" not in options
    )

    scaled_begin = options.index("if (scaled_or_hybrid")
    scaled_end = options.index("if (execution_plan.nonlocal_correlation", scaled_begin)
    scaled_gate = options[scaled_begin:scaled_end]
    assert "!execution_plan.range_exchange" in scaled_gate

    helper = cpp_function_definition(source, "cuda_rsh_provider_compatible")
    # Locate the actual CUDA range admission statement by its inputs, rather
    # than the scalar facade or its position among unrelated owner methods.
    guard = re.search(
        r"if\s*\(backend_\s*==\s*GENERATIVEQC_BACKEND_CUDA\s*&&"
        r"\s*execution_plan_\.range_exchange\s*&&",
        source,
    )
    assert guard is not None
    admission = source[guard.start() : source.index("#endif", guard.start())]
    cpp, binary = tmp_path / "admission.cpp", tmp_path / "admission"
    cpp.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include "generativeqc/generativeqc.hpp"
#include "scf/fock_build.hpp"
namespace generativeqc::scf {
// Provider doubles expose only the resolved strategy, prepared device binding
// and system identity consumed by the actual admission function.
struct Binding {
  bool available{true}; int device_id{0}; std::size_t nbf{8};
  explicit operator bool() const { return available; }
};
struct PreparedFockPlan {
  ResolvedFockBuild resolved;
  Binding binding;
  unsigned identity{1};
  const auto& strategy() const { return resolved; }
  auto system() const { return identity; }
  bool matches_system(unsigned other) const { return identity == other; }
};
auto prepared_cuda_fock_binding(const PreparedFockPlan& plan) { return plan.binding; }
}
namespace scf = generativeqc::scf;
using MethodError = generativeqc::Error;
HELPER
struct Owner {
  scf::PreparedFockPlan fock_;
  std::optional<scf::ResolvedFockBuild> range_strategy_;
  std::unique_ptr<scf::PreparedFockPlan> range_correction_;
  generativeqc_backend backend_{GENERATIVEQC_BACKEND_CUDA};
  struct { bool range_exchange{true}; } execution_plan_;
  bool admits() {
    try {
      ADMISSION
      return true;
    } catch (const MethodError& error) {
      assert(error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
      return false;
    }
  }
};
void check(scf::PreparedFockPlan primary, scf::ResolvedFockBuild correction,
           scf::PreparedFockPlan range, bool has_range, bool expected) {
  auto* selected = has_range ? &range : nullptr;
  assert(cuda_rsh_provider_compatible(primary, correction, selected) == expected);
  Owner owner{primary, correction,
              has_range ? std::make_unique<scf::PreparedFockPlan>(range) : nullptr};
  assert(owner.admits() == expected);
  owner.range_strategy_.reset();
  assert(!owner.admits());
  owner.execution_plan_.range_exchange = false;
  assert(owner.admits());
}
int main() {
  using namespace scf;
  for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
    for (bool fitted : {false, true}) {
      for (bool exchange : {false, true}) {
        PreparedFockPlan primary, range;
        auto& p = primary.resolved;
        p.backend = FockBackend::Cuda;
        p.spec.spin = spin; p.spec.derivative_order = 0;
        p.spec.coulomb.approximation = fitted ? FockApproximation::DensityFitted
                                            : FockApproximation::Exact;
        p.spec.exchange.approximation = p.spec.coulomb.approximation;
        p.spec.exchange.present = exchange;
        ResolvedFockBuild correction = p;
        correction.spec.coulomb.present = false;
        correction.spec.exchange.present = true;
        correction.spec.exchange.approximation = FockApproximation::Exact;
        correction.spec.exchange.op = FockOperator::LongRange;
        correction.spec.exchange.omega = 0.3;
        range.resolved = correction;
        check(primary, correction, range, fitted, true);
        check(primary, correction, range, !fitted, false);
        for (unsigned fault = 0; fault < 28; ++fault) {
          auto bad = primary; auto c = correction; auto r = range;
          switch (fault) {
            case 0: bad.binding.available = false; break;
            case 1: bad.resolved.backend = FockBackend::Cpu; break;
            case 2: bad.resolved.spec.derivative_order = 1; break;
            case 3: bad.resolved.spec.coulomb.present = false; break;
            case 4: bad.resolved.spec.coulomb.coefficient = 0.5; break;
            case 5: bad.resolved.spec.coulomb.op = FockOperator::LongRange; break;
            case 6: bad.resolved.spec.coulomb.omega = 0.2; break;
            case 7: c.backend = FockBackend::Cpu; break;
            case 8: c.spec.spin = spin == FockSpin::Restricted ? FockSpin::Unrestricted
                                                            : FockSpin::Restricted; break;
            case 9: c.spec.derivative_order = 1; break;
            case 10: c.spec.coulomb.present = true; break;
            case 11: c.spec.exchange.present = false; break;
            case 12: c.spec.exchange.approximation = FockApproximation::DensityFitted; break;
            case 13: c.spec.exchange.op = FockOperator::FullRange; break;
            case 14: c.spec.exchange.omega = 0; break;
            case 15: c.spec.exchange.omega = -0.1; break;
            case 16: c.spec.exchange.omega = std::numeric_limits<double>::quiet_NaN(); break;
            case 17: c.spec.exchange.coefficient = std::numeric_limits<double>::infinity(); break;
            case 18: c.screening_tolerance *= 2; break;
            case 19: bad.resolved.spec.coulomb.approximation = FockApproximation::SeminumericalCosx; break;
            case 20: r.resolved.screening_tolerance *= 2; break;
            case 21: r.identity = 2; break;
            case 22: r.binding.available = false; break;
            case 23: r.binding.device_id = 1; break;
            case 24: ++r.binding.nbf; break;
            case 25: bad.resolved.spec.exchange.approximation = fitted
                       ? FockApproximation::Exact : FockApproximation::DensityFitted; break;
            case 26: bad.resolved.spec.exchange.op = FockOperator::LongRange; break;
            case 27: bad.resolved.spec.exchange.omega = 0.2; break;
          }
          if (fault < 20 || (fault < 25 && fitted) || (fault >= 25 && exchange))
            check(bad, c, r, fitted, false);
        }
      }
    }
  }
}
""".replace("HELPER", helper).replace("ADMISSION", admission)
    )
    native_cxx.build_executable(
        [cpp],
        binary,
        compile_args=("-std=c++20", f"-I{root / 'include'}", f"-I{root / 'src'}"),
    )
    checked = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("lda-rks", True),
        ("pbe-rks", True),
        ("lda-uks", False),
        ("pbe-uks", False),
        ("pbe0-rks", False),
        ("r2scan-rks", False),
        ("pbe-d4-rks", False),
    ],
)
def test_stationary_second_order_preserves_spin_and_method_domain(
    selector: str, expected: bool
) -> None:
    from generativeqc.ks import stationary_second_order_eligible

    options = resolve_ks_options(selector, KsOptions(grid=GridSpec()))
    assert stationary_second_order_eligible(options.method_ir) is expected


def test_stationary_second_order_rejects_scaled_pbe() -> None:
    from generativeqc.ks import stationary_second_order_eligible

    graph = resolve_method(
        MethodSpec(
            "scaled-second-order",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
        ),
        spin="unpolarized",
    )
    assert not stationary_second_order_eligible(graph)


def test_cpu_stationary_force_preserves_intrinsic_d4_admission() -> None:
    from generativeqc.ks import cpu_stationary_all_electron_force_eligible

    options = resolve_ks_options("pbe-d4-rks", KsOptions(grid=GridSpec()))
    assert cpu_stationary_all_electron_force_eligible(options.method_ir)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_native_device_xc_selector_preserves_both_spin_flows(spin: str) -> None:
    from types import SimpleNamespace

    from generativeqc_compiler.dft.native_semilocal import (
        device_feature_ingredients,
        legacy_grid_xc_selector,
    )
    from generativeqc_compiler.xc.prepared import _native_device_xc
    from generativeqc_compiler.xc.spec import functional

    spec = functional("PBE", spin=spin)
    assert device_feature_ingredients(spec) == ("rho", "gradient")
    assert legacy_grid_xc_selector(spec) == 1
    program = SimpleNamespace(
        spec=spec,
        contract=SimpleNamespace(request=SimpleNamespace(observable="potential")),
    )
    assert _native_device_xc(program, object(), object())
    opposite = "polarized" if spin == "unpolarized" else "unpolarized"
    with pytest.raises(ValueError, match="spin"):
        legacy_grid_xc_selector(spec, spin=opposite)


def test_native_device_xc_selector_falls_back_for_unsupported_graph() -> None:
    from types import SimpleNamespace

    from generativeqc_compiler.xc.prepared import _native_device_xc
    from generativeqc_compiler.xc.spec import functional

    program = SimpleNamespace(
        spec=functional("R2SCAN"),
        contract=SimpleNamespace(request=SimpleNamespace(observable="potential")),
    )
    assert not _native_device_xc(program, object(), object())


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("include_forces", [False, True])
def test_ks_planning_rejects_unaccounted_intrinsic_post_scf_owner(
    backend: str, include_forces: bool
) -> None:
    from generativeqc.resources_ks import ks_resource_request

    with pytest.raises(NotImplementedError, match="post-SCF corrections"):
        ks_resource_request(
            [[("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]],
            method="pbe-d4-rks",
            backend=backend,
            basis="sto-3g",
            ks_options=KsOptions(grid=GridSpec()),
            include_forces=include_forces,
        )
