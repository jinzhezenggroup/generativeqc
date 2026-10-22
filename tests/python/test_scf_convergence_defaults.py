"""Default solver controls require a compatible owner and preserve tight requests."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conftest import NativeCxx

import pytest
from generativeqc import Calculator, GridSpec, KsOptions, PrecisionCalculator, _native
from generativeqc._scf_convergence import default_scf_controls

from tools.generativeqc_validation.native_registry_stub import install_registry_stub


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Resolve real model identities without allocating a CUDA context."""

    def available(method: int, pointer: object) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    native = SimpleNamespace(
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_standard_scf_convergence_version_v1=lambda: 1,
        generativeqc_method_available=available,
    )
    install_registry_stub(native, frozenset({"energy"}))
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: native)
    return native


@pytest.mark.parametrize(
    "method", ["rhf", "pbe-rks", "pbe0-rks", "b3lyp-rks", "r2scan-rks", "m06-2x-rks"]
)
def test_default_controls_and_explicit_tight_overrides(
    library: SimpleNamespace, method: str
) -> None:
    calculator = Calculator(method=method, device="cuda")
    assert calculator._energy_tolerance == 1e-9
    assert calculator._density_tolerance == 1e-5
    assert calculator._screening_tolerance == 1e-12
    for controls in ({"energy_tolerance": 1e-12}, {"density_tolerance": 1e-10}):
        explicit = Calculator(method=method, device="cuda", **controls)
        for name, value in controls.items():
            assert getattr(explicit, "_" + name) == value


@pytest.mark.parametrize(
    "method,device,options",
    [
        ("rhf", "cpu", {}),
        ("uhf", "cuda", {}),
        ("mp2", "cuda", {}),
        ("pbe-uks", "cuda", {}),
        ("wb97m-v-rks", "cuda", {}),
        ("pbe-rks", "cuda", {"ks_options": KsOptions(grid=GridSpec.treutler(4))}),
    ],
)
def test_other_models_and_grids_keep_strict_controls(
    library: SimpleNamespace, method: str, device: str, options: dict[str, Any]
) -> None:
    calculator = Calculator(method=method, device=device, **options)
    assert calculator._energy_tolerance == 1e-10
    assert calculator._density_tolerance == 1e-8


@pytest.mark.parametrize("version", [None, 0, 2])
def test_missing_or_unknown_native_capability_is_strict(
    library: SimpleNamespace, version: int | None
) -> None:
    if version is None:
        del library.generativeqc_standard_scf_convergence_version_v1
    else:
        library.generativeqc_standard_scf_convergence_version_v1 = lambda: version
    calculator = Calculator(method="pbe-rks", device="cuda")
    assert default_scf_controls(calculator)["density_tolerance"] == 1e-8
    assert calculator._energy_tolerance == 1e-10
    assert calculator._density_tolerance == 1e-8


@pytest.mark.parametrize("profile", ["standard", "fp64", "fast"])
def test_profiles_share_defaults_and_never_loosen_caller_controls(
    library: SimpleNamespace, profile: str
) -> None:
    default = PrecisionCalculator(method="rhf", device="cuda", policy=profile)
    assert default._candidate._energy_tolerance == 1e-9
    assert default._candidate._density_tolerance == 1e-5
    explicit = PrecisionCalculator(
        method="rhf",
        device="cuda",
        policy=profile,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    assert explicit._candidate._energy_tolerance == 1e-12
    assert explicit._candidate._density_tolerance == 1e-10
    assert explicit._options["energy_tolerance"] == 1e-12


def test_physical_limits_are_capped_and_tight_controls_restore_strict_policy(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    source = tmp_path / "limits.cpp"
    source.write_text(r"""
#include <cassert>
#include <limits>
#include <initializer_list>
#include "scf/convergence_limits.hpp"
int main() {
  using namespace generativeqc::scf;
  assert(standard_scf_convergence_requested(1e-9,1e-5));
  // Loose public controls cannot grant owner qualification. The unsupported
  // UHF/ECP residual from the regression must fail while qualified RHF passes.
  assert(hf_stationarity_tolerance(1e-9,1e-5,false)==1e-8);
  assert(hf_stationarity_tolerance(1e-9,1e-5,true)==1e-5);
  assert(5e-6 > hf_stationarity_tolerance(1e-9,1e-5,false));
  assert(5e-6 < hf_stationarity_tolerance(1e-9,1e-5,true));
  assert(hf_stationarity_tolerance(1e-12,1e-5,true)==1e-8);
  assert(hf_stationarity_tolerance(1e-9,1e-12,false)==1e-12);
  assert(scf_stationarity_tolerance(1e-9,1e-5,1e-9)==1e-5);
  assert(scf_stationarity_tolerance(1e-3,1e-2,1e-9)==1e-5);
  assert(scf_stationarity_tolerance(1e-10,1e-5,1e-9)==1e-9);
  assert(scf_stationarity_tolerance(1e-9,1e-8,1e-9)==1e-9);
  assert(scf_stationarity_tolerance(1e-9,1e-12,1e-9)==1e-12);
  for (double invalid : {0., -1., std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
    assert(!standard_scf_convergence_requested(invalid,1e-5));
    assert(!standard_scf_convergence_requested(1e-9,invalid));
  }
}
""")
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "limits",
        compile_args=["-std=c++20", f"-I{Path(__file__).resolve().parents[2] / 'src'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)
