"""Default solver controls require a compatible owner and preserve tight requests."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conftest import NativeCxx

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, PrecisionCalculator, _native
from generativeqc._scf_defaults_native import default_controls

from tools.generativeqc_validation.native_registry_stub import (
    install_registry_stub,
    install_scf_defaults_stub,
)


def test_restricted_mo_gradient_from_nonorthogonal_physical_commutator() -> None:
    """An independent MO evaluation fixes the normalization, not AO RMS numbers."""
    random = np.random.default_rng(2987)
    rotations, _ = np.linalg.qr(random.normal(size=(8, 8)))
    overlap = rotations @ np.diag(np.linspace(0.2, 3.0, 8)) @ rotations.T
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    orthogonalizer = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    frame, _ = np.linalg.qr(random.normal(size=(8, 8)))
    coefficients = orthogonalizer @ frame
    occupied, virtual = coefficients[:, :3], coefficients[:, 3:]
    density = 2 * occupied @ occupied.T
    fock = random.normal(size=(8, 8))
    fock = (fock + fock.T) / 2
    commutator = fock @ density @ overlap - overlap @ density @ fock
    transformed = orthogonalizer.T @ commutator @ orthogonalizer
    expected = np.linalg.norm(2 * virtual.T @ fock @ occupied)
    assert np.linalg.norm(transformed) / np.sqrt(2) == pytest.approx(
        expected, rel=1e-13
    )


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch, policy_library: Any) -> SimpleNamespace:
    """Borrow real model/policy queries without allocating a CUDA context."""

    def available(method: int, pointer: object) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    native = SimpleNamespace(
        generativeqc_precision_controls_query_v1=policy_library.generativeqc_precision_controls_query_v1,
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_method_available=available,
        generativeqc_precision_admission_query_v1=(
            policy_library.generativeqc_precision_admission_query_v1
        ),
        generativeqc_precision_execution_run_v1=(
            policy_library.generativeqc_precision_execution_run_v1
        ),
        generativeqc_observable_comparison_query_v1=(
            policy_library.generativeqc_observable_comparison_query_v1
        ),
        generativeqc_precision_candidate_query_v1=(
            policy_library.generativeqc_precision_candidate_query_v1
        ),
    )
    install_registry_stub(native, frozenset({"energy"}))
    install_scf_defaults_stub(native, (1e-9, 1e-5, 1e-12), qualified=True)
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: native)
    return native


@pytest.mark.parametrize(
    "method",
    [
        "rhf",
        "pbe-rks",
        "pbe0-rks",
        "b3lyp-rks",
        "r2scan-rks",
        "m06-2x-rks",
        "wb97m-v-rks",
    ],
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
        ("pbe-rks", "cuda", {"ks_options": KsOptions(grid=GridSpec.treutler(4))}),
    ],
)
def test_calculator_copies_strict_native_controls_for_contexts(
    library: SimpleNamespace, method: str, device: str, options: dict[str, Any]
) -> None:
    install_scf_defaults_stub(library)
    calculator = Calculator(method=method, device=device, **options)
    assert calculator._energy_tolerance == 1e-10
    assert calculator._density_tolerance == 1e-8


@pytest.mark.parametrize("legacy_version", [None, 1, 2])
def test_missing_required_query_cannot_use_the_legacy_capability(
    library: SimpleNamespace, legacy_version: int | None
) -> None:
    del library.generativeqc_scf_defaults_query_v1
    if legacy_version is not None:
        library.generativeqc_standard_scf_convergence_version_v1 = lambda: (
            legacy_version
        )
    with pytest.raises(
        NotImplementedError, match="required SCF default-controls query"
    ):
        Calculator(method="pbe-rks", device="cuda")
    with pytest.raises(
        NotImplementedError, match="required SCF default-controls query"
    ):
        default_controls(SimpleNamespace(_library=library))


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
#include "dft/ks_stopping_policy.hpp"
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
  using namespace generativeqc::dft;
  KsStoppingState state{2, false, true, true, 1e-10, 1e-3, 1e-3, 1e-3,
                        1e-9, 1e-5, 1e-5};
  assert(orbital_ks_converged(state, 3e-5));
  assert(!ordinary_ks_converged(state));
  assert(!orbital_ks_converged(state, 4e-5));
  assert(!orbital_ks_converged(state, std::sqrt(state.energy_tolerance)));
  state.target_ready = false;
  assert(!orbital_ks_converged(state, 0));
  state.target_ready = true;
  state.full_energy_history = false;
  assert(!orbital_ks_converged(state, 0));
  state.full_energy_history = true;
  state.energy_change = state.energy_tolerance;
  assert(!orbital_ks_converged(state, 0));
  state.energy_change = 0;
  assert(!orbital_ks_converged(state, std::numeric_limits<double>::quiet_NaN()));
  assert(!orbital_ks_converged(state, -1));
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
