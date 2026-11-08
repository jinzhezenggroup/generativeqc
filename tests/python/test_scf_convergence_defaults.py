"""Shared solver defaults preserve explicit controls and independent physical gates."""

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


def test_cold_hf_automatic_proposal_preserves_domain_and_page_bounds(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Execute the real host policy without a native library or GPU context."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_one_electron_kernels.py"),
            "--derivatives",
            "--derivative-policy-output",
            str(tmp_path / "generated_one_electron_derivative_policy.cuh"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = tmp_path / "cold_policy.cpp"
    source.write_text(r"""
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/hf_bounded_page_policy.hpp"
int main() {
  using namespace generativeqc::scf::cuda_policy;
  const ColdHfScreeningDomain cold{768, 1, 2, 100, true, false, false,
                                  false, true, false, false, false, false};
  for (const char* value : {static_cast<const char*>(nullptr), "1"}) {
    if (value) setenv("GENERATIVEQC_HF_COLD_SCREENING", value, 1);
    else unsetenv("GENERATIVEQC_HF_COLD_SCREENING");
    const auto admitted = resolve_cold_hf_screening(cold, 1e-12);
    assert(admitted.maximum_iterations == 8 && admitted.tolerance == 8e-8);
    for (std::size_t size : {383U, 384U, 511U, 512U}) {
      auto trial = cold;
      trial.nbf = size;
      const bool expected = size >= 512;
      assert((resolve_cold_hf_screening(trial, 1e-12).maximum_iterations != 0) == expected);
    }
    for (auto member : {&ColdHfScreeningDomain::warm_seed,
                        &ColdHfScreeningDomain::unrestricted,
                        &ColdHfScreeningDomain::physical_reference,
                        &ColdHfScreeningDomain::persistent_eri,
                        &ColdHfScreeningDomain::resident_values,
                        &ColdHfScreeningDomain::mixed_precision,
                        &ColdHfScreeningDomain::incremental_jk}) {
      auto ineligible = cold;
      ineligible.*member = true;
      assert(resolve_cold_hf_screening(ineligible, 1e-12).maximum_iterations == 0);
    }
    for (double cutoff : {0.0, 8e-8, -1.0,
                          std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()}) {
      const auto policy = resolve_cold_hf_screening(cold, cutoff);
      assert((cutoff == 0.0) == (policy.maximum_iterations != 0));
    }
    for (unsigned allowance : {1U, 8U, 23U}) {
      auto short_solve = cold;
      short_solve.maximum_iterations = allowance;
      assert(resolve_cold_hf_screening(short_solve, 1e-12).maximum_iterations == 0);
    }
  }
  unsetenv("GENERATIVEQC_HF_COLD_SCREENING");
  const auto ordinary_page = resolve_hf_direct_tuning_profile(false);
  assert(ordinary_page.bounded_streaming.maximum_task_capacity == 8U * 1024U * 1024U);
  assert(ordinary_page.bounded_streaming.maximum_arena_bytes == (std::size_t{3} << 29));
  generativeqc::runtime::CudaTargetInfo target;
  target.total_global_memory = std::size_t{32} << 30;
  const HfBoundedPageDomain page_domain{768, 1, 2, 100, false, false, false, false};
  auto page = resolve_hf_bounded_page_policy(target, page_domain);
  assert(page.bounded_streaming.task_capacity_ceiling == 32U * 1024U * 1024U);
  assert(page.bounded_streaming.arena_maximum_bytes == (std::size_t{6} << 30));
  for (auto member : {&HfBoundedPageDomain::unrestricted,
                      &HfBoundedPageDomain::physical_reference,
                      &HfBoundedPageDomain::mixed_precision,
                      &HfBoundedPageDomain::incremental_jk}) {
    auto ineligible = page_domain;
    ineligible.*member = true;
    assert(resolve_hf_bounded_page_policy(target, ineligible)
               .bounded_streaming.task_capacity_ceiling == 8U * 1024U * 1024U);
  }
  for (auto ineligible : {HfBoundedPageDomain{511, 1, 2, 100},
                          HfBoundedPageDomain{768, 2, 2, 100},
                          HfBoundedPageDomain{768, 1, 3, 100},
                          HfBoundedPageDomain{768, 1, 2, 23}})
    assert(resolve_hf_bounded_page_policy(target, ineligible)
               .bounded_streaming.task_capacity_ceiling == 8U * 1024U * 1024U);
  for (std::size_t bytes : {std::size_t{0}, std::size_t{16} << 30}) {
    target.total_global_memory = bytes;
    page = resolve_hf_bounded_page_policy(target, page_domain);
    assert(page.bounded_streaming.arena_maximum_bytes ==
           (bytes ? bytes / 16 : std::size_t{256} << 20));
  }
  target.total_global_memory = std::size_t{32} << 30;
  for (const char* value : {"0", "", "none", "invalid", " 1", "1 "}) {
    setenv("GENERATIVEQC_HF_COLD_SCREENING", value, 1);
    const auto refused = resolve_cold_hf_screening(cold, 1e-12);
    assert(refused.maximum_iterations == 0 && refused.tolerance == 0.0);
    assert(resolve_hf_bounded_page_policy(target, page_domain)
               .bounded_streaming.task_capacity_ceiling == 8U * 1024U * 1024U);
  }
}
""")
    executable = required_native_cxx.build_executable(
        [
            source,
            root / "src/scf/cuda/rhf_policy.cpp",
            root / "src/runtime/cuda_provider.cpp",
        ],
        tmp_path / "cold-policy",
        compile_args=(
            "-std=c++20",
            f"-I{root / 'src'}",
            f"-I{root / 'include'}",
            f"-I{tmp_path}",
        ),
    )
    subprocess.run([executable], check=True, capture_output=True, timeout=20)


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
def test_calculator_copies_uniform_native_controls_for_contexts(
    library: SimpleNamespace, method: str, device: str, options: dict[str, Any]
) -> None:
    install_scf_defaults_stub(library, (1e-9, 1e-5, 1e-12), qualified=True)
    calculator = Calculator(method=method, device=device, **options)
    assert calculator._energy_tolerance == 1e-9
    assert calculator._density_tolerance == 1e-5


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


def test_rhf_block_convergence_strict_selection(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Compile the actual private parser/resolver, without a CUDA or SCF run."""
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_one_electron_kernels.py"),
            "--derivatives",
            "--derivative-policy-output",
            str(tmp_path / "generated_one_electron_derivative_policy.cuh"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = tmp_path / "block_policy.cpp"
    source.write_text(r"""
#include "scf/rhf_block_convergence_policy.hpp"
#include <cassert>
#include <cstdlib>
#include <initializer_list>
int main() {
  using namespace generativeqc::scf::cuda_policy;
  using R = RhfBlockConvergenceRequest;
  assert(parse_rhf_block_convergence_request(nullptr) == R::Off);
  assert(parse_rhf_block_convergence_request("0") == R::Off);
  assert(parse_rhf_block_convergence_request("1") == R::On);
  for (const char* bad : {"", "01", "true", "auto", " 1", "1 ", "-1", "2"})
    assert(parse_rhf_block_convergence_request(bad) == R::Invalid);
  unsetenv("GENERATIVEQC_HF_BLOCK_CONVERGENCE");
  assert(rhf_block_convergence_requested() == R::Off);
  setenv("GENERATIVEQC_HF_BLOCK_CONVERGENCE", "1", 1);
  assert(rhf_block_convergence_requested() == R::On);
  for (unsigned mask = 0; mask != 32; ++mask) {
    const bool unrestricted = mask & 1, mixed = mask & 2, incremental = mask & 4;
    const bool quartet = mask & 8, physical = mask & 16;
    assert(resolve_rhf_block_convergence(R::On, unrestricted, mixed, incremental,
                                        quartet, physical) == (mask == 8));
    for (auto request : {R::Off, R::Invalid})
      assert(!resolve_rhf_block_convergence(request, unrestricted, mixed, incremental,
                                            quartet, physical));
  }
}
""")
    executable = required_native_cxx.build_executable(
        [
            source,
            root / "src/scf/cuda/rhf_policy.cpp",
            root / "src/runtime/cuda_provider.cpp",
        ],
        tmp_path / "block-policy",
        compile_args=(
            "-std=c++20",
            f"-I{root / 'src'}",
            f"-I{root / 'include'}",
            f"-I{tmp_path}",
        ),
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)


def test_rhf_block_convergence_is_part_of_cuda_resource_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from generativeqc import estimate_hf_resources
    from test_hf_cuda_fleet_resources import H2, layout_library

    def identity(backend: str) -> Any:
        return (
            estimate_hf_resources([H2], backend=backend, library=layout_library())
            .requests[0]
            .identity
        )

    cpu = identity("cpu")
    schedules = []
    for value in (None, "0", "1", "invalid"):
        if value is None:
            monkeypatch.delenv("GENERATIVEQC_HF_BLOCK_CONVERGENCE", raising=False)
        else:
            monkeypatch.setenv("GENERATIVEQC_HF_BLOCK_CONVERGENCE", value)
        current = identity("cuda")
        assert (
            json.loads(current.schedule)["runtime_policy"][
                "GENERATIVEQC_HF_BLOCK_CONVERGENCE"
            ]
            == value
        )
        assert identity("cpu") == cpu
        schedules.append(current.schedule)
    assert len(set(schedules)) == 4


def test_nonlocal_confirmation_requires_active_stationarity_and_no_rejected_proof(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """A physical idempotent target cannot be held below only the AO bounds."""
    occupied = np.eye(8)[:, :4]
    virtual = np.eye(8)[:, 4:]
    density = 2 * occupied @ occupied.T
    fock = np.diag([-1.0] * 4 + [1.0] * 4)
    fock[:4, 4:] = 4.5e-6
    fock[4:, :4] = 4.5e-6
    _, coefficients = np.linalg.eigh(fock)
    proposal = 2 * coefficients[:, :4] @ coefficients[:, :4].T
    commutator = fock @ density - density @ fock
    gradient = np.linalg.norm(2 * virtual.T @ fock @ occupied)
    density_rms = np.linalg.norm(proposal - density) / 8
    residual_rms = np.linalg.norm(commutator) / 8
    maximum_residual = np.max(np.abs(commutator))
    assert np.linalg.norm(density @ density - 2 * density) < 1e-14
    assert max(density_rms, residual_rms, maximum_residual) < 1e-5
    assert gradient > np.sqrt(1e-9)

    source = tmp_path / "confirmation.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>
#include "dft/ks_stopping_policy.hpp"
int main() {
  using namespace generativeqc::dft;
  KsStoppingState state{20, false, true, true, 0.0,
"""
        + f"{density_rms:.17g}, {residual_rms:.17g}, {maximum_residual:.17g},\n"
        + r"""
                        1e-9, 1e-5, 1e-5};
"""
        + f"  const double gradient = {gradient:.17g};\n"
        + r"""
  auto confirm = [&](bool orbital, double norm, bool rejected = false,
                     double tolerance = 1e-5) {
    return nonlocal_ks_density_confirmation(state, tolerance, orbital, norm, rejected);
  };
  assert(confirm(false, gradient));
  assert(!orbital_ks_converged(state, gradient));
  assert(!confirm(true, gradient));
  const double stationary = gradient * 0.5;
  assert(confirm(true, stationary));
  assert(orbital_ks_converged(state, stationary));
  assert(!confirm(true, stationary, true));
  assert(!confirm(false, stationary, true));
  assert(!confirm(true, std::sqrt(state.energy_tolerance)));
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    assert(!confirm(true, invalid));
  state.iterations = 1;
  state.full_energy_history = state.target_ready = false;
  state.energy_change = std::numeric_limits<double>::infinity();
  assert(confirm(true, stationary));
  assert(!orbital_ks_converged(state, stationary));
  assert(!confirm(false, stationary, false, 1e-9));
  state.density_change = state.residual = state.maximum_residual = 1e-10;
  assert(confirm(false, std::numeric_limits<double>::quiet_NaN(), false, 1e-9));
  state.maximum_residual = 1e-9;
  assert(!confirm(false, stationary, false, 1e-9));
  state.maximum_residual = 1e-10;
  state.density_change = state.density_tolerance;
  assert(!confirm(true, stationary));
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "confirmation",
        compile_args=["-std=c++20", f"-I{Path(__file__).resolve().parents[2] / 'src'}"],
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)
