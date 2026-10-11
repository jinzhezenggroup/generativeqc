"""Metadata-only automatic CUDA KS admission; doubles are not GPU evidence."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc import KsOptions, resources_ks
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from test_ks_point_batch_resources import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


@pytest.fixture(scope="module")
def derivative_reservation(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Any:
    """Compile the real factory policy, not a duplicated Python decision."""
    root = Path(__file__).resolve().parents[2]
    definition = _definition(
        (root / "src/methods/dft_method.cpp").read_text(),
        "unsigned ks_direct_derivative_order(",
    )
    folder = tmp_path_factory.mktemp("automatic-ks-derivative-policy")
    source = folder / "policy.cpp"
    source.write_text(
        '#include "scf/fock_build.hpp"\n'
        '#include "generativeqc/generativeqc.h"\n'
        "using namespace generativeqc;\n"
        "struct NativeKsExecutionPlan { const void* automatic_program{}; };\n"
        + definition
        + r"""
extern "C" unsigned reservation(bool automatic, bool cuda, bool exact) {
  scf::ResolvedFockBuild strategy{};
  strategy.backend = cuda ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
  strategy.spec.coulomb.present = true;
  strategy.spec.exchange.present = false;
  strategy.spec.coulomb.approximation = exact ? scf::FockApproximation::Exact
                                             : scf::FockApproximation::DensityFitted;
  NativeKsExecutionPlan plan{automatic ? &strategy : nullptr};
  return ks_direct_derivative_order(strategy, cuda ? GENERATIVEQC_BACKEND_CUDA
                                                  : GENERATIVEQC_BACKEND_CPU_REFERENCE, plan);
}
"""
    )
    path = native_cxx.build_shared(
        [source],
        folder / "policy.so",
        compile_args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
        ),
    )
    library = ctypes.CDLL(str(path))
    library.reservation.argtypes = [ctypes.c_bool] * 3
    library.reservation.restype = ctypes.c_uint
    return library.reservation


@pytest.mark.parametrize("automatic", (False, True))
@pytest.mark.parametrize("cuda", (False, True))
@pytest.mark.parametrize("exact", (False, True))
def test_energy_only_automatic_owner_does_not_reserve_derivative_sources(
    derivative_reservation: Any, automatic: bool, cuda: bool, exact: bool
) -> None:
    assert derivative_reservation(automatic, cuda, exact) == int(
        not automatic and cuda and exact
    )


@pytest.fixture(scope="module")
def final_model_admission(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Any:
    """Execute the shared model gate with descriptor-only registry doubles.

    Grid/Fock validation doubles isolate this metadata gate; the real endpoint
    suite independently exercises canonicality, physical state and numerics.
    No CUDA allocation, compiler or device is needed for these regressions.
    """
    root = Path(__file__).resolve().parents[2]
    definition = _definition(
        (root / "src/dft/ks_final_state.cpp").read_text(), "bool valid_model("
    )
    folder = tmp_path_factory.mktemp("automatic-ks-final-model")
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t = void*;\n"
    )
    source = folder / "policy.cpp"
    source.write_text(
        r"""
#include <cmath>
#include <string_view>
#include "dft/cuda_xc_point_program.hpp"
#include "dft/ks_final_state.hpp"
#include "dft/semilocal_family.hpp"
#include "dft/xc.hpp"
#include "scf/mean_field.hpp"
namespace generativeqc::dft {
void validate_grid_spec(const GridSpec&) {}
namespace generated {
struct AutomaticLibxcEntry {
  const SemilocalPointProgram* program{};
  explicit operator bool() const { return program != nullptr; }
};
SemilocalPointProgram cpu{"metadata-only", "same-science", 7, 17, nullptr};
CudaXcPointProgram cuda;
bool missing{};
AutomaticLibxcEntry automatic_libxc_entry(std::uint32_t code) {
  return {code == 123 ? &cpu : nullptr};
}
const CudaXcPointProgram* automatic_cuda_point_entry(std::uint32_t code) {
  return code == 123 && !missing ? &cuda : nullptr;
}
bool split_hybrid_registered(std::uint32_t) { return false; }
struct SplitComposition {
  bool matched{};
  unsigned exact_exchange_numerator{}, exact_exchange_denominator{};
};
SplitComposition split_hybrid_composition(std::uint32_t) { return {}; }
}
}
namespace generativeqc::scf {
void validate_resolved_fock_build(const ResolvedFockBuild&) {}
void require_wb97mv_composition(const ResolvedFockBuild&, const ResolvedFockBuild&,
                                const dft::nlc::Vv10Parameters&) {}
}
namespace generativeqc::dft {
"""
        + definition
        + r"""
extern "C" bool final_model(bool cuda_backend, unsigned spins, unsigned mutation) {
  generated::cuda = {};
  generated::cuda.expression_identity = "same-science";
  generated::cuda.functional_code = 123;
  generated::cuda.ingredient_mask = 7;
  generated::cuda.domain_version = 17;
  generated::missing = mutation == 1;
  if (mutation == 2) generated::cuda.expression_identity = "other-science";
  if (mutation == 3) ++generated::cuda.ingredient_mask;
  if (mutation == 4) ++generated::cuda.domain_version;
  if (mutation == 5) ++generated::cuda.functional_code;
  if (mutation == 6) generated::cuda.expression_identity = nullptr;
  KsFinalStateIdentity identity;
  auto& fock = identity.determinant.model;
  fock.backend = cuda_backend ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
  fock.spec.spin = spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
  fock.spec.derivative_order = 0;
  fock.spec.exchange.present = mutation == 7;
  fock.spec.exchange.coefficient = -0.5;
  identity.determinant.occupied.resize(spins, 1);
  identity.model = {1, 17, {}, 37, 123, spins, cuda_backend ? 0 : -1, 1};
  if (mutation == 8) identity.model.semilocal_exchange_scale = 0.5;
  if (mutation == 9) identity.model.range_correction.emplace();
  if (mutation == 10) identity.model.nonlocal_correlation.emplace();
  if (mutation == 11) identity.model.device = cuda_backend ? -1 : 0;
  if (mutation == 12) ++identity.model.scf_domain_version;
  return valid_model(identity);
}
}
"""
    )
    path = native_cxx.build_shared(
        [source],
        folder / "policy.so",
        compile_args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(folder),
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
        ),
    )
    library = ctypes.CDLL(str(path))
    library.final_model.argtypes = [ctypes.c_bool, ctypes.c_uint, ctypes.c_uint]
    library.final_model.restype = ctypes.c_bool
    return library.final_model


@pytest.mark.parametrize("cuda", (False, True))
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("mutation", range(13))
def test_automatic_final_model_authenticates_actual_backend_program(
    final_model_admission: Any, cuda: bool, spins: int, mutation: int
) -> None:
    # CPU registration remains independent of optional CUDA packaging; CUDA
    # requires matching code/science/ingredients/domain, not just a device ID.
    admitted = mutation == 0 or (not cuda and 1 <= mutation <= 6)
    assert final_model_admission(cuda, spins, mutation) == admitted


def _library(calls: list[tuple[Any, ...]], status: int = 0) -> SimpleNamespace:
    """Provide only shape transports and record the exact functional selector."""

    def query(*arguments: Any) -> int:
        calls.append(arguments[:-2])
        if not status:
            arguments[-2][:] = [1024, 2048, 4096]
        return status

    def provider(nao: int, output: Any) -> int:
        output._obj.value = 128
        return 0

    def quadrature(atoms: int, points: int, output: Any) -> int:
        output._obj.value = 128
        return 0

    def forbidden(*arguments: Any) -> int:
        pytest.fail("automatic XC was reduced to the legacy LDA/PBE shape flag")

    return SimpleNamespace(
        generativeqc_ks_resource_inventory_version_v1=lambda: 1,
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_resource_ks_cuda_v1=forbidden,
        generativeqc_resource_ks_cuda_v2=query,
        generativeqc_resource_quadrature_cuda_v1=quadrature,
        generativeqc_resource_ks_matrix_provider_cuda_v1=provider,
    )


@pytest.mark.parametrize("name", ("LDA_C_CHACHIYO", "GGA_X_APBE", "MGGA_X_LTA"))
@pytest.mark.parametrize("spins", (1, 2))
def test_automatic_resource_shape_binds_exact_static_program(
    monkeypatch: pytest.MonkeyPatch, name: str, spins: int
) -> None:
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    calls: list[tuple[Any, ...]] = []
    request = resources_ks.ks_resource_request(
        [H2],
        method=f"libxc-{'rks' if spins == 1 else 'uks'}:{name}",
        backend="cuda",
        library=_library(calls),
        charges=[0 if spins == 1 else 1],
        multiplicities=[spins],
    )

    assert request.unsupported_reason is None
    assert request.identity.observables == ("energy",)
    assert json.loads(request.identity.schedule)["history_iterations"] == 104
    assert (
        next(
            estimate.bytes
            for estimate in request.candidates[0].estimates
            if estimate.name == "all KS host history"
        )
        == 256 * 104
    )
    assert calls[0][6] == spins
    assert calls[0][7] == automatic_functional_code(name)
    assert not any(
        "force device staging" in estimate.name
        for estimate in request.candidates[0].estimates
    )


@pytest.mark.parametrize("missing_query", (False, True))
def test_missing_packaged_provider_has_no_cpu_or_legacy_shape_fallback(
    monkeypatch: pytest.MonkeyPatch, missing_query: bool
) -> None:
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    calls: list[tuple[Any, ...]] = []
    library = _library(calls, status=1)
    if missing_query:
        del library.generativeqc_resource_ks_cuda_v2
    request = resources_ks.ks_resource_request(
        [H2], method="libxc:MGGA_X_LTA", backend="cuda", library=library
    )
    assert request.candidates == ()
    assert request.unsupported_reason
    assert (
        "v2" in request.unsupported_reason
        if missing_query
        else ("static point provider" in request.unsupported_reason)
    )
    if not missing_query:
        assert (
            "GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON" in request.unsupported_reason
        )


@pytest.mark.parametrize(
    ("options", "reason"),
    (
        ({"precision": "auto"}, "strict-FP64"),
        ({"ks_options": KsOptions(xc_schedule="host_unfused")}, "device-fused"),
        ({"include_forces": True}, "forces are not qualified"),
    ),
)
def test_unqualified_policy_fails_before_loading_a_library(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], reason: str
) -> None:
    from generativeqc import _native

    def forbidden(*arguments: Any, **keywords: Any) -> None:
        pytest.fail("unqualified CUDA policy initialized a native context")

    monkeypatch.setattr(_native, "load_library", forbidden)
    with pytest.raises(NotImplementedError, match=reason):
        resources_ks.ks_resource_request(
            [H2], method="libxc:GGA_X_APBE", backend="cuda", **options
        )
