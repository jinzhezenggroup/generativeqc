"""Reference-selection and capability contracts without loading a GPU runtime."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest
from generativeqc import Atom, Calculator, CBSBasis, _native
from generativeqc.basis import BasisShell
from generativeqc.basis_capabilities import basis_capability

from tools.generativeqc_validation.native_registry_stub import install_registry_stub


@pytest.fixture
def available_mp2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub library discovery, but use the real constructor/descriptor policy."""

    def available(method: int, output: object) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(generativeqc_method_available=available)
    install_registry_stub(library, frozenset({"energy", "forces"}))
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)


@pytest.mark.usefixtures("available_mp2")
def test_explicit_correlation_only_df_mode_preserves_force_and_reference_contract() -> (
    None
):
    owner = Calculator(
        method="mp2",
        device="cuda",
        basis="sto-3g",
        auxiliary_basis="def2-svp",
        density_fitting="correlation_cuda",
        correlation_memory_budget_bytes=1 << 30,
    )
    descriptor = owner._method_descriptor(auxiliary_basis=ctypes.c_void_p(1234))
    assert descriptor.density_fitting_mode == _native.DENSITY_FITTING_CORRELATION_CUDA
    assert descriptor.density_fitting_auxiliary_basis == 1234
    assert descriptor.correlation_memory_budget_bytes == 1 << 30
    assert owner.capabilities.supported_properties == frozenset({"energy", "forces"})
    assert not owner.capabilities.supports_batch
    with pytest.raises(NotImplementedError, match="prepared batches"):
        owner.prepare_batch([])


@pytest.mark.usefixtures("available_mp2")
@pytest.mark.parametrize(
    "options,error,match",
    (
        ({"device": "cpu"}, NotImplementedError, "requires CUDA"),
        ({"auxiliary_basis": None}, ValueError, "explicit auxiliary"),
        ({"method": "rhf"}, ValueError, "MP2 reference contract"),
        ({"precision": "auto"}, ValueError, "fp64"),
    ),
)
def test_correlation_only_df_refuses_unsupported_contract(
    options: dict, error: type[Exception], match: str
) -> None:
    arguments = {
        "method": "mp2",
        "device": "cuda",
        "auxiliary_basis": "sto-3g",
        "density_fitting": "correlation_cuda",
        **options,
    }
    with pytest.raises(error, match=match):
        Calculator(**arguments)


@pytest.mark.usefixtures("available_mp2")
def test_legacy_cuda_fitting_selector_is_not_reinterpreted() -> None:
    owner = Calculator(
        method="mp2", device="cuda", auxiliary_basis="sto-3g", density_fitting="cuda"
    )
    descriptor = owner._method_descriptor(auxiliary_basis=ctypes.c_void_p(1234))
    assert descriptor.density_fitting_mode == _native.DENSITY_FITTING_CUDA


@pytest.mark.usefixtures("available_mp2")
@pytest.mark.parametrize("method", ("mp2", "df-ccsd(t)"))
def test_full_df_selector_is_a_distinct_scientific_model(method: str) -> None:
    owner = Calculator(
        method=method,
        device="cuda",
        auxiliary_basis="def2-svp",
        density_fitting="full_cuda",
    )
    descriptor = owner._method_descriptor(auxiliary_basis=ctypes.c_void_p(1234))
    assert descriptor.density_fitting_mode == _native.DENSITY_FITTING_FULL_CUDA
    assert descriptor.density_fitting_auxiliary_basis == 1234


@pytest.mark.usefixtures("available_mp2")
@pytest.mark.parametrize("method", ("mp2", "df-ccsd(t)"))
@pytest.mark.parametrize("operator", ("df_metric", "df_three_center"))
@pytest.mark.parametrize("derivative_order", (0, 1))
def test_auxiliary_g_admission_is_scoped_to_the_correlation_consumer(
    method: str,
    operator: str,
    derivative_order: int,
) -> None:
    orbital = CBSBasis("sto-3g", 2).basis
    auxiliary = replace(
        orbital,
        name="synthetic-g-fit",
        elements=(
            replace(
                orbital.by_element[1], shells=(BasisShell(4, ("1.0",), (("1.0",),)),)
            ),
        ),
    )
    atoms = (Atom.from_value(("H", (0.0, 0.0, 0.0))),)
    request = {
        "backend": "cuda",
        "role": "auxiliary",
        "operator": operator,
        "derivative_order": derivative_order,
    }
    assert not basis_capability(auxiliary, atoms, **request)["eligible"]
    assert basis_capability(auxiliary, atoms, consumer="correlation_df", **request)[
        "eligible"
    ]
    owner = Calculator(
        method=method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        device="cuda",
        density_fitting="correlation_cuda" if method == "mp2" else "cuda",
    )
    owner._preflight_hf_basis(atoms, compute_forces=True)
    shells = owner._shells_for_atoms(
        atoms, auxiliary, operator=operator, derivative_order=derivative_order
    )
    assert len(shells) == 1 and shells[0].angular_momentum == 4
    with pytest.raises(NotImplementedError, match="l<=3"):
        owner._shells_for_atoms(atoms, auxiliary, operator="eri")
    legacy = Calculator(
        method="mp2",
        basis=orbital,
        auxiliary_basis=auxiliary,
        device="cuda",
        density_fitting="cuda",
    )
    with pytest.raises(NotImplementedError, match="l<=3"):
        legacy._preflight_hf_basis(atoms, compute_forces=True)
    with pytest.raises(NotImplementedError, match="l<=3"):
        legacy._shells_for_atoms(
            atoms, auxiliary, operator=operator, derivative_order=derivative_order
        )
