from __future__ import annotations

import ctypes
import json
import os
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Calculator, _generated_methods, _native, method_capabilities
from generativeqc._api_types import MethodCapabilities

ROOT = Path(__file__).resolve().parents[2]


def test_df_rccsdt_manifest_is_distinct_energy_force_method() -> None:
    payload = json.loads((ROOT / "manifests/public_methods.json").read_text())
    row = next(
        method for method in payload["methods"] if method["name"] == "df-rccsd(t)"
    )
    assert row == {
        "name": "df-rccsd(t)",
        "symbol": "DF_RCCSD_T",
        "abi_id": 19,
        "family": "coupled_cluster",
        "provider": "df_rccsdt",
        "properties": ["energy", "forces"],
        "supports_batch": False,
        "aliases": ["df-ccsd(t)"],
    }
    assert _generated_methods.METHOD_NAME_TO_ID["df-rccsd(t)"] == 19
    assert _generated_methods.METHOD_NAME_TO_ID["df-ccsd(t)"] == 19


def test_df_rccsdt_public_owner_forwards_force_requests() -> None:
    source = (ROOT / "src/methods/df_rccsdt_method.cpp").read_text()
    assert (
        "run_df_ccsdt_native(execution_, system_, auxiliary_, descriptor_, compute_forces)"
        in source
    )
    assert "public DF-RCCSD(T) forces remain unqualified" not in source
    assert (
        "descriptor_.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE" in source
    )
    assert "descriptor_.density_fitting_auxiliary_basis = nullptr" in source


def test_df_rccsdt_public_capability_is_energy_and_forces() -> None:
    caps = method_capabilities("df-rccsd(t)")
    alias = method_capabilities("df-ccsd(t)")
    assert caps.available and not caps.supports_batch
    assert caps.family == "coupled_cluster"
    assert caps.supported_properties == frozenset({"energy", "forces"})
    assert alias.supported_properties == caps.supported_properties


@pytest.fixture
def available_cc_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub library discovery only; exercise the actual public constructor."""

    def available(_method: int, result: typing.Any) -> int:
        ctypes.cast(result, ctypes.POINTER(ctypes.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    def capabilities(method: str) -> MethodCapabilities:
        return MethodCapabilities(
            method, "coupled_cluster", True, False, frozenset({"energy", "forces"})
        )

    library = SimpleNamespace(generativeqc_method_available=available)
    monkeypatch.setattr(_native, "load_library", lambda **_kwargs: library)
    monkeypatch.setattr("generativeqc.calculator.method_capabilities", capabilities)


@pytest.mark.usefixtures("available_cc_library")
@pytest.mark.parametrize("method", ("df-rccsd(t)", "df-ccsd(t)"))
@pytest.mark.parametrize("fitting", (None, "cuda", "auto", "none", True, False))
def test_df_rccsdt_constructor_and_descriptor(
    method: str, fitting: str | bool | None
) -> None:
    options = {} if fitting is None else {"density_fitting": fitting}
    calc = Calculator(
        method=method,
        device="cuda",
        auxiliary_basis="sto-3g",
        correlation_memory_budget_bytes=32 << 20,
        ccsd_max_iterations=75,
        ccsd_diis_history=4,
        **options,
    )
    auxiliary = ctypes.c_void_p(1234)
    descriptor = calc._method_descriptor(auxiliary_basis=auxiliary)
    assert descriptor.method == _native.METHOD_DF_RCCSD_T
    assert descriptor.density_fitting_mode == (
        _native.DENSITY_FITTING_AUTO
        if fitting == "auto"
        else _native.DENSITY_FITTING_CUDA
    )
    assert descriptor.density_fitting_auxiliary_basis == auxiliary.value
    assert descriptor.density_fitting_relative_threshold == 1e-10
    assert descriptor.density_fitting_memory_budget_bytes == 0
    assert descriptor.correlation_memory_budget_bytes == 32 << 20
    assert descriptor.ccsd_max_iterations == 75
    assert descriptor.ccsd_diis_history == 4
    assert descriptor.ccsd_energy_tolerance == 1e-11
    assert descriptor.ccsd_residual_tolerance == 1e-9
    assert descriptor.precision_mode == _native.PRECISION_FP64
    assert descriptor.screening_tolerance == 0


@pytest.mark.usefixtures("available_cc_library")
@pytest.mark.parametrize(
    "options,error,match",
    (
        ({"device": "cpu"}, NotImplementedError, "requires device='cuda'"),
        ({"auxiliary_basis": None}, ValueError, "explicit auxiliary_basis"),
        ({"density_fitting": "cpu"}, NotImplementedError, "CUDA density fitting"),
        (
            {"density_fitting": "cpu_reference"},
            NotImplementedError,
            "CUDA density fitting",
        ),
        ({"precision": "auto"}, ValueError, "require precision='fp64'"),
        ({"screening_tolerance": 1e-12}, ValueError, "screening_tolerance=0"),
        ({"ccsd_frozen_core": 1}, NotImplementedError, "frozen-core"),
        ({"ccsd_max_iterations": 0}, ValueError, "ccsd_max_iterations"),
        ({"ccsd_diis_history": 1}, ValueError, "ccsd_diis_history"),
    ),
)
def test_df_rccsdt_constructor_preserves_validation(
    options: dict[str, typing.Any], error: type[Exception], match: str
) -> None:
    kwargs = {"device": "cuda", "auxiliary_basis": "sto-3g", **options}
    with pytest.raises(error, match=match):
        Calculator(method="df-rccsd(t)", **kwargs)


@pytest.mark.usefixtures("available_cc_library")
@pytest.mark.parametrize("method", ("rccsd", "rccsd(t)"))
def test_conventional_cc_constructor_still_rejects_fitting(method: str) -> None:
    calc = Calculator(method=method, device="cuda")
    assert (
        calc._method_descriptor().density_fitting_mode == _native.DENSITY_FITTING_NONE
    )
    with pytest.raises(NotImplementedError, match="density fitting is not implemented"):
        Calculator(method=method, device="cuda", density_fitting="cuda")
    with pytest.raises(ValueError, match="auxiliary_basis requires density_fitting"):
        Calculator(method=method, device="cuda", auxiliary_basis="sto-3g")


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_COMPLETE_FORCE_TEST") != "1",
    reason="requires finite Slurm real-GPU allocation and native DF force owner",
)
def test_public_df_rccsdt_force_matches_public_energy_difference() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calc = Calculator(
        method="df-rccsd(t)",
        basis="sto-3g",
        auxiliary_basis="sto-3g",
        device="cuda",
        correlation_memory_budget_bytes=1 << 30,
        ccsd_max_iterations=100,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
    )
    result = calc.singlepoint(atoms, properties=("energy", "forces"))
    assert result.converged and result.forces is not None
    assert result.correlation is not None
    assert result.correlation.response_absolute_residual < 1e-9

    force = np.asarray(result.forces)
    analytic = -float(force[1, 2])
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1.0, 1.0):
            moved = [
                ("H", (0.0, 0.0, -0.7)),
                ("H", (0.0, 0.0, 0.7 + sign * step)),
            ]
            energies.append(calc.singlepoint(moved, properties=("energy",)).energy)
        numeric = (energies[1] - energies[0]) / (2 * step)
        np.testing.assert_allclose(numeric, analytic, atol=3e-7, rtol=3e-7)
