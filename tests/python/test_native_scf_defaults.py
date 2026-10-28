"""Actual native policy queries without constructing a GPU execution context."""

from __future__ import annotations

import ctypes
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import Calculator, GridSpec, _native
from generativeqc import _scf_convergence as policy
from generativeqc import _scf_defaults_native as transport

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def query_library(required_native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Link the real adapter to the selected library's catalog and grid owners."""
    native = _native.load_library()
    library = Path(native._name).resolve()
    output = tmp_path_factory.mktemp("scf-defaults") / "query.so"
    required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_scf_defaults.cpp"],
        output,
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    return ctypes.CDLL(str(output))


@pytest.fixture
def library(query_library: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    native = _native.load_library()
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: native)
    monkeypatch.setattr(
        native,
        "generativeqc_scf_defaults_query_v1",
        query_library.generativeqc_scf_defaults_query_v1,
        raising=False,
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("selected policy retried Python default selection")

    monkeypatch.setattr(policy, "_compatibility_default_scf_controls", forbidden)
    monkeypatch.setattr(policy, "_standard_models", forbidden)
    return native


def metadata(library: Any, method: str) -> Any:
    """Resolve actual model inputs on CPU; ask only for hypothetical CUDA metadata."""
    calculator = Calculator(method=method, device="cpu")
    return SimpleNamespace(
        _library=library,
        _method=calculator._method,
        _ks_options=calculator._ks_options,
        _basis=calculator._basis,
        _device_name="cuda",
    )


@pytest.mark.parametrize(
    "method", ("rhf", "pbe-rks", "pbe0-rks", "b3lyp-rks", "r2scan-rks", "m06-2x-rks")
)
def test_native_defaults_use_complete_catalog_and_grid(
    library: Any, method: str
) -> None:
    calculator = metadata(library, method)
    assert policy.default_scf_controls(calculator) == {
        "energy_tolerance": 1e-9,
        "density_tolerance": 1e-5,
        "screening_tolerance": 1e-12,
    }
    calculator._device_name = "cpu"
    assert policy.default_scf_controls(calculator)["density_tolerance"] == 1e-8


@pytest.mark.parametrize("method", ("uhf", "mp2", "pbe-uks", "wb97m-v-rks"))
def test_unqualified_models_keep_strict_defaults(library: Any, method: str) -> None:
    assert (
        policy.default_scf_controls(metadata(library, method))["density_tolerance"]
        == 1e-8
    )


@pytest.mark.parametrize(
    "change", ("level", "radius", "partition", "coincident", "schedule", "identity")
)
def test_altered_model_or_grid_cannot_reuse_qualified_defaults(
    library: Any, change: str
) -> None:
    calculator = metadata(library, "pbe-rks")
    options = calculator._ks_options
    grid = options.grid
    if change == "level":
        grid = GridSpec.treutler(4)
    elif change == "radius":
        radii = list(grid.element_radii)
        z, value = radii[-1]
        radii[-1] = (z, math.nextafter(value, math.inf))
        grid = replace(grid, element_radii=tuple(radii))
    elif change == "partition":
        grid = replace(grid, partition_iterations=4)
    elif change == "coincident":
        grid = replace(
            grid,
            coincident_tolerance=math.nextafter(grid.coincident_tolerance, math.inf),
        )
    calculator._ks_options = SimpleNamespace(
        grid=grid,
        xc_schedule="host_unfused" if change == "schedule" else options.xc_schedule,
        method_ir=SimpleNamespace(identity="0" * 64)
        if change == "identity"
        else options.method_ir,
    )
    assert policy.default_scf_controls(calculator)["density_tolerance"] == 1e-8


def test_native_failure_and_malformed_receipts_are_authoritative(library: Any) -> None:
    calculator = metadata(library, "pbe-rks")
    calculator._ks_options = SimpleNamespace(
        grid=calculator._ks_options.grid,
        xc_schedule="device_fused",
        method_ir=SimpleNamespace(identity="not-a-model-identity"),
    )
    with pytest.raises(RuntimeError, match="invalid argument"):
        policy.default_scf_controls(calculator)
    calculator._library = SimpleNamespace(
        generativeqc_scf_defaults_query_v1=lambda *args: 0
    )
    with pytest.raises(RuntimeError, match="invalid receipt"):
        policy.default_scf_controls(calculator)


def test_absent_protocol_retains_the_explicit_compatibility_boundary() -> None:
    calculator = SimpleNamespace(_library=SimpleNamespace())
    assert transport.default_controls(calculator) is None
    assert policy.default_scf_controls(calculator)["density_tolerance"] == 1e-8
