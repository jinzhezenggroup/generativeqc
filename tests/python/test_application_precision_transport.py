"""Execute the private HF/KS request ABI without admitting unqualified arithmetic.

The host bridge consumes the real C++ parser, so Python layout, bounded reads,
validation and copied request lifetime are tested across the language boundary.
Real-device endpoint qualification is deliberately a separate gate.
"""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    PrecisionCalculator,
    TargetErrorBudget,
    _native,
)
from generativeqc.initial_guess import _initial_guess_precision

from tools.generativeqc_validation.native_registry_stub import install_registry_stub

ROOT = Path(__file__).resolve().parents[2]
BUDGET = TargetErrorBudget(1e-5, 1e-4, 3e-5)


@pytest.fixture(scope="module")
def parser(required_native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Retain an independently owned native request after marshalling expires."""
    folder = tmp_path_factory.mktemp("application-precision-transport")
    source = folder / "transport.cpp"
    source.write_text(r"""
#include "methods/application_precision.hpp"
namespace detail = generativeqc::methods::detail;
namespace scf = generativeqc::scf;
static scf::ScfOptions retained;
extern "C" std::size_t method_size() { return sizeof(generativeqc_method_descriptor); }
extern "C" std::size_t extended_size() { return sizeof(detail::ApplicationPrecisionMethodOptions); }
extern "C" int prepare(const generativeqc_method_descriptor* request) {
  try {
    retained.application_precision_budget = detail::application_precision_budget(*request);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const generativeqc::methods::MethodError& error) {
    return error.status();
  }
}
extern "C" double allowance(unsigned index) {
  return retained.application_precision_budget.at(index);
}
""")
    binary = folder / "transport.so"
    required_native_cxx.build_shared(
        [source],
        binary,
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    library = ctypes.CDLL(str(binary))
    library.prepare.argtypes = (ctypes.POINTER(_native.MethodDescriptor),)
    library.prepare.restype = ctypes.c_int
    library.allowance.argtypes = (ctypes.c_uint,)
    library.allowance.restype = ctypes.c_double
    for name in ("method_size", "extended_size"):
        getattr(library, name).argtypes = ()
        getattr(library, name).restype = ctypes.c_size_t
    return library


@pytest.fixture
def constructor(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Mock availability only; retain the actual public scientific constructor."""

    def available(method: int, pointer: object) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_application_precision_version_v1=lambda: 1,
        generativeqc_application_precision_version_v2=lambda: 2,
        generativeqc_method_available=available,
    )
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    install_registry_stub(library, frozenset({"energy", "forces"}))
    return library


def request(*, extended: bool = True) -> Any:
    """Construct storage large enough for exactly the declared private ABI."""
    descriptor_type = (
        _native.ApplicationPrecisionMethodDescriptor
        if extended
        else _native.MethodDescriptor
    )
    descriptor = descriptor_type()
    descriptor.struct_size = ctypes.sizeof(descriptor_type)
    descriptor.abi_version = _native.ABI_VERSION
    descriptor.method = _native.METHOD_RHF
    descriptor.precision_mode = _native.PRECISION_FP64
    return descriptor


def test_legacy_and_zero_extension_keep_strict_defaults(parser: Any) -> None:
    assert parser.method_size() == ctypes.sizeof(_native.MethodDescriptor)
    assert parser.extended_size() == ctypes.sizeof(
        _native.ApplicationPrecisionMethodDescriptor
    )
    assert parser.extended_size() == parser.method_size() + 3 * ctypes.sizeof(
        ctypes.c_double
    )
    for descriptor in (request(extended=False), request()):
        assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_SUCCESS
        assert [parser.allowance(index) for index in range(3)] == [0.0] * 3


def test_truncated_prefix_and_extension_are_rejected_before_reading(
    parser: Any,
) -> None:
    descriptor = request()
    for size in range(parser.extended_size()):
        if size == parser.method_size():
            continue
        descriptor.struct_size = size
        assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_ABI_MISMATCH
    descriptor.struct_size = parser.extended_size()
    descriptor.abi_version = _native.ABI_VERSION + 1
    assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_ABI_MISMATCH


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("invalid", (-1.0, float("nan"), float("inf")))
def test_invalid_allowance_never_changes_retained_request(
    parser: Any, index: int, invalid: float
) -> None:
    descriptor = request()
    descriptor.application_energy_budget = BUDGET.energy_abs
    assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_SUCCESS
    field = (
        "application_energy_budget",
        "application_max_force_budget",
        "application_rms_force_budget",
    )[index]
    setattr(descriptor, field, invalid)
    assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_INVALID_ARGUMENT
    assert parser.allowance(0) == BUDGET.energy_abs


def test_force_only_request_requires_energy_allowance(parser: Any) -> None:
    descriptor = request()
    descriptor.application_max_force_budget = BUDGET.force_max_abs
    assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_INVALID_ARGUMENT


@pytest.mark.parametrize("method", ("rhf", "pbe0-rks"))
@pytest.mark.parametrize("fitted", (False, True))
def test_real_descriptor_copies_budget_without_requiring_auto(
    parser: Any, constructor: Any, method: str, fitted: bool
) -> None:
    calculator = Calculator(
        method=method,
        device="cuda",
        initial_guess=None,
        precision="fp64",
        density_fitting="cuda" if fitted else None,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)) if method != "rhf" else None,
        _application_precision_budget=BUDGET,
    )
    descriptor = calculator._method_descriptor()
    del calculator
    gc.collect()
    assert type(descriptor) is _native.MethodDescriptor
    assert descriptor.struct_size == parser.extended_size()
    assert descriptor.precision_mode == _native.PRECISION_FP64
    assert bool(descriptor.density_fitting_mode) is fitted
    if method != "rhf":
        assert descriptor.ks_options.contents.semilocal_component_count > 0
    assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_SUCCESS
    descriptor._extension_owner.application_energy_budget = 10.0
    del descriptor
    gc.collect()
    assert tuple(parser.allowance(index) for index in range(3)) == (
        BUDGET.energy_abs,
        BUDGET.force_max_abs,
        BUDGET.force_rms,
    )


@pytest.mark.parametrize("version", (None, 1, 3))
def test_constructor_requires_verified_transport_version(
    constructor: Any, version: int | None
) -> None:
    if version is None:
        del constructor.generativeqc_application_precision_version_v2
    else:
        constructor.generativeqc_application_precision_version_v2 = lambda: version
    with pytest.raises(NotImplementedError, match="transport"):
        Calculator(
            device="cuda", initial_guess=None, _application_precision_budget=BUDGET
        )


def test_two_budget_owners_are_rejected(constructor: Any) -> None:
    with pytest.raises(ValueError, match="one method/KS transport owner"):
        Calculator(
            method="pbe0-rks",
            device="cuda",
            initial_guess=None,
            precision="auto",
            ks_options=KsOptions(
                grid=GridSpec(16, 8, 16), _precision_budget=(1e-5, 1e-4, 3e-5)
            ),
            _application_precision_budget=BUDGET,
        )


def test_budget_selects_strict_seed_without_changing_target_precision() -> None:
    calculator = SimpleNamespace(
        _device_name="cuda",
        _precision_mode=_native.PRECISION_AUTO,
        _application_precision_budget=BUDGET,
        _ks_options=None,
    )
    assert _initial_guess_precision(calculator) == _native.PRECISION_FP64
    assert calculator._precision_mode == _native.PRECISION_AUTO
    calculator._application_precision_budget = None
    assert _initial_guess_precision(calculator) == _native.PRECISION_AUTO


@pytest.mark.parametrize(
    "method",
    (
        "rhf",
        "pbe-rks",
        "pbe0-rks",
        "b3lyp-rks",
        "r2scan-rks",
        "m06-2x-rks",
        "wb97m-v-rks",
    ),
)
@pytest.mark.parametrize("fitted", (False, True))
def test_facade_v2_keeps_budget_out_of_strict_reference(
    constructor: Any, method: str, fitted: bool
) -> None:
    """Unavailable AUTO must not erase a derivative budget or narrow the model."""
    calculator = PrecisionCalculator(
        method=method,
        device="cuda",
        initial_guess=None,
        policy="fast",
        density_fitting="cuda" if fitted else None,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)) if method != "rhf" else None,
    )
    candidate = calculator._candidate
    assert candidate._application_precision_budget == BUDGET
    if method == "rhf" or fitted:
        assert calculator._candidate_precision == "fp64"
    reference = calculator._reference()
    assert reference._application_precision_budget is None
    assert reference._precision_mode == _native.PRECISION_FP64
    assert reference._density_fitting_mode == candidate._density_fitting_mode
    if method != "rhf":
        assert candidate.ks_options._precision_budget is None
        assert reference.ks_options._precision_budget is None
        assert (
            reference.ks_options.method_ir.identity
            == candidate.ks_options.method_ir.identity
        )


def test_facade_refuses_private_candidate_request_leaking_into_audits() -> None:
    with pytest.raises(ValueError, match="through policy"):
        PrecisionCalculator(policy="fast", _application_precision_budget=BUDGET)
    with pytest.raises(ValueError, match="through policy"):
        PrecisionCalculator(
            policy="fast", ks_options=KsOptions(_precision_budget=(1e-5, 1e-4, 3e-5))
        )


def test_resource_descriptor_uses_the_same_native_budget_parser(parser: Any) -> None:
    from generativeqc.ks import resolve_ks_options
    from generativeqc.resources_ks import _cuda_stationary_descriptor

    model = resolve_ks_options("pbe0-rks", KsOptions(grid=GridSpec(16, 8, 16)))
    descriptor = _cuda_stationary_descriptor(
        model,
        precision="fp64",
        max_iterations=100,
        diis_history=8,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
        application_precision_budget=BUDGET,
    )
    gc.collect()
    assert parser.prepare(ctypes.byref(descriptor)) == _native.STATUS_SUCCESS
    assert tuple(parser.allowance(index) for index in range(3)) == (
        BUDGET.energy_abs,
        BUDGET.force_max_abs,
        BUDGET.force_rms,
    )
    assert descriptor.ks_options.contents.semilocal_component_count > 0


def test_candidate_budget_changes_prepared_identity_and_reaches_resource_planning(
    constructor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import resources_ks

    settings = {
        "method": "pbe0-rks",
        "device": "cuda",
        "initial_guess": None,
        "ks_options": KsOptions(grid=GridSpec(16, 8, 16)),
    }
    strict = Calculator(**settings)
    fast = Calculator(**settings, _application_precision_budget=BUDGET)
    assert fast._model_signature() != strict._model_signature()
    captured = {}

    def request(systems: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(resources_ks, "ks_resource_request", request)
    fast._base_resource_request([[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]])
    assert captured["_application_precision_budget"] is BUDGET
    assert captured["precision"] == "fp64"
