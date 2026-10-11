"""Automatic selection consumes native policy, rather than reimplementing it."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
from generativeqc import Calculator, InitialGuessSpec, _native
from generativeqc.initial_guess import (
    _automatic_minao_model_eligible,
    _native_automatic_minao_eligible,
    initial_guess_for_systems,
)


class PolicyQuery:
    """Record exactly the ABI facts presented to an independent native owner."""

    def __init__(self, kind: int, status: int = 0) -> None:
        self.kind = kind
        self.status = status
        self.requests: list[dict] = []

    def __call__(self, request_pointer: object, output_pointer: object) -> int:
        request = ctypes.cast(
            request_pointer, ctypes.POINTER(_native.InitialGuessAutoPolicyDescriptor)
        ).contents
        self.requests.append(
            {
                "method": request.method,
                "backend": request.backend,
                "restricted": request.restricted,
                "has_ecp": request.has_ecp,
                "atomic_numbers": tuple(
                    request.atomic_numbers[index] for index in range(request.atom_count)
                ),
            }
        )
        ctypes.cast(
            output_pointer, ctypes.POINTER(ctypes.c_uint32)
        ).contents.value = self.kind
        return self.status


def calculator(query: PolicyQuery) -> SimpleNamespace:
    """Provide marshalling facts without a Python capability-selection authority."""
    return SimpleNamespace(
        _library=SimpleNamespace(generativeqc_initial_guess_auto_kind_v1=query),
        _method=_native.METHOD_RHF,
        _device_name="cpu",
        _precision_mode=_native.PRECISION_FP64,
        _density_fitting_mode=_native.DENSITY_FITTING_NONE,
        _initial_guess=InitialGuessSpec("minao"),
        _automatic_initial_guess=True,
        _basis="sto-3g",
    )


@pytest.mark.parametrize("kind,expected", [(0, False), (3, True)])
def test_model_selection_uses_native_result(kind: int, expected: bool) -> None:
    query = PolicyQuery(kind)
    assert (
        _automatic_minao_model_eligible(
            calculator(query), restricted=True, has_ecp=False
        )
        is expected
    )
    assert query.requests == [
        {
            "method": _native.METHOD_RHF,
            "backend": _native.BACKEND_CPU_REFERENCE,
            "restricted": 1,
            "has_ecp": 0,
            "atomic_numbers": (),
        }
    ]


def test_element_selection_does_not_override_native_policy() -> None:
    query = PolicyQuery(0)
    assert initial_guess_for_systems(calculator(query), [[("He", (0, 0, 0))]]) is None
    assert query.requests[0]["atomic_numbers"] == (2,)


def test_mixed_batch_retains_conservative_shared_descriptor() -> None:
    query = PolicyQuery(3)
    calc = calculator(query)
    assert (
        initial_guess_for_systems(calc, [[("H", (0, 0, 0))], [("He", (0, 0, 0))]])
        is calc._initial_guess
    )
    assert [request["atomic_numbers"] for request in query.requests] == [(1,), (2,)]


def test_ecp_facts_include_terms_without_removed_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import ecp

    monkeypatch.setattr(ecp, "resolve_ecp", lambda basis, atoms: ((0,), (object(),)))
    query = PolicyQuery(0)
    assert initial_guess_for_systems(calculator(query), [[("He", (0, 0, 0))]]) is None
    assert query.requests[0]["has_ecp"] == 1


def test_explicit_choice_does_not_query_automatic_policy() -> None:
    query = PolicyQuery(0)
    calc = calculator(query)
    calc._automatic_initial_guess = False
    assert initial_guess_for_systems(calc, [[("He", (0, 0, 0))]]) is calc._initial_guess
    assert not query.requests


def test_absent_query_is_only_compatibility_case() -> None:
    calc = calculator(PolicyQuery(3))
    calc._library = SimpleNamespace()
    assert (
        _native_automatic_minao_eligible(calc, restricted=True, has_ecp=False) is None
    )
    assert initial_guess_for_systems(calc, [[("He", (0, 0, 0))]]) is calc._initial_guess


def test_unknown_policy_result_does_not_silently_fall_back() -> None:
    with pytest.raises(RuntimeError, match="unsupported native automatic"):
        _automatic_minao_model_eligible(
            calculator(PolicyQuery(99)), restricted=True, has_ecp=False
        )


def test_failed_query_does_not_silently_fall_back() -> None:
    query = PolicyQuery(0, _native.STATUS_INVALID_ARGUMENT)
    calc = calculator(query)
    calc._library.generativeqc_status_message = lambda status: b"invalid policy"
    with pytest.raises(RuntimeError, match="invalid policy"):
        _automatic_minao_model_eligible(calc, restricted=True, has_ecp=False)


def test_public_calculator_does_not_reselect_native_declined_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _native.load_library(device="cpu")
    if not hasattr(library, "generativeqc_initial_guess_auto_kind_v1"):
        pytest.skip("native automatic-policy export required")
    query = PolicyQuery(0)
    monkeypatch.setattr(library, "generativeqc_initial_guess_auto_kind_v1", query)
    monkeypatch.setattr(_native, "load_library", lambda *args, **kwargs: library)
    calc = Calculator("rhf", "sto-3g", device="cpu")
    assert calc.initial_guess is None
    assert query.requests and query.requests[0]["method"] == _native.METHOD_RHF
    assert not calc._method_descriptor(systems=[[("He", (0, 0, 0))]]).initial_guess
