"""Automatic selection consumes native policy, rather than reimplementing it."""

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from generativeqc import Calculator, InitialGuessSpec, _native
from generativeqc.initial_guess import (
    _native_automatic_minao_eligible,
    _uses_native_automatic_cpu,
    automatic_initial_guess_candidate,
    initial_guess_for_systems,
    with_initial_guess_resources,
)

DEFAULT_QUERY = "generativeqc_initial_guess_default_kind_v1"


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


class SystemPolicyQuery(PolicyQuery):
    """Record population arguments without computing an eligibility decision."""

    def __call__(
        self,
        request_pointer: object,
        electron_count: int,
        multiplicity: int,
        output_pointer: object,
    ) -> int:
        status = super().__call__(request_pointer, output_pointer)
        self.requests[-1].update(
            electron_count=electron_count, multiplicity=multiplicity
        )
        return status


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


@pytest.mark.parametrize(
    "kind,expected", [(0, (None, 0)), (4, (InitialGuessSpec("minao"), 4))]
)
def test_model_selection_uses_native_result(
    kind: int, expected: tuple[InitialGuessSpec | None, int]
) -> None:
    query = PolicyQuery(kind)
    calc = calculator(PolicyQuery(99))
    setattr(calc._library, DEFAULT_QUERY, query)
    assert (
        automatic_initial_guess_candidate(calc, restricted=True, has_ecp=False)
        == expected
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


def test_missing_eligibility_query_does_not_reselect_in_python() -> None:
    calc = calculator(PolicyQuery(3))
    calc._library = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="policy lacks.*auto_kind_v1"):
        initial_guess_for_systems(calc, [[("He", (0, 0, 0))]])


def test_unknown_policy_result_does_not_silently_fall_back() -> None:
    with pytest.raises(RuntimeError, match="unsupported native automatic"):
        initial_guess_for_systems(calculator(PolicyQuery(99)), [[("He", (0, 0, 0))]])


def test_failed_query_does_not_silently_fall_back() -> None:
    query = PolicyQuery(0, _native.STATUS_INVALID_ARGUMENT)
    calc = calculator(query)
    calc._library.generativeqc_status_message = lambda status: b"invalid policy"
    with pytest.raises(RuntimeError, match="invalid policy"):
        initial_guess_for_systems(calc, [[("He", (0, 0, 0))]])


def test_automatic_requires_the_combined_native_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _native.load_library(device="cpu")
    query = PolicyQuery(3)
    monkeypatch.setattr(library, DEFAULT_QUERY, None, raising=False)
    monkeypatch.setattr(library, "generativeqc_initial_guess_auto_kind_v1", query)
    monkeypatch.setattr(_native, "load_library", lambda *args, **kwargs: library)
    with pytest.raises(NotImplementedError, match=DEFAULT_QUERY):
        Calculator("rhf", "sto-3g", device="cpu")
    calc = Calculator("rhf", "sto-3g", device="cpu", initial_guess=None)
    assert calc.initial_guess is None
    assert not calc._method_descriptor(systems=[[("He", (0, 0, 0))]]).initial_guess
    assert not query.requests


@pytest.mark.parametrize("kind,provider", [(0, None), (4, "minao"), (6, "huckel")])
def test_public_calculator_uses_native_default_transport(
    monkeypatch: pytest.MonkeyPatch, kind: int, provider: str | None
) -> None:
    library = _native.load_library(device="cpu")
    if not hasattr(library, DEFAULT_QUERY):
        pytest.skip("requires the combined native model policy")
    policy = PolicyQuery(kind)
    monkeypatch.setattr(library, DEFAULT_QUERY, policy)

    def forbidden(*_args: object) -> int:
        pytest.fail("a combined native decision must not restart provider discovery")

    monkeypatch.setattr(library, "generativeqc_initial_guess_auto_kind_v1", forbidden)
    monkeypatch.setattr(library, "generativeqc_initial_guess_auto_kind_v2", forbidden)
    monkeypatch.setattr(
        library, "generativeqc_initial_guess_capabilities_v1", forbidden
    )
    monkeypatch.setattr(_native, "load_library", lambda *args, **kwargs: library)
    calc = Calculator("rhf", "sto-3g", device="cpu")
    descriptor = calc._method_descriptor(systems=[[("He", (0, 0, 0))]])
    if provider is None:
        assert calc.initial_guess is None and not descriptor.initial_guess
    else:
        assert calc.initial_guess.kind == provider
        assert descriptor.initial_guess.contents.kind == kind
    assert len(policy.requests) == 1 and policy.requests[0]["atomic_numbers"] == ()
    explicit = Calculator(
        "rhf", "sto-3g", device="cpu", initial_guess=InitialGuessSpec("minao")
    )
    assert (
        explicit._method_descriptor(
            systems=[[("He", (0, 0, 0))]]
        ).initial_guess.contents.kind
        == 3
    )
    assert len(policy.requests) == 1


@pytest.mark.parametrize(
    "kind,status", [(99, 0), (5, 0), (0, _native.STATUS_INVALID_ARGUMENT)]
)
def test_invalid_combined_policy_does_not_enter_legacy_discovery(
    kind: int, status: int
) -> None:
    legacy = PolicyQuery(3)
    calc = calculator(legacy)
    setattr(calc._library, DEFAULT_QUERY, PolicyQuery(kind, status))
    calc._library.generativeqc_status_message = lambda _status: b"invalid policy"
    with pytest.raises(
        RuntimeError, match="invalid policy|unsupported native automatic"
    ):
        automatic_initial_guess_candidate(calc, restricted=True, has_ecp=False)
    assert not legacy.requests


@pytest.mark.parametrize("kind,expected", [(0, False), (3, True)])
def test_system_population_is_marshaled_without_python_reselection(
    kind: int, expected: bool
) -> None:
    query = SystemPolicyQuery(kind)
    calc = calculator(PolicyQuery(3))
    calc._library.generativeqc_initial_guess_auto_system_kind_v1 = query
    assert (
        _native_automatic_minao_eligible(
            calc,
            restricted=True,
            has_ecp=False,
            atomic_numbers=(2,),
            electron_count=2,
            multiplicity=1,
        )
        is expected
    )
    assert query.requests[0]["electron_count"] == 2
    assert query.requests[0]["multiplicity"] == 1
    assert query.requests[0]["atomic_numbers"] == (2,)


def test_native_per_item_execution_does_not_preselect_the_shared_descriptor() -> None:
    query = PolicyQuery(0)
    calc = calculator(query)
    calc._native_automatic_initial_guess_kind = _native.INITIAL_GUESS_AUTO_CPU
    assert _uses_native_automatic_cpu(calc)
    assert (
        initial_guess_for_systems(calc, [[("He", (0, 0, 0))], [("Ca", (0, 0, 0))]])
        is calc._initial_guess
    )
    assert not query.requests
    assert calc._initial_guess.native(automatic_cpu=True).kind == 4
    calc._native_automatic_initial_guess_kind = _native.INITIAL_GUESS_MINAO
    assert not _uses_native_automatic_cpu(calc)


def test_missing_system_query_does_not_guess_capacity_policy() -> None:
    calc = calculator(PolicyQuery(3))
    with pytest.raises(NotImplementedError, match="policy lacks.*auto_system_kind_v1"):
        _native_automatic_minao_eligible(
            calc,
            restricted=True,
            has_ecp=False,
            atomic_numbers=(2,),
            electron_count=2,
            multiplicity=1,
        )


@pytest.mark.parametrize("version,transport", [(1, 4), (2, 6)])
def test_shape_v1_resource_planner_obeys_native_system_policy(
    monkeypatch: pytest.MonkeyPatch, version: int, transport: int
) -> None:
    """An eligible element/population must not override a native Hcore decision."""
    library = _native.load_library(device="cpu")
    monkeypatch.setattr(library, DEFAULT_QUERY, PolicyQuery(transport))
    monkeypatch.setattr(
        library, "generativeqc_resource_preliminary_cpu_shape_v2", None, raising=False
    )
    monkeypatch.setattr(_native, "load_library", lambda *args, **kwargs: library)
    calc = Calculator("rhf", "sto-3g", device="cpu")
    systems = [[("He", (0, 0, 0))]]
    target = calc._base_resource_request(systems, properties=("energy",))
    topology = json.loads(target.identity.topology)
    topology["items"][0]["orbital"]["ecp_terms"] = 1
    target = replace(
        target, identity=replace(target.identity, topology=json.dumps(topology))
    )
    query = SystemPolicyQuery(0)
    monkeypatch.setattr(
        calc._library, f"generativeqc_initial_guess_auto_system_kind_v{version}", query
    )
    request = with_initial_guess_resources(target, calc, systems, [0], [1])
    assert request.candidates == target.candidates
    assert json.loads(request.identity.schedule)["initial_guess_resolved_kinds"] == [0]
    assert query.requests[0]["has_ecp"] == 1
    assert query.requests[0]["electron_count"] == 2
    assert query.requests[0]["multiplicity"] == 1


@pytest.mark.parametrize("kind,status", [(99, 0), (0, _native.STATUS_INVALID_ARGUMENT)])
def test_invalid_system_query_does_not_become_compatibility(
    kind: int, status: int
) -> None:
    calc = calculator(PolicyQuery(3))
    calc._library.generativeqc_initial_guess_auto_system_kind_v1 = SystemPolicyQuery(
        kind, status
    )
    calc._library.generativeqc_status_message = lambda status: b"invalid policy"
    with pytest.raises(
        RuntimeError, match="invalid policy|unsupported native automatic"
    ):
        _native_automatic_minao_eligible(
            calc,
            restricted=True,
            has_ecp=False,
            atomic_numbers=(2,),
            electron_count=2,
            multiplicity=1,
        )
