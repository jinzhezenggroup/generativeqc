"""Native-free admission, API routing and budgets for optional CPU KS forces.

Native calls return fixed dummy results. These are compatibility/control-flow
regressions, not numerical qualification or performance measurements.
"""

from __future__ import annotations

import ctypes
import json
import typing
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from generativeqc import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    GridSpec,
    KsOptions,
    MethodCapabilities,
    ResourceBudget,
    _native,
    resources_native,
)
from generativeqc import batch as batch_module
from generativeqc import calculator as calculator_module
from generativeqc._cpu_force_resources import cpu_force_inventory
from generativeqc.batch import PreparedBatch
from generativeqc_compiler.method import (
    MethodSpec,
    original_nonlocal_correlation,
    resolve_method,
)

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
WATER = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7)), ("H", (1.6, 0.0, -0.5))]
ENERGY = frozenset({"energy"})
FORCES = frozenset({"energy", "forces"})


@pytest.fixture
def fake_native(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Use the real Python facade with no native library, compilation or SCF."""
    state = SimpleNamespace(fail=False, force_fail=False, force_calls=0, observed=[])

    def available(method: int, output: object) -> int:
        output._obj.value = 1
        return 0

    def handle(*args: object) -> int:
        args[-1]._obj.value = 1
        return 0

    def fill(output: object) -> None:
        output.energy = -1.0
        output.converged = 1
        output.executed_backend = _native.BACKEND_CPU_REFERENCE

    def execute_batch(
        handle: object, inputs: object, count: int, outputs: object, n: int
    ) -> int:
        for output in outputs:
            fill(output)
            output.status = _native.STATUS_NUMERICAL_FAILURE if state.fail else 0
        return 0

    def execute_calculation(handle: object, output: object) -> int:
        fill(output._obj)
        return 0

    library = SimpleNamespace(
        generativeqc_method_available=available,
        generativeqc_ks_options_version=Mock(return_value=1),
        generativeqc_ks_resource_inventory_version_v1=Mock(return_value=1),
        generativeqc_context_create=handle,
        generativeqc_context_destroy=Mock(return_value=0),
        generativeqc_system_destroy=Mock(return_value=0),
        generativeqc_batch_prepare=handle,
        generativeqc_batch_execute=Mock(side_effect=execute_batch),
        generativeqc_batch_destroy=Mock(return_value=0),
        generativeqc_batch_get_system_count=Mock(return_value=1),
        generativeqc_batch_get_last_fock_builds=Mock(
            return_value=_native.STATUS_NOT_IMPLEMENTED
        ),
        generativeqc_batch_get_scf_diagnostic=Mock(
            return_value=_native.STATUS_NOT_IMPLEMENTED
        ),
        generativeqc_calculation_prepare=handle,
        generativeqc_calculation_execute=Mock(side_effect=execute_calculation),
        generativeqc_calculation_destroy=Mock(return_value=0),
        generativeqc_calculation_get_scf_diagnostic=Mock(
            return_value=_native.STATUS_NOT_IMPLEMENTED
        ),
        generativeqc_status_message=Mock(return_value=b"dummy native status"),
    )
    state.library = library
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)

    def capabilities(method: str) -> MethodCapabilities:
        hf = method in ("rhf", "uhf")
        return MethodCapabilities(
            method,
            "hartree_fock" if hf else "density_functional",
            True,
            True,
            FORCES if hf else ENERGY,
        )

    monkeypatch.setattr(calculator_module, "method_capabilities", capabilities)
    monkeypatch.setattr(
        Calculator, "_create_native_system", lambda *args, **kwargs: ctypes.c_void_p(1)
    )
    monkeypatch.setattr(Calculator, "_precision_report", lambda *args: None)
    monkeypatch.setattr(
        Calculator, "_incremental_direct_jk_diagnostic", lambda *args: None
    )
    for module in (batch_module,):
        monkeypatch.setattr(module, "read_ks_diagnostic", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            module, "read_ks_transport_diagnostic", lambda *args, **kwargs: None
        )
    from generativeqc import ks_diagnostics

    monkeypatch.setattr(
        ks_diagnostics, "read_ks_diagnostic", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        ks_diagnostics, "read_ks_transport_diagnostic", lambda *args, **kwargs: None
    )

    def observe(
        library: object,
        plan: object,
        ledger: object,
        callback: object,
        **kwargs: object,
    ) -> tuple:
        state.observed.append(plan)
        return callback(), {"plan": plan.to_dict()}

    monkeypatch.setattr(resources_native, "observe_method_call", observe)

    def force(batch: PreparedBatch, index: int, atoms: object) -> tuple:
        state.force_calls += 1
        if state.force_fail:
            raise ValueError("injected force failure")
        return np.ones((len(atoms), 3)), {}

    state.force = force
    return state


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
@pytest.mark.parametrize("basis", ["sto-3g", "def2-svp"])
def test_energy_defaults_preserve_one_shot_and_batch_domains(
    fake_native: typing.Any,
    monkeypatch: typing.Any,
    method: typing.Any,
    basis: typing.Any,
) -> typing.Any:
    calculator = Calculator(
        method=method, basis=basis, resource_budget=ResourceBudget(host_bytes=128 << 20)
    )
    assert calculator.capabilities.supported_properties == FORCES
    force = Mock(side_effect=AssertionError("energy default entered force consumer"))
    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", force)
    for properties in (None, ENERGY):
        assert calculator.singlepoint(WATER, properties=properties).forces is None
        with calculator.prepare_batch([WATER]) as batch:
            assert (
                batch.execute(strict=True, properties=properties).items[0].forces
                is None
            )
            assert batch.resource_plan.requests[0].identity.observables == ("energy",)
    assert calculator.batch_singlepoint([WATER], strict=True).items[0].forces is None
    force.assert_not_called()


def test_explicit_small_force_routes_and_budget_transitions(
    fake_native: typing.Any, monkeypatch: typing.Any
) -> typing.Any:
    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", fake_native.force)
    calc = Calculator(
        method="pbe-rks", resource_budget=ResourceBudget(host_bytes=512 << 20)
    )
    assert calc.singlepoint(H2, properties=FORCES).forces is not None
    with calc.prepare_batch([H2]) as batch:
        for properties in (FORCES, ENERGY, FORCES):
            result = batch.execute(strict=True, properties=properties)
            assert (result.items[0].forces is not None) == (properties == FORCES)
            assert (
                set(batch.resource_plan.requests[0].identity.observables) == properties
            )
            assert (
                set(fake_native.observed[-1].requests[0].identity.observables)
                == properties
            )
        assert batch.execute(strict=True).items[0].forces is None
    assert fake_native.force_calls == 3


def test_force_upgrade_over_budget_preserves_plan_and_skips_replay(
    fake_native: typing.Any, monkeypatch: typing.Any
) -> typing.Any:
    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", fake_native.force)
    calc = Calculator(
        method="pbe-rks", resource_budget=ResourceBudget(host_bytes=128 << 20)
    )
    with calc.prepare_batch([H2]) as batch:
        original = batch.resource_plan
        with pytest.raises(MemoryError, match="CPU forces"):
            batch.execute(strict=True, properties=FORCES)
        assert batch.resource_plan is original
        fake_native.library.generativeqc_batch_execute.assert_not_called()
        assert batch.execute(strict=True).items[0].forces is None
    with pytest.raises(MemoryError):
        calc.singlepoint(H2, properties=FORCES)
    assert fake_native.force_calls == 0


@pytest.mark.parametrize("fail_force", [False, True])
def test_failed_replay_does_not_commit_output_plan(
    fake_native: typing.Any, monkeypatch: typing.Any, fail_force: typing.Any
) -> typing.Any:
    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", fake_native.force)
    calc = Calculator(
        method="pbe-rks", resource_budget=ResourceBudget(host_bytes=512 << 20)
    )
    with calc.prepare_batch([H2]) as batch:
        initial = batch.resource_plan
        fake_native.force_fail = fail_force
        fake_native.fail = not fail_force
        with pytest.raises(RuntimeError):
            batch.execute(strict=True, properties=FORCES)
        assert batch.resource_plan is initial
        assert (
            set(
                batch.resource_diagnostics["plan"]["requests"][0]["identity"][
                    "observables"
                ]
            )
            == FORCES
        )
        fake_native.fail = fake_native.force_fail = False
        batch.execute(strict=True, properties=FORCES)
        force_plan = batch.resource_plan
        fake_native.force_fail = True
        with pytest.raises(RuntimeError, match="injected force failure"):
            batch.execute(strict=True, properties=FORCES)
        assert batch.resource_plan is force_plan
        fake_native.force_fail = False
        fake_native.fail = True
        result = batch.execute(properties=ENERGY)
        assert not result.items[0].succeeded
        assert batch.resource_plan is force_plan
        assert (
            set(
                batch.resource_diagnostics["plan"]["requests"][0]["identity"][
                    "observables"
                ]
            )
            == ENERGY
        )
        fake_native.fail = False
        batch.execute(strict=True, properties=ENERGY)
        assert batch.resource_plan.requests[0].identity.observables == ("energy",)


def test_estimates_track_requested_outputs(fake_native: typing.Any) -> typing.Any:
    calc = Calculator(method="pbe-rks", basis="def2-svp")
    for properties in (None, ENERGY):
        plan = calc.estimate_resources(
            [WATER], properties=properties, budget=ResourceBudget(host_bytes=128 << 20)
        ).require_feasible()
        assert plan.requests[0].identity.observables == ("energy",)
    small = Calculator(method="pbe-rks")
    force = small.estimate_resources([H2], properties=FORCES)
    assert force.requests[0].identity.observables == ("energy", "forces")
    assert force.peak_bytes["host"] >= 256 << 20
    assert (
        small.estimate_resources(
            [H2], properties=FORCES, budget=ResourceBudget(host_bytes=128 << 20)
        ).status
        == "infeasible"
    )


@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks"])
@pytest.mark.parametrize("kind", ["vv10", "d3"])
def test_composed_contexts_do_not_advertise_rejected_forces(
    fake_native: typing.Any, method: typing.Any, kind: typing.Any
) -> typing.Any:
    spin = "polarized" if method.endswith("uks") else "unpolarized"
    graph = (
        resolve_method("PBE-D3(BJ)", spin=spin)
        if kind == "d3"
        else resolve_method(
            MethodSpec(
                "PBE+vv10",
                (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
                nonlocal_correlation=original_nonlocal_correlation("vv10"),
            ),
            spin=spin,
        )
    )
    calc = Calculator(
        method=graph,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=3, angular_polar=2, angular_azimuth=4)
        ),
    )
    assert calc.capabilities.supported_properties == ENERGY
    assert calc._default_properties() == ENERGY
    with pytest.raises(ValueError, match="does not support properties: forces"):
        calc.singlepoint(H2, properties=FORCES)


def test_existing_ecp_and_hf_defaults_stay_force_enabled(
    fake_native: typing.Any,
) -> typing.Any:
    ecp = BasisSet(
        "synthetic",
        (
            ElementBasis(1, (BasisShell(0, ("1",), (("1",),)),)),
            ElementBasis(
                11,
                (BasisShell(0, ("1",), (("1",),)),),
                ecp_core_electrons=10,
                ecp_data=json.dumps(
                    [
                        {
                            "ecp_type": "scalar_ecp",
                            "angular_momentum": [0],
                            "r_exponents": [2],
                            "gaussian_exponents": ["0.8"],
                            "coefficients": [["-2"]],
                        }
                    ]
                ),
            ),
        ),
        BasisProvenance("synthetic", "1", "CC0", "0" * 64),
    )
    for method, basis in (("pbe-rks", ecp), ("rhf", "sto-3g"), ("uhf", "sto-3g")):
        calc = Calculator(method=method, basis=basis)
        assert calc._default_properties() == FORCES
        assert calc._default_properties(batch=True) == FORCES


def test_explicit_force_numeric_caps_stay_closed(
    fake_native: typing.Any, monkeypatch: typing.Any
) -> typing.Any:
    calc = Calculator(method="pbe-rks", basis="def2-svp")
    from generativeqc_compiler import dft

    basis = SimpleNamespace(nao=25, natom=3, nprimitive=22)

    class AO:
        def __enter__(self) -> typing.Any:
            return basis

        def __exit__(self, *args: object) -> typing.Any:
            return None

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: AO())
    with (
        calc.prepare_batch([WATER]) as batch,
        pytest.raises(RuntimeError, match="dense-export domain exceeded"),
    ):
        batch.execute(strict=True, properties=FORCES)
    with pytest.raises(RuntimeError, match="dense-export domain exceeded"):
        calc.singlepoint(WATER, properties=FORCES)
    for shape in ((17, 3, 22), (2, 9, 6), (2, 2, 129)):
        with pytest.raises(ValueError, match="dense-export domain exceeded"):
            cpu_force_inventory(
                SimpleNamespace(nao=shape[0], natom=shape[1], nprimitive=shape[2]),
                grid_points=10,
                ecp_terms=0,
            )


@pytest.mark.parametrize("failure", [MemoryError, ValueError])
def test_constructor_failure_still_releases_handles(
    fake_native: typing.Any, failure: type[Exception]
) -> None:
    original = fake_native.library.generativeqc_batch_prepare

    def fail(*args: typing.Any) -> int:
        original(*args)
        raise failure("injected preparation failure")

    fake_native.library.generativeqc_batch_prepare = fail
    calculator = Calculator(method="pbe-rks", resource_budget=ResourceBudget())
    with pytest.raises(failure, match="injected preparation failure"):
        calculator.prepare_batch([H2])
    fake_native.library.generativeqc_batch_destroy.assert_called_once()
    fake_native.library.generativeqc_context_destroy.assert_called_once()


def test_supplied_force_plan_and_other_owners_survive_output_changes(
    fake_native: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from generativeqc_compiler.common.resources import plan_resources

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", fake_native.force)
    calculator = Calculator(method="pbe-rks")
    force_plan = calculator.estimate_resources([H2], properties=FORCES)
    unrelated = replace(force_plan.requests[0], name="unrelated")
    supplied = plan_resources(
        (*force_plan.requests, unrelated), ResourceBudget(host_bytes=1 << 30)
    )
    with calculator.prepare_batch([H2], resource_plan=supplied) as batch:
        assert batch.resource_plan is supplied
        batch.execute(strict=True, properties=ENERGY)
        assert batch.resource_plan.selections == supplied.selections
        assert batch.resource_plan.requests[1] == unrelated
        assert batch.resource_plan.requests[0].identity.observables == ("energy",)
        batch.execute(strict=True, properties=FORCES)
        assert batch.resource_plan == supplied


def test_existing_hybrid_and_fitted_defaults_remain_conservative(
    fake_native: typing.Any,
) -> None:
    hybrid = Calculator(method="pbe0-rks", ks_options=KsOptions(grid=GridSpec()))
    assert hybrid._default_properties() == FORCES
    assert hybrid._default_properties(batch=True) == FORCES
    for properties in (None, ENERGY, FORCES):
        assert hybrid.estimate_resources([H2], properties=properties).requests[
            0
        ].identity.observables == ("energy", "forces")
    fitted = Calculator(
        method="pbe-rks", density_fitting="cpu", auxiliary_basis="sto-3g"
    )
    assert fitted._default_properties() == FORCES
    assert fitted._default_properties(batch=True) == FORCES


@pytest.mark.parametrize("method,ao_order", [("lda-rks", 0), ("pbe-rks", 1)])
def test_preparation_and_force_replay_admit_only_requested_derivatives(
    fake_native: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    ao_order: int,
) -> None:
    calculator = Calculator(method=method)
    calls = []
    require_basis = calculator_module.require_basis

    def record(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        calls.append((kwargs["operator"], kwargs["derivative_order"]))
        return require_basis(*args, **kwargs)

    monkeypatch.setattr(calculator_module, "require_basis", record)
    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", fake_native.force)
    with calculator.prepare_batch([H2]) as batch:
        assert ("ao", ao_order) in calls
        assert all(order == 0 for operator, order in calls if operator != "ao")
        calls.clear()
        batch.execute(strict=True, properties=FORCES)
        assert all(
            (operator, 1) in calls
            for operator in ("overlap", "kinetic", "nuclear_attraction", "eri")
        )
    calls.clear()
    plan = calculator.estimate_resources([H2], properties=FORCES)
    with calculator.prepare_batch([H2], resource_plan=plan):
        assert ("eri", 1) in calls


def test_force_derivative_rejection_precedes_native_replay(
    fake_native: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = Calculator(method="pbe-rks")
    require_basis = calculator_module.require_basis

    def reject_derivative(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if kwargs["operator"] != "ao" and kwargs["derivative_order"]:
            raise NotImplementedError("injected unavailable force derivative")
        return require_basis(*args, **kwargs)

    monkeypatch.setattr(calculator_module, "require_basis", reject_derivative)
    with calculator.prepare_batch([H2]) as batch:
        with pytest.raises(NotImplementedError, match="unavailable force derivative"):
            batch.execute(strict=True, properties=FORCES)
        fake_native.library.generativeqc_batch_execute.assert_not_called()
        assert batch.execute(strict=True, properties=ENERGY).items[0].forces is None
