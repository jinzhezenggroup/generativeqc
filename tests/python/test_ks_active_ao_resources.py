"""Public dense KS reservations cannot fund optional retained local-AO maps.

Compile the actual native admission/accounting block and use the public planner
with the real bind/unbind/allocation ledger. CUDA allocation doubles do not
execute discovery, kernels, molecular forces or GPU performance qualification.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any

import pytest
import test_ks_point_batch_resources as point_budget
from generativeqc import ResourceBudget, resources_ks
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import plan_resources

native_probe = point_budget.native_probe
ACTIVE = "GENERATIVEQC_CUDA_KS_ACTIVE_AO"


def _selection(monkeypatch: pytest.MonkeyPatch, mode: str | None) -> None:
    if mode is None:
        monkeypatch.delenv(ACTIVE, raising=False)
    else:
        monkeypatch.setenv(ACTIVE, mode)


def _bind(probe: Any, ledger: NativeDeviceLedger) -> None:
    probe.generativeqc_resource_tracking_begin_v1.argtypes = [ctypes.c_uint]
    assert probe.generativeqc_resource_tracking_begin_v1(1) == 0
    assert probe.generativeqc_resource_ledger_bind_v1(ledger.handle) == 0


def _unbind(probe: Any) -> None:
    probe.generativeqc_resource_tracking_end_v1.argtypes = [
        ctypes.POINTER(ctypes.c_uint64)
    ] * 2
    peak, samples = ctypes.c_uint64(), ctypes.c_uint64()
    assert (
        probe.generativeqc_resource_tracking_end_v1(
            ctypes.byref(peak), ctypes.byref(samples)
        )
        == 0
    )


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("mode", [None, "0", "1"])
@pytest.mark.parametrize("count", [1, 2048])
@pytest.mark.parametrize("explicit_limit", [False, True])
def test_public_map_fallback_preserves_fleet_force_and_rebuild(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    mode: str | None,
    count: int,
    explicit_limit: bool,
) -> None:
    _selection(monkeypatch, mode)
    for name in point_budget.CONTROLS:
        monkeypatch.delenv(name, raising=False)
    request, library = point_budget._request(
        native_probe, monkeypatch, count, method="pbe-uks" if spins == 2 else "pbe-rks"
    )
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    if explicit_limit:
        plan = plan_resources(
            [request], ResourceBudget(device_bytes=plan.peak_bytes["device"])
        ).require_feasible()
    decisions = dict(request.candidates[0].decisions)
    assert decisions["xc_ao_selection"] == (
        "dense under the public device ledger; no optional map allowance"
    )
    row = json.loads(decisions["item_device_inventory"])[0]
    mandatory = sum(row[key] for key in ("state", "xc", "coulomb"))
    force, cycles = 512 << 20, 3
    ledger = NativeDeviceLedger(library, plan, owner="ks")
    try:
        result = (ctypes.c_uint64 * 6)()
        native_probe.active_fleet(
            ledger.handle, count, mandatory, force, spins, cycles, result
        )
        assert list(result) == [
            count * cycles,
            0,
            force,
            count * mandatory + force,
            0,
            0,
        ]
        assert ledger.to_dict()["live_bytes"] == 0
        _bind(native_probe, ledger)
        try:
            policy = (ctypes.c_uint64 * 9)()
            assert native_probe.active_policy(spins, False, policy) == 0
            requested = int(mode != "0")
            assert list(policy) == [
                requested,
                0,
                policy[2],
                policy[2],
                requested,
                0,
                0,
                49152 * 2 * 2,
                policy[2],
            ]
        finally:
            _unbind(native_probe)
        # Unbound preparation must recover the ordinary default. The ledger
        # object still exists, so this also rejects a process-global sticky veto.
        policy = (ctypes.c_uint64 * 9)()
        assert native_probe.active_policy(spins, False, policy) == 0
        assert list(policy[:2]) == [requested, requested]
        assert policy[3] - policy[2] == (3072 if requested else 0)
    finally:
        ledger.close()


@pytest.mark.parametrize("mode", [None, "0", "1", "", "yes", "2", "-1", "01"])
@pytest.mark.parametrize("host_unfused", [False, True])
def test_budget_fallback_does_not_bypass_control_or_capability_validation(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
    host_unfused: bool,
) -> None:
    _selection(monkeypatch, mode)
    request, library = point_budget._request(native_probe, monkeypatch, 1)
    ledger = NativeDeviceLedger(
        library, plan_resources([request], ResourceBudget()), owner="ks"
    )
    try:
        _bind(native_probe, ledger)
        try:
            result = (ctypes.c_uint64 * 9)()
            invalid = mode not in (None, "0", "1") or (mode == "1" and host_unfused)
            assert native_probe.active_policy(1, host_unfused, result) == int(invalid)
        finally:
            _unbind(native_probe)
    finally:
        ledger.close()


def test_map_diagnostic_does_not_add_a_cpu_policy() -> None:
    request = resources_ks.ks_resource_request([point_budget.H2])
    assert dict(request.candidates[0].decisions)["xc_ao_selection"] == "not applicable"
