"""Selected-inventory ledger admission; all checks are host-only metadata work."""

from __future__ import annotations

import ctypes
import random
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native, resources_native
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import (
    MAX_BYTES,
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourcePlan,
    ResourceRequest,
    _account,
    plan_resources,
)


@pytest.fixture(scope="module")
def library() -> Any:
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ledger_create_from_buffers_v1")
    return loaded


def selected_plan(estimates: tuple[ResourceEstimate, ...]) -> ResourcePlan:
    identity = ResourceIdentity(
        "test", "ledger", "cuda", "fp64", '{"shape": 1}', ("energy",), "fixed"
    )
    request = ResourceRequest(
        "hf", identity, (ResourceCandidate("fixed", "resident", estimates),)
    )
    return plan_resources((request,), ResourceBudget()).require_feasible()


@pytest.mark.parametrize("seed", range(80))
def test_native_limit_matches_independent_lifetime_oracle(
    library: Any, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    randomizer = random.Random(seed)
    device = randomizer.choice((0, 7, 2**31 - 1))
    estimates = [ResourceEstimate("device", 0, f"device:{device}", 0, 0)]
    for index in range(randomizer.randrange(1, 15)):
        first = randomizer.randrange(12)
        estimates.append(
            ResourceEstimate(
                f"buffer-{index}",
                randomizer.randrange(1000),
                randomizer.choice(("pinned", "pageable", f"device:{device}")),
                first,
                first + randomizer.randrange(5),
                accounting=randomizer.choice(("capacity_bound", "runtime_allowance")),
            )
        )
    plan = selected_plan(tuple(estimates))
    numeric = tuple(
        estimate for estimate in estimates if estimate.accounting != "runtime_allowance"
    )
    expected = _account(numeric)[0].get(f"device:{device}", 0)

    def forbidden_portable_accounting(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "discovered native ledger admission retried portable peaks"
        )

    monkeypatch.setattr(resources_native, "_account", forbidden_portable_accounting)
    ledger = NativeDeviceLedger(library, plan)
    try:
        assert ledger.device == device and ledger.limit == expected
        receipt = ledger.to_dict()
        assert receipt["capacity_admission"] == {
            "source": "native-device-ledger-plan-v1",
            "declared_device_peak": plan.peak_bytes.get(f"device:{device}", 0),
        }
        assert receipt["live_bytes"] == receipt["allocations"] == 0
        assert receipt["admission"]["limit_bytes"] == expected
    finally:
        ledger.close()


def test_selected_choice_and_noncoincident_allowances_are_preserved(
    library: Any,
) -> None:
    plan = selected_plan(
        (
            ResourceEstimate("cold", 64, "device:7", 0, 0),
            ResourceEstimate("later", 32, "device:7", 1, MAX_BYTES),
            ResourceEstimate(
                "library", 100, "device:7", 1, MAX_BYTES, accounting="runtime_allowance"
            ),
        )
    )
    request = plan.requests[0]
    cheaper = replace(
        request.candidates[0],
        name="cheaper",
        estimates=(ResourceEstimate("empty", 0, "device:0", 0, 0),),
    )
    plan = replace(
        plan, requests=(replace(request, candidates=(cheaper, *request.candidates)),)
    )
    ledger = NativeDeviceLedger(library, plan)
    try:
        assert ledger.limit == 64 and ledger.device == 7
        assert ledger.to_dict()["capacity_admission"]["declared_device_peak"] == 132
        assert plan.selections == (("hf", "fixed"),)
    finally:
        ledger.close()


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("amount", (0, 100))
def test_allowance_only_inventory_funds_no_numeric_storage(
    library: Any, legacy: bool, amount: int
) -> None:
    class LegacyLibrary:
        def __getattr__(self, name: str) -> Any:
            if name == "generativeqc_resource_ledger_create_from_buffers_v1":
                raise AttributeError(name)
            return getattr(library, name)

    plan = selected_plan(
        (
            ResourceEstimate(
                "opaque", amount, "device:0", 0, 0, accounting="runtime_allowance"
            ),
        )
    )
    ledger = NativeDeviceLedger(LegacyLibrary() if legacy else library, plan)
    try:
        assert ledger.limit == 0 and ledger.to_dict()["limit_bytes"] == 0
        assert ("capacity_admission" in ledger.to_dict()) != legacy
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "spaces", (("pageable",), ("device:0", "device:1"), ("device:2147483648",))
)
@pytest.mark.parametrize("legacy", (False, True))
def test_unsupported_or_out_of_domain_owner_does_not_create_a_ledger(
    library: Any, spaces: tuple[str, ...], legacy: bool
) -> None:
    class LoadedLibrary:
        def __getattr__(self, name: str) -> Any:
            if legacy and name == "generativeqc_resource_ledger_create_from_buffers_v1":
                raise AttributeError(name)
            return getattr(library, name)

    plan = selected_plan(
        tuple(ResourceEstimate(space, 1, space, 0, 0) for space in spaces)
    )
    with pytest.raises((ValueError, NotImplementedError)):
        NativeDeviceLedger(LoadedLibrary(), plan)


@pytest.mark.parametrize("status", (0, 1, 2, 3, 4, 99))
def test_native_rejection_never_uses_old_creation_or_portable_policy(
    status: int,
) -> None:
    class Reject:
        def __call__(self, *_args: Any) -> int:
            return status

    class Forbidden:
        def __call__(self, *_args: Any) -> Any:
            raise AssertionError(
                "rejected native admission invoked legacy ledger creation"
            )

    library = SimpleNamespace(
        generativeqc_resource_ledger_create_from_buffers_v1=Reject(),
        generativeqc_resource_ledger_create_v1=Forbidden(),
        generativeqc_resource_ledger_destroy_v1=Forbidden(),
        generativeqc_resource_ledger_bind_v1=Forbidden(),
        generativeqc_resource_ledger_read_v1=Forbidden(),
    )
    plan = selected_plan((ResourceEstimate("numeric", 8, "device:0", 0, 0),))
    expected = (
        MemoryError
        if status == 3
        else NotImplementedError
        if status == 4
        else ValueError
    )
    with pytest.raises(expected):
        NativeDeviceLedger(library, plan)


@pytest.mark.parametrize(
    "receipt", ((1, 8, 8), (2**31, 8, 8), (0, 9, 8), (0, 8, MAX_BYTES + 1))
)
def test_malformed_native_receipt_releases_unpublished_metadata(
    receipt: tuple[int, ...],
) -> None:
    released = []

    class Create:
        def __call__(
            self, _buffers: Any, _count: int, handle: Any, values: Any, _size: int
        ) -> int:
            ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 7
            values[:] = receipt
            return 0

    class Destroy:
        def __call__(self, handle: Any) -> None:
            released.append(handle.value)

    class Unused:
        def __call__(self, *_args: Any) -> int:
            return 0

    library = SimpleNamespace(
        generativeqc_resource_ledger_create_from_buffers_v1=Create(),
        generativeqc_resource_ledger_destroy_v1=Destroy(),
        generativeqc_resource_ledger_bind_v1=Unused(),
        generativeqc_resource_ledger_read_v1=Unused(),
    )
    plan = selected_plan((ResourceEstimate("numeric", 8, "device:0", 0, 0),))
    with pytest.raises(ValueError, match="invalid capacity receipt"):
        NativeDeviceLedger(library, plan)
    assert released == [7]
