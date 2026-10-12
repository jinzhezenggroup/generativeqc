"""Metadata-only native force planning; no fake GPU arithmetic qualification."""

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    ResourceBudget,
    _native,
    resources_ks,
)
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import plan_resources

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
HE = [(2, (0.0, 0.0, 0.0))]


@pytest.fixture
def planning_library(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Double only allocation inventories and prepared capacity authentication."""
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})

    def state(*args: Any) -> int:
        args[-2][:] = (256, 512, 1024)
        return _native.STATUS_SUCCESS

    def quadrature(atoms: int, points: int, output: Any) -> int:
        output._obj.value = 128
        return _native.STATUS_SUCCESS

    def matrix(aos: int, output: Any) -> int:
        output._obj.value = 2048
        return _native.STATUS_SUCCESS

    receipts: list[tuple[int, ...]] = []
    descriptors: list[Any] = []
    limits: list[tuple[int, int]] = []

    def stationary(
        descriptor: Any, shape: Any, count: int, output: Any, size: int
    ) -> int:
        assert count == 7 and size == 9
        descriptors.append(descriptor._obj)
        atoms, aos = shape[0], shape[1]
        grid, provider, minimum, cached = 8192 * atoms, 4096, 64 * aos, 64 * aos + 4096
        receipt = (
            grid,
            provider,
            minimum,
            cached,
            grid + provider + minimum,
            grid + provider + cached,
            64,
            64,
            32,
        )
        receipts.append(receipt)
        output[:] = receipt
        return _native.STATUS_SUCCESS

    def prepared(batch: Any, index: int, output: Any, count: int) -> int:
        output[:] = receipts[index]
        return _native.STATUS_SUCCESS

    def limit(batch: Any, index: int, maximum: int) -> int:
        limits.append((index, maximum))
        return _native.STATUS_SUCCESS

    return SimpleNamespace(
        generativeqc_ks_resource_inventory_version_v1=lambda: 1,
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_resource_ks_cuda_v1=state,
        generativeqc_resource_quadrature_cuda_v1=quadrature,
        generativeqc_resource_ks_matrix_provider_cuda_v1=matrix,
        generativeqc_resource_ks_stationary_cuda_v1=stationary,
        generativeqc_ks_batch_stationary_cuda_capacity_v1=prepared,
        generativeqc_ks_batch_stationary_cuda_limit_v1=limit,
        receipts=receipts,
        descriptors=descriptors,
        limits=limits,
    )


@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe-uks", "pbe0-uks"))
def test_native_fleet_candidates_reserve_all_consumers(
    planning_library: SimpleNamespace, method: str
) -> None:
    request = resources_ks.ks_resource_request(
        [H2, HE, H2],
        method=method,
        backend="cuda",
        library=planning_library,
        ks_options=KsOptions(grid=GridSpec()) if "pbe0" in method else None,
    )
    assert not request.unsupported_reason
    cached = plan_resources([request], ResourceBudget()).require_feasible()
    assert dict(cached.selections)["ks"].endswith("-cached")
    minimum_candidate = next(
        item for item in request.candidates if item.name.endswith("-minimum")
    )
    cached_bytes = cached.peak_bytes["device"]
    saved = sum(receipt[5] - receipt[4] for receipt in planning_library.receipts)
    minimum = plan_resources(
        [request], ResourceBudget(device_bytes=cached_bytes - saved)
    ).require_feasible()
    assert dict(minimum.selections)["ks"] == minimum_candidate.name
    assert minimum.peak_bytes["device"] == cached_bytes - saved
    assert (
        plan_resources(
            [request], ResourceBudget(device_bytes=cached_bytes - saved - 1)
        ).status
        == "infeasible"
    )
    assert all(
        "staging cap" not in item.name
        for item in cached.estimates
        if item.space.startswith("device:")
    )
    inventory = json.loads(dict(minimum_candidate.decisions)["item_device_inventory"])
    assert len(inventory) == 3 and all(item["stationary"] for item in inventory)
    resources_ks.configure_native_cuda_stationary_batch(planning_library, None, minimum)
    assert planning_library.limits == [
        (index, receipt[4]) for index, receipt in enumerate(planning_library.receipts)
    ]
    for descriptor in planning_library.descriptors:
        options = descriptor.ks_options.contents
        assert options.spin_channels == (2 if method.endswith("uks") else 1)
        assert descriptor.precision_mode == _native.PRECISION_FP64


def test_mixed_admission_preserves_entire_legacy_fleet(
    planning_library: SimpleNamespace,
) -> None:
    admitted = planning_library.generativeqc_resource_ks_stationary_cuda_v1

    def query(descriptor: Any, shape: Any, count: int, output: Any, size: int) -> int:
        if shape[0] == 1:
            return _native.STATUS_NOT_IMPLEMENTED
        return admitted(descriptor, shape, count, output, size)

    planning_library.generativeqc_resource_ks_stationary_cuda_v1 = query
    request = resources_ks.ks_resource_request(
        [H2, HE], backend="cuda", library=planning_library
    )
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    assert len(request.candidates) == 1
    assert any(item.bytes == 512 << 20 for item in plan.estimates)
    resources_ks.configure_native_cuda_stationary_batch(planning_library, None, plan)
    assert not planning_library.limits


def test_missing_query_preserves_legacy_semilocal_and_rejects_hybrid(
    planning_library: SimpleNamespace,
) -> None:
    del planning_library.generativeqc_resource_ks_stationary_cuda_v1
    assert not resources_ks.ks_resource_request(
        [H2], backend="cuda", library=planning_library
    ).unsupported_reason
    assert resources_ks.ks_resource_request(
        [H2],
        method="pbe0-rks",
        backend="cuda",
        library=planning_library,
        ks_options=KsOptions(grid=GridSpec()),
    ).unsupported_reason


def test_prepared_capacity_mismatch_is_transactional(
    planning_library: SimpleNamespace,
) -> None:
    request = resources_ks.ks_resource_request(
        [H2, HE], backend="cuda", library=planning_library
    )
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    planning_library.receipts[1] = tuple(
        value + 1 for value in planning_library.receipts[1]
    )
    with pytest.raises(ValueError, match="differs from the plan"):
        resources_ks.configure_native_cuda_stationary_batch(
            planning_library, None, plan
        )
    assert not planning_library.limits


def test_stationary_reservations_remain_inside_native_ledger_limit(
    planning_library: SimpleNamespace,
) -> None:
    request = resources_ks.ks_resource_request(
        [H2, HE], backend="cuda", library=planning_library
    )
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    actual = Calculator()._library
    ledger = NativeDeviceLedger(actual, plan, owner="ks")
    try:
        assert ledger.limit == plan.peak_bytes["device"] - 2 * 2048
        receipt = ledger.to_dict()
        assert receipt["allocations"] == 0 and receipt["requested_bytes"] == 0
        assert receipt["admission"]["limit_bytes"] == ledger.limit
        assert receipt["admission"]["provider_reserved_bytes"] == 0
        assert receipt["admission"]["combined_peak_bytes"] == 0
    finally:
        ledger.close()


def test_unbuilt_descriptor_capacity_preserves_output_and_validates_abi() -> None:
    """Exercise the compiled host ABI, not the metadata doubles above."""
    calculator = Calculator(method="pbe-rks")
    library = calculator._library
    query = library.generativeqc_resource_ks_stationary_cuda_v1
    query.argtypes = [
        ctypes.POINTER(_native.MethodDescriptor),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    descriptor = calculator._method_descriptor(systems=[H2])
    shape = (ctypes.c_uint64 * 7)(2, 2, 6, 2 * 24 * 12 * 24, 0, 0, 0)
    output = (ctypes.c_uint64 * 9)(*[991] * 9)
    assert (
        query(ctypes.byref(descriptor), shape, 7, output, 9)
        == _native.STATUS_NOT_IMPLEMENTED
    )
    assert list(output) == [991] * 9
    descriptor.abi_version += 1
    assert (
        query(ctypes.byref(descriptor), shape, 7, output, 9)
        == _native.STATUS_ABI_MISMATCH
    )
    descriptor.abi_version -= 1
    shape[0] = 0
    assert (
        query(ctypes.byref(descriptor), shape, 7, output, 9)
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert list(output) == [991] * 9


def test_cpu_prepared_capacity_is_unqualified_and_transactional() -> None:
    calculator = Calculator(method="pbe-rks")
    library = calculator._library
    query = library.generativeqc_ks_batch_stationary_cuda_capacity_v1
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    limit = library.generativeqc_ks_batch_stationary_cuda_limit_v1
    limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    limit.restype = ctypes.c_int
    with calculator.prepare_batch([H2]) as batch:
        output = (ctypes.c_uint64 * 9)(*[992] * 9)
        assert query(batch._batch, 0, output, 9) == _native.STATUS_NOT_IMPLEMENTED
        assert limit(batch._batch, 0, 1 << 20) == _native.STATUS_NOT_IMPLEMENTED
        assert query(batch._batch, 1, output, 9) == _native.STATUS_INVALID_ARGUMENT
        assert query(batch._batch, 0, output, 8) == _native.STATUS_INVALID_ARGUMENT
        assert list(output) == [992] * 9
        assert batch.execute(properties=("energy",), strict=True).items[0].succeeded
