"""KS retained-grid inventory, using real native shapes and host allocation doubles.

The fleet proof is the dense incumbent with optional panels disabled by the
public ledger. It does not qualify optional active-AO maps or GPU execution.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any

import pytest
import test_ks_point_batch_resources as point_budget
from generativeqc import ResourceBudget
from generativeqc.ks import KsOptions
from generativeqc.resources_native import NativeDeviceLedger
from generativeqc_compiler.common.resources import plan_resources

native_probe = point_budget.native_probe
_request = point_budget._request


def _shapes(
    probe: Any,
    points: int,
    spins: int,
    functional: int,
    tile: int,
    *,
    response: bool = False,
    mixed: bool = False,
) -> list[int]:
    output = (ctypes.c_uint64 * 3)()
    assert (
        probe.grid_shapes(points, spins, functional, tile, response, mixed, output) == 0
    )
    return list(output)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("functional", [0, 1])
@pytest.mark.parametrize(
    "points,tile", [(1, 256), (255, 256), (257, 256), (49152, 256)]
)
def test_native_ks_slot_counts_shared_grid_once(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    functional: int,
    points: int,
    tile: int,
) -> None:
    _, library = _request(native_probe, monkeypatch, 1)
    output = (ctypes.c_uint64 * 3)()
    assert (
        library.generativeqc_resource_ks_cuda_v1(
            2, 2, 2, 6, points, 8, spins, functional, tile, output, 3
        )
        == 0
    )
    direct, borrowed, grid = _shapes(native_probe, points, spins, functional, tile)
    assert grid == 5 * points * 8
    assert direct == borrowed + 4 * points * 8
    assert output[1] == borrowed + grid == direct + points * 8


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize(
    "functional,response,mixed",
    [
        (0, False, False),
        (1, False, False),
        (2, False, False),
        (0, True, False),
        (1, True, False),
        (0, False, True),
        (1, False, True),
    ],
)
def test_direct_response_and_optional_family_storage_remain_independent(
    native_probe: Any,
    spins: int,
    functional: int,
    response: bool,
    mixed: bool,
) -> None:
    points, tile, nao = 257, 256, 2
    direct, borrowed, grid = _shapes(
        native_probe, points, spins, functional, tile, response=response, mixed=mixed
    )
    jets = 1 if functional == 0 else 4
    work_jets = 4 if functional == 2 else 1
    features = (1, 4, 5)[functional]
    # Independent count of the established private arena, including tau/response
    # arrays. Neither this layout nor a standalone CPU-grid upload owns raw weights.
    expected = 8 * (
        3 * 2
        + 2 * 6
        + 16 * nao
        + (jets + spins * work_jets) * tile * nao
        + ((3 if response else 2) * spins * features + 3) * tile
        + spins * nao * nao
        + 4
    )
    assert borrowed == expected
    assert direct == expected + 4 * points * 8
    assert borrowed + grid == direct + points * 8


def test_native_grid_owner_keeps_fifth_array_until_last_borrower(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, library = _request(native_probe, monkeypatch, 1)
    plan = plan_resources([request], ResourceBudget()).require_feasible()
    ledger = NativeDeviceLedger(library, plan, owner="ks")
    try:
        output = (ctypes.c_uint64 * 3)()
        native_probe.grid_lifetime(ledger.handle, 49152, output)
        assert list(output) == [1966080, 1966080, 0]
        assert ledger.to_dict()["live_bytes"] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
def test_public_inventory_propagates_one_combined_slot(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    precision: str,
) -> None:
    options = {"method": method, "precision": precision}
    request, _ = _request(native_probe, monkeypatch, 3, **options)
    candidate = request.candidates[0]
    rows = json.loads(dict(candidate.decisions)["item_device_inventory"])
    _, borrowed, grid = _shapes(
        native_probe,
        49152,
        2 if method.endswith("uks") else 1,
        int(method.startswith("pbe")),
        256,
    )
    assert [row["xc"] for row in rows] == [borrowed + grid] * 3
    estimates = {item.name: item for item in candidate.estimates}
    assert estimates["all KS device xc"].bytes == 3 * (borrowed + grid)
    assert estimates["all KS device xc"].kind == "persistent"
    assert estimates["serialized KS transient device phase peak"].bytes == max(
        max(0, row["setup"] - sum(row[key] for key in ("state", "xc", "coulomb")))
        for row in rows
    )
    host, _ = _request(
        native_probe,
        monkeypatch,
        3,
        **options,
        ks_options=KsOptions(xc_schedule="host_unfused"),
    )
    host_rows = json.loads(dict(host.candidates[0].decisions)["item_device_inventory"])
    assert [row["xc"] for row in host_rows] == [0] * 3


@pytest.mark.parametrize("batch_bytes", [None, "0", "33554432"])
@pytest.mark.parametrize("spins", [1, 2])
def test_dense_fleet_preserves_full_force_reservation_and_explicit_caps(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    batch_bytes: str | None,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "0")
    if batch_bytes is None:
        monkeypatch.delenv("GENERATIVEQC_CUDA_XC_BATCH_BYTES", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_CUDA_XC_BATCH_BYTES", batch_bytes)
    count, points, force = 2048, 49152, 512 << 20
    request, library = _request(
        native_probe, monkeypatch, count, method="pbe-uks" if spins == 2 else "pbe-rks"
    )
    row = json.loads(dict(request.candidates[0].decisions)["item_device_inventory"])[0]
    _, borrowed, grid = _shapes(native_probe, points, spins, 1, 256)
    # Derive actual retention independently of the public XC slot under test.
    mandatory = row["state"] + borrowed + grid + row["coulomb"]
    unlimited = plan_resources([request], ResourceBudget()).require_feasible()
    peak = unlimited.peak_bytes["device"]
    for budget in (
        ResourceBudget(device_bytes=peak - 1),
        ResourceBudget(per_device_bytes=((0, peak - 1),)),
        ResourceBudget(device_bytes=3860795004),
    ):
        rejected = plan_resources([request], budget)
        assert rejected.status == "infeasible"
        with pytest.raises(MemoryError, match="no supported plan fits"):
            NativeDeviceLedger(library, rejected, owner="ks")
    exact = plan_resources(
        [request], ResourceBudget(device_bytes=peak)
    ).require_feasible()
    ledger = NativeDeviceLedger(library, exact, owner="ks")
    try:
        assert count * mandatory + force <= ledger.limit
        result = (ctypes.c_uint64 * 5)()
        native_probe.fleet(ledger.handle, count, mandatory, force, result)
        assert list(result) == [count, 0, force, count * mandatory + force, 0]
        assert ledger.to_dict()["live_bytes"] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize("extra_point", [0, 1])
def test_combined_native_inventory_rejects_overflow_transactionally(
    native_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    extra_point: int,
) -> None:
    _, library = _request(native_probe, monkeypatch, 1)
    # Reject both the combined sum and the five-array multiplication overflow.
    points = (2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1) // 40 + extra_point
    output = (ctypes.c_uint64 * 3)(17, 19, 23)
    assert (
        library.generativeqc_resource_ks_cuda_v1(
            2, 2, 2, 6, points, 8, 1, 1, 256, output, 3
        )
        == 1
    )
    assert list(output) == [17, 19, 23]
