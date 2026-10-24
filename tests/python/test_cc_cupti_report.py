"""Editable receipt assertions cannot promote a partial activity ledger."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import cc_cupti_report as reporting


@pytest.mark.parametrize(
    "phase", [11, reporting.OBSERVER_FENCE_ID, reporting.OBSERVER_FENCE_ID + 1, 0]
)
@pytest.mark.parametrize("kind", [10, 1, 38])
@pytest.mark.parametrize("assertions", [False, True])
def test_only_exact_wait_fence_is_observer_owned_and_no_ledger_is_qualified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: int,
    kind: int,
    assertions: bool,
) -> None:
    kinds = ("activity", "source", "graph", "legacy")
    traces = {
        name: SimpleNamespace(identity=(index, index), sha256=name)
        for index, name in enumerate(kinds)
    }
    traces["graph"].metadata = (0, 0, 0, 0, 0, 0, 1, 0)
    traces["legacy"].metadata = (0, 0, 0, 0, 0, 1, 28, 2)
    monkeypatch.setattr(
        reporting, "open_trace", lambda path: traces[path.name.split(".")[-2]]
    )
    source = {
        "pid": 1,
        "pid_starttime": 2,
        "phases": {11: {"words": (0, 0, 0, 0, 1)}},
        "root": {"start": 20, "end": 80},
    }
    monkeypatch.setattr(reporting, "source_inventory", lambda _: source)
    monkeypatch.setattr(
        reporting, "validate_cc_census", lambda *args: {"scope": "synthetic"}
    )
    row = (kind, 0, 0, 30, 40, 5)
    activity = {
        "gpu": [] if kind == 38 else [row],
        "waits": [row] if kind == 38 else [],
        "external": {(1, 5): phase},
        "queued_completed": "UNAVAILABLE",
    }
    monkeypatch.setattr(reporting, "activity_inventory", lambda _: activity)
    receipt = {
        "schema": "generativeqc.cc-cupti-caller-receipt.v1",
        "cupti_parent_window": {
            "start_ns": 10,
            "end_ns": 100,
            "clock": "cupti28-normalized-default-ns",
        },
        "artifacts": {name: {"sha256": name} for name in kinds},
        "native_pid": 1,
        "native_starttime": 2,
        "allowed_triples_lowerings": [0],
        "cupti_abi": {"custom0": 1, "concurrent_kernel": 10, "memset": 2},
        "supervised_child_wall": {"scope": "synthetic"},
        **dict.fromkeys(
            (
                "allocation_verified",
                "same_host_clock_verified",
                "device_clock_calibration_passed",
                "independent_science_passed",
                "original_work_gates_passed",
                "observer_fence_verified",
                "independent_tail_verified",
            ),
            assertions,
        ),
    }
    result = reporting.report(tmp_path / "capture", {}, (1, 1, 1), receipt)
    assert result["campaign_science"].startswith("INCOMPLETE")
    assert result["whole_endpoint_attribution"].startswith("INCOMPLETE")
    assert result["preserved_unknown_activity"].get(
        "explicit_observer_region", 0
    ) == int(phase == reporting.OBSERVER_FENCE_ID and kind == 38)
    if phase not in (11, reporting.OBSERVER_FENCE_ID) or (
        phase == reporting.OBSERVER_FENCE_ID and kind != 38
    ):
        assert result["preserved_unknown_activity"]["unattributed_activity"] == 1
