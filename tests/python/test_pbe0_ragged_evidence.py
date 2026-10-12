"""Ragged evidence must isolate scheduling from SCF algorithm experiments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import pbe0_xc_trajectories as trajectories
from benchmarks.pbe0_xc_trajectories import (
    PreparedBatchReceipts,
    check_record,
    prepared_snapshots,
    selection,
)


def test_ragged_selection_exercises_default_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both arms use normal SCF chunks; external settings cannot leak into A/B."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "7")
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_RAGGED_BATCH", "9")
    for enabled in (False, True):
        with selection(enabled, None, "ragged"):
            assert "GENERATIVEQC_CUDA_KS_CHUNK" not in os.environ
            assert os.environ.get("GENERATIVEQC_CUDA_XC_RAGGED_BATCH") == (
                None if enabled else "0"
            )
            assert os.environ["GENERATIVEQC_CUDA_XC_COMPACT_BATCH"] == "1"
        assert os.environ["GENERATIVEQC_CUDA_KS_CHUNK"] == "7"
        assert os.environ["GENERATIVEQC_CUDA_XC_RAGGED_BATCH"] == "9"


def test_batch_receipts_decode_completed_prepared_plans(tmp_path: Path) -> None:
    """Preparation receipts are metadata, not additive iteration counters."""
    common = {"name": "cuda_xc_batch_plan", "id": 4}
    events = [
        common | {"event": "BEGIN"},
        common | {"event": "VALUE", "key": "ragged", "value": 1},
        common | {"event": "VALUE", "key": "compact_nonempty_tiles", "value": 41},
        common | {"event": "END", "status": "returned"},
    ]
    path = tmp_path / "batch.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert prepared_snapshots(path, "cuda_xc_batch_plan") == [
        {"ragged": 1, "compact_nonempty_tiles": 41}
    ]


def test_failed_endpoint_without_forces_cannot_pass() -> None:
    """Failure evidence remains serializable and never masquerades as E+F."""
    reference = {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]], "converged": True}
    actual = {"energy": None, "forces": None, "converged": False, "status": 1}
    assert not check_record(actual, reference)["gate"]


def append_receipt(path: Path, identifier: int, *, ragged: bool, large: int) -> None:
    """Emit the actual cumulative receipt format without a native/GPU owner."""
    common = {"name": "cuda_xc_batch_plan", "id": identifier}
    events = [
        common | {"event": "BEGIN"},
        common | {"event": "VALUE", "key": "ragged", "value": int(ragged)},
        common | {"event": "VALUE", "key": "compact_large_tiles", "value": large},
        common | {"event": "END", "status": "returned"},
    ]
    previous = path.read_text() if path.exists() else ""
    path.write_text(previous + "".join(json.dumps(event) + "\n" for event in events))


def test_batch_receipts_are_bound_to_the_current_geometry(tmp_path: Path) -> None:
    """A moved fallback cannot borrow cold evidence; warm reuse remains legal."""
    path = tmp_path / "batch.jsonl"
    receipts = PreparedBatchReceipts()
    assert receipts.observe(path, 0) == []
    append_receipt(path, 1, ragged=True, large=41)
    cold = [{"ragged": 1, "compact_large_tiles": 41}]
    assert receipts.observe(path, 0) == cold
    assert receipts.observe(path, 0) == cold
    assert receipts.observe(path, 1) == []
    assert receipts.observe(path, 1) == []
    append_receipt(path, 2, ragged=True, large=43)
    moved = [{"ragged": 1, "compact_large_tiles": 43}]
    observed = receipts.observe(path, 1)
    assert observed == moved
    observed[0]["compact_large_tiles"] = 0
    assert receipts.observe(path, 1) == moved
    append_receipt(path, 3, ragged=False, large=0)
    assert receipts.observe(path, 1) == [{"ragged": 0, "compact_large_tiles": 0}]
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        receipts.observe(path, 1)


@pytest.mark.parametrize(
    "scenario",
    ("healthy", "moved-fallback", "failed-endpoint", "malformed-diagnostic"),
)
def test_main_retains_endpoint_and_geometry_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Run the real trajectory controller against a synchronous fake owner."""
    from generativeqc import _native

    scientific = {
        "geometries_bohr": [[[1, [0.0, 0.0, 0.0]]], [[1, [0.0, 0.0, 0.001]]]]
    }
    oracle = {
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]],
        "converged": True,
    }
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "stage": "complete",
                "protocol": scientific,
                "records": [
                    oracle | {"geometry": 0, "phase": "cold"},
                    oracle | {"geometry": 1, "phase": "moved"},
                ],
            }
        )
    )
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pbe0_ragged",
            "--atoms",
            "48",
            "--basis-file",
            str(reference),
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--feasibility",
            "--repeats",
            "1",
        ],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
            )
        ),
    )
    monkeypatch.setattr(
        _native, "load_library", lambda **kwargs: SimpleNamespace(_name=__file__)
    )
    monkeypatch.setattr(trajectories, "environment_metadata", lambda **kwargs: {})
    monkeypatch.setattr(trajectories, "cuda_accelerator_metadata", lambda cp: {})
    monkeypatch.setattr(trajectories, "scaling_cases", lambda: {"water-48": None})
    monkeypatch.setattr(
        trajectories, "load_comparison_basis", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(trajectories, "protocol", lambda *args, **kwargs: scientific)
    monkeypatch.setattr(
        trajectories,
        "read_trace",
        lambda path: [json.loads(line) for line in path.read_text().splitlines()],
    )
    monkeypatch.setattr(trajectories, "aggregate", lambda rows: {"calls": len(rows)})
    owners = []

    class Owner:
        def __init__(self) -> None:
            self.trace = os.environ.get("GENERATIVEQC_DF_PROGRESS_TRACE")
            self.ragged = os.environ.get("GENERATIVEQC_CUDA_XC_RAGGED_BATCH") != "0"
            self.calls = 0
            self.closed = False
            self.fail = bool(
                self.trace and self.ragged and scenario == "failed-endpoint"
            )
            if self.trace and not self.fail:
                append_receipt(
                    Path(self.trace),
                    1,
                    ragged=self.ragged,
                    large=41 if self.ragged else 0,
                )

        def execute(self, *args: object, **kwargs: object) -> SimpleNamespace:
            if self.trace and not self.fail:
                component = Path(self.trace + ".cuda")
                previous = component.read_text() if component.exists() else ""
                component.write_text(
                    "{invalid\n"
                    if self.ragged and scenario == "malformed-diagnostic"
                    else previous + json.dumps({"operation": "cuda_ks_xc"}) + "\n"
                )
                if self.calls == 2 and not (
                    self.ragged and scenario == "moved-fallback"
                ):
                    append_receipt(
                        Path(self.trace),
                        2,
                        ragged=self.ragged,
                        large=43 if self.ragged else 0,
                    )
            self.calls += 1
            item = SimpleNamespace(
                energy=None if self.fail else -1.0,
                forces=None if self.fail else trajectories.np.zeros((1, 3)),
                iterations=1,
                fock_builds=1,
                converged=not self.fail,
                status=9 if self.fail else 0,
                status_message="original native failure" if self.fail else "",
                warm_start_used=self.calls > 1,
                warm_start_fallback=False,
                ks_diagnostic=None if self.fail else SimpleNamespace(to_payload=dict),
            )
            return SimpleNamespace(items=[item])

        def close(self) -> None:
            self.closed = True

    class Calculator:
        def __init__(self, **kwargs: object) -> None:
            pass

        def prepare_batch(self, *args: object, **kwargs: object) -> Owner:
            owner = Owner()
            owners.append(owner)
            return owner

    monkeypatch.setattr(trajectories, "Calculator", Calculator)
    if scenario == "healthy":
        trajectories.main()
    elif scenario == "malformed-diagnostic":
        with pytest.raises(json.JSONDecodeError, match="Expecting property name"):
            trajectories.main()
    else:
        message = (
            "requested Ragged XC did not execute"
            if scenario == "moved-fallback"
            else r"independent E\+F gate failed"
        )
        with pytest.raises(AssertionError, match=message):
            trajectories.main()
    evidence = json.loads(output.read_text())
    assert all(owner.closed for owner in owners)
    diagnostic = [
        row
        for row in evidence["samples"]
        if row["diagnostic"] and row["arm"] == "ragged"
    ]
    if scenario == "healthy":
        assert evidence["stage"] == "complete"
        assert [
            row["batch_receipts"][0]["compact_large_tiles"] for row in diagnostic
        ] == [41, 41, 43, 43]
    elif scenario == "moved-fallback":
        assert evidence["stage"] == "ragged/1/moved"
        assert diagnostic[-1]["phase"] == "moved"
        assert diagnostic[-1]["batch_receipts"] == []
        assert diagnostic[-1]["gate"]
    elif scenario == "malformed-diagnostic":
        assert evidence["stage"] == "ragged/1/cold"
        assert len(diagnostic) == 1
        assert diagnostic[0]["gate"]
        assert "component_profile" not in diagnostic[0]
    else:
        assert evidence["stage"] == "ragged/1/cold"
        failed = diagnostic[-1]
        assert failed["status"] == 9
        assert failed["status_message"] == "original native failure"
        assert failed["forces"] is None
        assert failed["ks_diagnostic"] is None
        assert not failed["gate"]
        assert "component_profile" not in failed
