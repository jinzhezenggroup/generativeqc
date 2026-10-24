"""Host tests for the observation-only complete PBE0 experiment adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from benchmarks import selective_eri_pbe0 as endpoint

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("invalid", [None, "incomplete", "fp32", "fast"])
def test_adapter_preserves_solver_and_independent_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str | None
) -> None:
    """Only observation changes; replay source receipts are not inferred gains."""
    trace = tmp_path / "trace.jsonl"
    events = [
        {"id": 1, "event": "BEGIN", "name": "batch_execute"},
        {"id": 2, "event": "BEGIN", "name": "cuda_ks_resident_values_prepare"},
        {"id": 2, "event": "VALUE", "key": "source_values_completed", "value": 5},
        {"id": 2, "event": "END", "status": "admitted"},
        {
            "id": 1,
            "event": "VALUE",
            "key": "phase_resident_fock_actions",
            "value": 12,
        },
        {
            "id": 3,
            "event": "VALUE",
            "key": "phase_direct_values_per_action",
            "value": 7,
        },
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", "5461")
    precision = {
        "complete": invalid != "incomplete",
        "operator_inventory_complete": True,
        "operators": [
            {
                "storage": "fp32" if invalid == "fp32" else "fp64",
                "compute": "fp64",
                "accumulation": "fp64",
                "reduction": "fp64",
                "arithmetic_mode": "fast" if invalid == "fast" else "strict",
            }
        ],
    }
    item = SimpleNamespace(precision=precision)
    returned = SimpleNamespace(items=[item])
    solver_calls, oracle_calls = [], []
    owner = object()
    oracle = object()
    row = {"phase": "cold"}

    def execute(actual_owner: object, *args: object, **kwargs: object) -> object:
        solver_calls.append((actual_owner, args, kwargs))
        return returned

    def check_record(actual_row: object, actual_oracle: object) -> dict:
        oracle_calls.append((actual_row, actual_oracle))
        return {"gate": "passed"}

    def benchmark() -> None:
        assert endpoint.PreparedBatch.execute(owner, "forces", warm=True) is returned
        assert endpoint.readme_omol25.check_record(row, oracle) == {"gate": "passed"}

    monkeypatch.setattr(endpoint.PreparedBatch, "execute", execute)
    monkeypatch.setattr(endpoint.readme_omol25, "check_record", check_record)
    monkeypatch.setattr(endpoint.readme_pbe0, "main", benchmark)
    if invalid:
        with pytest.raises(RuntimeError, match="precision|FP64"):
            endpoint.main()
        assert not oracle_calls
    else:
        endpoint.main()
        assert oracle_calls == [(row, oracle)]
        assert row["source_work"] == {"phase_resident_fock_actions": 12}
        assert len(row["source_receipts"]) == 2
        assert row["canonical_work_receipts"] == [events[-1]]
        assert row["source_selection"]["GENERATIVEQC_KS_RESIDENT_BUCKET_MASK"] == "5461"
    assert solver_calls == [(owner, ("forces",), {"warm": True})]
    assert endpoint.PreparedBatch.execute is execute
    assert endpoint.readme_omol25.check_record is check_record
