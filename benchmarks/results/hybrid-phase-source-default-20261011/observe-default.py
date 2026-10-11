"""Observe completed endpoints without supplying a production density or oracle."""

import json
import os
from pathlib import Path
from typing import Any

from generativeqc.batch import PreparedBatch

from benchmarks import readme_omol25, readme_pbe0

original_execute = PreparedBatch.execute
original_check = readme_omol25.check_record
observed: dict[str, Any] = {}


def execute(owner: Any, *args: Any, **kwargs: Any) -> Any:
    result = original_execute(owner, *args, **kwargs)
    observed["item"] = result.items[0]
    return result


def check_record(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Require real cold reuse and prove warm/seeded source setup was skipped."""
    item = observed["item"]
    for field in (
        "initial_guess",
        "incremental_direct_jk",
        "precision",
        "executed_backend",
    ):
        row[field] = getattr(item, field)
    precision = row["precision"]
    assert precision["complete"] and precision["operator_inventory_complete"]
    operators = precision["operators"]
    assert operators and all(
        all(
            record[dtype] == "fp64"
            for dtype in ("storage", "compute", "accumulation", "reduction")
        )
        and record["arithmetic_mode"] == "strict"
        for record in operators
    )
    source = sum(
        record["count"]
        for record in operators
        if record["name"] == "coulomb_recurrence"
    )
    trace = [
        json.loads(line)
        for line in Path(os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"])
        .read_text()
        .splitlines()
    ]
    begin = next(
        index
        for index in range(len(trace) - 1, -1, -1)
        if trace[index]["event"] == "BEGIN" and trace[index]["name"] == "batch_execute"
    )
    batch = trace[begin]
    events = trace[begin:]
    work = {
        event["key"]: event["value"]
        for event in events
        if event["id"] == batch["id"] and event["event"] == "VALUE"
    }
    prepare_ids = {
        event["id"]
        for event in events
        if event["event"] == "BEGIN"
        and event["name"] == "cuda_ks_resident_values_prepare"
    }
    skip_ids = {
        event["id"]
        for event in events
        if event["event"] == "BEGIN" and event["name"] == "cuda_ks_resident_values_skip"
    }
    rebase_ids = {
        event["id"]
        for event in events
        if event["event"] == "BEGIN" and event["name"] == "cuda_ks_warm_energy_rebase"
    }
    rebase_actions = sum(
        event["value"]
        for event in events
        if event["event"] == "VALUE"
        and event["id"] in rebase_ids
        and event["key"] == "fock_actions"
    )
    actions = work.get("phase_resident_fock_actions", 0)
    value_bytes = work.get("phase_resident_value_bytes", 0)
    admitted = source > 0 and actions > 0 and value_bytes > 0
    row.update(
        source_values=source,
        source_fock_actions=actions,
        source_value_bytes=value_bytes,
        source_reuse=admitted,
        source_prepare_attempts=len(prepare_ids),
        source_setup_skips=len(skip_ids),
        warm_energy_rebase_focks=rebase_actions,
    )
    selection = os.environ.get("GENERATIVEQC_KS_RESIDENT_VALUES")
    selected = selection in (None, "auto")
    row["resident_selection"] = "unset" if selection is None else selection
    expected = (
        selected
        and row["phase"] == "cold"
        and os.environ.get("EXPECT_SOURCE_COLD", "1") == "1"
    )
    assert admitted == expected, row
    if expected:
        assert len(prepare_ids) == 1 and not skip_ids
        assert rebase_actions == len(rebase_ids) == 1
        assert row["fock_builds"] == actions + rebase_actions
        assert (
            sum(
                record["kind"] == "post_scf_fock"
                for record in precision["scf_fock_timeline"]
            )
            == 1
        )
        assert precision["post_scf_fock_builds"] == 1
        assert any(
            event["event"] == "VALUE"
            and event["id"] in prepare_ids
            and event["key"] == "selection"
            and event["value"] == "auto"
            for event in events
        )
    else:
        assert rebase_actions == 0 and not rebase_ids
        assert source == actions == value_bytes == 0
        assert not prepare_ids
        if selected:
            assert skip_ids
    if selected and row["phase"] in ("warm", "moved-warm"):
        assert row["fock_builds"] == 1, row
    assert row["fock_builds"] > 0
    return original_check(row, oracle)


PreparedBatch.execute = execute
readme_omol25.check_record = check_record
try:
    readme_pbe0.main()
finally:
    PreparedBatch.execute = original_execute
    readme_omol25.check_record = original_check
