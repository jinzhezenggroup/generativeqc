"""Fresh original replay can borrow a smaller tile without displacing the audit."""

from __future__ import annotations

import json
import typing

import numpy as np
import pytest
from test_df_cc_lambda import case, run
from test_df_cc_lambda import probe as _native_probe

if typing.TYPE_CHECKING:
    from pathlib import Path

probe = _native_probe


@pytest.mark.parametrize("q", [32, 35])
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("core_reuse", [False, True])
@pytest.mark.parametrize("serial_scratch_reuse", [False, True])
def test_smaller_replay_preserves_operator_cache_and_complete_response(
    probe: typing.Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    q: int,
    source: bool,
    core_reuse: bool,
    serial_scratch_reuse: bool,
) -> None:
    """At the audit-only budget, preserve every published adjoint and resource choice.

    A whole auxiliary batch and an unequal tail exercise independently prepared
    replay descriptors. The disabled-replay control supplies the actual native
    capacity, avoiding assumptions about the compiler's arena coloring policy.
    """
    # This shape refuses same-batch expansion even before admitting the core
    # cache, so both cache settings exercise the independent replay fallback.
    _, _, arrays = case(2, 12, q)
    common = {
        "source": source,
        "batch_large": True,
        "serial_scratch_reuse": serial_scratch_reuse,
        "core_reuse": core_reuse,
    }
    status, expected, values, control, error = run(probe, arrays, **common)
    assert status == 0, error
    assert control[24] == 1  # An independent matrix audit owns reusable scratch.
    budget = int(control[2])
    trace = tmp_path / "replay.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    status, actual, replay_values, candidate, error = run(
        probe, arrays, **common, primal_matrix=True, budget=budget
    )
    assert status == 0, error
    assert candidate[26] == 1
    for index in (2, 3, 13, 14, 20, 21, 24):
        assert candidate[index] == control[index]
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
    np.testing.assert_allclose(replay_values, values, atol=2e-11, rtol=0)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    batches = [
        row["value"] for row in rows if row.get("key") == "df_lambda_primal_batch_size"
    ]
    assert len(batches) == 1 and 1 <= batches[0] < candidate[14]


def test_audit_refusal_retains_scalar_original_replay(probe: typing.Any) -> None:
    """Independent tiling cannot create a new provider or evade audit admission."""
    _, _, arrays = case(2, 3, 5)
    common = {"source": True, "audit_matrix": False, "serial_scratch_reuse": True}
    status, expected, _, control, error = run(probe, arrays, **common)
    assert status == 0, error
    status, actual, _, candidate, error = run(
        probe, arrays, **common, primal_matrix=True, budget=int(control[2])
    )
    assert status == 0, error
    assert candidate[24] == candidate[26] == 0
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_distinct_replay_tail_is_charged_to_host_budget(
    probe: typing.Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An odd whole-Q operator needs an extra descriptor variant for Q/2 replay."""
    _, _, arrays = case(2, 24, 17)
    common = {
        "batch_large": True,
        "serial_scratch_reuse": True,
        "core_reuse": False,
    }
    status, expected, _, control, error = run(probe, arrays, **common)
    assert status == 0, error
    assert control[14] == 17 and control[24] == 1
    for slack in (0, 65536):
        trace = tmp_path / f"tail-{slack}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
        budget = int(control[2]) + slack
        status, actual, _, candidate, error = run(
            probe, arrays, **common, primal_matrix=True, budget=budget
        )
        assert status == 0, error
        assert candidate[26] == 1 and candidate[2] <= budget
        for index in (3, 13, 14, 20, 21, 24, 25):
            assert candidate[index] == control[index]
        for actual_array, expected_array in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        batches = [
            row["value"]
            for row in rows
            if row.get("key") == "df_lambda_primal_batch_size"
        ]
        assert len(batches) == 1
        if slack:
            assert 1 < batches[0] < 17
            assert candidate[2] > control[2]
        else:
            assert batches[0] == 1
            assert candidate[2] == control[2]
