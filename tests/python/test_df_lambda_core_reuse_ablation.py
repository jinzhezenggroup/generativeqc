"""No-GPU checks for parent-consistent experimental evidence admission."""

from __future__ import annotations

import json
import typing

import pytest

from benchmarks.df_lambda_core_reuse_ablation import trace_categories

if typing.TYPE_CHECKING:
    from pathlib import Path


def journal() -> list[dict]:
    """Synthetic immediate parents and nested, completed physical actions."""
    scopes = [
        (1, -1, "cc_lambda_complete", 10),
        (2, 1, "cc_lambda_initialization", 2),
        (3, 1, "cc_lambda_gmres", 4),
        (4, 3, "gmres_arnoldi_action", 1),
        (5, 3, "gmres_exact_residual_replay", 2),
        (6, 1, "cc_lambda_independent_audit", 1),
        (7, 1, "cc_lambda_parameter_factor_vjp", 2),
    ]
    rows = []
    for identity, parent, name, seconds in scopes:
        base = {"id": identity, "parent": parent, "name": name}
        rows.extend(
            [
                {**base, "event": "BEGIN", "elapsed_ms": 0},
                {**base, "event": "END", "elapsed_ms": seconds * 1000},
            ]
        )
    rows.append({"id": 4, "event": "VALUE", "key": "lambda_gemm_calls", "value": 12})
    return rows


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_nested_actions_are_not_added_to_the_lambda_parent(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    write_trace(path, journal())
    result = trace_categories(path)
    assert result["parent_seconds"] == 10
    assert sum(result["immediate_children_seconds"].values()) == 9
    assert result["unattributed_parent_seconds"] == 1
    assert [row["seconds"] for row in result["nested_physical_actions"]] == [1, 2]
    assert result["nested_physical_actions"][0]["work"] == {"lambda_gemm_calls": 12}


@pytest.mark.parametrize("missing", [1, 2, 4])
def test_incomplete_parent_phase_or_action_is_rejected(
    tmp_path: Path, missing: int
) -> None:
    rows = [
        row for row in journal() if not (row["id"] == missing and row["event"] == "END")
    ]
    path = tmp_path / "trace.jsonl"
    write_trace(path, rows)
    with pytest.raises(ValueError, match="incomplete"):
        trace_categories(path)


def test_overlapping_immediate_child_totals_are_rejected(tmp_path: Path) -> None:
    rows = journal()
    next(row for row in rows if row["id"] == 2 and row["event"] == "END")[
        "elapsed_ms"
    ] = 8000
    path = tmp_path / "trace.jsonl"
    write_trace(path, rows)
    with pytest.raises(ValueError, match="exceed"):
        trace_categories(path)
