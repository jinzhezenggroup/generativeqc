from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeqc_compiler.xc import retained_public_evidence as retained


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rejects_incomplete_matrix(tmp_path: Path) -> None:
    _write(
        tmp_path / "summary.json",
        {
            "schema": retained.MATRIX_SCHEMA,
            "complete": False,
            "functionals": [],
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        retained.load_retained_public_evidence(tmp_path)


def test_rejects_stale_capability_before_stage_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Capability:
        identity = "current"

    monkeypatch.setattr(retained, "functional_capability", lambda *args, **kwargs: Capability())
    _write(
        tmp_path / "summary.json",
        {
            "schema": retained.MATRIX_SCHEMA,
            "complete": True,
            "functionals": [
                {
                    "name": "GGA_X_FAKE",
                    "status": "pass",
                    "capability_identity": "stale",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="stale"):
        retained.load_retained_public_evidence(tmp_path)


def test_loads_only_revalidated_public_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Capability:
        identity = "identity"
        public_dft = True

    monkeypatch.setattr(retained, "functional_capability", lambda *args, **kwargs: Capability())
    name = "GGA_X_FAKE"
    _write(
        tmp_path / "summary.json",
        {
            "schema": retained.MATRIX_SCHEMA,
            "complete": True,
            "functionals": [
                {
                    "name": name,
                    "status": "pass",
                    "capability_identity": "identity",
                },
                {
                    "name": "GGA_X_BLOCKED",
                    "status": "blocked",
                    "capability_identity": "unused",
                },
            ],
        },
    )
    for stage in retained.PUBLIC_STAGES:
        _write(
            tmp_path / name / f"{stage}.json",
            {
                "stage_evidence": {
                    "stage": stage,
                    "status": "pass",
                }
            },
        )

    result = retained.load_retained_public_evidence(tmp_path)
    assert tuple(result) == (name,)
    assert tuple(result[name]) == retained.PUBLIC_STAGES
