"""Tool acceptance gates, not host substitutes for CUDA qualification."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks import ks_matrix_provider_qualification as qualification

if TYPE_CHECKING:
    from pathlib import Path


def sample() -> qualification.Case:
    return qualification.Case(2, 2, 2, False, True, False, True, "mixed", False, True)


def test_transpose_and_physical_broadcast_have_known_independent_values() -> None:
    case = sample()
    left = np.array([[[[1, 2], [3, 4]], [[2, 0], [1, 3]]]] * 2, dtype=float)
    right = np.array([[[[5, 6], [7, 8]]]] * 2, dtype=float)
    expected = qualification.oracle(case, left, right, np.array([1, 0]))
    np.testing.assert_array_equal(
        expected[0], [[[26, 30], [38, 44]], [[17, 20], [21, 24]]]
    )
    assert np.all(expected[1] == qualification.SENTINEL)
    # Distinct system/spin blocks remain contiguous; each 2x2 matrix is column major.
    np.testing.assert_array_equal(
        np.frombuffer(qualification.encode(left[:1]), dtype="<f8"),
        [1, 3, 2, 4, 2, 1, 0, 3],
    )


@pytest.mark.parametrize("damage", ["mask", "nan", "finite-error"])
def test_acceptance_rejects_mask_overwrite_and_bad_active_arithmetic(
    damage: str,
) -> None:
    case = sample()
    left, right, active = qualification.inputs(case)
    expected = qualification.oracle(case, left, right, active)
    actual = expected.astype(float)
    if damage == "mask":
        actual[1, 0, 0, 0] += 1.0
    elif damage == "nan":
        actual[0, 0, 0, 0] = np.nan
    else:
        actual[0, 0, 0, 0] += 1e-6
    assert qualification.assess(case, actual, expected, active)["status"] == "FAIL"


def test_failed_item_poison_does_not_enter_oracle_or_successful_neighbor() -> None:
    case = replace(sample(), mask="failed")
    left, right, active = qualification.inputs(case)
    assert np.isnan(left[-1]).all() and np.isnan(right[-1]).all()
    expected = qualification.oracle(case, left, right, active)
    assert np.isfinite(expected).all()
    assert (
        qualification.assess(case, expected.astype(float), expected, active)["status"]
        == "PASS"
    )


def test_protocol_contains_every_provider_pair_and_boundary() -> None:
    rows = qualification.cases()
    assert len(rows) == len(set(rows))
    assert {row.n for row in rows} == {3, 16, 17, 19, 97}
    assert {row.mask for row in rows} == {"active", "mixed", "inactive", "failed"}
    for row in rows:
        assert replace(row, library=not row.library) in rows
        assert replace(row, graph=not row.graph) in rows
    spin = [row for row in rows if not row.ordinary and row.spins == 2]
    assert {(row.left_spin, row.right_spin) for row in spin} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


@pytest.fixture
def tiny_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(qualification, "cases", lambda: (sample(),))
    folder = tmp_path / "run"
    qualification.prepare(folder)
    return folder


def write_observations(folder: Path, case_rows: list[dict]) -> None:
    rows = (
        [{"kind": "device", "ordinal": 0}]
        + case_rows
        + [
            {"kind": "owner", "probe": name, "pass": True}
            for name in ("small", "admit", "allocation", "memory-error", "capture")
        ]
    )
    (folder / "device.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_missing_device_case_is_incomplete(tiny_manifest: Path) -> None:
    write_observations(tiny_manifest, [])
    result = qualification.collect(tiny_manifest)
    assert result["status"] == "NOT_QUALIFIED"
    assert result["incomplete_count"] == 1
    assert result["complete_endpoint_qualified"] is False


def test_duplicate_device_case_is_rejected(tiny_manifest: Path) -> None:
    row = {"kind": "case", "id": 0, "status": 1}
    write_observations(tiny_manifest, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        qualification.collect(tiny_manifest)


def test_tampered_fixture_is_rejected(tiny_manifest: Path) -> None:
    write_observations(tiny_manifest, [])
    (tiny_manifest / "0.input").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        qualification.collect(tiny_manifest)


def test_shortened_coverage_cannot_claim_pass(tiny_manifest: Path) -> None:
    manifest = json.loads((tiny_manifest / "manifest.json").read_text())
    manifest["rows"] = []
    (tiny_manifest / "manifest.json").write_text(json.dumps(manifest))
    write_observations(tiny_manifest, [])
    with pytest.raises(ValueError, match="fixed protocol"):
        qualification.collect(tiny_manifest)


def test_bad_second_graph_replay_is_retained(tiny_manifest: Path) -> None:
    case = sample()
    left, right, active = qualification.inputs(case)
    output = qualification.oracle(case, left, right, active).astype(float)
    bad = output.copy()
    bad[0, 0, 0, 0] += 1.0
    (tiny_manifest / "0.output").write_bytes(
        qualification.encode(output) + qualification.encode(bad)
    )
    write_observations(tiny_manifest, [{"kind": "case", "id": 0, "status": 0}])
    result = qualification.collect(tiny_manifest)
    assert result["fail_count"] == 1
    assert result["rows"][0]["gates"][0]["status"] == "PASS"
    assert result["rows"][0]["gates"][1]["status"] == "FAIL"
    assert result["status"] == "NOT_QUALIFIED"


def test_owner_probe_missing_prevents_acceptance(tiny_manifest: Path) -> None:
    case = sample()
    left, right, active = qualification.inputs(case)
    output = qualification.oracle(case, left, right, active).astype(float)
    (tiny_manifest / "0.output").write_bytes(qualification.encode(output) * 2)
    write_observations(tiny_manifest, [{"kind": "case", "id": 0, "status": 0}])
    lines = (tiny_manifest / "device.jsonl").read_text().splitlines()
    (tiny_manifest / "device.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    result = qualification.collect(tiny_manifest)
    assert result["owner_probes_accepted"] is False
    assert result["status"] == "NOT_QUALIFIED"


def test_final_state_stationarity_is_independent_of_convergence_flag() -> None:
    model = {"atomic_numbers": [2], "charge": 0}
    metadata = (3, 2, 1, 1, 0, 0, 0, 0, 1, 1, 4, 4, 0, 0, 1, 0)
    density = np.diag([2.0, 0.0])[None]
    fock = np.diag([-1.0, 1.0])[None]
    coefficients = np.eye(2)[None]
    energies = np.array([[-1.0, 1.0]])
    occupations = np.array([[2.0, 0.0]])
    prefix = np.zeros(6)
    values = np.concatenate(
        [
            prefix,
            density.ravel(),
            fock.ravel(),
            coefficients.ravel(),
            energies.ravel(),
            occupations.ravel(),
            (-density).ravel(),
            np.eye(2).ravel(),
        ]
    )
    assert qualification.final_state_gates(metadata, values, model)["accepted"]
    # Physical Fock no longer commutes with the accepted density.
    values[11] = 0.1
    values[12] = 0.1
    assert not qualification.final_state_gates(metadata, values, model)["accepted"]


def test_final_state_generation_mismatch_rejected() -> None:
    model = {"atomic_numbers": [2], "charge": 0}
    metadata = (3, 1, 1, 1, 0, 0, 0, 0, 1, 1, 4, 5, 0, 0, 1, 0)
    values = np.array([0.0] * 6 + [2.0, -1.0, 1.0, -1.0, 2.0, -2.0, 1.0])
    result = qualification.final_state_gates(metadata, values, model)
    assert not result["density_orbital_generation_match"]
    assert not result["accepted"]
