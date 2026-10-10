"""Tool acceptance gates, not host substitutes for CUDA qualification."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
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


@pytest.mark.parametrize("field", ["provider", "graph", "scale", "missing", "extra"])
def test_changed_executed_protocol_is_rejected(tiny_manifest: Path, field: str) -> None:
    case = sample()
    # A correct native/no-Graph result must not be attributed to altered commands.
    left, right, active = qualification.inputs(case)
    output = qualification.oracle(case, left, right, active).astype(float)
    (tiny_manifest / "0.output").write_bytes(qualification.encode(output) * 2)
    write_observations(tiny_manifest, [{"kind": "case", "id": 0, "status": 0}])
    protocol = tiny_manifest / "cases.txt"
    fields = protocol.read_text().split()
    if field == "provider":
        fields[9] = "0" if fields[9] == "1" else "1"
    elif field == "graph":
        fields[8] = "0" if fields[8] == "1" else "1"
    elif field == "scale":
        fields[10] = "3.0"
    protocol.write_text(
        ""
        if field == "missing"
        else (" ".join(fields) + "\n") * (2 if field == "extra" else 1)
    )
    with pytest.raises(ValueError, match="command protocol"):
        qualification.collect(tiny_manifest)


def test_long_double_receipt_scalars_are_serializable(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    qualification.write_json(
        path, {"limit": np.longdouble("1e-12"), "error": np.longdouble("nan")}
    )
    receipt = json.loads(path.read_text())
    assert receipt["limit"] == 1e-12
    assert receipt["error"] == {"nonfinite": "nan"}


def test_forged_protocol_digest_is_rejected(tiny_manifest: Path) -> None:
    path = tiny_manifest / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["command_protocol_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    write_observations(tiny_manifest, [])
    with pytest.raises(ValueError, match="command protocol"):
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


@pytest.mark.parametrize("mismatch", ["none", "source", "unmapped", "dirty"])
def test_endpoint_identity_checks_selected_library_not_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    requested = tmp_path / "requested.so"
    selected = tmp_path / "selected.so"
    requested.write_bytes(b"requested library")
    selected.write_bytes(b"actually selected library")
    maps = tmp_path / "maps"
    mapped = requested if mismatch == "unmapped" else selected
    maps.write_text(f"1-2 r-xp 00000000 00:00 1 {mapped}\n")
    native = SimpleNamespace(
        _name=str(selected),
        generativeqc_get_source_identity=lambda: (
            b"wrong" if mismatch == "source" else b"expected"
        ),
    )
    calculator = SimpleNamespace(
        _library=native, profile_diagnostics={"source": "local"}
    )
    monkeypatch.setattr(
        qualification.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "head\n"
            if "rev-parse" in args
            else " M tracked\n"
            if mismatch == "dirty"
            else ""
        ),
    )
    identity = qualification.endpoint_identity(
        tmp_path, requested, calculator, "expected", maps=maps
    )
    assert identity["selected_library"] == str(selected.resolve())
    assert identity["selected_library_sha256"] == qualification.sha256(selected)
    assert identity["selected_library_sha256"] != identity["requested_library_sha256"]
    assert identity["source_matched_identity_available"] == (mismatch == "none")
    assert (tmp_path / "loaded-maps.txt").read_bytes() == maps.read_bytes()
