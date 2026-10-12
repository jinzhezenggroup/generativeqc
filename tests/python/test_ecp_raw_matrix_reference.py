"""Integrity and negative controls for authentic fixed ECP matrix receipts."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from _ecp_raw_matrix_reference import REFERENCE_DIR, load_reference
from test_ecp import _assert_raw_matrix_results, _record_from_resolved_basis

if TYPE_CHECKING:
    from pathlib import Path


def _copy_reference(root: Path) -> None:
    for name in ("manifest.json", "inputs.json", "reference.json"):
        (root / name).write_bytes((REFERENCE_DIR / name).read_bytes())


def _load(root: Path = REFERENCE_DIR) -> Any:
    return load_reference(representation="cartesian", d_shell=True, root=root)


def test_all_four_fixed_matrices_and_basis_load_without_pyscf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_pyscf(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("fixed ECP loading imported PySCF")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pyscf)
    for d_shell in (False, True):
        for representation in ("spherical", "cartesian"):
            atoms, basis, ecp, matrix = load_reference(
                representation=representation, d_shell=d_shell
            )
            record = _record_from_resolved_basis(
                basis, ecp, representation, frozen=True
            )
            nao = (15 if representation == "cartesian" else 14) if d_shell else 9
            assert np.asarray(matrix).shape == (nao, nao)
            assert np.isfinite(matrix).all()
            assert atoms == [["Na", [0.13, -0.21, 0.17]], ["H", [0.43, 0.19, 3.2]]]
            assert record.representation == representation
            assert record.by_element[11].ecp_core_electrons == 10


@pytest.mark.parametrize("name", ("manifest.json", "inputs.json", "reference.json"))
def test_missing_reference_fails_closed(tmp_path: Path, name: str) -> None:
    _copy_reference(tmp_path)
    (tmp_path / name).unlink()
    with pytest.raises(FileNotFoundError):
        _load(tmp_path)


@pytest.mark.parametrize("name", ("manifest.json", "inputs.json", "reference.json"))
def test_nonobject_reference_fails_closed(tmp_path: Path, name: str) -> None:
    _copy_reference(tmp_path)
    (tmp_path / name).write_text("[]\n")
    with pytest.raises(TypeError, match="JSON object"):
        _load(tmp_path)


@pytest.mark.parametrize("name", ("inputs.json", "reference.json"))
def test_changed_reference_bytes_fail_checksum(tmp_path: Path, name: str) -> None:
    _copy_reference(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        _load(tmp_path)


@pytest.mark.parametrize(
    ("representation", "d_shell"),
    (("unknown", False), ("spherical", 1), ("cartesian", None)),
)
def test_unknown_case_has_no_nearby_fallback(representation: Any, d_shell: Any) -> None:
    with pytest.raises(ValueError, match="selector"):
        load_reference(representation=representation, d_shell=d_shell)


@pytest.mark.parametrize(
    "field",
    (
        "shape",
        "ragged",
        "nonfinite",
        "duplicate",
        "basis",
        "ecp",
        "atom",
        "ao_order",
        "normalization",
        "replay",
        "provenance",
    ),
)
def test_semantic_receipt_mutants_fail_closed(tmp_path: Path, field: str) -> None:
    _copy_reference(tmp_path)
    path = tmp_path / "reference.json"
    receipt = json.loads(path.read_text())
    record = receipt["records"][-1]
    if field == "shape":
        record["ecp_normalized"]["shape"] = [1, record["nao"] ** 2]
    elif field == "ragged":
        record["ecp_normalized"]["values"][0].pop()
    elif field == "nonfinite":
        record["ecp_normalized"]["values"][0][0] = float("nan")
    elif field == "duplicate":
        receipt["records"][-1] = copy.deepcopy(receipt["records"][0])
    elif field == "basis":
        record["input"]["basis"]["Na"][0][1][0] += 0.001
    elif field == "ecp":
        record["input"]["ecp"]["Na"][1][0][1][1][0][1] += 0.001
    elif field == "atom":
        record["molecule_atoms"][0][1][0] += 0.001
    elif field == "ao_order":
        record["ao_labels"][0], record["ao_labels"][1] = (
            record["ao_labels"][1],
            record["ao_labels"][0],
        )
    elif field == "normalization":
        record["normalization"]["values"] = [1.0] * record["nao"]
        record["normalization"]["sha256_le_f64"] = hashlib.sha256(
            np.ones(record["nao"], dtype="<f8").tobytes()
        ).hexdigest()
    elif field == "replay":
        receipt["replay"]["process_ids"][1] = receipt["replay"]["process_ids"][0]
    else:
        receipt["provenance"]["generator_sha256"] = "0" * 64
    # Rebind only the outer byte envelope so semantic validation, rather
    # than the independently tested file checksum, rejects each unit mutant.
    raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["receipt_bytes"] = len(raw)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _load(tmp_path)


def test_loading_returns_independent_mutable_copies() -> None:
    atoms, basis, ecp, matrix = _load()
    expected = matrix[0][0]
    atoms[0][1][0] += 1
    basis["Na"][0][1][0] += 1
    ecp["Na"][0] = 0
    matrix[0][0] += 1
    fresh = _load()
    assert fresh[0][0][1][0] == 0.13
    assert fresh[2]["Na"][0] == 10
    assert fresh[3][0][0] == expected


@pytest.mark.parametrize(
    "field", ("matrix", "unnormalized", "local_derivative", "nonlocal_derivative")
)
def test_real_numerical_gate_rejects_mutants(field: str) -> None:
    _, _, _, expected = _load()
    matrix = np.array(expected, dtype=np.float64)
    derivative = np.zeros((2, 3, len(matrix), len(matrix)))
    # Synthetic endpoint objects exercise the actual unchanged scientific
    # assertions; every expected matrix number is from the genuine receipt.
    actual = SimpleNamespace(
        matrix=matrix.copy(),
        local=matrix.copy(),
        nonlocal_=np.zeros_like(matrix),
        local_derivative=derivative.copy(),
        nonlocal_derivative=derivative.copy(),
    )
    refined = copy.deepcopy(actual)
    _assert_raw_matrix_results(actual, refined, expected)
    if field == "matrix":
        # Preserve symmetry and refined quadrature parity; only the independent
        # full-matrix assertion catches this common-mode symmetric defect.
        actual.matrix[0, 1] += 1e-4
        actual.matrix[1, 0] += 1e-4
    elif field == "unnormalized":
        receipt = json.loads((REFERENCE_DIR / "reference.json").read_text())
        actual.matrix = np.array(receipt["records"][-1]["ecp_raw"]["values"])
    else:
        values = getattr(actual, field)
        # Keep translational cancellation intact while changing the derivative.
        values[0, 0, 0, 0] += 1e-5
        values[1, 0, 0, 0] -= 1e-5
    with pytest.raises(AssertionError):
        _assert_raw_matrix_results(actual, refined, expected)


def test_oversized_receipt_fails_before_json_parsing(tmp_path: Path) -> None:
    _copy_reference(tmp_path)
    (tmp_path / "reference.json").write_bytes(b"x" * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="byte limit"):
        _load(tmp_path)


def _assert_retained_replicas(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    reference = json.loads((root / "reference.json").read_text())
    replay = reference["replay"]
    assert [entry["path"] for entry in manifest["replicas"]] == [
        "acquisition-first.json",
        "acquisition-second.json",
    ]
    assert replay["replicas"] == ["first", "second"]
    assert replay["receipt_sha256"] == [
        entry["sha256"] for entry in manifest["replicas"]
    ]
    assert len(replay["process_ids"]) == 2
    assert replay["process_ids"][0] != replay["process_ids"][1]
    for entry, replica, process_id in zip(
        manifest["replicas"], replay["replicas"], replay["process_ids"], strict=True
    ):
        raw = (root / entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        original = json.loads(raw)
        expected = copy.deepcopy(reference)
        expected.pop("replay")
        expected["replica"] = replica
        expected["process_id"] = process_id
        # This compares every input, matrix, AO field and provenance field,
        # not only selected summary numbers or the raw file hashes.
        assert original == expected


def _copy_with_replicas(root: Path) -> None:
    _copy_reference(root)
    for name in ("acquisition-first.json", "acquisition-second.json"):
        (root / name).write_bytes((REFERENCE_DIR / name).read_bytes())


def test_both_complete_raw_replicas_match_validated_reference() -> None:
    _assert_retained_replicas(REFERENCE_DIR)


@pytest.mark.parametrize("name", ("acquisition-first.json", "acquisition-second.json"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_raw_replica_loss_or_corruption_fails(
    tmp_path: Path, name: str, damage: str
) -> None:
    _copy_with_replicas(tmp_path)
    path = tmp_path / name
    if damage == "missing":
        path.unlink()
        error = FileNotFoundError
    else:
        # Preserve length so this control reaches the raw-file SHA check.
        path.write_bytes(b"[" + path.read_bytes()[1:])
        error = AssertionError
    with pytest.raises(error):
        _assert_retained_replicas(tmp_path)


@pytest.mark.parametrize("field", ("identity", "records", "provenance"))
def test_raw_replica_semantic_mutants_fail(tmp_path: Path, field: str) -> None:
    _copy_with_replicas(tmp_path)
    path = tmp_path / "acquisition-second.json"
    original = json.loads(path.read_text())
    if field == "identity":
        original["process_id"] += 1
    elif field == "records":
        original["records"][-1]["ecp_raw"]["values"][0][0] += 0.01
    else:
        original["provenance"]["generator_sha256"] = "0" * 64
    raw = (json.dumps(original, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    # Bind the synthetic envelope to this mutant so full-object comparison,
    # rather than the separately tested byte checksum, rejects the defect.
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["replicas"][1]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["replicas"][1]["bytes"] = len(raw)
    manifest_path.write_text(json.dumps(manifest))
    reference_path = tmp_path / "reference.json"
    reference = json.loads(reference_path.read_text())
    reference["replay"]["receipt_sha256"][1] = manifest["replicas"][1]["sha256"]
    reference_path.write_text(json.dumps(reference))
    with pytest.raises(AssertionError):
        _assert_retained_replicas(tmp_path)
