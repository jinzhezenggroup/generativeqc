"""Integrity and negative controls for the authentic fixed Li/UHF receipt."""

from __future__ import annotations

import base64
import builtins
import copy
import hashlib
import json
import shutil
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _li_uhf_random_reference import (
    BLOCK_SHAPES,
    MAX_BYTES,
    REFERENCE_DIR,
    SHAPES,
    _array,
    _read,
    _validate,
    load_reference,
)

if TYPE_CHECKING:
    from pathlib import Path


def _documents() -> tuple[dict, dict, dict]:
    return tuple(
        _read(REFERENCE_DIR / name)[0]
        for name in ("reference.json", "inputs.json", "manifest.json")
    )


def _copy_reference(target: Path) -> Path:
    for name in (
        "reference.json",
        "inputs.json",
        "manifest.json",
        "acquisition-first.json",
        "acquisition-second.json",
    ):
        shutil.copyfile(REFERENCE_DIR / name, target / name)
    return target


def _write(path: Path, value: dict) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _reseal_array(
    receipt: dict, manifest: dict, name: str, values: list[float]
) -> None:
    """Synthetic negative controls only; never produce a numerical fixture."""
    record = receipt["arrays"][name]
    raw = struct.pack(f"<{len(values)}d", *values)
    record["data"] = base64.b64encode(raw).decode("ascii")
    record["sha256_le_f64"] = hashlib.sha256(raw).hexdigest()
    manifest["arrays"][name] = {
        key: value for key, value in record.items() if key != "data"
    }


def test_complete_profile_and_both_original_processes_are_retained() -> None:
    arrays = load_reference()
    receipt, inputs, manifest = _documents()
    assert len(arrays) == 48
    assert set(arrays) == set(SHAPES)
    for name, shape in SHAPES.items():
        assert len(arrays[name]) == int(np.prod(shape))
        assert np.isfinite(arrays[name]).all()
    assert inputs["populations"] == {"oa": 2, "ob": 1, "va": 3, "vb": 4}
    assert receipt["molecule"]["basis"] == inputs["resolved_basis"]
    assert (
        receipt["molecule"]["ao_labels_structured"] == inputs["ao_identity"]["labels"]
    )
    assert receipt["molecule"]["ao_locations"] == [0, 1, 2, 5]
    assert arrays["amplitudes.t2bb"] == (0.0,) * 16
    assert (
        arrays["channels.aaa"]
        == arrays["channels.bbb"]
        == arrays["channels.abb"]
        == (0.0,)
    )
    assert abs(arrays["channels.aab"][0]) > 1e-6
    assert receipt["replay"]["max_absolute_difference"] == 0
    assert receipt["replay"]["process_ids"] == [2251, 2268]
    for binding in manifest["replicas"]:
        original, raw = _read(REFERENCE_DIR / binding["path"])
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
        expected = {key: value for key, value in receipt.items() if key != "replay"}
        expected.update(replica=binding["replica"], process_id=binding["process_id"])
        assert original == expected


def test_reader_has_no_live_pyscf_import_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("fixed reader attempted to import PySCF")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert len(load_reference()) == 48


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_complete_array_digest_detects_changed_values(name: str) -> None:
    receipt, inputs, manifest = _documents()
    record = receipt["arrays"][name]
    raw = bytearray(base64.b64decode(record["data"]))
    raw[0] ^= 1
    record["data"] = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError, match="array byte identity"):
        _validate(receipt, inputs, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("shape", [1, 25]),
        ("shape", [5, True]),
        ("order", "F"),
        ("encoding", "base64-native-f64"),
        ("bytes", 199),
        ("bytes", True),
        ("data", "*"),
        ("sha256_le_f64", "0" * 64),
    ],
)
def test_array_metadata_is_closed(field: str, value: object) -> None:
    receipt, _, _ = _documents()
    record = receipt["arrays"]["overlap"]
    record[field] = value
    with pytest.raises((ValueError, TypeError)):
        _array(record, (5, 5))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_encoded_scalar_is_rejected(value: float) -> None:
    receipt, inputs, manifest = _documents()
    _reseal_array(receipt, manifest, "correlation_energy", [value])
    with pytest.raises(ValueError, match="nonfinite"):
        _validate(receipt, inputs, manifest)


@pytest.mark.parametrize("name", ["reference.json", "inputs.json", "manifest.json"])
def test_nonobject_json_is_rejected(name: str, tmp_path: Path) -> None:
    root = _copy_reference(tmp_path)
    (root / name).write_text("[]\n")
    with pytest.raises(TypeError, match="JSON object"):
        load_reference(root=root)


def test_duplicate_json_fields_and_oversize_input_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "invalid.json"
    target.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        _read(target)
    target.write_bytes(b" " * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        _read(target)


@pytest.mark.parametrize("name", ["acquisition-first.json", "acquisition-second.json"])
@pytest.mark.parametrize("mode", ["missing", "corrupt", "reidentified"])
def test_each_raw_replica_is_required_and_byte_bound(
    name: str, mode: str, tmp_path: Path
) -> None:
    root = _copy_reference(tmp_path)
    target = root / name
    if mode == "missing":
        target.unlink()
        with pytest.raises(FileNotFoundError):
            load_reference(root=root)
        return
    original, raw = _read(target)
    if mode == "corrupt":
        # Preserve the length and valid JSON while breaking exact source bytes.
        target.write_bytes(raw.replace(b'"schema"', b'"schemb"', 1))
        with pytest.raises(ValueError, match="raw replica byte identity"):
            load_reference(root=root)
        return
    original["process_id"] += 100
    changed = _write(target, original)
    manifest, _ = _read(root / "manifest.json")
    binding = next(item for item in manifest["replicas"] if item["path"] == name)
    binding.update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    receipt, _ = _read(root / "reference.json")
    receipt["replay"]["receipt_sha256"] = [
        item["sha256"] for item in manifest["replicas"]
    ]
    changed_receipt = _write(root / "reference.json", receipt)
    manifest.update(
        receipt_bytes=len(changed_receipt),
        receipt_sha256=hashlib.sha256(changed_receipt).hexdigest(),
    )
    _write(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="raw replica differs"):
        load_reference(root=root)


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "spin_order",
        "missing_array",
        "duplicate_pid",
        "source_hash",
        "configuration",
        "ao_order",
        "basis",
        "state",
    ],
)
def test_input_provenance_and_resolved_identity_mutations_fail(mutation: str) -> None:
    receipt, inputs, manifest = _documents()
    if mutation == "seed":
        inputs["random"]["seed"] = 38
        receipt["inputs"] = copy.deepcopy(inputs)
    elif mutation == "spin_order":
        inputs["spin_orbital_order"].reverse()
        receipt["inputs"] = copy.deepcopy(inputs)
    elif mutation == "missing_array":
        receipt["arrays"].pop("antisymmetrized_eri")
    elif mutation == "duplicate_pid":
        receipt["replay"]["process_ids"][1] = receipt["replay"]["process_ids"][0]
    elif mutation == "source_hash":
        receipt["provenance"]["source_sha256"][inputs["sources"][0]["path"]] = "0" * 64
    elif mutation == "configuration":
        receipt["provenance"]["configuration"]["policy"] = "external"
    elif mutation == "ao_order":
        receipt["molecule"]["ao_labels_structured"].reverse()
    elif mutation == "basis":
        receipt["molecule"]["basis"]["Li"][0][1][0] += 0.01
    else:
        receipt["molecule"]["spin"] = 0
    with pytest.raises(ValueError):
        _validate(receipt, inputs, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "occupation",
        "fock_order",
        "eri_symmetry",
        "packed_amplitude",
        "same_spin_zero",
        "residual",
        "impossible_channel",
        "optimized_total",
    ],
)
def test_scientific_relations_reject_resealed_synthetic_mutants(mutation: str) -> None:
    receipt, inputs, manifest = _documents()
    names = {
        "occupation": "occupations_alpha",
        "fock_order": "fock_spin",
        "eri_symmetry": "antisymmetrized_eri",
        "packed_amplitude": "packed_amplitudes",
        "same_spin_zero": "amplitudes.t2bb",
        "residual": "residuals.t1a",
        "impossible_channel": "channels.aaa",
        "optimized_total": "triples_optimized_total",
    }
    name = names[mutation]
    values = list(_array(receipt["arrays"][name], SHAPES[name]))
    index = 1 if mutation == "eri_symmetry" else 0
    values[index] += 0.001
    _reseal_array(receipt, manifest, name, values)
    with pytest.raises(ValueError):
        _validate(receipt, inputs, manifest)


@pytest.mark.parametrize(
    "name", ["correlation_energy", *[f"residuals.{block}" for block in BLOCK_SHAPES]]
)
def test_actual_equation_gate_rejects_each_changed_expected_output(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_uccsd_equations as equations

    layout, amplitudes, arrays = equations._fixed_li_random_profile()
    arrays[name] = arrays[name].copy()
    arrays[name].flat[0] += 1e-4
    monkeypatch.setattr(
        equations, "_fixed_li_random_profile", lambda: (layout, amplitudes, arrays)
    )
    with pytest.raises(AssertionError):
        equations.test_random_amplitudes_against_pyscf_uccsd("Li 0 0 0", 1, False)


@pytest.mark.parametrize("channel", ["aaa", "bbb", "aab", "abb"])
def test_actual_channel_gate_rejects_each_changed_expected_channel(
    channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_uccsd_triples as triples

    layout, amplitudes, arrays = triples._fixed_li_random_profile()
    arrays[f"channels.{channel}"] = arrays[f"channels.{channel}"].copy() + 1e-4
    monkeypatch.setattr(
        triples, "_fixed_li_random_profile", lambda: (layout, amplitudes, arrays)
    )
    with pytest.raises(AssertionError):
        triples.test_all_spin_channels_and_partitioned_tiles_match_pyscf(
            "Li 0 0 0", 1, False
        )


@pytest.mark.parametrize("case", ["equations", "triples"])
def test_fixed_numerical_cases_execute_with_pyscf_blocked(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    import test_uccsd_equations as equations
    import test_uccsd_triples as triples

    original_import = builtins.__import__
    original_module = importlib.import_module
    original_skip = pytest.importorskip

    def blocked(name: str) -> None:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("fixed numerical case attempted live PySCF or a skip")

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        blocked(name)
        return original_import(name, *args, **kwargs)

    def guarded_module(name: str, *args: object, **kwargs: object) -> object:
        blocked(name)
        return original_module(name, *args, **kwargs)

    def guarded_skip(name: str, *args: object, **kwargs: object) -> object:
        blocked(name)
        return original_skip(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_module)
    monkeypatch.setattr(pytest, "importorskip", guarded_skip)
    if case == "equations":
        equations.test_random_amplitudes_against_pyscf_uccsd("Li 0 0 0", 1, False)
    else:
        triples.test_all_spin_channels_and_partitioned_tiles_match_pyscf(
            "Li 0 0 0", 1, False
        )
