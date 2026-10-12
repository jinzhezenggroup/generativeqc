"""Complete-tensor integrity gates for the authentic converged Li Lambda fixture."""

from __future__ import annotations

import base64
import builtins
import copy
import hashlib
import shutil
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _li_uhf_lambda_reference import (
    BLOCK_SHAPES,
    MAX_BYTES,
    REFERENCE_DIR,
    SHAPES,
    _array,
    _read,
    _validate,
    frozen_problem,
    load_reference,
)

if TYPE_CHECKING:
    from pathlib import Path


def _documents() -> tuple:
    return tuple(
        _read(REFERENCE_DIR / name)[0]
        for name in (
            "reference.json",
            "inputs.json",
            "manifest.json",
        )
    )


def _copy_reference(root: Path) -> None:
    for name in (
        "reference.json",
        "inputs.json",
        "manifest.json",
        "acquisition-first.json",
        "acquisition-second.json",
    ):
        shutil.copyfile(REFERENCE_DIR / name, root / name)


def test_full_spin_resolved_primal_and_dual_arrays_are_retained() -> None:
    arrays = {
        name: np.asarray(value).reshape(SHAPES[name])
        for name, value in load_reference().items()
    }
    assert len(arrays) == 50
    for group in (
        "amplitudes",
        "updated_amplitudes",
        "residuals",
        "multipliers",
        "updated_multipliers",
    ):
        for name, shape in BLOCK_SHAPES.items():
            value = arrays[f"{group}.{name}"]
            assert value.shape == shape
            assert np.isfinite(value).all()
        np.testing.assert_array_equal(arrays[f"{group}.t2bb"], np.zeros((1, 1, 4, 4)))
        value = arrays[f"{group}.t2aa"]
        np.testing.assert_allclose(value, -value.swapaxes(0, 1), atol=1e-13, rtol=0)
        np.testing.assert_allclose(value, -value.swapaxes(2, 3), atol=1e-13, rtol=0)
    for spin in ("alpha", "beta"):
        c = arrays[f"coefficients_{spin}"]
        np.testing.assert_allclose(
            c.T @ arrays["overlap"] @ c, np.eye(5), atol=3e-12, rtol=0
        )
        np.testing.assert_allclose(
            c @ np.diag(arrays[f"occupations_{spin}"]) @ c.T,
            arrays[f"density_{spin}"],
            atol=3e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            c.T @ arrays[f"fock_ao_{spin}"] @ c,
            arrays[f"fock_mo_{spin}"],
            atol=3e-12,
            rtol=0,
        )
    np.testing.assert_allclose(
        arrays["ao_eri"], arrays["ao_eri"].transpose(2, 3, 0, 1), atol=3e-12, rtol=0
    )
    assert arrays["scf_gradient_norm"] < 1e-10
    receipt, _, manifest = _documents()
    assert receipt["molecule"]["cc_converged"] is True
    assert receipt["molecule"]["lambda_converged"] is True
    assert len(manifest["replicas"]) == 2
    assert len(set(receipt["replay"]["process_ids"])) == 2


def test_frozen_gqc_inputs_need_no_pyscf_or_import_based_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import, original_skip = builtins.__import__, pytest.importorskip

    def refuse_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("frozen Lambda fixture attempted a PySCF import")
        return original_import(name, *args, **kwargs)

    def refuse_skip(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("frozen Lambda fixture attempted a PySCF skip")
        return original_skip(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_import)
    monkeypatch.setattr(pytest, "importorskip", refuse_skip)
    snapshot, source, expected = frozen_problem()
    assert snapshot.nocc("alpha") == 2 and snapshot.nocc("beta") == 1
    assert tuple(expected) == ("l1a", "l1b", "l2aa", "l2ab", "l2bb")
    assert source is not None
    np.testing.assert_array_equal(expected["l2bb"], np.zeros((1, 1, 4, 4)))


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_full_array_detects_byte_tampering(name: str) -> None:
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
def test_array_envelopes_fail_closed(field: str, value: object) -> None:
    receipt, _, _ = _documents()
    record = receipt["arrays"]["overlap"]
    record[field] = value
    with pytest.raises((ValueError, TypeError)):
        _array(record, (5, 5))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_encoded_nonfinite_is_rejected(value: float) -> None:
    receipt, inputs, manifest = _documents()
    record = receipt["arrays"]["hf_energy"]
    raw = struct.pack("<d", value)
    record["data"] = base64.b64encode(raw).decode("ascii")
    record["sha256_le_f64"] = hashlib.sha256(raw).hexdigest()
    manifest["arrays"]["hf_energy"] = {k: v for k, v in record.items() if k != "data"}
    with pytest.raises(ValueError, match="nonfinite"):
        _validate(receipt, inputs, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_spin",
        "wrong_polish",
        "missing_block",
        "unconverged_cc",
        "unconverged_lambda",
        "same_pid",
        "wrong_order",
        "wrong_generator",
        "wrong_source",
        "wrong_run",
    ],
)
def test_profile_and_provenance_drift_fail_closed(mutation: str) -> None:
    receipt, inputs, manifest = copy.deepcopy(_documents())
    if mutation == "wrong_spin":
        inputs["spin"] = 0
    elif mutation == "wrong_polish":
        inputs["scf"]["polish"]["conv_tol_grad"] = 1e-4
    elif mutation == "missing_block":
        del receipt["arrays"]["multipliers.t2bb"]
    elif mutation in ("unconverged_cc", "unconverged_lambda"):
        key = "cc_converged" if mutation == "unconverged_cc" else "lambda_converged"
        receipt["molecule"][key] = manifest["molecule"][key] = False
    elif mutation == "same_pid":
        receipt["replay"]["process_ids"][1] = receipt["replay"]["process_ids"][0]
    elif mutation == "wrong_order":
        inputs["block_order"].reverse()
    elif mutation == "wrong_generator":
        manifest["generator"]["sha256"] = "0" * 64
    elif mutation == "wrong_source":
        inputs["sources"][0]["sha256"] = "0" * 64
    else:
        manifest["acquisition"]["run_id"] += 1
    with pytest.raises(ValueError):
        _validate(receipt, inputs, manifest)


@pytest.mark.parametrize(
    "name",
    [
        "reference.json",
        "inputs.json",
        "acquisition-first.json",
        "acquisition-second.json",
    ],
)
@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_retained_files_are_required_and_byte_bound(
    name: str, mode: str, tmp_path: Path
) -> None:
    _copy_reference(tmp_path)
    target = tmp_path / name
    if mode == "missing":
        target.unlink()
        with pytest.raises(FileNotFoundError):
            load_reference(root=tmp_path)
    else:
        target.write_bytes(target.read_bytes().replace(b'"schema"', b'"schemb"', 1))
        with pytest.raises(ValueError):
            load_reference(root=tmp_path)


def test_duplicate_json_fields_and_oversize_input_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "invalid.json"
    target.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        _read(target)
    target.write_bytes(b" " * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        _read(target)
