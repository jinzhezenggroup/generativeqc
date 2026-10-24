"""Authentic orbital-f HF receipt integrity, full arrays and PySCF-free inputs."""

from __future__ import annotations

import base64
import builtins
import copy
import hashlib
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from _ecp_f_hf_reference import (
    PROFILES,
    REFERENCE_DIR,
    _array,
    _read,
    _validate,
    load_reference,
    shapes,
)

if TYPE_CHECKING:
    from pathlib import Path


def _documents(profile: str = "rhf-cartesian") -> tuple[dict, dict, dict, dict]:
    inputs, _ = _read(REFERENCE_DIR / "inputs.json")
    manifest, _ = _read(REFERENCE_DIR / "manifest.json")
    receipt, _ = _read(REFERENCE_DIR / f"{profile}-first.json")
    binding = next(item for item in manifest["profiles"] if item["profile"] == profile)
    return inputs, manifest, receipt, binding


@pytest.mark.parametrize("profile", PROFILES)
def test_complete_original_arrays_and_hf1e_conventions(profile: str) -> None:
    method, representation = profile.split("-")
    spin = int(method == "uhf")
    atoms, basis, ecp, decoded = load_reference(
        representation=representation, spin=spin
    )
    inputs, manifest, receipt, binding = _documents(profile)
    array_shapes = shapes(profile)
    arrays = {
        key: np.asarray(value).reshape(array_shapes[key])
        for key, value in decoded.items()
    }
    assert len(arrays) == 16
    assert (
        atoms == inputs["atoms"]
        and basis == inputs["resolved_basis"]
        and ecp == inputs["resolved_ecp"]
    )
    for key, shape in array_shapes.items():
        assert arrays[key].shape == shape
        assert np.isfinite(arrays[key]).all()
    np.testing.assert_array_equal(arrays["forces"], -arrays["gradient"])
    np.testing.assert_allclose(arrays["forces"].sum(axis=0), 0, atol=2e-7, rtol=0)
    n = array_shapes["overlap"][0]
    energy = 0.0
    for block in (0, 1) if spin else (None,):
        c, fock, density, occupations, orbital_energy = (
            arrays[key] if block is None else arrays[key][block]
            for key in (
                "coefficients",
                "fock",
                "density",
                "occupations",
                "orbital_energies",
            )
        )
        np.testing.assert_allclose(
            c.T @ arrays["overlap"] @ c, np.eye(n), atol=1e-10, rtol=0
        )
        np.testing.assert_allclose((c * occupations) @ c.T, density, atol=1e-12, rtol=0)
        # HF1e returns hcore eigenvectors, including unoccupied beta orbitals.
        # Never substitute a later Fock-canonicalized virtual spectrum.
        operator = arrays["hcore"] if spin else fock
        np.testing.assert_allclose(
            operator @ c, (arrays["overlap"] @ c) * orbital_energy, atol=3e-8, rtol=0
        )
        energy += 0.5 * np.einsum("ij,ji->", density, arrays["hcore"] + fock)
    assert energy + arrays["nuclear_energy"] == pytest.approx(
        arrays["energy"], abs=1e-10
    )
    assert receipt["scf"]["cycles"] == (0 if spin else 8)
    assert receipt["scf"]["initial_guess_used"] is (not spin)
    assert binding["replay"]["max_absolute_difference_by_array"] == dict.fromkeys(
        arrays, 0.0
    )
    assert (
        len(
            {
                replica["process_id"]
                for item in manifest["profiles"]
                for replica in item["replicas"]
            }
        )
        == 8
    )
    if spin:
        np.testing.assert_array_equal(arrays["density"][1], np.zeros((n, n)))


@pytest.mark.parametrize("profile", PROFILES)
def test_frozen_gqc_inputs_do_not_import_or_skip_pyscf(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    original_import, original_skip = builtins.__import__, pytest.importorskip

    def forbid_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("frozen fixture imported PySCF")
        return original_import(name, *args, **kwargs)

    def forbid_skip(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("frozen fixture requested a PySCF skip")
        return original_skip(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_import)
    monkeypatch.setattr(pytest, "importorskip", forbid_skip)
    from test_ecp import _record_from_resolved_basis

    method, representation = profile.split("-")
    atoms, resolved_basis, resolved_ecp, arrays = load_reference(
        representation=representation, spin=int(method == "uhf")
    )
    basis = _record_from_resolved_basis(
        resolved_basis, resolved_ecp, representation, frozen=True
    )
    assert len(atoms) == 2 and basis.representation == representation
    assert len(arrays["forces"]) == 6 and basis.by_element[11].ecp_core_electrons == 10


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("key", tuple(shapes("rhf-cartesian")))
def test_every_complete_array_rejects_byte_tampering(profile: str, key: str) -> None:
    inputs, manifest, receipt, binding = _documents(profile)
    record = receipt["arrays"][key]
    raw = bytearray(base64.b64decode(record["data"]))
    raw[0] ^= 1
    record["data"] = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError, match="array byte identity"):
        _validate(receipt, inputs, binding, manifest["provenance"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [6]),
        ("shape", [True, 6]),
        ("dtype", "float32"),
        ("bytes", 8),
        ("bytes", True),
        ("order", "F"),
        ("encoding", "json"),
    ],
)
def test_complete_force_envelope_rejects_shape_and_type_drift(
    field: str, value: Any
) -> None:
    _, _, receipt, _ = _documents()
    record = receipt["arrays"]["forces"]
    record[field] = value
    with pytest.raises(ValueError):
        _array(record, (2, 3))


def _reseal(receipt: dict, binding: dict, key: str, values: np.ndarray) -> None:
    """In-memory negative controls only; never write modified reference values."""
    raw = np.asarray(values, dtype="<f8").tobytes(order="C")
    record = receipt["arrays"][key]
    record["data"] = base64.b64encode(raw).decode("ascii")
    record["sha256_le_f64"] = hashlib.sha256(raw).hexdigest()
    binding["arrays"][key] = {
        key: value for key, value in record.items() if key != "data"
    }


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_force_is_rejected_even_after_resealing(value: float) -> None:
    inputs, manifest, receipt, binding = _documents()
    values = np.array(_array(receipt["arrays"]["forces"], (2, 3)))
    values[0] = value
    _reseal(receipt, binding, "forces", values)
    with pytest.raises(ValueError, match="nonfinite"):
        _validate(receipt, inputs, binding, manifest["provenance"])


def test_full_force_sign_is_not_a_scalar_summary() -> None:
    inputs, manifest, receipt, binding = _documents()
    values = np.array(_array(receipt["arrays"]["forces"], (2, 3)))
    values[2] *= -1
    _reseal(receipt, binding, "forces", values)
    with pytest.raises(ValueError, match="negative analytic gradient"):
        _validate(receipt, inputs, binding, manifest["provenance"])


@pytest.mark.parametrize(
    "mutation",
    [
        "profile",
        "spin",
        "cart",
        "ao_order",
        "basis",
        "ecp",
        "class",
        "convergence",
        "array_inventory",
        "package",
        "provider",
        "source",
        "run",
        "timing",
    ],
)
def test_scientific_identity_and_provenance_reject_mutations(mutation: str) -> None:
    inputs, manifest, receipt, binding = _documents("uhf-cartesian")
    if mutation == "profile":
        receipt["profile"] = "rhf-cartesian"
    elif mutation == "spin":
        receipt["molecule"]["spin"] = 0
    elif mutation == "cart":
        receipt["molecule"]["cart"] = False
    elif mutation == "ao_order":
        receipt["molecule"]["ao_labels"].reverse()
    elif mutation == "basis":
        receipt["molecule"]["basis"]["Na"][-1][1][1] += 0.1
    elif mutation == "ecp":
        receipt["molecule"]["ecp"]["Na"][0] = 8
    elif mutation == "class":
        receipt["scf"]["actual_class"] = "pyscf.scf.uhf.UHF"
    elif mutation == "convergence":
        receipt["scf"]["converged"] = False
    elif mutation == "array_inventory":
        del receipt["arrays"]["density"]
    elif mutation == "package":
        receipt["provenance"]["packages"]["pyscf"]["installed_source_native_sha256"] = (
            "0" * 64
        )
    elif mutation == "provider":
        receipt["provenance"]["threadpools"][0]["internal_api"] = "cuda"
    elif mutation == "source":
        receipt["provenance"]["generator_sha256"] = "0" * 64
    elif mutation == "run":
        receipt["provenance"]["execution"]["github_run_id"] = "1"
    else:
        receipt["timings"]["scf_seconds"] = -1
    with pytest.raises(ValueError):
        _validate(receipt, inputs, binding, manifest["provenance"])


@pytest.mark.parametrize(
    "filename",
    [
        "inputs.json",
        "manifest.json",
        *[
            f"{profile}-{replica}.json"
            for profile in PROFILES
            for replica in ("first", "second")
        ],
    ],
)
def test_every_retained_original_is_authenticated(
    tmp_path: Path, filename: str
) -> None:
    for source in REFERENCE_DIR.glob("*.json"):
        shutil.copyfile(source, tmp_path / source.name)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte identity"):
        load_reference(representation="cartesian", spin=0, root=tmp_path)


@pytest.mark.parametrize(
    "representation,spin",
    [("spinor", 0), ("spherical", True), ("cartesian", 2), ("cartesian", 0.0)],
)
def test_unknown_selectors_do_not_fall_back_to_live_oracle(
    representation: str, spin: Any
) -> None:
    with pytest.raises(ValueError, match="selector"):
        load_reference(representation=representation, spin=spin)


def test_reader_returns_fresh_inputs_and_arrays() -> None:
    first = load_reference(representation="cartesian", spin=0)
    expected = copy.deepcopy(first)
    first[0][0][1][0] += 1
    first[1]["Na"].clear()
    first[3]["energy"] = (123.0,)
    assert load_reference(representation="cartesian", spin=0) == expected
