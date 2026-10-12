"""Integrity and negative controls for genuine fixed high-l reference records."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _cpu_hf_high_l_reference import REFERENCE_DIR, load_reference
from test_cpu_hf_high_l import _assert_results_match_reference


def _arguments() -> dict[str, Any]:
    inputs = json.loads((REFERENCE_DIR / "inputs.json").read_text())
    return {
        "atoms": [(element, list(position)) for element, position in inputs["atoms"]],
        "basis": inputs["basis"],
        "representation": "cartesian",
        "method": "rhf",
        "charge": 1,
        "spin": 0,
        "displacement": 0.0,
    }


def _copy_reference(root: Path) -> None:
    for name in ("manifest.json", "inputs.json", "reference.json"):
        (root / name).write_bytes((REFERENCE_DIR / name).read_bytes())


def test_all_eight_fixed_references_load_without_pyscf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_pyscf(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("ordinary fixed-reference loading imported PySCF")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pyscf)
    for method, charge, spin in (("rhf", 1, 0), ("uhf", 0, 1)):
        for representation in ("cartesian", "spherical"):
            for displacement in (0.0, 0.013):
                arguments = _arguments()
                arguments.update(
                    method=method,
                    charge=charge,
                    spin=spin,
                    representation=representation,
                    displacement=displacement,
                )
                arguments["atoms"][-1][1][0] += displacement
                energy, forces = load_reference(**arguments)
                assert np.isfinite(energy)
                assert np.asarray(forces).shape == (2, 3)
                assert np.isfinite(forces).all()


@pytest.mark.parametrize("name", ("manifest.json", "inputs.json", "reference.json"))
def test_missing_fixed_reference_fails_closed(tmp_path: Path, name: str) -> None:
    for existing in ("manifest.json", "inputs.json", "reference.json"):
        if existing != name:
            (tmp_path / existing).write_bytes((REFERENCE_DIR / existing).read_bytes())
    with pytest.raises(FileNotFoundError):
        load_reference(**_arguments(), root=tmp_path)


@pytest.mark.parametrize("name", ("inputs.json", "reference.json"))
def test_modified_fixed_reference_bytes_fail_checksum(
    tmp_path: Path, name: str
) -> None:
    _copy_reference(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(**_arguments(), root=tmp_path)


@pytest.mark.parametrize(
    "field", ("atom", "basis", "state", "displacement", "representation")
)
def test_changed_scientific_input_has_no_nearby_fallback(field: str) -> None:
    arguments = _arguments()
    if field == "atom":
        arguments["atoms"][1][1][2] += 0.001
    elif field == "basis":
        arguments["basis"]["He"][0][1][0] += 0.001
    elif field == "state":
        arguments["charge"] = 0
    elif field == "displacement":
        arguments["displacement"] = 0.02
    else:
        arguments["representation"] = "unknown"
    with pytest.raises(ValueError, match="scientific input|representation"):
        load_reference(**arguments)


@pytest.mark.parametrize("field", ("convergence", "shape", "nonfinite", "duplicate"))
def test_semantically_invalid_receipt_fails_closed(tmp_path: Path, field: str) -> None:
    _copy_reference(tmp_path)
    receipt_path = tmp_path / "reference.json"
    receipt = json.loads(receipt_path.read_text())
    record = receipt["records"][0]
    if field == "convergence":
        record["converged"] = False
    elif field == "shape":
        record["forces"]["shape"] = [3, 2]
    elif field == "nonfinite":
        record["energy"]["values"] = float("nan")
    else:
        receipt["records"][-1] = copy.deepcopy(record)
    # Rebind only the outer byte envelope so the loader's semantic checks,
    # rather than its already-tested outer checksum, reject this unit mutant.
    raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    receipt_path.write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["receipt_bytes"] = len(raw)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_reference(**_arguments(), root=tmp_path)


@pytest.mark.parametrize("field", ("energy_only", "complete_energy", "forces"))
def test_endpoint_gate_rejects_numerical_mutants(field: str) -> None:
    reference_energy, reference_forces = load_reference(**_arguments())
    # Synthetic endpoint objects test the real assertion helper. Expected
    # numbers still come only from the authentic, integrity-checked receipt.
    energy = SimpleNamespace(
        converged=True, executed_backend="cpu", energy=reference_energy
    )
    complete = SimpleNamespace(
        converged=True,
        executed_backend="cpu",
        energy=reference_energy,
        forces=np.array(reference_forces, dtype=np.float64),
    )
    _assert_results_match_reference(
        energy, complete, reference_energy, reference_forces
    )
    if field == "energy_only":
        energy.energy += 1e-6
    elif field == "complete_energy":
        complete.energy += 1e-6
    else:
        # Keep translation intact: only the independent force comparison
        # catches this equal-and-opposite defect.
        complete.forces[0, 0] += 1e-5
        complete.forces[1, 0] -= 1e-5
    with pytest.raises(AssertionError):
        _assert_results_match_reference(
            energy, complete, reference_energy, reference_forces
        )


def test_oversized_receipt_fails_before_json_parsing(tmp_path: Path) -> None:
    _copy_reference(tmp_path)
    (tmp_path / "reference.json").write_bytes(b"x" * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="byte limit"):
        load_reference(**_arguments(), root=tmp_path)
