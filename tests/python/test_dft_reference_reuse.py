"""Reference reuse binds scientific data and provenance, not historical timings."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc import _dft_reference, tune_cli
from generativeqc._dft_autotune import numerical_gate
from generativeqc.profiles import file_hash

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def reference(tmp_path: Path) -> Any:
    pytest.importorskip("pyscf")
    workload = {
        "kind": "dft",
        "method": "pbe0-rks",
        "basis": "sto-3g",
        "atoms": [["H", [0.0, 0.0, -0.7]], ["H", [0.0, 0.0, 0.7]]],
        "representation": "spherical",
        "charge": 0,
        "multiplicity": 1,
        "batch": 1,
        "grid_level": 3,
    }
    model = _dft_reference.model(workload)
    sample = {
        "independent_reference": "synthetic unit-test oracle, not hardware evidence",
        "converged": True,
        "energies": [-1.0],
        "forces": [[[0.0, 0.0, 0.0]] * 2],
        "settings": {
            "xc_grid": model["grid"],
            "small_rho_cutoff": 0.0,
            "force_grid_response": True,
        },
    }
    document = {
        "schema": _dft_reference.SCHEMA,
        "model": model,
        "producer": {
            "engine": "gpu4pyscf",
            "versions": {"gpu4pyscf": "synthetic"},
            "source_sha256": "a" * 64,
        },
        "observations": [sample, copy.deepcopy(sample)],
    }
    path = tmp_path / "reference.json"

    def save() -> str:
        path.write_text(json.dumps(document))
        return file_hash(path)

    return workload, document, path, save


def test_reuse_keeps_every_observation_and_original_producer(reference: Any) -> None:
    workload, document, path, save = reference
    document["observations"][1]["forces"][0][0][0] = 2e-4
    cached = _dft_reference.load(path, save(), workload)
    checked = numerical_gate([document["observations"][0]], cached)
    assert not checked["passed"] and checked["maximum_force_error"] == 2e-4
    assert checked["reference_observation_count"] == 2
    assert cached["provenance"]["producer"] == document["producer"]


@pytest.mark.parametrize(
    "field,value",
    (
        ("basis", "def2-svp"),
        ("representation", "cartesian"),
        ("grid_level", 2),
        ("charge", 2),
        ("multiplicity", 3),
        ("method", "pbe-rks"),
        ("batch", 2),
    ),
)
def test_reuse_refuses_scientific_changes(
    reference: Any, field: str, value: Any
) -> None:
    workload, _document, path, save = reference
    digest = save()
    changed = {**workload, field: value}
    with pytest.raises(ValueError):
        _dft_reference.load(path, digest, changed)


@pytest.mark.parametrize(
    "failure",
    (
        "checksum",
        "geometry",
        "basis-contents",
        "provenance",
        "empty",
        "unconverged",
        "grid",
        "response",
        "nonfinite",
        "missing-forces",
        "wrong-layout",
    ),
)
def test_reuse_refuses_corruption_or_incomplete_evidence(
    reference: Any, failure: str
) -> None:
    workload, document, path, save = reference
    observation = document["observations"][0]
    if failure == "geometry":
        document["model"]["atoms_bohr"][0][1][0] = 0.1
    elif failure == "basis-contents":
        document["model"]["reference_gaussian_hash"] = "b" * 64
    elif failure == "provenance":
        document["producer"] = {}
    elif failure == "empty":
        document["observations"] = []
    elif failure == "unconverged":
        observation["converged"] = False
    elif failure == "grid":
        observation["settings"]["xc_grid"] = {"level": 0}
    elif failure == "response":
        observation["settings"]["force_grid_response"] = False
    elif failure == "nonfinite":
        observation["energies"] = [float("nan")]
    elif failure == "missing-forces":
        del observation["forces"]
    elif failure == "wrong-layout":
        observation["forces"] = [[[0.0, 0.0, 0.0]]]
    digest = save()
    if failure == "checksum":
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        _dft_reference.load(path, digest, workload)


def test_scientific_reference_does_not_transfer_hardware_performance(
    reference: Any,
) -> None:
    workload, document, path, save = reference
    cached = _dft_reference.load(
        path, save(), {**workload, "device_id": 7, "source_identity": "new-build"}
    )
    assert cached["observations"] == document["observations"]


@pytest.mark.parametrize(
    "arguments",
    (
        ["--reference-file", "unused.json"],
        ["--reference-sha256", "a" * 64],
        [
            "--reference-file",
            "unused.json",
            "--reference-sha256",
            "a" * 64,
            "--method",
            "rhf",
        ],
    ),
)
def test_reference_cli_requires_explicit_document_pin_and_dft(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        tune_cli.main(["unused.xyz", *arguments])
