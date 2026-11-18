"""Synthetic receipt controls, not additional scientific qualification runs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.validate_installed_cpu_hessian import main, validate


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _change(path: Path, **changes: Any) -> None:
    value = json.loads(path.read_text())
    value.update(changes)
    _write(path, value)


@pytest.fixture
def receipts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a translation-invariant raw operator with two distinct spin spaces."""
    root = Path(__file__).resolve().parents[2]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        (
            root / "tests/data/global_hybrid_ecp/native_curvature_manifest.json"
        ).read_bytes()
    )
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    specification, gates = manifest["cases"]["ni-uks"], manifest["gates"]
    reference_directory = tmp_path / "reference"
    reference_directory.mkdir()
    library = tmp_path / "synthetic-library"
    library.write_bytes(b"not an executable: unit-test provenance input")
    metadata = {
        name: specification[name]
        for name in ("atoms_bohr", "charge", "multiplicity", "max_eri_primitives")
    }
    metadata.update(
        method_identity="synthetic-operator",
        basis_manifest={"sha256": "fixture"},
        gates=gates,
    )
    reference = dict(
        metadata,
        accepted=True,
        stage="accepted",
        case="ni-uks",
        source_hashes={
            "tests/data/global_hybrid_ecp/native_curvature_manifest.json": manifest_sha
        },
        native_scf_policy={
            "max_iterations": specification["native_scf_max_iterations"]
        },
        minimum_occupied_root_overlap=0.999,
        central_errors={"density": 0.0, "energy": 0.0},
        oracle_scf_cross_seeding=False,
        production_displaced_scfs=0,
    )
    _write(reference_directory / "receipt.json", reference)
    diagnostic = {
        "public_calculator_endpoint": True,
        "numerical_runtime": "native-cpp-cpu",
        "runtime_compilation": False,
        "displaced_scf_evaluations": 0,
        "reference_engine_fallbacks": 0,
        "posthoc_symmetrization": False,
        "total_method_identity": metadata["method_identity"],
        "d3_pair_visits": 12,
        "response_residual_norm": 1e-12,
    }
    producer = dict(
        metadata,
        accepted=False,
        native_endpoints_complete=True,
        independent_validation_pending=True,
        stage="independent-validation-pending",
        oracle_modules_absent=dict.fromkeys(("pyscf", "dftd3", "cupy"), True),
        fixture_manifest_sha256=manifest_sha,
        scf_max_iterations=specification["native_scf_max_iterations"],
        library=str(library),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        matrix_diagnostics=diagnostic,
        hvp_0_diagnostics=diagnostic,
        hvp_1_diagnostics=diagnostic,
        seconds=1.0,
        maximum_rss_kib=1024,
    )
    producer_path = tmp_path / "installed-ni-uks.json"
    _write(producer_path, producer)
    coefficients = np.stack((np.eye(3), np.eye(3)))
    occupations = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    density = np.einsum("sik,sk,sjk->sij", coefficients, occupations, coefficients)
    np.savez(
        tmp_path / "installed-ni-uks-central.npz",
        coefficients=coefficients,
        occupations=occupations,
        density=density,
        energy=-1.01,
    )
    np.savez(
        reference_directory / "independent-state.npz",
        occupations=occupations,
        density=density,
        overlap=np.eye(3),
        electronic_energy=-1.0,
        d3_energy=-0.01,
    )
    matrix = np.kron(np.eye(4) - np.ones((4, 4)) / 4, np.eye(3)) * 0.1
    np.savez(tmp_path / "installed-ni-uks-matrix.npz", matrix=matrix)
    np.savez(reference_directory / "native-matrix.npz", matrix=matrix)
    for step in manifest["difference_steps_bohr"]:
        np.savez(reference_directory / f"independent-matrix-{step}.npz", matrix=matrix)
    for index, function in enumerate((np.sin, np.cos)):
        direction = function(np.arange(12) + 1).reshape(4, 3) * 0.1
        np.savez(
            tmp_path / f"installed-ni-uks-hvp-{index}.npz",
            direction=direction,
            value=(matrix @ direction.ravel()).reshape(4, 3),
        )
    return producer_path, reference_directory, manifest_path


def test_separate_acceptance_binds_inputs_without_relabeling_producer(
    receipts: tuple,
) -> None:
    producer, reference, manifest = receipts
    original = producer.read_bytes()
    result = validate(producer, reference, manifest, "ni-uks")
    assert result["accepted"] and not result["producer_modified"]
    assert not result["new_independent_oracles"]
    assert not result["ts_qualified"] and not result["scale_qualified"]
    assert result["errors"]["density"] == 0
    assert producer.read_bytes() == original
    assert (
        str((reference / "independent-matrix-0.003.npz").resolve())
        in result["referenced_members"]
    )


@pytest.mark.parametrize(
    "target,changes,match",
    [
        ("reference", {"accepted": False}, "not accepted"),
        ("reference", {"stage": "native-matrix-retained"}, "incomplete"),
        ("producer", {"native_endpoints_complete": False}, "not completed"),
        ("producer", {"accepted": True}, "not completed"),
        ("producer", {"failure": "SCF failure"}, "recorded a failure"),
        ("producer", {"method_identity": "other"}, "physical model"),
        ("producer", {"multiplicity": 1}, "physical input"),
        ("producer", {"fixture_manifest_sha256": "other"}, "manifest identity"),
        ("producer", {"gates": {"raw_hessian": 100.0}}, "frozen gates"),
        ("producer", {"scf_max_iterations": 601}, "SCF allowance"),
        ("producer", {"oracle_modules_absent": {"pyscf": False}}, "isolation"),
        ("reference", {"minimum_occupied_root_overlap": 0.98}, "root gate"),
        ("reference", {"oracle_scf_cross_seeding": True}, "independence contract"),
    ],
)
def test_unfinished_or_changed_receipts_fail_closed(
    receipts: tuple, target: str, changes: dict, match: str
) -> None:
    producer, reference, manifest = receipts
    _change(
        reference / "receipt.json" if target == "reference" else producer, **changes
    )
    with pytest.raises(ValueError, match=match):
        validate(producer, reference, manifest, "ni-uks")


@pytest.mark.parametrize(
    "panel", ["matrix", "hvp-0", "central", "library", "independent"]
)
def test_corrupt_or_wrong_state_outputs_are_not_accepted(
    receipts: tuple, panel: str
) -> None:
    producer, reference, manifest = receipts
    if panel == "library":
        (producer.parent / "synthetic-library").write_bytes(b"changed")
    else:
        path = (
            reference / "independent-matrix-0.003.npz"
            if panel == "independent"
            else producer.with_name(f"{producer.stem}-{panel}.npz")
        )
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files}
        key = {"hvp-0": "value", "central": "density"}.get(panel, "matrix")
        values[key] = values[key] + 0.01
        np.savez(path, **values)
    with pytest.raises(ValueError):
        validate(producer, reference, manifest, "ni-uks")


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j])
def test_nonreal_or_nonfinite_arrays_fail_closed(
    receipts: tuple, value: complex
) -> None:
    producer, reference, manifest = receipts
    np.savez(
        producer.with_name(f"{producer.stem}-matrix.npz"),
        matrix=np.full((12, 12), value),
    )
    with pytest.raises(ValueError, match="nonfinite or nonreal"):
        validate(producer, reference, manifest, "ni-uks")


@pytest.mark.parametrize(
    "field,value",
    [
        ("public_calculator_endpoint", False),
        ("runtime_compilation", True),
        ("displaced_scf_evaluations", 1),
        ("reference_engine_fallbacks", 1),
        ("posthoc_symmetrization", True),
        ("response_residual_norm", np.nan),
    ],
)
def test_native_endpoint_contract_is_required(
    receipts: tuple, field: str, value: Any
) -> None:
    producer, reference, manifest = receipts
    metadata = json.loads(producer.read_text())
    metadata["hvp_1_diagnostics"][field] = value
    _write(producer, metadata)
    with pytest.raises(ValueError, match="diagnostics failed"):
        validate(producer, reference, manifest, "ni-uks")


@pytest.mark.parametrize("accepted", [False, True])
def test_cli_preserves_inputs_and_refuses_to_overwrite_output(
    receipts: tuple, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    producer, reference, manifest = receipts
    if not accepted:
        _change(reference / "receipt.json", accepted=False)
    original = producer.read_bytes()
    output = producer.parent / "separate-validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--producer",
            str(producer),
            "--reference-directory",
            str(reference),
            "--manifest",
            str(manifest),
            "--case",
            "ni-uks",
            "--output",
            str(output),
        ],
    )
    assert main() == (0 if accepted else 1)
    assert json.loads(output.read_text())["accepted"] is accepted
    retained = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == retained
    assert producer.read_bytes() == original
