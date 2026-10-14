"""Supplemental scientific evidence must not complete a failed timing series."""

import hashlib
import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_readme_dft_benchmark import completed_sample

from benchmarks.readme_accuracy_oracle import SCHEMA as ORACLE_SCHEMA
from benchmarks.readme_accuracy_oracle import (
    merge_geometry_oracles,
    read_geometry_oracle,
)
from benchmarks.readme_dft import SCHEMA
from tools.render_readme_dft import collect_point


def receipt() -> dict[str, Any]:
    """Small independent geometry receipt with the same endpoint contract."""
    return {
        "schema": ORACLE_SCHEMA,
        "status": "complete",
        "geometry": 1,
        "protocol": {
            "method": "WB97M-V/RKS",
            "atoms": 3,
            "repeats": 5,
            "force_return": "host_array_in_timed_endpoint",
            "reference_fock_policy": "full-density-rebuild",
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "reference_gradient_tolerance": 1e-10,
            "max_iterations": 100,
        },
        "basis_sha256": "b" * 64,
        "initial_density": "GPU4PySCF default atomic guess; no native density, orbitals, energy or force input",
        "converged": True,
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "convergence": {
            "converged": True,
            "iterations": 19,
            "final_residuals": {
                "energy_change_hartree": 1e-13,
                "density_rms": 1e-12,
                "orbital_gradient_norm": 1e-11,
            },
        },
        "source_file_sha256": {"reference.py": "c" * 64},
        "driver_sha256": "d" * 64,
        "scheduler": {"SLURM_JOB_ID": "independent-reference-receipt"},
        "diagnostic_seconds": 1180.0,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "basis",
        "force_shape",
        "energy",
        "residual",
        "seed",
        "duplicate",
        "nonfinite_forces",
        "nonfinite_residual",
        "iterations",
        "driver_hash",
        "source_hash",
    ],
)
def test_invalid_scientific_oracle_cannot_qualify_a_native_sample(
    mutation: str,
) -> None:
    oracle = receipt()
    scientific = deepcopy(oracle["protocol"])
    baselines = {}
    if mutation == "protocol":
        oracle["protocol"]["density_tolerance"] = 1e-4
    elif mutation == "basis":
        oracle["basis_sha256"] = "wrong"
    elif mutation == "force_shape":
        oracle["forces"] = [[0.0, 0.0, 0.0]]
    elif mutation == "energy":
        oracle["energy"] = float("nan")
    elif mutation == "residual":
        oracle["convergence"]["final_residuals"]["orbital_gradient_norm"] = 1e-5
    elif mutation == "seed":
        oracle["initial_density"] = "native density"
    elif mutation == "nonfinite_forces":
        oracle["forces"][0][1] = float("inf")
    elif mutation == "nonfinite_residual":
        oracle["convergence"]["final_residuals"]["density_rms"] = float("nan")
    elif mutation == "iterations":
        oracle["convergence"]["iterations"] = 0.5
    elif mutation == "driver_hash":
        oracle["driver_sha256"] = "z" * 64
    elif mutation == "source_hash":
        oracle["source_file_sha256"] = {"reference.py": "not a digest"}
    else:
        baselines[1] = {}
    with pytest.raises(ValueError, match="geometry accuracy oracle"):
        merge_geometry_oracles(baselines, [oracle], scientific, basis_sha256="b" * 64)


def retained_point(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Retain a full native population and an incomplete reference population."""
    oracle = receipt()
    directory = tmp_path / "3"
    directory.mkdir()
    reference = {
        "schema": SCHEMA,
        "protocol": oracle["protocol"],
        "status": "running",
        "records": [dict(completed_sample(), energy=-1.0)],
    }
    reference_path = directory / "reference.json"
    reference_path.write_text(json.dumps(reference))
    (directory / "reference.outcome").write_text(json.dumps({"exit_code": 124}))
    oracle_path = directory / "accuracy-reference-moved.json"
    oracle_path.write_text(json.dumps(oracle))
    rows = [
        dict(completed_sample(phase, geometry), energy=-1.0, repeat=repeat)
        for phase, geometry, count in (
            ("cold", 0, 1),
            ("warm", 0, 5),
            ("moved", 1, 1),
            ("moved-warm", 1, 5),
        )
        for repeat in range(count)
    ]
    native = {
        "schema": SCHEMA,
        "protocol": oracle["protocol"],
        "status": "measured",
        "records": rows,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "basis_file_sha256": oracle["basis_sha256"],
        "accuracy_reference_files": [
            {
                "file": oracle_path.name,
                "sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (directory / "native.json").write_text(json.dumps(native))
    (directory / "native.outcome").write_text(json.dumps({"exit_code": 0}))
    return oracle, oracle_path


def test_geometry_oracle_qualifies_only_native_accuracy_and_is_hash_bound(
    tmp_path: Path,
) -> None:
    oracle, oracle_path = retained_point(tmp_path)
    point = collect_point(tmp_path, 3)
    assert point["engines"]["native"]["status"] == "measured"
    assert point["engines"]["reference"]["status"] == "timeout"
    assert len(point["engines"]["reference"]["records"]) == 1
    assert point["engines"]["reference"]["phase_status"]["moved"] == "incomplete"
    assert point["independent_accuracy_receipts"] == [oracle]
    baseline = merge_geometry_oracles(
        {}, [oracle], oracle["protocol"], basis_sha256="b" * 64
    )[1]
    assert "complete_seconds" not in baseline
    oracle_path.write_text(json.dumps(dict(oracle, diagnostic_seconds=1.0)))
    with pytest.raises(ValueError, match="different accuracy-only reference"):
        collect_point(tmp_path, 3)


def test_renderer_rejects_accuracy_receipt_symlink_escape(tmp_path: Path) -> None:
    _, path = retained_point(tmp_path)
    outside = tmp_path / path.name
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="accuracy reference"):
        collect_point(tmp_path, 3)


@pytest.mark.parametrize("alias", [False, True])
def test_geometry_driver_rejects_retained_output_before_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    alias: bool,
) -> None:
    from benchmarks import _support
    from benchmarks import readme_geometry_oracle as driver

    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    original = retained / "receipt.json"
    original.write_text("retained reference")
    destination = retained
    if alias:
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "host-only-fixture")
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "geometry-oracle",
            "--template",
            "unused-template.json",
            "--basis",
            "unused-basis.json",
            "--geometry",
            "1",
            "--output",
            str(destination / "receipt.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert "raw_output_path" in capsys.readouterr().err
    assert original.read_text() == "retained reference"


def test_receipt_hash_binds_exactly_the_parsed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, path = retained_point(tmp_path)
    old_bytes = path.read_bytes()
    changed = json.dumps(dict(original, energy=42.0)).encode()
    read_bytes = Path.read_bytes

    def replace_after_read(source: Path) -> bytes:
        payload = read_bytes(source)
        if source == path:
            source.write_bytes(changed)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    parsed, binding = read_geometry_oracle(path, directory=path.parent)
    assert parsed == original
    assert binding["sha256"] == hashlib.sha256(old_bytes).hexdigest()
    assert read_bytes(path) == changed


@pytest.mark.parametrize("quantity,error", [("energy", 2e-8), ("forces", 2e-7)])
def test_renderer_rechecks_accuracy_instead_of_trusting_native_gate(
    tmp_path: Path, quantity: str, error: float
) -> None:
    _, path = retained_point(tmp_path)
    native_path = path.parent / "native.json"
    native = json.loads(native_path.read_text())
    row = next(row for row in native["records"] if row["phase"] == "moved")
    if quantity == "energy":
        row["energy"] += error
    else:
        row["forces"][0][0] += error
    assert row["gate"] is True
    native_path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="missing or failed endpoint"):
        collect_point(tmp_path, 3)


@pytest.mark.parametrize("fitted", [False, True])
def test_geometry_driver_scf_uses_final_reference_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fitted: bool
) -> None:
    """GPU4PySCF DF construction copies instance attributes to a new owner."""
    from dataclasses import replace

    from generativeqc import GridSpec, load_r2scan3c_basis

    from benchmarks.readme_dft import specification
    from benchmarks.readme_omol25 import protocol
    from benchmarks.readme_wb97mv import configure_reference_full_fock

    cp = SimpleNamespace(
        asnumpy=np.asarray,
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    from benchmarks import readme_geometry_oracle as driver

    class Engine:
        converged = True
        grids = nlcgrids = SimpleNamespace(weights=[1.0])

        def __init__(self) -> None:
            self.owner = "direct"
            self.direct_scf = True

        def get_veff(self, **kwargs: Any) -> str:
            assert kwargs.get("dm_last") is None
            assert kwargs.get("vhf_last") is None
            return self.owner

        def density_fit(self, **kwargs: Any) -> "Engine":
            other = Engine()
            other.__dict__.update(self.__dict__)
            other.owner = "fitted"
            return other

    def reference_engine(*args: Any, full_fock: bool = False, **kwargs: Any) -> Engine:
        engine = Engine()
        if full_fock:
            configure_reference_full_fock(engine)
        return engine

    def endpoint(engine: Engine, density: Any, **kwargs: Any) -> tuple:
        assert density is None
        assert engine.direct_scf is False
        assert engine.get_veff(
            dm="reference density", dm_last="stale density", vhf_last="stale potential"
        ) == ("fitted" if fitted else "direct")
        assert getattr(engine, "mixed_exchange", False) is fitted
        return -1.0, np.zeros((3, 3)), 1.0

    basis = load_r2scan3c_basis()
    auxiliary = (
        SimpleNamespace(name="fixture auxiliary", identity="auxiliary identity")
        if fitted
        else None
    )
    p = protocol(
        3,
        basis,
        GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32),
        5,
        benchmark=replace(specification("wb97m-v"), density_fitting=fitted),
        auxiliary_basis=auxiliary,
    )
    basis_path = tmp_path / "basis.json"
    basis_path.write_text("host-only basis fixture")
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "protocol": p,
                "basis_file_sha256": hashlib.sha256(
                    basis_path.read_bytes()
                ).hexdigest(),
            }
        )
    )
    output = tmp_path / "oracle.json"
    monkeypatch.setenv("SLURM_JOB_ID", "host-only-fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "geometry-oracle",
            "--template",
            str(template),
            "--basis",
            str(basis_path),
            "--geometry",
            "1",
            "--output",
            str(output),
            *(["--auxiliary", str(basis_path)] if fitted else []),
        ],
    )
    monkeypatch.setattr(driver, "load_comparison_basis", lambda *a, **kw: (basis, {}))
    monkeypatch.setattr(
        driver, "load_retained_comparison_basis", lambda *a, **kw: (auxiliary, {})
    )
    monkeypatch.setattr(driver, "source_hashes", lambda: {"fixture": "a" * 64})
    monkeypatch.setattr(driver, "environment_metadata", lambda **kw: {})
    monkeypatch.setattr(driver, "cuda_accelerator_metadata", lambda cp: {})
    monkeypatch.setattr(driver, "reference_engine", reference_engine)
    monkeypatch.setattr(
        driver,
        "configure_reference_mixed_rsh_df",
        lambda e: setattr(e, "mixed_exchange", True),
    )
    monkeypatch.setattr(
        driver,
        "reference_work_counter",
        lambda e: nullcontext({"tracker": None, "scf_jk_builds": 1}),
    )
    monkeypatch.setattr(
        driver, "reference_vv10_domain", lambda threshold: nullcontext({})
    )
    monkeypatch.setattr(driver, "reference_energy_forces", endpoint)
    monkeypatch.setattr(
        driver, "gpu_convergence_payload", lambda *a: [receipt()["convergence"]]
    )
    driver.main()
    assert json.loads(output.read_text())["status"] == "complete"
