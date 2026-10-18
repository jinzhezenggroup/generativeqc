"""Production comparison gates stay independent of either candidate's grid."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.grid_pruning_gpu import (
    accuracy_gates,
    errors,
    main,
    qualification_errors,
)


def test_production_pruning_gates_preserve_per_water_energy_and_max_force() -> None:
    assert accuracy_gates("lda", 3) == {"energy": 1e-6, "force": 5e-5}
    assert accuracy_gates("pbe", 3) == {"energy": 1.5e-6, "force": 3e-5}
    for method in ("lda", "pbe"):
        small, large = accuracy_gates(method, 3), accuracy_gates(method, 96)
        assert large["energy"] == 32 * small["energy"]
        assert large["force"] == small["force"]
    with pytest.raises(ValueError, match="only LDA and PBE"):
        accuracy_gates("wb97m-v", 96)


def test_production_comparison_uses_total_energy_and_max_absolute_force() -> None:
    reference = {"energy": -100.0, "forces": [[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]]}
    candidate = {
        "energy": -99.0,
        "forces": np.array([[1.0, -2.5, 3.0], [0.0, 0.0, 0.0]]),
    }
    assert errors(candidate, reference) == {"energy": 1.0, "force": 0.5}


def test_qualification_reserves_observed_reference_variation() -> None:
    gate = accuracy_gates("pbe", 96)
    measured = {"energy": 3.4e-5, "force": 2.98e-5}
    stability = {"energy": 6e-7, "force": 5e-7}
    admitted = qualification_errors(measured, stability)
    assert measured["force"] < gate["force"] < admitted["force"]
    assert admitted["energy"] < gate["energy"]


def test_gpu_qualification_requires_scheduler_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="srun"):
        main(["--output", str(tmp_path / "qualification.json")])


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_gpu_qualification_rejects_retained_output_before_gpu_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, destination_kind: str
) -> None:
    """Reject retained paths before allocation checks, imports or evidence writes."""
    from benchmarks import _retention

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "qualification.json"
    existing.write_text("original evidence", encoding="utf-8")
    destination = retained
    if destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    elif destination_kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(destination / existing.name)])
    assert error.value.code == 2
    assert existing.read_text(encoding="utf-8") == "original evidence"
    assert not (tmp_path / ".artifacts").exists()
