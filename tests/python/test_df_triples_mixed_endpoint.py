"""Qualify approximate forward-W arithmetic through complete cold E+F calls.

Independent finite differences use the original physical FP64 Hamiltonian, not
the rounded candidate's noisy energy. All calls require a finite Slurm GPU job;
neither reference amplitudes nor orbitals enter the native production endpoint.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import typing

import numpy as np
import pytest
from test_df_complete_force import independent

from tools.generativeqc_posthf.fixtures import load_fixture

if typing.TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_MIXED_ENDPOINT_TEST") != "1",
    reason="requires finite Slurm allocation and the mixed-W endpoint executable",
)


def write_input(path: Path, metadata: dict, budget: int = 1 << 30) -> None:
    """Serialize the existing spherical molecular benchmark input contract."""
    inputs = metadata["inputs"]
    orbital, auxiliary = inputs["shells"], metadata["auxiliary_shells"]
    atoms = inputs["atomic_numbers"]
    lines = [f"{len(atoms)} {len(orbital)} {len(auxiliary)} {budget}"]
    for number, position in zip(atoms, inputs["coordinates"], strict=True):
        lines.append(" ".join(str(value) for value in (number, *position)))
    for shell in (*orbital, *auxiliary):
        primitives = shell["primitives"]
        lines.append(
            f"{shell['atom_index']} {shell['angular_momentum']} {len(primitives)}"
        )
        lines.extend(" ".join(str(value) for value in row) for row in primitives)
    path.write_text("\n".join(lines) + "\n")


def run_endpoint(
    tmp_path: Path,
    metadata: dict,
    *,
    mixed: bool,
    forces: bool = True,
    lambda_interval: int | None = 1,
) -> dict:
    """Use a fresh process with matched controls and no recycled solver state."""
    assert os.environ.get("SLURM_JOB_ID")
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    source = tmp_path / "input.txt"
    output = (
        tmp_path
        / f"result-{int(mixed)}-forces-{int(forces)}-lambda-{lambda_interval}.json"
    )
    write_input(source, metadata)
    command = [
        os.environ["GENERATIVEQC_DF_FORCE_ENDPOINT_BINARY"],
        str(source),
        str(output),
        "1",
        "1",
        str(int(forces)),
        "1",
        "8",
        "8",
        "8",
        "0",
        "1",
        "2",
        "1",
        "30",
        "0",
        "0",
        "1",
        "auto",
        "1",
        "0",
        "auto",
        "0",
        str(int(mixed)),
    ]
    if lambda_interval is not None:
        command.append(str(lambda_interval))
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text())


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
def test_complete_mixed_force_has_independent_energy_and_force_gates(
    name: str, tmp_path: Path
) -> None:
    metadata, _ = load_fixture(name)
    strict = run_endpoint(tmp_path, metadata, mixed=False)
    default_cadence = run_endpoint(tmp_path, metadata, mixed=False, lambda_interval=None)
    mixed = run_endpoint(tmp_path, metadata, mixed=True)
    cadenced = run_endpoint(tmp_path, metadata, mixed=True, lambda_interval=30)
    energy_only = run_endpoint(tmp_path, metadata, mixed=True, forces=False)
    expected = independent(metadata, True)[0]
    assert mixed["triples_w_fp32_requested"]
    assert mixed["triples_w_compute_bits"] == 32
    assert mixed["triples_w_accumulation_bits"] == 32
    assert mixed["triples_fp32_gemms"] > 0
    assert mixed["triples_precision_cast_elements"] > 0
    assert mixed["triples_w_precision_schedule_identity"]
    assert not mixed["triples_w_resource_fallback"]
    assert strict["lambda_true_residual_interval"] == 1
    assert default_cadence["lambda_true_residual_interval"] == 30
    assert default_cadence["triples_w_compute_bits"] == 64
    assert default_cadence["lambda_actions"] <= strict["lambda_actions"]
    if strict["lambda_iterations"] > 1:
        assert default_cadence["lambda_actions"] < strict["lambda_actions"]
    assert cadenced["lambda_true_residual_interval"] == 30
    assert cadenced["lambda_actions"] <= mixed["lambda_actions"]
    if mixed["lambda_iterations"] > 1:
        assert cadenced["lambda_actions"] < mixed["lambda_actions"]
    for key in ("lambda_residual", "z_residual", "stationarity"):
        assert default_cadence[key] < 1e-8
        assert cadenced[key] < 1e-8
    np.testing.assert_allclose(mixed["total_energy"], expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        mixed["total_energy"], strict["total_energy"], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        default_cadence["total_energy"], strict["total_energy"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        default_cadence["forces"], strict["forces"], atol=3e-7, rtol=0
    )
    np.testing.assert_allclose(mixed["forces"], strict["forces"], atol=3e-7, rtol=0)
    np.testing.assert_allclose(cadenced["forces"], strict["forces"], atol=3e-7, rtol=0)
    np.testing.assert_allclose(
        cadenced["total_energy"], mixed["total_energy"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        mixed["total_energy"], energy_only["total_energy"], atol=1e-12, rtol=0
    )
    forces = np.asarray(cadenced["forces"]).reshape(-1, 3)
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=3e-8, rtol=0)

    direction = np.random.default_rng(1764).normal(size=forces.shape)
    direction /= np.linalg.norm(direction)
    finite_differences = []
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            displaced = copy.deepcopy(metadata)
            displaced["inputs"]["coordinates"] = (
                np.asarray(metadata["inputs"]["coordinates"]) + sign * step * direction
            ).tolist()
            energies.append(independent(displaced, True)[0])
        finite_difference_force = -(energies[1] - energies[0]) / (2 * step)
        np.testing.assert_allclose(
            np.sum(forces * direction), finite_difference_force, atol=3e-7, rtol=0
        )
        finite_differences.append(
            {
                "step_bohr": step,
                "energies": energies,
                "force": finite_difference_force,
                "absolute_error": abs(
                    float(np.sum(forces * direction)) - finite_difference_force
                ),
            }
        )
    (tmp_path / "accuracy.json").write_text(
        json.dumps(
            {
                "oracle_total_energy": expected,
                "lambda_original_actions": mixed["lambda_actions"],
                "lambda_cadenced_actions": cadenced["lambda_actions"],
                "energy_error": abs(mixed["total_energy"] - expected),
                "force_error_vs_strict": float(
                    np.max(np.abs(forces.ravel() - np.asarray(strict["forces"])))
                ),
                "direction": direction.tolist(),
                "finite_differences": finite_differences,
            },
            indent=2,
        )
        + "\n"
    )
