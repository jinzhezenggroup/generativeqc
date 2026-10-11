"""Independent fixed-density CUDA E/Fock/replay gates, not public KS admission.

Run inside a finite Slurm allocation with an explicitly built native driver.
PySCF supplies independent AO jets and pinned Libxc E/vxc. Production only
executes static AOT launchers and the existing bounded CUDA contractions.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from test_libxc_automatic_point_native import NAMES, _inputs, _oracle


@pytest.fixture(scope="session")
def native_xc_driver() -> Path:
    """Require explicit real-device opt-in and preserve scheduler visibility."""
    configured = os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_XC_EXECUTABLE")
    if not configured:
        pytest.skip("opt-in native automatic CUDA XC gate")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU gate requires srun"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "preserve Slurm GPU visibility"
    executable = Path(configured).resolve()
    assert executable.is_file(), "native CUDA XC driver is not built"
    return executable


def _independent_xc(
    name: str, record: dict, displacement: float
) -> tuple[float, np.ndarray]:
    """Contract pinned Libxc coefficients with independently evaluated AO jets."""
    gto = pytest.importorskip("pyscf.gto")
    numint = pytest.importorskip("pyscf.dft.numint")
    molecule = gto.M(
        atom=[("H", (0, 0, 0)), ("H", (0.1, 0.2, 1.4 + displacement))],
        basis={
            "H": [
                [
                    0,
                    [3.425250914, 0.1543289673],
                    [0.6239137298, 0.5353281423],
                    [0.168855404, 0.4446345422],
                ]
            ]
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    points = np.asarray(record["points"]).reshape(-1, 3)
    weights = np.asarray(record["weights"])
    ao = numint.eval_ao(molecule, points, deriv=1)
    spins = record["spins"]
    density = np.asarray(record["density"]).reshape(spins, 2, 2)
    physical_density = density if spins == 2 else np.repeat(density / 2, 2, axis=0)
    rho = np.einsum("pi,sij,pj->ps", ao[0], physical_density, ao[0])
    gradient = 2 * np.einsum("api,sij,pj->psa", ao[1:4], physical_density, ao[0])
    tau = 0.5 * np.einsum("api,sij,apj->ps", ao[1:4], physical_density, ao[1:4])
    inputs = np.column_stack((rho, gradient.reshape(-1, 6), tau))
    expected = _oracle(name, inputs, spins - 1)
    energy = float(weights @ expected[:, 0])
    potential = np.zeros((spins, 2, 2))
    for spin in range(spins):
        rho_column = 1 if spins == 1 else 1 + spin
        gradient_column = 2 if spins == 1 else 3 + 3 * spin
        kinetic_column = 5 if spins == 1 else 9 + spin
        potential[spin] = np.einsum(
            "p,pi,pj->ij", weights * expected[:, rho_column], ao[0], ao[0]
        )
        for axis in range(3):
            coefficient = weights * expected[:, gradient_column + axis]
            potential[spin] += np.einsum(
                "p,pi,pj->ij", coefficient, ao[axis + 1], ao[0]
            ) + np.einsum("p,pi,pj->ij", coefficient, ao[0], ao[axis + 1])
        # Oracle kinetic coefficients already include the generalized-KS half.
        potential[spin] += np.einsum(
            "p,api,apj->ij", weights * expected[:, kinetic_column], ao[1:4], ao[1:4]
        )
    return energy, potential.reshape(-1)


def _pack_tiles(values: np.ndarray, tile_points: int) -> np.ndarray:
    """Pack point-major channels into consecutive compact channel-major tiles."""
    return np.concatenate(
        [
            values[begin : begin + tile_points].T.reshape(-1)
            for begin in range(0, len(values), tile_points)
        ]
    )


def _unpack_tiles(
    values: list[float], count: int, channels: int, tile_points: int
) -> np.ndarray:
    """Invert tile layout while preserving final partial-tile channel strides."""
    packed = np.asarray(values)
    result = np.empty((count, channels))
    for begin in range(0, count, tile_points):
        width = min(tile_points, count - begin)
        result[begin : begin + width] = (
            packed[begin * channels : (begin + width) * channels]
            .reshape(channels, width)
            .T
        )
    return result


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("tile_points", (0, 16))
def test_automatic_cuda_aot_point_layout_matches_independent_libxc(
    native_xc_driver: Path, name: str, spins: int, tile_points: int
) -> None:
    """Qualify direct AOT single/batched E/vxc, vacuum, empty spin and work floors."""
    provider = emit_automatic_cuda_provider(name)
    inputs = _inputs(name, spins - 1)
    count = len(inputs)
    physical = np.stack(
        [inputs[:, [0, 2, 3, 4, 8]], inputs[:, [1, 5, 6, 7, 9]]], axis=1
    )
    if spins == 1:
        physical = physical.sum(axis=1, keepdims=True)
    features = physical[:, :, : provider.feature_terms].reshape(count, -1)
    weights = np.linspace(0.1, 1.1, count)
    packed = _pack_tiles(features, tile_points or count)
    submitted = (
        str(count) + " " + " ".join(str(value) for value in np.r_[packed, weights])
    )
    completed = subprocess.run(
        [
            str(native_xc_driver),
            "--automatic-points-json",
            str(automatic_functional_code(name)),
            str(spins),
            str(tile_points),
        ],
        input=submitted,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["error"] == 0
    coefficients = _unpack_tiles(
        record["coefficients"], count, features.shape[1], tile_points or count
    )
    totals = _unpack_tiles(record["totals"], count, 3, tile_points or count)
    oracle = _oracle(name, inputs, spins - 1)
    expected_coefficients = (
        oracle[:, 1:6]
        if spins == 1
        else np.column_stack([oracle[:, [1, 3, 4, 5, 9]], oracle[:, [2, 6, 7, 8, 10]]])
    )
    expected_coefficients = expected_coefficients.reshape(count, spins, 5)[
        :, :, : provider.feature_terms
    ].reshape(count, -1)
    np.testing.assert_allclose(
        coefficients, expected_coefficients, rtol=4e-11, atol=4e-11
    )
    np.testing.assert_allclose(
        totals[:, 0], weights * oracle[:, 0], rtol=4e-11, atol=4e-11
    )
    np.testing.assert_allclose(
        totals[:, 1:], weights[:, None] * inputs[:, :2], rtol=1e-15, atol=1e-30
    )


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA"))
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("tile_points", (0, 16))
def test_automatic_cuda_invalid_point_is_reported(
    native_xc_driver: Path, name: str, spins: int, tile_points: int
) -> None:
    """Preserve sticky device-domain errors instead of silently flooring negatives."""
    provider = emit_automatic_cuda_provider(name)
    features = np.zeros(spins * provider.feature_terms)
    features[0] = -0.1
    submitted = "1 " + " ".join(str(value) for value in np.r_[features, 1.0])
    completed = subprocess.run(
        [
            str(native_xc_driver),
            "--automatic-points-json",
            str(automatic_functional_code(name)),
            str(spins),
            str(tile_points),
        ],
        input=submitted,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["error"] == 3
    np.testing.assert_array_equal(record["coefficients"], np.zeros_like(features))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("displacement", (0.0, 0.08))
def test_automatic_cuda_fixed_density_energy_fock_and_replay(
    native_xc_driver: Path,
    name: str,
    spins: int,
    displacement: float,
    tmp_path: Path,
) -> None:
    """Exercise real AO/features/XC/Fock work, partial tiles and changed geometry."""
    command = [
        str(native_xc_driver),
        "--automatic-xc-json",
        str(automatic_functional_code(name)),
        str(spins),
        str(displacement),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    provider = emit_automatic_cuda_provider(name)
    assert record["expression_identity"] == provider.point.expression_identity
    assert record["source_identity"] == provider.source_identity
    assert record["point_count"] % record["tile_points"] != 0
    actual_potential = np.asarray(record["potential"])
    expected_energy, expected_potential = _independent_xc(name, record, displacement)
    energy_error = abs(record["energy"] - expected_energy)
    potential_error = float(np.max(np.abs(actual_potential - expected_potential)))
    assert energy_error < 2e-10
    np.testing.assert_allclose(
        actual_potential, expected_potential, rtol=2e-9, atol=2e-10
    )
    assert record["energy"] == record["replay_energy"]
    np.testing.assert_array_equal(actual_potential, record["replay_potential"])
    direction = np.asarray(record["direction"])
    directional = (record["plus_energy"] - record["minus_energy"]) / (
        2 * record["step"]
    )
    directional_error = abs(directional - float(actual_potential @ direction))
    assert directional_error < 2e-8
    tiles = (record["point_count"] + record["tile_points"] - 1) // record["tile_points"]
    assert record["evaluations"] == 4
    assert record["potential_calls"] == 4 * spins * tiles
    work_jets = 4 if provider.feature_terms == 5 else 1
    assert (
        record["potential_summands"]
        == 4 * 6 * record["point_count"] * work_jets * spins
    )
    (tmp_path / "native-xc-evidence.json").write_text(
        json.dumps(
            {
                "stage": "native-fixed-density-xc-only",
                "name": name,
                "spins": spins,
                "displacement": displacement,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "binary_sha256": file_hash(native_xc_driver),
                "expression_identity": provider.point.expression_identity,
                "source_identity": provider.source_identity,
                "device_bytes": record["device_bytes"],
                "evaluations": record["evaluations"],
                "potential_calls": record["potential_calls"],
                "potential_summands": record["potential_summands"],
                "energy_error": energy_error,
                "potential_error": potential_error,
                "directional_error": directional_error,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
