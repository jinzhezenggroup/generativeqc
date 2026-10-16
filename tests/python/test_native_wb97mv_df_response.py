"""Slurm-only fitted-primary algebra against expanded independent NumPy ERIs.

The executable runs with Python discovery disabled. Synthetic integral motion
isolates metric/source AD from the separate raw nuclear-integral consumer.
These tests do not qualify a molecular Hessian or endpoint performance.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from test_wb97mv_df_response_compiler import df_case, df_oracle


def _native(case: dict, path: Path, *, reject: bool = False) -> tuple:
    executable = os.environ.get("GENERATIVEQC_CUDA_WB97MV_DF_PROBE")
    if not executable:
        pytest.skip("requires native DF probe and an allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    spins, nao, _ = case["density"].shape
    naux = case["whitening"].shape[0]
    source, result = path / "source.bin", path / "output.bin"
    result.unlink(missing_ok=True)
    with source.open("wb") as output:
        np.array([nao, naux, spins, reject], dtype=np.uint64).tofile(output)
        for name in (
            "density",
            "three_center",
            "whitening",
            "eigenvalues",
            "d_density",
            "d_three_center",
            "d_metric",
        ):
            np.asarray(case[name], dtype=np.float64).tofile(output)
        np.array([case.get("threshold", 1e-10)], dtype=np.float64).tofile(output)
    completed = subprocess.run(
        [executable, str(source), str(result)],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    receipt = json.loads(completed.stdout)
    if reject:
        assert receipt == {"prepare_rejected": True} and not result.exists()
        return (), receipt
    assert receipt["lifecycle"]
    return _split(np.fromfile(result), spins, nao, naux), receipt


def _split(values: np.ndarray, spins: int, nao: int, naux: int) -> dict:
    shapes = (
        (),
        (spins, nao, nao),
        (spins, nao, nao),
        (nao, nao, naux),
        (naux, naux),
        (nao, nao, naux),
        (naux, naux),
    )
    names = (
        "energy",
        "fock",
        "fock_direction",
        "weight_three_center",
        "weight_metric",
        "weight_direction_three_center",
        "weight_direction_metric",
    )
    actual, offset = {}, 0
    for name, shape in zip(names, shapes, strict=True):
        size = int(np.prod(shape))
        actual[name] = values[offset : offset + size].reshape(shape)
        offset += size
    assert offset == values.size
    return actual


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("nao,naux", ((2, 3), (5, 7), (9, 13)))
def test_native_fitted_sources_and_frozen_fock(
    spins: int, nao: int, naux: int, tmp_path: Path
) -> None:
    case = df_case(spins, nao, naux)
    case["eigenvalues"] = np.linalg.eigvalsh(case["metric"])
    actual, receipt = _native(case, tmp_path)
    expected = df_oracle(case["density"], case["three_center"], case["metric"])
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, atol=5e-12, rtol=3e-12)
    for step in (3e-4, 1e-4):
        plus, minus = {}, {}
        for sign, output in ((1, plus), (-1, minus)):
            shifted = {
                name: case[name] + sign * step * case[f"d_{name}"]
                for name in ("density", "three_center", "metric")
            }
            moving = df_oracle(
                shifted["density"], shifted["three_center"], shifted["metric"]
            )
            fixed = df_oracle(
                case["density"], shifted["three_center"], shifted["metric"]
            )
            output["fock_direction"] = fixed["fock"]
            for field in ("three_center", "metric"):
                output[f"weight_direction_{field}"] = moving[f"weight_{field}"]
        for name, value in plus.items():
            np.testing.assert_allclose(
                actual[name], (value - minus[name]) / (2 * step), atol=2e-6, rtol=2e-6
            )
    assert receipt["workspace_bytes"] > (2 * nao**2 * naux + 2 * naux**2) * 8
    assert receipt["reference_terms"] > 0 and receipt["action_terms"] > 0


@pytest.mark.parametrize(
    "malformed", ("cutoff", "rank", "eigenvalue", "whitening", "density", "values")
)
def test_native_fitted_reference_fails_closed(malformed: str, tmp_path: Path) -> None:
    case = df_case(2)
    case["eigenvalues"] = np.linalg.eigvalsh(case["metric"])
    if malformed == "cutoff":
        case["threshold"] = case["eigenvalues"][0] / case["eigenvalues"][-1] / 1.05
    elif malformed == "rank":
        case["eigenvalues"][0] = 0
    elif malformed == "eigenvalue":
        case["eigenvalues"][0] = np.nan
    elif malformed == "whitening":
        case["whitening"][0, 0] = np.inf
    elif malformed == "density":
        case["density"][0, 0, 1] += 1
    else:
        case["three_center"][0, 1, 0] += 1
    _native(case, tmp_path, reject=True)


@pytest.mark.parametrize("spins", (1, 2))
def test_actual_native_ks_df_owner_sources_and_revocation(spins: int) -> None:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("requires actual native KS probe and an allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    completed = subprocess.run(
        [executable, "rks" if spins == 1 else "uks", "df-response"],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    result = json.loads(completed.stdout)
    assert result["revoked_without_publication"]
    gto = pytest.importorskip("pyscf.gto")
    df = pytest.importorskip("pyscf.df")
    from pyscf import lib

    lib.num_threads(1)
    centers = np.array(result["basis"][: 3 * result["atoms"]]).reshape(-1, 3)
    molecule = gto.M(
        atom=[("H", position) for position in centers],
        unit="Bohr",
        spin=spins - 1,
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
        verbose=0,
    )
    nao, naux = result["nao"], result["df_naux"]
    assert naux == nao == molecule.nao
    values = df.incore.aux_e2(molecule, molecule, aosym="s1")
    metric = molecule.intor("int2c2e")
    density = np.array(result["density"]).reshape(spins, nao, nao)
    delta_density = np.array(result["direction"]).reshape(density.shape)
    delta_values = np.array(result["df_delta_raw"]).reshape(values.shape)
    delta_metric = np.array(result["df_delta_metric"]).reshape(metric.shape)
    actual = _split(np.array(result["df_values"]), spins, nao, naux)
    for name, value in df_oracle(density, values, metric).items():
        np.testing.assert_allclose(actual[name], value, atol=2e-10, rtol=2e-9)
    step = 1e-4
    plus = df_oracle(
        density + step * delta_density,
        values + step * delta_values,
        metric + step * delta_metric,
    )
    minus = df_oracle(
        density - step * delta_density,
        values - step * delta_values,
        metric - step * delta_metric,
    )
    for field in ("three_center", "metric"):
        np.testing.assert_allclose(
            actual[f"weight_direction_{field}"],
            (plus[f"weight_{field}"] - minus[f"weight_{field}"]) / (2 * step),
            atol=3e-9,
            rtol=2e-7,
        )
    fixed_plus = df_oracle(
        density, values + step * delta_values, metric + step * delta_metric
    )
    fixed_minus = df_oracle(
        density, values - step * delta_values, metric - step * delta_metric
    )
    np.testing.assert_allclose(
        actual["fock_direction"],
        (fixed_plus["fock"] - fixed_minus["fock"]) / (2 * step),
        atol=3e-9,
        rtol=2e-7,
    )


@pytest.mark.parametrize("spins", (1, 2))
def test_actual_direct_owner_cannot_substitute_for_fitted_primary(spins: int) -> None:
    executable = os.environ.get("GENERATIVEQC_KS_WB97MV_GRID_PROBE")
    if not executable:
        pytest.skip("requires actual native KS probe and an allocated Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    completed = subprocess.run(
        [executable, "rks" if spins == 1 else "uks", "direct-df-response"],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert json.loads(completed.stdout)["df_direct_rejected"]
