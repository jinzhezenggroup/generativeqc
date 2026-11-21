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
        np.array(
            [
                nao,
                naux,
                spins,
                reject,
                not case.get("coulomb", True),
                "eigenvectors" in case,
            ],
            dtype=np.uint64,
        ).tofile(output)
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
        np.array(
            [case.get("threshold", 1e-10), case["exchange_fraction"]], dtype=np.float64
        ).tofile(output)
        if "eigenvectors" in case:
            np.asarray(case["eigenvectors"], dtype=np.float64).ravel(order="F").tofile(
                output
            )
    completed = subprocess.run(
        [executable, str(source), str(result)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    if reject:
        assert receipt["prepare_rejected"] and not result.exists()
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
@pytest.mark.parametrize("nao,naux", ((2, 3), (5, 7), (9, 13), (17, 19)))
@pytest.mark.parametrize(
    "fraction,coulomb", ((0.0, True), (0.15, True), (0.25, True), (0.85, False))
)
def test_native_fitted_sources_and_frozen_fock(
    spins: int, nao: int, naux: int, fraction: float, coulomb: bool, tmp_path: Path
) -> None:
    case = df_case(spins, nao, naux)
    case["eigenvalues"] = np.linalg.eigvalsh(case["metric"])
    case["exchange_fraction"] = np.asarray(fraction)
    case["coulomb"] = coulomb
    actual, receipt = _native(case, tmp_path)
    expected = df_oracle(
        case["density"], case["three_center"], case["metric"], fraction, coulomb=coulomb
    )
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
                shifted["density"],
                shifted["three_center"],
                shifted["metric"],
                fraction,
                coulomb=coulomb,
            )
            fixed = df_oracle(
                case["density"],
                shifted["three_center"],
                shifted["metric"],
                fraction,
                coulomb=coulomb,
            )
            output["fock_direction"] = fixed["fock"]
            for field in ("three_center", "metric"):
                output[f"weight_direction_{field}"] = moving[f"weight_{field}"]
        for name, value in plus.items():
            np.testing.assert_allclose(
                actual[name], (value - minus[name]) / (2 * step), atol=2e-6, rtol=2e-6
            )
    assert 0 < receipt["workspace_bytes"] <= 512 << 20
    assert receipt["reference_terms"] > 0 and receipt["action_terms"] > 0
    assert receipt["matrix_reference_terms"] <= 1_000_000_000
    assert receipt["matrix_action_terms"] <= 1_000_000_000
    if fraction and nao >= 17 and naux >= 19:
        assert receipt["matrix_reference_terms"] and receipt["matrix_action_terms"]


@pytest.mark.parametrize(
    "malformed", ("cutoff", "rank", "eigenvalue", "whitening", "density", "values")
)
@pytest.mark.parametrize("coulomb", (True, False))
def test_native_fitted_reference_fails_closed(
    malformed: str, coulomb: bool, tmp_path: Path
) -> None:
    case = df_case(2)
    case["coulomb"] = coulomb
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
    _, receipt = _native(case, tmp_path, reject=True)
    expected_check = {
        "cutoff": "metric spectrum",
        "rank": "metric spectrum",
        "eigenvalue": "metric spectrum",
        "whitening": "SCF whitening",
        "density": "reference density",
        "values": "raw three-center source",
    }
    assert expected_check[malformed] in receipt["detail"]


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("fraction", (0.0, 0.15))
def test_ill_conditioned_full_rank_weights(
    spins: int, fraction: float, tmp_path: Path
) -> None:
    """Resolved small metric eigenvalues must not make source weights asymmetric."""
    nao, naux = 8, 20
    case = df_case(spins, nao=nao, naux=naux)
    rng = np.random.default_rng(2397)
    frame, _ = np.linalg.qr(rng.normal(size=(naux, naux)))
    eigenvalues = np.geomspace(5e-8, 1.0, naux)
    metric = (frame * eigenvalues) @ frame.T
    case.update(
        metric=metric,
        exchange_fraction=fraction,
        eigenvalues=eigenvalues,
        whitening=frame.T / np.sqrt(eigenvalues[:, None]),
        d_metric=metric @ case["d_metric"] @ metric,
    )
    # Model physical integrals with weak components along the almost-null
    # eigenvectors. The analytic oracle below solves the metric separately.
    for field in ("three_center", "d_three_center"):
        case[field] = case[field] @ metric
    actual, _ = _native(case, tmp_path)

    def weights(density: np.ndarray, values: np.ndarray) -> tuple:
        fitted = np.linalg.solve(metric, values.reshape(-1, naux).T).T.reshape(
            values.shape
        )
        total = density.sum(axis=0)
        fitted_charge = np.einsum("ij,ijp->p", total, fitted)
        exchange = fraction / (4 if spins == 1 else 2)
        raw = total[:, :, None] * fitted_charge - 2 * exchange * np.einsum(
            "sik,sjl,klp->ijp", density, density, fitted
        )
        metric_weight = -0.5 * np.outer(
            fitted_charge, fitted_charge
        ) + exchange * np.einsum(
            "sik,sjl,ijp,klq->pq", density, density, fitted, fitted, optimize=True
        )
        return raw, metric_weight

    density, values = case["density"], case["three_center"]
    expected = weights(density, values)
    # Complex-step avoids subtracting two independently ill-conditioned
    # solves. The weight formula is polynomial except for the linear solve.
    step = 1e-12
    metric = metric + 1j * step * case["d_metric"]
    shifted = weights(
        density + 1j * step * case["d_density"],
        values + 1j * step * case["d_three_center"],
    )
    for index, field in enumerate(("three_center", "metric")):
        np.testing.assert_allclose(
            actual[f"weight_{field}"], expected[index], atol=1e-8, rtol=1e-8
        )
        np.testing.assert_allclose(
            actual[f"weight_direction_{field}"],
            shifted[index].imag / step,
            atol=1e-8,
            rtol=1e-8,
        )


def _spectral_case(spins: int, fraction: float, coulomb: bool, naux: int = 4) -> dict:
    case = df_case(spins, nao=3, naux=naux)
    _, vectors = np.linalg.eigh(case["metric"])
    # Both clusters are degenerate, but the relative cutoff separates them.
    keep = np.arange(naux) >= naux // 2
    values = np.where(keep, 1.27, 0.12)
    roots = np.where(keep, 1 / np.sqrt(values), 0.0)
    case.update(
        metric=(vectors * values) @ vectors.T,
        eigenvectors=vectors,
        eigenvalues=values,
        whitening=roots[:, None] * vectors.T,
        threshold=0.28,
        exchange_fraction=fraction,
        coulomb=coulomb,
    )
    return case


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("naux", (4, 19), ids=("small", "tiled-tail"))
@pytest.mark.parametrize(
    "fraction,coulomb",
    ((0.0, True), (0.15, True), (0.85, False)),
    ids=("coulomb", "hybrid", "exchange-only"),
)
def test_native_separated_spectral_fitted_sources(
    spins: int, naux: int, fraction: float, coulomb: bool, tmp_path: Path
) -> None:
    case = _spectral_case(spins, fraction, coulomb, naux)
    actual, _ = _native(case, tmp_path)

    def fitted(density: np.ndarray, raw: np.ndarray, metric: np.ndarray) -> dict:
        values, vectors = np.linalg.eigh(metric)
        keep = values > case["threshold"] * values[-1]
        assert np.count_nonzero(keep) == naux - naux // 2
        inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
        return df_oracle(
            density, raw, metric, fraction, coulomb=coulomb, inverse=inverse
        )

    def primal(density: np.ndarray, raw: np.ndarray, metric: np.ndarray) -> dict:
        values = fitted(density, raw, metric)
        # First metric weights use energy differences after rebuilding the
        # truncated inverse. No Loewner formulas enter the reference.
        # A wider inner stencil limits energy cancellation in the subsequent
        # directional difference for the larger, degenerate fixture.
        inner_step = 1.6e-3 if naux > 4 else 2e-4
        weight = np.zeros_like(metric)
        for row in range(metric.shape[0]):
            for column in range(row, metric.shape[0]):
                seed = np.zeros_like(metric)
                seed[row, column] += 0.5
                seed[column, row] += 0.5
                slopes = []
                for step in (inner_step, inner_step / 2):
                    plus = fitted(density, raw, metric + step * seed)["energy"]
                    minus = fitted(density, raw, metric - step * seed)["energy"]
                    slopes.append((plus - minus) / (2 * step))
                weight[row, column] = weight[column, row] = (
                    4 * slopes[1] - slopes[0]
                ) / 3
        values["weight_metric"] = weight
        return values

    feeds = [case[name] for name in ("density", "three_center", "metric")]
    directions = [case[f"d_{name}"] for name in ("density", "three_center", "metric")]
    expected = primal(*feeds)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, atol=2e-9, rtol=3e-10)
    estimates = []
    for step in (0.002, 0.001):
        signs = []
        for sign in (1, -1):
            shifted = [
                value + sign * step * motion
                for value, motion in zip(feeds, directions, strict=True)
            ]
            moving = primal(*shifted)
            fixed = fitted(feeds[0], *shifted[1:])
            signs.append(
                (fixed["fock"], moving["weight_three_center"], moving["weight_metric"])
            )
        estimates.append(
            [(plus - minus) / (2 * step) for plus, minus in zip(*signs, strict=True)]
        )
    keys = (
        "fock_direction",
        "weight_direction_three_center",
        "weight_direction_metric",
    )
    for name, coarse, fine in zip(keys, *estimates, strict=True):
        np.testing.assert_allclose(
            actual[name], (4 * fine - coarse) / 3, atol=5e-8, rtol=3e-8
        )


@pytest.mark.parametrize("side", (0.99, 1.01))
def test_native_spectral_metric_rejects_either_side_of_cutoff(
    side: float, tmp_path: Path
) -> None:
    case = _spectral_case(1, 0.15, True)
    case["eigenvalues"][1] = side * case["threshold"] * case["eigenvalues"][-1]
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
