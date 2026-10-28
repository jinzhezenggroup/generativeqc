"""Opt-in, Slurm-only qualification of production terminal DIIS observations."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_CUDA_DIIS_WORK_TEST") != "1",
    reason="requires explicit finite Slurm allocation and native CUDA library",
)
TRACE = "GENERATIVEQC_DF_PROGRESS_TRACE"
WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]
MOVED = [WATER[0], (1, (1.47, 0.02, 1.11)), WATER[2]]
GRID = GridSpec(24, 8, 16)


def receipts(path: Path) -> list[dict]:
    """Reject partial receipts, including failed terminal-copy publication."""
    scopes: dict[int, list[dict]] = {}
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["name"] == "cuda_diis_work":
            scopes.setdefault(event["id"], []).append(event)
    result = []
    for events in scopes.values():
        assert events[0]["event"] == "BEGIN"
        assert events[-1]["event"] == "END"
        assert events[-1]["status"] == "completed"
        assert all(e["execution"] == "cuda_completed" for e in events)
        row = {e["key"]: e["value"] for e in events if e["event"] == "VALUE"}
        assert row["live_history"] <= row["history_capacity"]
        assert row["history_head"] < row["history_capacity"]
        if row["work_coverage"] == "observed":
            assert row["residual_vector_elements"] == (
                row["gram_dot_reductions"] * row["nbf"] ** 2 * row["spins"]
            )
        result.append(row)
    return result


def calculator(method: str, history: int = 2, **options: object) -> Calculator:
    settings = {
        "method": method,
        "basis": "sto-3g",
        "device": "cuda",
        "precision": "fp64",
        "initial_guess": None,
        "diis_history": history,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "max_iterations": 200,
    }
    if method != "rhf":
        settings["ks_options"] = KsOptions(grid=GRID, tile_points=256)
    return Calculator(**(settings | options))


@pytest.fixture(scope="module")
def references() -> dict:
    """Independent CPU SCF and analytic moving-grid gradients at both geometries."""
    from generativeqc._model_resolution import snapshot_basis
    from pyscf import gto, lib, scf

    from benchmarks.pbe0_precision import reference

    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    lib.num_threads(2)
    # Share coefficient inputs, not integrals: library STO-3G tables can differ
    # in decimal precision enough to exceed the independent energy gate.
    basis = {
        {1: "H", 8: "O"}[element.atomic_number]: [
            [
                shell.angular_momentum,
                *[
                    [float(exponent), *[float(row[i]) for row in shell.coefficients]]
                    for i, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in snapshot_basis("sto-3g", "cartesian").elements
        if element.atomic_number in (1, 8)
    }
    result = {}
    for geometry, atoms in enumerate((WATER, MOVED)):
        molecule = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
        hf = scf.RHF(molecule)
        hf.conv_tol, hf.conv_tol_grad, hf.max_cycle = 1e-13, 1e-10, 200
        energy = hf.kernel()
        assert hf.converged
        result["rhf", geometry] = (energy, -hf.nuc_grad_method().kernel())
        ks = reference(
            "diis-water",
            GRID,
            2,
            functional="PBE",
            atoms=atoms,
            basis="sto-3g",
            case_snapshot=SimpleNamespace(
                charge=0, multiplicity=1, basis_representation="cartesian"
            ),
        )
        result["pbe-rks", geometry] = (ks["energy"], np.array(ks["forces"]))
    return result


@pytest.mark.parametrize(
    "method,chunk", [("rhf", "1"), ("pbe-rks", "1"), ("pbe-rks", "2")]
)
def test_complete_endpoints_reset_and_replay(
    method: str,
    chunk: str,
    references: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM", "1")
    monkeypatch.setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION", "ordered")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", chunk)
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_REPLAY", "small-native")
    # Different convergence rates exercise independent and inactive histories.
    all_results, endpoints, transports = [], [], []
    for enabled in (False, True):
        path = tmp_path / "work.jsonl"
        if enabled:
            monkeypatch.setenv(TRACE, str(path))
        else:
            monkeypatch.delenv(TRACE, raising=False)
        rows = []
        with calculator(method).prepare_batch([WATER, MOVED]) as batch:
            for phase in ("cold", "warm", "moved", "rollback", "reset"):
                if phase == "reset":
                    batch.clear_warm_starts()
                geometries = [MOVED, WATER] if phase == "moved" else [WATER, MOVED]
                result = batch.execute(
                    coordinates=[[p for _, p in atoms] for atoms in geometries],
                    properties=("energy", "forces"),
                    strict=True,
                ).items
                for item, atoms in zip(result, geometries, strict=True):
                    expected = references[method, int(atoms == MOVED)]
                    assert item.converged and item.executed_backend == "cuda"
                    np.testing.assert_allclose(
                        item.energy, expected[0], atol=1e-8, rtol=0
                    )
                    np.testing.assert_allclose(
                        item.forces, expected[1], atol=1e-6, rtol=0
                    )
                    endpoints.append(
                        {
                            "diagnostics": enabled,
                            "phase": phase,
                            "geometry": int(atoms == MOVED),
                            "energy": item.energy,
                            "forces": np.asarray(item.forces).tolist(),
                            "iterations": item.iterations,
                            "reference_energy": float(expected[0]),
                            "reference_forces": expected[1].tolist(),
                        }
                    )
                rows.append(result)
            if method != "rhf":
                transports.append(batch.ks_transport_diagnostics)
        all_results.append(rows)
    for disabled, enabled in zip(*all_results, strict=True):
        for before, after in zip(disabled, enabled, strict=True):
            assert before.iterations == after.iterations
            # Direct GPU reductions need not repeat bit for bit. These tight
            # paired gates supplement the unchanged independent physical gates.
            np.testing.assert_allclose(before.energy, after.energy, atol=1e-12, rtol=0)
            np.testing.assert_allclose(before.forces, after.forces, atol=1e-10, rtol=0)
    observed = receipts(path)
    assert len(observed) >= 10
    assert all(row["work_coverage"] == "observed" for row in observed)
    if method == "rhf" or chunk == "2":
        assert any(row["graph_launches_lifetime"] > 0 for row in observed)
    assert any(
        row["gram_dot_reductions"] > row["history_capacity"] ** 2 for row in observed
    )
    # A reset reruns the same cold science and must not retain prior work.
    assert [r["gram_dot_reductions"] for r in observed[:2]] == [
        r["gram_dot_reductions"] for r in observed[-2:]
    ]
    if transports:
        for before, after in zip(*transports, strict=True):
            assert after.scalar_d2h_bytes == before.scalar_d2h_bytes + 5 * 24
            assert after.synchronizations == before.synchronizations + 5
            assert after.matrix_d2h_bytes == before.matrix_d2h_bytes
    record_property(
        "qualification",
        json.dumps(
            {
                "method": method,
                "chunk": chunk,
                "endpoints": endpoints,
                "receipts": observed,
            }
        ),
    )


@pytest.mark.parametrize("method", ["rhf", "pbe-rks"])
def test_nonconvergence_and_retry_do_not_publish_stale_work(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    path = tmp_path / "failed.jsonl"
    monkeypatch.setenv(TRACE, str(path))
    monkeypatch.setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM", "1")
    monkeypatch.setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION", "ordered")
    with calculator(method, max_iterations=1).prepare_batch([WATER]) as batch:
        for _ in range(2):
            item = batch.execute(properties=("energy", "forces"), strict=False).items[0]
            assert not item.converged and not item.succeeded
    failed = receipts(path)
    assert len(failed) == 2
    assert all(row["converged"] == 0 for row in failed)
    assert failed[0]["gram_dot_reductions"] == failed[1]["gram_dot_reductions"]
    with calculator(method).prepare_batch([WATER]) as batch:
        item = batch.execute(properties=("energy", "forces"), strict=True).items[0]
        assert item.converged
    assert receipts(path)[-1]["converged"] == 1


@pytest.mark.parametrize("incremental", ["0", "1"])
def test_fallback_does_not_claim_observed_zero(
    incremental: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    path = tmp_path / "fallback.jsonl"
    monkeypatch.setenv(TRACE, str(path))
    monkeypatch.setenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM", incremental)
    monkeypatch.setenv(
        "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION", "cooperative"
    )
    with calculator("rhf").prepare_batch([WATER]) as batch:
        assert (
            batch.execute(properties=("energy", "forces"), strict=True)
            .items[0]
            .converged
        )
    (row,) = receipts(path)
    assert row["work_coverage"] == "unavailable"
    assert "gram_dot_reductions" not in row
