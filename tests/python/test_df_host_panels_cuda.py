"""Bounded raw DF host panels preserve independent J/K and resource fallback."""

from __future__ import annotations

import json
import os
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest

from benchmarks.issue308_stage_probe import prepare

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU allocation and a cached-build native probe",
)


@pytest.fixture(scope="module")
def host_fixture(tmp_path_factory: typing.Any) -> tuple[Path, Path]:
    """Use a separately compiled probe so GPU tests never silently compile uncached."""
    assert os.environ.get("SLURM_JOB_ID")
    binary = Path(os.environ["GENERATIVEQC_DF_STAGE_PROBE"])
    assert binary.is_file()
    fixture = tmp_path_factory.mktemp("host-df-source") / "input"
    prepare("water-tetramer-def2-svp-spherical", fixture, None)
    return binary, fixture


@pytest.mark.parametrize(
    "pairs,auxiliary,retained", [(3552, 13, False), (9216, 5, True)]
)
@pytest.mark.parametrize("host_limit", [None, "0", "1", str((1 << 64) - 1)])
def test_host_panels_match_independent_jk_and_preserve_gpu_priority(
    host_fixture: tuple[Path, Path],
    pairs: int,
    auxiliary: int,
    retained: bool,
    host_limit: str | None,
    tmp_path: Path,
) -> None:
    """Repeated dense/occupied actions reuse A only when the complete host source fits."""
    binary, fixture = host_fixture
    trace, arrays = tmp_path / "trace.jsonl", tmp_path / "arrays.bin"
    env = dict(
        os.environ,
        GENERATIVEQC_DF_TRACE=str(trace),
        GENERATIVEQC_TEST_DF_HOST_BUDGET=str(64 << 20),
    )
    if host_limit is None:
        env.pop("GENERATIVEQC_DF_HOST_MAX_BYTES", None)
    else:
        env["GENERATIVEQC_DF_HOST_MAX_BYTES"] = host_limit
    result = subprocess.run(
        [
            str(binary),
            str(fixture / "input.txt"),
            "2",
            str(auxiliary),
            str(pairs),
            str(arrays),
            "all",
            str(int(retained)),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    setup = next(
        json.loads(line)
        for line in result.stdout.splitlines()
        if json.loads(line)["operation"] == "setup"
    )
    cached = host_limit in (None, str((1 << 64) - 1)) and not retained
    assert setup["host_raw_factor_bytes"] == (96**3 * 8 if cached else 0)
    actual = np.fromfile(arrays).reshape(4, 96, 96)
    with np.load(fixture / "reference.npz") as reference:
        for values, key in zip(actual, ["density", "j", "k", "k"], strict=True):
            np.testing.assert_allclose(values, reference[key], atol=1e-9, rtol=0)
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    for record in (r for r in records if r["operation"] in ("ri_k", "ri_k_occupied")):
        counters = record["counters"]
        assert bool(counters.get("host_raw_panel_hits", 0)) == cached
        if cached:
            assert counters["host_raw_panel_h2d_bytes"] > 0
            assert counters.get("raw_panel_source_auxiliary_evaluations", 0) == 0


@pytest.mark.parametrize("limit", ["bad", "-1", str(1 << 65)])
def test_invalid_host_policy_never_publishes_jk(
    host_fixture: tuple[Path, Path], limit: str, tmp_path: Path
) -> None:
    """Malformed capacity does not silently truncate or publish scientific arrays."""
    binary, fixture = host_fixture
    arrays = tmp_path / "arrays.bin"
    result = subprocess.run(
        [
            str(binary),
            str(fixture / "input.txt"),
            "1",
            "13",
            "3552",
            str(arrays),
            "all",
        ],
        env=dict(
            os.environ,
            GENERATIVEQC_TEST_DF_HOST_BUDGET=str(64 << 20),
            GENERATIVEQC_DF_HOST_MAX_BYTES=limit,
        ),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode != 0 and "host cache byte ceiling" in result.stderr
    assert not arrays.exists()


def test_uhf_host_source_preserves_independent_force_and_spin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changing open-shell density must never reuse density-dependent projections."""
    from generativeqc import Calculator
    from pyscf import gto, scf

    from benchmarks._cases import benchmark_cases

    assert os.environ.get("SLURM_JOB_ID")
    atoms = benchmark_cases()["oh-def2-svp-spherical-uhf"].atoms
    molecule = gto.M(
        atom=atoms,
        unit="Bohr",
        basis="def2-tzvp",
        charge=0,
        spin=1,
        cart=False,
        verbose=0,
    )
    reference = scf.UHF(molecule).density_fit(auxbasis="def2-tzvp")
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-12, 1e-10, 150
    reference.kernel()
    assert reference.converged
    force = -reference.nuc_grad_method().kernel()
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    monkeypatch.delenv("GENERATIVEQC_DF_HOST_MAX_BYTES", raising=False)
    calculator = Calculator(
        method="uhf",
        basis="def2-tzvp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=5 << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=150,
    )
    trace = tmp_path / "uhf.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    with calculator.prepare_batch([atoms], charges=[0], multiplicities=[2]) as batch:
        for _ in range(2):
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            assert abs(result.energy - reference.e_tot) < 1e-9
            np.testing.assert_allclose(result.forces, force, atol=2e-8, rtol=0)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert (
        sum(row.get("counters", {}).get("host_raw_panel_hits", 0) for row in rows) > 0
    )
