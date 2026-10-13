"""Screened iteration stages must publish a fully unscreened physical frame."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not os.environ.get("GENERATIVEQC_RHF_SCREENING_PROBE"),
    reason="requires a compiled native probe and a finite Slurm GPU allocation",
)


@pytest.mark.parametrize("distance", (1.8, 8.0))
@pytest.mark.parametrize("bounded", (False, True), ids=("automatic", "bounded"))
def test_screened_reference_refines_and_reuses_geometry(
    tmp_path: Path, distance: float, bounded: bool
) -> None:
    """Check cold, warm, moved, and changed-policy states against Libcint RHF."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(1)
    basis = [
        [0, [1.5, 1.0], [0.45, -0.18]],
        *[[angular, [1.5 - 0.2 * angular, 1.0]] for angular in range(1, 4)],
    ]
    positions = [(0.0, 0.0, 0.0), (0.3, -0.2, distance)]
    lines = [f"2 {2 * len(basis)}"]
    lines += [" ".join(map(str, (2, *position))) for position in positions]
    for atom in range(2):
        for angular, *primitives in basis:
            lines.append(f"{atom} {angular} {len(primitives)}")
            lines += [" ".join(map(str, primitive)) for primitive in primitives]
    source = tmp_path / "input.txt"
    source.write_text("\n".join(lines) + "\n")
    journal = tmp_path / "progress.jsonl"
    environment = {
        **os.environ,
        "GENERATIVEQC_RHF_RESIDENT_VALUES": "0",
        "GENERATIVEQC_DF_PROGRESS_TRACE": str(journal),
    }
    environment.pop("GENERATIVEQC_BOUNDED_DIRECT_STREAMING", None)
    if bounded:
        environment["GENERATIVEQC_BOUNDED_DIRECT_STREAMING"] = "force"
    result = subprocess.run(
        [os.environ["GENERATIVEQC_RHF_SCREENING_PROBE"], str(source)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 5
    assert [row["reused"] for row in rows] == [0, 0, 1, 1, 0]
    expected = []
    for displacement in (0.0, 0.02):
        moved = [positions[0], (*positions[1][:2], distance + displacement)]
        molecule = pyscf.gto.M(
            atom=[("He", position) for position in moved],
            basis={"He": basis},
            unit="Bohr",
            verbose=0,
        )
        solver = molecule.RHF()
        solver.conv_tol = 1e-13
        solver.conv_tol_grad = 1e-10
        solver.direct_scf_tol = 0
        energy = solver.kernel()
        assert solver.converged
        expected.append((energy, solver.make_rdm1(), solver.mo_energy))
    for phase, row in enumerate(rows):
        energy, density, orbital_energies = expected[int(phase >= 3)]
        assert row["energy"] == pytest.approx(energy, abs=2e-9)
        np.testing.assert_allclose(row["density"], density.ravel(), rtol=0, atol=3e-8)
        np.testing.assert_allclose(
            row["orbital_energies"], orbital_energies, rtol=0, atol=3e-8
        )
        assert row["commutator"] < 1e-8 and row["canonical_drift"] < 1e-8
        assert row["work_valid"] and row["focks"] == row["iterations"]
        if phase in (0, 4):
            assert not row["strict_refinement"] and row["refinement"] == 0
        else:
            assert row["strict_refinement"]
            assert 0 < row["refinement"] < row["iterations"]
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    cutoffs = [
        float(record["value"])
        for record in records
        if record.get("key") == "iteration_screening_tolerance"
    ]
    assert cutoffs == [0, 1e-12, 1e-12, 1e-12, 0]
