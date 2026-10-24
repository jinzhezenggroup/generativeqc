"""Exercise automatic occupied-XC admission and rollback through complete SCF."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_OCCUPIED_XC") != "1",
    reason="requires a finite Slurm GPU allocation and independent PySCF",
)


@pytest.mark.parametrize("unrestricted", (False, True))
def test_automatic_frame_admission_rollback_and_geometry(
    unrestricted: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate the public default, explicit controls and density-frame lifetime.

    Trace counters prove admission without fixing iteration or internal call
    counts. Cold and displaced solves have independently converged CPU oracles;
    held warm density must not borrow a retired canonical proposal frame.
    Analytic force and finite-difference coverage lives in test_dft_complete_cuda.
    """
    from generativeqc import Calculator
    from pyscf import dft, gto, lib

    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID") and "CUDA_VISIBLE_DEVICES" in os.environ
    lib.num_threads(1)
    atoms = scaling_cases()["water-3"].atoms
    moved = np.asarray([xyz for _, xyz in atoms]).copy()
    moved[1, 2] += 0.02
    geometries = [
        atoms,
        [(symbol, xyz) for (symbol, _), xyz in zip(atoms, moved, strict=True)],
    ]
    charge = int(unrestricted)
    references = []
    for geometry in geometries:
        molecule = gto.M(
            atom=geometry,
            basis="def2-svp",
            unit="Bohr",
            cart=False,
            charge=charge,
            spin=charge,
            verbose=0,
        )
        reference = dft.UKS(molecule) if unrestricted else dft.RKS(molecule)
        reference.xc = "R2SCAN"
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-9
        reference.max_cycle = 150
        references.append(reference.kernel())
        assert reference.converged

    for selector in (None, "1", "0", "invalid"):
        with monkeypatch.context() as scope:
            if selector is None:
                scope.delenv("GENERATIVEQC_CUDA_KS_OCCUPIED_XC", raising=False)
            else:
                scope.setenv("GENERATIVEQC_CUDA_KS_OCCUPIED_XC", selector)
            calculator = Calculator(
                method="r2scan-uks" if unrestricted else "r2scan-rks",
                device="cuda",
                basis="def2-svp",
                basis_representation="spherical",
                energy_tolerance=1e-10,
                density_tolerance=1e-8,
                max_iterations=150,
            )
            try:
                with calculator.prepare_batch(
                    [atoms],
                    charges=[charge],
                    multiplicities=[charge + 1],
                    warm_start=True,
                ) as owner:
                    for phase in ("cold", "warm", "moved"):
                        trace = tmp_path / f"{selector}-{phase}.jsonl"
                        scope.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                        if phase == "moved":
                            owner.set_warm_start_updates(True)
                        result = owner.execute(
                            [moved] if phase == "moved" else None,
                            properties=("energy",),
                            strict=True,
                        ).items[0]
                        assert result.succeeded and result.converged
                        assert result.executed_backend == "cuda"
                        expected = references[int(phase == "moved")]
                        assert abs(result.energy - expected) / len(atoms) < 1e-5
                        rows = [
                            json.loads(line) for line in trace.read_text().splitlines()
                        ]
                        frames = [
                            row["counters"]["occupied_xc_frame_generation"]
                            for row in rows
                            if row["operation"] == "cuda_ks_xc_shared"
                        ]
                        assert frames
                        if selector not in (None, "1"):
                            assert not any(frames)
                        elif phase == "warm":
                            # The held input has no live proposal frame. UKS
                            # may reconverge through another accepted proposal,
                            # which can legitimately supply a later frame.
                            assert frames[0] == 0
                        else:
                            assert any(frames)
                        if phase == "cold":
                            owner.set_warm_start_updates(False)
            finally:
                calculator.clear_cache()
