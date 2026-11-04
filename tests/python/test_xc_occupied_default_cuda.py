"""Exercise provisional occupied XC and strict matrix publication through SCF."""

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
def test_occupied_controls_and_strict_handoff_across_geometry(
    unrestricted: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate opt-in proposals and matrix publication for every control.

    Trace counters prove no occupied frame is admitted without fixing iteration
    or internal call counts. Cold and displaced solves have independently
    converged CPU oracles; held warm and moved geometries retain matrix work.
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
                        if selector == "1" and phase != "warm":
                            assert any(frames)
                            last_proposal = max(
                                index for index, value in enumerate(frames) if value
                            )
                            assert last_proposal < len(frames) - 1
                            assert not any(frames[last_proposal + 1 :])
                        else:
                            assert not any(frames)
                        if phase == "cold":
                            owner.set_warm_start_updates(False)
            finally:
                calculator.clear_cache()


@pytest.mark.parametrize("unrestricted", (False, True))
def test_explicit_request_matrix_default_grid_energy_and_forces(
    unrestricted: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare complete native E+F with original D and independent PySCF.

    An explicit occupied request must retain matrix work through native owners,
    force scratch and geometry updates. Preserve the complete-CUDA gates of
    2e-9 Eh and 1e-7 Eh/Bohr and pair the ordinary matrix route at those same
    gates. Parallel force reductions do not promise bitwise reproducibility.
    Lower-level frame/certificate tests independently retain the experiment.
    """
    from generativeqc import Calculator
    from pyscf import dft, gto, lib
    from test_dft_complete_cuda import no_cpu_derivatives

    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID") and "CUDA_VISIBLE_DEVICES" in os.environ
    lib.num_threads(1)
    atoms = scaling_cases()["water-3"].atoms
    moved = np.asarray([position for _, position in atoms]).copy()
    moved[1, 2] += 0.02
    geometries = (
        atoms,
        [(symbol, point) for (symbol, _), point in zip(atoms, moved, strict=True)],
    )
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
        reference.max_cycle = 200
        energy = reference.kernel()
        assert reference.converged
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        references.append((energy, -gradient.kernel()))

    baseline = []
    observations = []
    for selector in ("0", "1"):
        with monkeypatch.context() as scope:
            scope.setenv("GENERATIVEQC_CUDA_KS_OCCUPIED_XC", selector)
            calculator = Calculator(
                method="r2scan-uks" if unrestricted else "r2scan-rks",
                device="cuda",
                basis="def2-svp",
                basis_representation="spherical",
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                max_iterations=200,
            )
            try:
                with calculator.prepare_batch(
                    [atoms],
                    charges=[charge],
                    multiplicities=[charge + 1],
                    warm_start=True,
                ) as owner:
                    for index, (energy, forces) in enumerate(references):
                        trace = tmp_path / f"forces-{selector}-{index}.jsonl"
                        scope.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                        with no_cpu_derivatives():
                            result = owner.execute(
                                [moved] if index else None,
                                properties=("energy", "forces"),
                                strict=True,
                            ).items[0]
                        frames = [
                            row["counters"]["occupied_xc_frame_generation"]
                            for line in trace.read_text().splitlines()
                            if (row := json.loads(line))["operation"]
                            == "cuda_ks_xc_shared"
                        ]
                        observations.append(
                            {
                                "selector": selector,
                                "geometry": index,
                                "energy": result.energy,
                                "forces": np.asarray(result.forces).tolist(),
                                "reference_energy": energy,
                                "reference_forces": forces.tolist(),
                                "occupied_frame_generations": frames,
                            }
                        )
                        if selector == "1":
                            original_energy, original_forces = baseline[index]
                            difference = np.asarray(result.forces) - original_forces
                            observations[-1].update(
                                matrix_pair_energy_error=abs(
                                    result.energy - original_energy
                                ),
                                matrix_pair_force_max_error=float(
                                    np.max(np.abs(difference))
                                ),
                                matrix_pair_force_rms_error=float(
                                    np.sqrt(np.mean(difference**2))
                                ),
                            )
                        (
                            tmp_path / "native-default-grid-energy-forces.json"
                        ).write_text(json.dumps(observations, indent=2) + "\n")
                        assert result.succeeded and result.converged
                        assert result.executed_backend == "cuda"
                        assert frames and not frames[-1]
                        assert any(frames) == (selector == "1")
                        assert result.energy == pytest.approx(energy, abs=2e-9, rel=0)
                        np.testing.assert_allclose(
                            result.forces, forces, atol=1e-7, rtol=0
                        )
                        if selector == "0":
                            baseline.append(
                                (result.energy, np.asarray(result.forces).copy())
                            )
                        else:
                            original_energy, original_forces = baseline[index]
                            assert result.energy == pytest.approx(
                                original_energy, abs=2e-9, rel=0
                            )
                            np.testing.assert_allclose(
                                result.forces,
                                original_forces,
                                atol=1e-7,
                                rtol=0,
                            )
            finally:
                calculator.clear_cache()
