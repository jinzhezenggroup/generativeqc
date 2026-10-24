"""Independent full-range E+F and same-owner host cache lifetime checks."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("functional,spin", [("pbe0", 0), ("pbe0", 1), ("b3lyp", 0)])
def test_full_host_cache_complete_endpoints(
    functional: str, spin: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep force/geometry oracles independent of every native cache policy."""
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires a finite Slurm GPU allocation")
    assert os.environ.get("SLURM_JOB_ID")
    import numpy as np
    from generativeqc import Calculator
    from pyscf import dft, gto, lib

    lib.num_threads(1)
    atoms = [
        ("O", (0.02, -0.03, 0.04)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    moved = [
        (s, (x + (0.001 if i == 2 else 0), y, z))
        for i, (s, (x, y, z)) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(
            atom=geometry,
            basis="def2-svp",
            unit="Bohr",
            charge=spin,
            spin=spin,
            verbose=0,
        )
        solver = (dft.UKS if spin else dft.RKS)(mol)
        solver.xc = functional
        solver.conv_tol, solver.conv_tol_grad = 1e-12, 1e-9
        solver.direct_scf = False
        solver.max_cycle = 150
        solver.kernel()
        if not solver.converged:
            # The water cation can stall in DIIS at this tight reference gate.
            # Newton resolves the same independent UKS equations; no native
            # density or result enters this fallback.
            solver = solver.newton()
            solver.max_cycle = 150
            solver.kernel()
        assert solver.converged
        gradient = solver.nuc_grad_method()
        gradient.grid_response = True
        references.append((solver.e_tot, -gradient.kernel()))

    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
    monkeypatch.setenv("GENERATIVEQC_KS_HOST_TILE_BYTES", "4096")
    observations = {}
    receipts = {}
    for tier in (
        "direct",
        "gpu",
        "host",
        "default",
        "refused",
        "full",
        "fit",
        "disk",
        "disk-auto",
    ):
        trace = tmp_path / f"{tier}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
        monkeypatch.setenv(
            "GENERATIVEQC_KS_DISK_MAX_BYTES", str(2 << 30) if tier == "disk" else "0"
        )
        if tier == "disk-auto":
            monkeypatch.delenv("GENERATIVEQC_KS_DISK_MAX_BYTES")
        monkeypatch.setenv(
            "GENERATIVEQC_KS_RESIDENT_VALUES", "0" if tier == "direct" else "1"
        )
        monkeypatch.setenv(
            "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
            "0xfffffff" if tier == "full" else "0x1555",
        )
        monkeypatch.setenv(
            "GENERATIVEQC_KS_RESIDENT_MAX_BYTES",
            str(8192 if tier == "fit" else 8 << 30),
        )
        monkeypatch.setenv(
            "GENERATIVEQC_KS_HOST_MAX_BYTES",
            str(2 << 30) if tier == "host" else "1" if tier == "refused" else "0",
        )
        if tier in {"default", "full", "fit"}:
            monkeypatch.delenv("GENERATIVEQC_KS_HOST_MAX_BYTES")
        if tier == "fit":
            monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK")
        calculator = Calculator(
            method=f"{functional}-{'uks' if spin else 'rks'}",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            max_iterations=100,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        try:
            with calculator.prepare_batch(
                [atoms], charges=[spin], multiplicities=[spin + 1], warm_start=True
            ) as owner:
                cold = owner.execute(strict=True).items[0]
                owner.set_warm_start_updates(False)
                warm = owner.execute(strict=True).items[0]
                owner.set_warm_start_updates(True)
                displaced = owner.execute(
                    coordinates=np.array([[p for _, p in moved]]), strict=True
                ).items[0]
                owner.set_warm_start_updates(False)
                moved_warm = owner.execute(
                    coordinates=np.array([[p for _, p in moved]]), strict=True
                ).items[0]
            observations[tier] = (cold, warm, displaced, moved_warm)
            for result, (energy, forces) in zip(
                observations[tier],
                [references[0]] * 2 + [references[1]] * 2,
                strict=True,
            ):
                assert result.converged
                assert abs(result.energy - energy) / len(atoms) < 1e-5
                delta = result.forces - forces
                assert np.max(np.abs(delta)) < 1e-4
                assert np.sqrt(np.mean(delta**2)) < 3e-5
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            host = [
                row["value"]
                for row in rows
                if row.get("key") == "host_source_value_bytes"
            ]
            assert any(host) == (tier in {"host", "default", "fit"})
            disk_bytes = [
                row["value"]
                for row in rows
                if row.get("key") == "disk_source_value_bytes"
            ]
            assert any(disk_bytes) == (tier in {"disk", "disk-auto"})
            receipts[tier] = {
                "host_bytes": host,
                "energies": [r.energy for r in observations[tier]],
                "forces": [r.forces.tolist() for r in observations[tier]],
            }
        finally:
            calculator.clear_cache()
    for tier, values in observations.items():
        for direct, cached in zip(observations["direct"], values, strict=True):
            assert abs(direct.energy - cached.energy) < 1e-8, tier
            np.testing.assert_allclose(direct.forces, cached.forces, atol=1e-7, rtol=0)
    (tmp_path / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
