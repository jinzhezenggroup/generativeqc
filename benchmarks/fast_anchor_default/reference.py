"""Produce an independent matched-grid oracle for one cold anchor endpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from benchmarks._retention import raw_output_path


def main() -> None:
    """Evaluate GPU4PySCF without reading native densities, energies or forces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(12, 24, 48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--coarse-seed", action="store_true")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Independent GPU oracle requires Slurm-assigned visibility")

    import cupy as cp
    from generativeqc import GridSpec

    from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
    from benchmarks.fast_anchor_default.endpoint import publish_json
    from benchmarks.readme_dft import specification
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_omol25 import protocol, source_hashes
    from benchmarks.readme_wb97mv import (
        reference_engine,
        reference_sample,
        reference_vv10_domain,
    )

    case = scaling_cases()[f"water-{args.atoms}"]
    basis, reference_basis = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(48, 16, 32)
    scientific = protocol(
        args.atoms, basis, grid, 1, benchmark=specification("wb97m-v")
    )
    engine = reference_engine(case.atoms, reference_basis, grid, full_fock=True)
    engine.conv_tol = 1e-12
    engine.conv_tol_grad = 1e-10
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 100
    with reference_vv10_domain(scientific["nonlocal_density_threshold"]):
        seed = None
        if args.coarse_seed:
            coarse = reference_engine(
                case.atoms, reference_basis, GridSpec(12, 8, 16), full_fock=True
            )
            coarse.conv_tol = 1e-9
            coarse.conv_tol_grad = 1e-7
            coarse.kernel()
            if not coarse.converged:
                raise RuntimeError("Independent coarse-grid seed did not converge")
            seed = coarse.make_rdm1().copy()
        sample = reference_sample(engine, cp, density=seed)
    record = {
        "schema": "generativeqc.fast-anchor-independent-oracle.v1",
        "engine": "GPU4PySCF full-density rebuild, matched moving grid",
        "status": "complete",
        "protocol": scientific,
        "basis_file_sha256": hashlib.sha256(args.basis_file.read_bytes()).hexdigest(),
        "initial_density": (
            "independent GPU4PySCF 12x8x16 coarse-grid solve; no native state input"
            if args.coarse_seed
            else "independent GPU4PySCF atomic guess; no native state input"
        ),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_file_sha256": source_hashes(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "records": [
            {
                "energy": sample["energies_hartree"][0],
                "forces": sample["forces_hartree_per_bohr"][0],
                "converged": bool(engine.converged),
                "convergence": sample["convergence"][0],
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_json(args.output, record)
    print(f"Independent water-{args.atoms} energy/force oracle published", flush=True)


if __name__ == "__main__":
    main()
