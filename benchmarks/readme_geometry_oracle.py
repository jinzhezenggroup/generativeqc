"""Independent accuracy-only geometry solve; never a timed population replacement."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np
from generativeqc import GridSpec

from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks._support import cuda_accelerator_metadata, environment_metadata
from benchmarks.readme_dft import specification
from benchmarks.readme_omol25 import (
    gpu_convergence_payload,
    load_comparison_basis,
    protocol,
    reference_energy_forces,
    reference_work_counter,
    scaling_cases,
    source_hashes,
)
from benchmarks.readme_wb97mv import (
    configure_reference_mixed_rsh_df,
    reference_engine,
    reference_vv10_domain,
)


def main() -> None:
    """Reconverge one geometry from the reference engine atomic guess.

    The retained protocol fixes grid, basis, stopping tolerances and Hamiltonian.
    This command never consumes native density or changes a reference timing run.
    Run through srun with an explicit finite time limit; the diagnostic duration
    is deliberately excluded from published cold/warm reference populations.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU execution requires a finite Slurm allocation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--geometry", type=int, choices=(0, 1), required=True)
    parser.add_argument("--auxiliary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    original = json.loads(a.template.read_text())
    p = original["protocol"]
    assert p["method"] == "WB97M-V/RKS"
    case = scaling_cases()[p["case"]]
    basis, reference_basis = load_comparison_basis(
        a.basis, case, role="orbital", compute_forces=True
    )
    assert (
        hashlib.sha256(a.basis.read_bytes()).hexdigest()
        == original["basis_file_sha256"]
    )
    spec = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    benchmark = replace(specification("wb97m-v"), density_fitting=p["density_fitting"])
    aux = reference_aux = None
    if p["density_fitting"]:
        assert a.auxiliary
        aux, reference_aux = load_retained_comparison_basis(
            a.auxiliary, case, role="auxiliary", compute_forces=True
        )
    actual = protocol(
        p["atoms"], basis, spec, p["repeats"], benchmark=benchmark, auxiliary_basis=aux
    )
    assert actual == p, "scientific or timing protocol template changed"
    record = {
        "schema": "generativeqc.readme-independent-geometry-oracle.v1",
        "purpose": "accuracy-only independently reconverged geometry; no population timing or speedup claim",
        "protocol": p,
        "geometry": a.geometry,
        "status": "running",
        "stage": "construct",
        "initial_density": "GPU4PySCF default atomic guess; no native density, orbitals, energy or force input",
        "template_sha256": hashlib.sha256(a.template.read_bytes()).hexdigest(),
        "basis_sha256": hashlib.sha256(a.basis.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_file_sha256": source_hashes(),
        "scheduler": {
            k: os.environ.get(k)
            for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_GPUS_ON_NODE")
        },
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x",),
                "cupy": ("cupy-cuda12x",),
                "cutensor": ("cutensor-cu12",),
                "numpy": ("numpy",),
            }
        ),
        "accelerator": cuda_accelerator_metadata(cp),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        a.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    engine = reference_engine(
        p["geometries_bohr"][a.geometry], reference_basis, spec, full_fock=True
    )
    if reference_aux is not None:
        engine = engine.density_fit(auxbasis=reference_aux)
        from benchmarks.readme_wb97mv import configure_reference_full_fock

        configure_reference_full_fock(engine)
        configure_reference_mixed_rsh_df(engine)
    engine.conv_tol = p["energy_tolerance"]
    engine.conv_tol_grad = p["reference_gradient_tolerance"]
    engine.direct_scf_tol = p["reference_direct_scf_tolerance"]
    engine.max_cycle = p["max_iterations"]
    record["stage"] = "independent-scf-and-gradient"
    save()
    with (
        reference_work_counter(engine) as work,
        reference_vv10_domain(p["nonlocal_density_threshold"]) as domain,
    ):
        e, f, seconds = reference_energy_forces(
            engine, None, export=cp.asnumpy, synchronize=cp.cuda.Stream.null.synchronize
        )
        convergence = gpu_convergence_payload([engine], [work["tracker"]])[0]
    record.update(
        status="complete",
        stage="complete",
        energy=e,
        forces=f.tolist(),
        converged=bool(engine.converged),
        convergence=convergence,
        scf_jk_builds=work["scf_jk_builds"],
        reference_vv10_domain=domain,
        diagnostic_seconds=seconds,
        grid_points={
            "semilocal": len(engine.grids.weights),
            "vv10": len(engine.nlcgrids.weights),
        },
    )
    assert engine.converged and np.isfinite(e) and np.isfinite(f).all()
    save()
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "geometry",
                    "energy",
                    "diagnostic_seconds",
                    "convergence",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
