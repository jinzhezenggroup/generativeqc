"""Explicit Slurm gate for public hybrid AOT, including spherical s/p/d forces."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_STATIONARY_AOT_CUDA_TEST") != "1",
    reason="explicit Slurm stationary AOT qualification gate",
)


@pytest.mark.parametrize("method", ["pbe0", "b3lyp"])
@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp"])
def test_public_hybrid_aot_energy_force_reuse_and_displacement(
    monkeypatch: pytest.MonkeyPatch, method: str, spin: str, basis_name: str
) -> None:
    """Time complete host-return endpoints, never moving preparation outside them."""
    import pyscf
    from generativeqc import Calculator, GridSpec, KsOptions, _stationary_cuda
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.method import stationary_cuda
    from generativeqc_compiler.method.stationary_gradient import StationaryGradientPlan
    from pyscf.dft import libxc
    from test_dft_complete_cpu import independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives

    assert pyscf.__version__ == "2.14.0"
    assert libxc.libxc_version() == "7.0.0"

    assert os.environ.get("SLURM_JOB_ID"), (
        "real GPU tests require finite Slurm allocations"
    )
    # Oxygen exercises p primitives even in the compact s/p artifact; the
    # spherical def2-SVP cases additionally exercise the full d-shell domain.
    atoms = [
        ("O", (0.02, -0.01, 0.0)),
        ("H", (0.04, 0.02, 1.81)),
        ("H", (1.72, 0.01, -0.58)),
    ]
    # The independent neutral-OH reference did not converge at the frozen
    # tolerances; bent water cation retains open-shell and p/d coverage.
    charge = 1 if spin == "uks" else 0
    multiplicity = 2 if spin == "uks" else 1
    calculator = Calculator(
        method=f"{method}-{spin}",
        device="cuda",
        basis=basis_name,
        basis_representation="spherical",
        precision="fp64",
        ks_options=KsOptions(
            grid=GridSpec(
                radial_points=32,
                angular_polar=10,
                angular_azimuth=20,
            )
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        max_iterations=300,
    )
    samples = []
    force_work = []
    with calculator.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        original_force = batch._public_dft_cuda_force

        def observed_force(*args: object, **kwargs: object) -> tuple:
            forces, work = original_force(*args, **kwargs)
            force_work.append(work)
            return forces, work

        monkeypatch.setattr(batch, "_public_dft_cuda_force", observed_force)

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail(
                "public packaged endpoint invoked compiler discovery or IR/AD/source generation"
            )

        moved = np.asarray([position for _, position in atoms])
        moved[-1] += (0.02, -0.01, 0.03)
        prepared = None
        for stage, coordinates in (
            ("cold", None),
            ("reuse", None),
            ("displaced", moved),
        ):
            # Clear the identity memoization as well: a warm host fingerprint
            # must not disguise AD/source generation in the loader.
            stationary_cuda.stationary_aot_profile_contract_identity.cache_clear()
            with monkeypatch.context() as guard, no_cpu_derivatives():
                guard.setattr(batch, "_stationary_cuda_compiler", forbidden)
                guard.setattr(_stationary_cuda, "compile_stationary_cuda", forbidden)
                guard.setattr(_stationary_cuda, "emit_first_derivative_cuda", forbidden)
                guard.setattr(_stationary_cuda, "derivative_cuda_sources", forbidden)
                guard.setattr(
                    stationary_cuda, "emit_stationary_wrapper_cuda", forbidden
                )
                guard.setattr(StationaryGradientPlan, "integral_block", forbidden)
                guard.setattr(StationaryGradientPlan, "reduction_program", forbidden)
                started = time.perf_counter()
                result = batch.execute(
                    coordinates=None if coordinates is None else (coordinates,),
                    properties=("energy", "forces"),
                    strict=True,
                ).items[0]
                endpoint_seconds = time.perf_counter() - started
            assert result.converged and result.executed_backend == "cuda"
            if prepared is None:
                prepared = batch._stationary_cuda_execution
            assert batch._stationary_cuda_execution is prepared
            current_atoms = (
                atoms
                if coordinates is None
                else [
                    (atom[0], position)
                    for atom, position in zip(atoms, coordinates, strict=True)
                ]
            )
            with NativeAO(
                current_atoms,
                basis=basis_name,
                representation="spherical",
                charge=charge,
                multiplicity=multiplicity,
            ) as basis:
                state = StationaryKsState.from_native(batch, basis)
                try:
                    reference_energy, reference_gradient = (
                        independent_global_hybrid_gradient(
                            basis, state, f"{method}-{spin}", xc=method.upper()
                        )
                    )
                finally:
                    state._source.close()
            energy_error = abs(result.energy - reference_energy)
            force_error = float(np.max(np.abs(result.forces + reference_gradient)))
            assert energy_error <= 1e-8
            assert force_error <= 1e-7
            assert force_work[-1]["stationary_artifact_kind"] == "packaged-aot"
            samples.append(
                {
                    "stage": stage,
                    "atoms": current_atoms,
                    "endpoint_seconds": endpoint_seconds,
                    "energy_error_hartree": energy_error,
                    "max_force_error_hartree_per_bohr": force_error,
                    "work": force_work[-1],
                }
            )
        assert prepared._lease.executions == 3
        if directory := os.environ.get("GENERATIVEQC_STATIONARY_AOT_EVIDENCE"):
            output = Path(directory) / f"{method}-{spin}-{basis_name}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "slurm_job": os.environ["SLURM_JOB_ID"],
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "method": method,
                        "spin": spin,
                        "charge": charge,
                        "multiplicity": multiplicity,
                        "basis": basis_name,
                        "representation": "spherical",
                        "precision": "fp64",
                        "samples": samples,
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
