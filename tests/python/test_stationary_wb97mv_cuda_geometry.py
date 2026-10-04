"""Opt-in real CUDA semilocal geometry gate against independent Libxc energies.

Run with GENERATIVEQC_TEST_RANGE_CUDA=1 inside Slurm, with nvcc on PATH and a native
GenerativeQC library available for basis preparation. This qualifies the semilocal
slice only; SR/LR exchange and VV10 are intentionally absent from this energy.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize(
    "ao_route", ("host-full", "resident-full", "resident-subset", "resident-empty")
)
def test_cuda_b97m_geometry_matches_independent_energy_differences(
    tmp_path: Path, spin: str, ao_route: str, large_basis: bool = False
) -> None:
    """Exercise actual tau/gradient pullbacks for fixed positive density matrices.

    Grid points move with their owner atoms while their explicit weights remain
    fixed. Separate center and point differences catch cancellation between the
    two sources, as well as spin and kinetic-density factors of two.
    """
    if os.environ.get("GENERATIVEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("native CUDA compiler unavailable")
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("independent oracle is pinned to Libxc 7.0.0")
    import cupy as cp
    from generativeqc._stationary_cuda import _CudaSources
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.dft.cuda import CudaGrid
    from generativeqc_compiler.dft.cuda import compile_cuda as compile_grid
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method.stationary_cuda import compile_stationary_cuda
    from generativeqc_compiler.method.stationary_gradient import (
        StationaryGradientPlan,
        StationaryMeanField,
    )
    from pyscf import gto
    from pyscf.dft import numint

    coordinates = np.array(((0.13, -0.17, -0.71), (-0.09, 0.11, 0.79)))
    # LiH supplies both s and p AOs in the admitted public basis domain.
    symbols = ("Li", "H")
    atoms = list(zip(symbols, coordinates, strict=True))
    native_basis = "sto-3g"
    reference_basis = "sto-3g"
    representation = "cartesian"
    if large_basis:
        from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
        from benchmarks.readme_hf_scaling import scaling_cases

        case = scaling_cases()["water-96"]
        native_basis, reference_basis = load_comparison_basis(
            Path(__file__).parents[2]
            / "benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json",
            case,
            role="orbital",
            compute_forces=True,
        )
        atoms = case.atoms
        symbols, coordinates = zip(*atoms, strict=True)
        coordinates = np.asarray(coordinates)
        representation = "spherical"
    points = np.array(
        (
            (0.31, 0.22, -0.41),
            (-0.27, 0.41, 0.65),
            (0.82, -0.31, 0.13),
            (-0.63, -0.32, -0.19),
            (0.24, 0.72, 0.31),
        )
    )
    owners = np.array((0, 1, 0, 1, 0), dtype=np.int64)
    if large_basis:
        # Probe both sides of the previous 1024-AO boundary and the last atom.
        # Five local points keep this a geometry gate, not a full-grid timing.
        owners = np.array((0, 53, 95, 53, 95), dtype=np.int64)
        points += coordinates[owners]
    weights = np.array((0.13, 0.27, 0.41, 0.19, 0.31))
    compiler = CudaCompilerAdapter(Path(nvcc), cuda_target_info("sm_120"))
    method = resolve_method("WB97M-V", spin=spin)
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
    )
    # No integral derivative is part of this isolated slice. Fail closed if
    # geometry execution ever calls into that unrelated provider.
    primitive = """#include <cuda_runtime.h>
__device__ bool first_derivative(unsigned, const double*, const double*, double*) {
  return false;
}
"""
    artifact = compile_stationary_cuda(
        primitive,
        functional=4,
        plan=plan,
        iterations=3,
        compiler=compiler,
        cache=tmp_path,
    )
    with NativeAO(atoms, basis=native_basis, representation=representation) as basis:
        if large_basis:
            assert basis.nao == 1856 and basis.natom == 96
        # PySCF stably groups shells by angular momentum on each atom. The
        # TZVPD snapshot appends diffuse s/p/d shells after f, while NativeAO
        # preserves snapshot order. This fixed-density oracle must map both
        # density indices; independently reconverged SCF does not share D.
        offsets = np.concatenate(
            ([0], np.cumsum([2 * s.angular_momentum + 1 for s in basis.shells]))
        )
        oracle_order = (
            np.array(
                [
                    ao
                    for shell in sorted(
                        range(len(basis.shells)),
                        key=lambda i: (
                            basis.shells[i].atom_index,
                            basis.shells[i].angular_momentum,
                        ),
                    )
                    for ao in range(offsets[shell], offsets[shell + 1])
                ]
            )
            if large_basis
            else np.arange(basis.nao)
        )
        if large_basis:
            oracle_mol = gto.M(
                atom=atoms, basis=reference_basis, unit="Bohr", cart=False, verbose=0
            )
            # Check all AO jets independently before using the metadata map;
            # do not infer a representation change from the force results.
            np.testing.assert_allclose(
                basis.evaluate(points, 2)[:, :, oracle_order],
                numint.eval_ao(oracle_mol, points, deriv=2),
                atol=3e-12,
                rtol=3e-12,
            )
        rng = np.random.default_rng(1389)
        matrices = rng.normal(
            size=(plan.spin_blocks, basis.nao, 8 if large_basis else basis.nao)
        )
        density = np.ascontiguousarray(
            0.15 * matrices @ matrices.transpose(0, 2, 1) + 0.1 * np.eye(basis.nao)
        )
        # A noncontiguous map spans both centers and retains cross-center D[I,I]
        # entries. The independent oracle zeros omitted rows/columns, while the
        # CUDA owner receives the original dense matrix and must gather itself.
        selected = (
            np.array(
                (0, 2, 1023, 1024, basis.nao - 1)
                if large_basis
                else (0, 2, basis.nao - 1),
                dtype=np.uintp,
            )
            if ao_route == "resident-subset"
            else np.empty(0, dtype=np.uintp)
            if ao_route == "resident-empty"
            else None
        )
        oracle_density = density.copy()
        if selected is not None:
            omitted = np.ones(basis.nao, dtype=bool)
            omitted[selected] = False
            oracle_density[:, omitted, :] = 0
            oracle_density[:, :, omitted] = 0
        oracle_density = np.ascontiguousarray(
            oracle_density[:, oracle_order, :][:, :, oracle_order]
        )
        device_points = cp.asarray(points) if ao_route.startswith("resident") else None
        with (
            _CudaSources(
                basis,
                artifact,
                compiler,
                0,
                len(points),
                1,
                (256 if large_basis else 16) << 20,
                spin_blocks=plan.spin_blocks,
                # Through-f forces borrow the prepared native integral owner.
                # This isolated slice must not build the SPD-only diagnostic
                # primitive descriptors (nor any AO^4 inventory).
                integral_derivatives=not large_basis,
            ) as sources,
            CudaGrid(
                basis,
                compile_grid(compiler, tmp_path),
                order=2,
                tile_points=len(points),
                # The global D matrices remain dense even for a short local
                # map; the small-fixture 256 MiB default cannot admit 1856 AOs.
                budget_bytes=(4 << 30) if large_basis else None,
                active_ao_capacity=basis.nao
                if selected is None
                else max(1, len(selected)),
                ingredients=("rho", "gradient", "tau"),
            ) as grid,
        ):
            grid.set_density(density[0] if plan.spin_blocks == 1 else density)
            sources.reset(1e-12, density, np.zeros_like(density))
            lease = (
                grid.feature_task_device_points(
                    device_points.data.ptr,
                    len(points),
                    selected,
                    ("rho", "gradient", "tau"),
                )
                if device_points is not None
                else grid.xc_task(points, None, "WB97M-V")
            )
            with lease as task:
                sources.geometry(
                    task, owners, weights, np.zeros_like(weights), functional=4
                )
            actual = sources.finish()

    def energy(centers: np.ndarray, grid_points: np.ndarray) -> float:
        mol = gto.M(
            atom=list(zip(symbols, centers, strict=True)),
            basis=reference_basis,
            unit="Bohr",
            cart=representation == "cartesian",
            verbose=0,
        )
        ao = numint.eval_ao(mol, grid_points, deriv=1)
        rho = np.array(
            [
                numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=False)
                for dm in oracle_density
            ]
        )
        exc = libxc.eval_xc(
            "HYB_MGGA_XC_WB97M_V",
            rho[0] if plan.spin_blocks == 1 else rho,
            spin=plan.spin_blocks - 1,
            deriv=0,
        )[0]
        return float(weights @ (exc * rho[:, 0].sum(axis=0)))

    expected_basis = np.zeros((len(atoms), 3))
    expected_points = np.zeros_like(expected_basis)
    step = 2e-5
    checked_atoms = [0, 53, 95] if large_basis else list(range(len(atoms)))
    for atom in checked_atoms:
        for axis in range(3):
            displacement = np.zeros_like(coordinates)
            displacement[atom, axis] = step
            motion = displacement[owners]
            expected_basis[atom, axis] = (
                energy(coordinates + displacement, points)
                - energy(coordinates - displacement, points)
            ) / (2 * step)
            expected_points[atom, axis] = (
                energy(coordinates, points + motion)
                - energy(coordinates, points - motion)
            ) / (2 * step)
    np.testing.assert_allclose(
        actual["xc_ao"][checked_atoms],
        expected_basis[checked_atoms],
        atol=3e-8,
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        actual["xc_grid"][checked_atoms],
        expected_points[checked_atoms],
        atol=3e-8,
        rtol=3e-7,
    )
    np.testing.assert_array_equal(actual["xc_weight"], 0)
    np.testing.assert_allclose(
        (actual["xc_ao"] + actual["xc_grid"]).sum(axis=0), 0, atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("ao_route", ("resident-full", "resident-subset"))
def test_cuda_1856_ao_geometry_has_independent_derivative_gate(
    tmp_path: Path, spin: str, ao_route: str
) -> None:
    """Qualify large global strides/maps; no complete SCF speed claim follows."""
    test_cuda_b97m_geometry_matches_independent_energy_differences(
        tmp_path, spin, ao_route, large_basis=True
    )
