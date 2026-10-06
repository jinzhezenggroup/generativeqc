"""Complete public CUDA hybrid forces against independent moving-grid PySCF.

Run with GENERATIVEQC_HYBRID_FORCE_CUDA_TEST=1 in a finite Slurm GPU allocation.
Each case checks the converged endpoint, both reconverged finite-difference
steps, source accounting, and reuse of the prepared force owner.
GENERATIVEQC_HYBRID_BECKE_PRIMITIVE_TEST=1 uses physical hydrogen clusters inside
the existing phased-cache domain; it does not change any admission threshold.
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_HYBRID_FORCE_CUDA_TEST") != "1",
    reason="explicit Slurm CUDA hybrid-force gate",
)


def _record_evidence(name: str, payload: dict) -> None:
    directory = os.environ.get("GENERATIVEQC_HYBRID_FORCE_EVIDENCE")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps({"slurm_job": os.environ["SLURM_JOB_ID"], **payload}, indent=2)
            + "\n"
        )


@pytest.fixture(scope="module", autouse=True)
def pinned_reference() -> None:
    """Pin the independent MGGA worker semantics used by this acceptance gate."""
    import pyscf
    from pyscf.dft import libxc

    assert pyscf.__version__ == "2.14.0"
    assert libxc.libxc_version() == "7.0.0"


@pytest.fixture(autouse=True)
def small_physical_becke_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the authenticated active-prefix primitive on physical RKS/UKS.

    The native phased cache deliberately retains its small-atom fallback.
    Changing the automatic threshold cannot admit two/three-atom owners; the
    opted-in gate instead uses a physical 35/36-atom cluster in the actual domain.
    """
    if os.environ.get("GENERATIVEQC_HYBRID_BECKE_PRIMITIVE_TEST") == "1":
        monkeypatch.setenv("GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE", "coefficients")


def primitive_physical_cluster(
    spin: str,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Return neutral separated H2 fragments and one open-shell H for UKS.

    Closed-shell 36-atom and doublet 35-atom clusters exercise the existing
    greater-than-32 cache admission with a modest independent-oracle AO domain.
    Neither native/compiler resource guards nor production defaults are patched.
    """
    pairs = 18 if spin == "rks" else 17
    atoms = []
    for pair in range(pairs):
        center = (8.0 * (pair % 3), 8.0 * ((pair // 3) % 3), 8.0 * (pair // 9))
        for offset in (-0.7, 0.7):
            atoms.append(("H", (center[0], center[1], center[2] + offset)))
    if spin == "uks":
        atoms.append(("H", (24.0, 24.0, 16.0)))
    return atoms


@pytest.mark.parametrize(
    "options",
    ({"density_fitting": "cuda"}, {"host_unfused": True}),
)
def test_global_hybrid_force_preserves_execution_boundaries(
    options: dict,
) -> None:
    from generativeqc import Calculator, GridSpec, KsOptions

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    options = dict(options)
    schedule = "host_unfused" if options.pop("host_unfused", False) else "device_fused"
    calc = Calculator(
        method="pbe0-rks",
        device="cuda",
        ks_options=KsOptions(grid=GridSpec(), xc_schedule=schedule),
        **options,
    )
    if options.get("density_fitting"):
        # The separately qualified DF response owner now supports FP64 forces;
        # it must not inherit the direct owner's component-wise AUTO admission.
        assert "forces" in calc.capabilities.supported_properties
        with pytest.raises(NotImplementedError, match="requires precision='fp64'"):
            Calculator(
                method="pbe0-rks",
                device="cuda",
                precision="auto",
                ks_options=KsOptions(grid=GridSpec(), xc_schedule=schedule),
                **options,
            )
        return
    assert "forces" not in calc._capabilities.supported_properties
    with pytest.raises((ValueError, NotImplementedError)):
        calc.singlepoint(
            [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], properties=("energy", "forces")
        )


def test_pbe0_auto_force_uses_strict_final_state() -> None:
    """AUTO may lower qualified SCF components; the published force uses the FP64-refined state."""
    from generativeqc import Calculator, GridSpec, KsOptions

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    calc = Calculator(
        method="pbe0-rks",
        device="cuda",
        basis="sto-3g",
        precision="auto",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    result = calc.singlepoint(
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        properties=("energy", "forces"),
    )
    assert result.converged and np.isfinite(result.forces).all()
    assert result.precision is not None
    assert result.precision["requested_mode"] == "auto"
    assert result.precision["effective_bits"] == 32
    assert result.precision["strict_refinement_applied"] is True
    assert result.precision["refinement_iterations"] >= 1


@pytest.mark.parametrize("name", ("M06-2X", "MN15"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_split_hybrid_auto_force_remains_fail_closed(name: str, spin: str) -> None:
    """Generated split hybrids may use AUTO energy, but force AUTO is not qualified."""
    from generativeqc import Calculator, GridSpec, KsOptions

    calc = Calculator(
        method=f"{name.lower()}-{spin}",
        device="cuda",
        basis="sto-3g",
        precision="auto",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
    )
    assert "forces" not in calc.capabilities.supported_properties


@pytest.mark.parametrize(
    "name,precision",
    (
        ("PBE0", "fp64"),
        ("PBE0", "auto"),
        ("B3LYP", "fp64"),
        ("B3LYP", "auto"),
        ("M06-2X", "fp64"),
        ("MN15", "fp64"),
        ("PBE0-alias", "fp64"),
        ("PBE0-alias", "auto"),
    ),
)
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_public_cuda_global_hybrid_force(
    name: str, spin: str, precision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from generativeqc_compiler.method import MethodSpec, resolve_method
    from test_dft_complete_cpu import independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [
        ("H", (0.13, -0.21, -1.3)),
        ("H", (-0.08, 0.16, 0.24)),
        *(([("H", (0.18, -0.04, 1.51))]) if spin == "uks" else []),
    ]
    if os.environ.get("GENERATIVEQC_HYBRID_BECKE_PRIMITIVE_TEST") == "1":
        atoms = primitive_physical_cluster(spin)
    multiplicity = 2 if spin == "uks" else 1
    method = f"{name.lower()}-{spin}"
    composition = None
    reference_xc = name
    if name == "PBE0-alias":
        # An admitted hybrid deliberately uses a semilocal public label.
        # Both energy and force must consume the owner's actual MethodIR.
        method = f"pbe-{spin}"
        reference_xc = "PBE0"
        composition = resolve_method(
            MethodSpec(
                "PBE0-label-independence",
                (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
                exact_exchange=Fraction(1, 4),
            ),
            spin="polarized" if spin == "uks" else "unpolarized",
        )
    calc = Calculator(
        method=method,
        device="cuda",
        basis="sto-3g",
        precision=precision,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20),
            composition=composition,
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=200,
    )
    assert "forces" in calc._capabilities.supported_properties
    with calc.prepare_batch(
        [atoms], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        # Retain the executed owner evidence without changing the force route.
        # A descriptor count alone cannot establish which native source ran.
        force_work = []
        force_consumer = batch._public_dft_cuda_force

        def observed_force(index: int, force_atoms: object) -> tuple:
            forces, work = force_consumer(index, force_atoms)
            force_work.append(work)
            return forces, work

        monkeypatch.setattr(batch, "_public_dft_cuda_force", observed_force)
        with no_cpu_derivatives():
            public = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        with NativeAO(atoms, multiplicity=multiplicity) as basis:
            state = StationaryKsState.from_native(batch, basis)
            try:
                reference_energy, reference_gradient = (
                    independent_global_hybrid_gradient(
                        basis,
                        state,
                        method,
                        xc=reference_xc,
                    )
                )
            finally:
                state._source.close()
        assert public.executed_backend == "cuda" and public.converged
        if precision == "auto":
            assert public.precision is not None
            assert public.precision["requested_mode"] == "auto"
            assert public.precision["strict_refinement_applied"] is True
            assert public.precision["refinement_iterations"] >= 1
        assert public.energy == pytest.approx(reference_energy, abs=2e-8)
        np.testing.assert_allclose(
            public.forces, -reference_gradient, atol=2e-7, rtol=0
        )
        np.testing.assert_allclose(public.forces.sum(axis=0), 0, atol=1e-9, rtol=0)
        with no_cpu_derivatives():
            replay = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        np.testing.assert_allclose(replay.forces, public.forces, atol=2e-8, rtol=0)
        assert batch._stationary_cuda_execution._lease.executions == 2
        # Exact exchange remains a logical source slot even though its
        # derivative arithmetic belongs to the prepared native integral owner.
        assert "exact_exchange" in batch._stationary_cuda_execution.sources.source_names
        # The prepared native integral owner supplies one-electron/Pulay and
        # full-range J'/K'. The generated stationary descriptor owner retains
        # only nuclear pair work; grid geometry is accounted separately.
        per_execution = len(atoms) * (len(atoms) - 1) // 2
        assert len(force_work) == 2
        for work in force_work:
            if os.environ.get("GENERATIVEQC_HYBRID_BECKE_PRIMITIVE_TEST") == "1":
                assert work["becke_primitive_requested"] == 1
                assert work["becke_primitive_selected"] == 1
                assert work["becke_primitive_batches"] > 0
                assert work["becke_reverse_pair_visits"] > 0
                assert (
                    work["becke_primitive_reverse_pair_visits"]
                    == work["becke_reverse_pair_visits"]
                )
            assert (
                work["stationary_integral_derivative_route"]
                == "prepared-native-complete"
            )
            assert work["stationary_native_integral_sources"] == (
                "one_electron",
                "overlap_pulay",
                "coulomb",
                "exact_exchange",
            )
            assert work["stationary_task_executor"]["sources"] == ()
            assert (
                work["stationary_task_executor"]["logical_primitive_records"]
                == per_execution
            )
            assert work["primitive_records"] == per_execution
        assert (
            batch._stationary_cuda_execution.sources.metrics()["primitive_records"]
            == 2 * per_execution
        )
        moved_xyz = np.asarray([position for _, position in atoms])
        moved_xyz[-1] += (0.02, -0.01, 0.03)
        with no_cpu_derivatives():
            moved = batch.execute(
                coordinates=(moved_xyz,), properties=("energy", "forces"), strict=True
            ).items[0]
            fresh = calc.singlepoint(
                [
                    (atom[0], position)
                    for atom, position in zip(atoms, moved_xyz, strict=True)
                ],
                multiplicity=multiplicity,
                properties=("energy", "forces"),
            )
        np.testing.assert_allclose(moved.forces, fresh.forces, atol=2e-8, rtol=0)
        geometry_reuse_error = float(np.max(np.abs(moved.forces - fresh.forces)))
        assert batch._stationary_cuda_execution._lease.refreshes == 1
        artifacts = {
            str(artifact.library): artifact.metadata["binary_sha256"]
            for artifact in batch._stationary_cuda_execution.artifacts
        }

    xyz = np.asarray([position for _, position in atoms])
    direction_pattern = np.array(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )
    direction = np.tile(direction_pattern, ((len(atoms) + 2) // 3, 1))[: len(atoms)]
    estimates = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (atom[0], position)
                for atom, position in zip(
                    atoms, xyz + sign * step * direction, strict=True
                )
            ]
            energies.append(
                calc.singlepoint(
                    moved, multiplicity=multiplicity, properties=("energy",)
                ).energy
            )
        estimates.append((energies[0] - energies[1]) / (2 * step))
    analytic = -float(np.sum(public.forces * direction))
    assert abs(estimates[-1] - estimates[-2]) < 1e-6
    assert abs(estimates[-1] - analytic) < 1e-6
    _record_evidence(
        f"{name.lower()}-{spin}-{precision}",
        {
            "method": method,
            "reference_xc": reference_xc,
            "atoms": len(atoms),
            "geometry_bohr": atoms,
            "precision": precision,
            "energy_error": abs(public.energy - reference_energy),
            "gradient_max_error": float(
                np.max(np.abs(public.forces + reference_gradient))
            ),
            "translation_residual": float(np.max(np.abs(public.forces.sum(axis=0)))),
            "primitive_records_per_execution": per_execution,
            "stationary_integral_derivative_route": force_work[0][
                "stationary_integral_derivative_route"
            ],
            "stationary_native_integral_sources": force_work[0][
                "stationary_native_integral_sources"
            ],
            "replay_max_error": float(np.max(np.abs(replay.forces - public.forces))),
            "geometry_reuse_max_error": geometry_reuse_error,
            "finite_difference": estimates,
            "analytic_directional_derivative": analytic,
            "artifacts": artifacts,
            "becke_small_physical_admission": os.environ.get(
                "GENERATIVEQC_HYBRID_BECKE_PRIMITIVE_TEST"
            )
            == "1",
            "becke_force_work": [
                {
                    key: value
                    for key, value in work.items()
                    if key.startswith(("becke_", "phased_becke_"))
                }
                for work in force_work
            ],
        },
    )


@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_force_coverage_does_not_bypass_native_composition_admission(spin: str) -> None:
    """The generic pullback does not promote unqualified CUDA SCF fractions."""
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc_compiler.method import MethodSpec, resolve_method

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    composition = resolve_method(
        MethodSpec(
            "PBE50-unqualified-cuda-test",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        spin="polarized" if spin == "uks" else "unpolarized",
    )
    calc = Calculator(
        method=f"pbe-{spin}",
        device="cuda",
        ks_options=KsOptions(grid=GridSpec(), composition=composition),
    )
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    with pytest.raises(NotImplementedError):
        calc.singlepoint(
            atoms,
            charge=1 if spin == "uks" else 0,
            multiplicity=2 if spin == "uks" else 1,
            properties=("energy", "forces"),
        )


def test_public_cuda_hybrid_p_shell_oracle() -> None:
    """Exercise non-s AO derivatives and Coulomb/exchange density permutations."""
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import ATOMS, independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    calc = Calculator(
        method="b3lyp-rks",
        device="cuda",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=200,
    )
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        with no_cpu_derivatives():
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        state = StationaryKsState.from_native(batch, basis)
        try:
            energy, gradient = independent_global_hybrid_gradient(
                basis, state, "b3lyp-rks", xc="B3LYP"
            )
        finally:
            state._source.close()
        assert result.energy == pytest.approx(energy, abs=2e-8)
        np.testing.assert_allclose(result.forces, -gradient, atol=2e-7, rtol=0)
        _record_evidence(
            "b3lyp-rks-water",
            {
                "method": "b3lyp-rks",
                "energy_error": abs(result.energy - energy),
                "gradient_max_error": float(np.max(np.abs(result.forces + gradient))),
            },
        )
