"""Separate SiH3 UKS/D3 saddle candidate with independent final-curvature gates."""

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.response_solver import GMRESOptions
from generativeqc.saddle import SaddleOptions, optimize_saddle
from generativeqc.vibrations import analyze_harmonic_modes
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import DispersionCorrectionPrimitive, resolve_method
from generativeqc_compiler.method._generated_parameters import d3_parameters
from generativeqc_compiler.method.dispersion import D3Spec
from test_dft_complete_cpu import independent_global_hybrid_gradient
from test_native_canonical_heavy_hessian import (
    _independent_state,
    _occupied_root_overlap,
)


def _json(value: object) -> object:
    """Serialize detached arrays and immutable observation mappings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return dict(value)


def _independent_ao_order(basis: Any) -> np.ndarray:
    """Map PySCF's stable atom/angular shell order to native spherical AO slots.

    Native SP shell records may interleave s and p contractions. PySCF groups
    shells by angular momentum even when its input preserves that interleaving.
    This is a validation-coordinate map, never an electronic state adjustment.
    """
    assert basis.representation == "real_spherical"
    starts = np.cumsum([0, *(2 * shell.angular_momentum + 1 for shell in basis.shells)])
    shell_order = sorted(
        range(len(basis.shells)),
        key=lambda index: (
            basis.shells[index].atom_index,
            basis.shells[index].angular_momentum,
        ),
    )
    return np.concatenate(
        [np.arange(starts[index], starts[index + 1]) for index in shell_order]
    )


def test_independent_ao_order_preserves_interleaved_sp_radial_contractions() -> None:
    basis = SimpleNamespace(
        representation="real_spherical",
        shells=tuple(
            SimpleNamespace(atom_index=atom, angular_momentum=angular)
            for atom, angular in ((0, 0), (0, 0), (0, 1), (0, 0), (0, 1), (1, 0))
        ),
    )
    np.testing.assert_array_equal(
        _independent_ao_order(basis), [0, 1, 5, 2, 3, 4, 6, 7, 8, 9]
    )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RUN_SIH3_UKS_SADDLE_ORACLE") != "1",
    reason="complete native-default SiH3 UKS TS controls are opt-in",
)
@pytest.mark.parametrize("basis_record", ["sto-3g", "def2-svp"])
def test_sih3_doublet_saddle_and_unscaled_frequencies(
    tmp_path: Path, record_property: object, basis_record: str
) -> None:
    """Qualify a light-element control, not Ni/ECP or a Python-free CLI."""
    pytest.importorskip("pyscf")
    dispersion = pytest.importorskip("dftd3.interface")
    started = perf_counter()
    basis_path = (
        Path(__file__).resolve().parents[2] / "python/generativeqc/data/basis_pack.json"
    )
    basis_manifest = {
        "source": str(basis_path),
        "sha256": hashlib.sha256(basis_path.read_bytes()).hexdigest(),
    }
    electronic = resolve_method("B3LYP", spin="polarized")
    parameters = d3_parameters("B3LYP-D3(BJ)")
    method = replace(
        electronic,
        primitives=(
            *electronic.primitives,
            DispersionCorrectionPrimitive(D3Spec(**parameters)),
        ),
    )
    damping = dispersion.RationalDampingParam(
        **{name: parameters[name] for name in ("s6", "s8", "a1", "a2", "s9")}
    )
    normal = np.ones(3) / np.sqrt(3.0)
    radius = 2.80
    atoms = (
        ("Si", tuple(0.02 * normal)),
        ("H", (radius / np.sqrt(2.0), -radius / np.sqrt(2.0), 0.0)),
        ("H", (0.0, radius / np.sqrt(2.0), -radius / np.sqrt(2.0))),
        ("H", (-radius / np.sqrt(2.0), 0.0, radius / np.sqrt(2.0))),
    )
    masses = [27.97692653465, *([1.00782503223] * 3)]
    options = SaddleOptions(project_rigid_motions=True, raw_symmetry_tolerance=2e-8)
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    gates = {
        "energy": 2e-8,
        "density": 2e-7,
        "gradient": 1e-5,
        "independent_orbital_gradient": 2e-9,
        "minimum_occupied_root_overlap": 0.99,
        "ao_jet_alignment": 5e-11,
        "overlap_alignment": 5e-11,
        "raw_matrix": 1e-5,
        "raw_symmetry": 2e-8,
        "translation": 2e-8,
        "hessian_hvp": 2e-8,
        "response_residual": 2e-10,
        "frequency_cm1": 1.0,
    }
    root = Path(__file__).resolve().parents[2]
    receipt = {
        "accepted": False,
        "installed_ts_cli_qualified": False,
        "requested_ni_ecp_domain_qualified": False,
        "production_callables_replaced": False,
        "production_displacements": 0,
        "independent_scf_cross_seeding": False,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                root / "python/generativeqc/_molecular_saddle.py",
                root / "python/generativeqc/saddle.py",
                root / "python/generativeqc/composite_rks_hessian.py",
                root / "python/generativeqc_compiler/xc/b3lyp_production_policy.py",
                root / "tests/python/test_native_canonical_heavy_hessian.py",
            )
        },
        "basis_manifest": basis_manifest,
        "basis": basis_record,
        "basis_scope": f"real-spherical {basis_record} Si/H control; not Ni/ECP qualification",
        "requested_basis_qualified": False,
        "max_eri_primitives": 1_000_000_000,
        "method_identity": method.identity,
        "charge": 0,
        "multiplicity": 2,
        "grid": "native-default",
        "initial_atoms_bohr": atoms,
        "initial_mode_screen": "Separate SiH3 same-method stationary-input screen; no failed H3 result is reclassified",
        "grid_orientation": "D3 proper rotations are signed coordinate permutations: normal [111], hydrogen directions [1,-1,0], [0,1,-1], [-1,0,1]",
        "prior_h3_runs_reclassified_as_pass": False,
        "gates": gates,
        "difference_steps_bohr": [0.003, 0.001, 0.0003],
        "independent_matrices": [],
    }

    def save(stage: str, **details: object) -> None:
        receipt.update(stage=stage, seconds=perf_counter() - started, **details)
        (tmp_path / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False, default=_json) + "\n"
        )

    def observe(event: object) -> None:
        with (tmp_path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event), allow_nan=False, default=_json) + "\n")

    calculator = Calculator(
        method=method,
        basis=basis_record,
        basis_representation="spherical",
        device="cpu",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    grid_spec = calculator.ks_options.grid
    cache = tmp_path / "compiler-cache"

    def independent(current: tuple[Atom, ...]) -> tuple:
        """Cold independent Libxc SCF and moving-grid gradients plus one D3."""
        grid = MolecularGrid(current, grid_spec, 0, 2).explicit()
        weights = np.concatenate(
            [
                MolecularGrid((atom,), grid_spec, 0, 1).explicit().weights
                for atom in current
            ]
        )
        state = SimpleNamespace(
            grid=grid,
            _source=SimpleNamespace(grid_spec=grid_spec, atomic_weights=weights),
        )
        with NativeAO(
            current, basis=basis_record, representation="spherical", multiplicity=2
        ) as basis:
            energy, gradient, reference = independent_global_hybrid_gradient(
                basis,
                state,
                "b3lyp-uks",
                xc="0.81*GGA_C_LYP+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.08*LDA_X+0.2*HF",
                return_mean_field=True,
                scf_gradient_tolerance=1e-11,
            )
        correction = dispersion.DispersionModel(
            np.array([atom.atomic_number for atom in current]),
            np.asarray([atom.position for atom in current]),
        ).get_dispersion(damping, grad=True)
        return (
            energy + float(correction["energy"]),
            gradient + correction["gradient"],
            reference,
        )

    try:
        save("native-saddle-start")
        result = optimize_saddle(
            calculator,
            atoms,
            masses=masses,
            multiplicity=2,
            options=options,
            initial_mode=np.outer([3.0, -1.0, -1.0, -1.0], normal),
            solver_options=solver,
            grid_budget_bytes=128 << 20,
            max_eri_primitives=receipt["max_eri_primitives"],
            evaluation_progress=observe,
        )
        native = result.optimizer
        np.savez_compressed(
            tmp_path / "last-native.npz",
            coordinates=native.coordinates,
            gradient=native.gradient,
            matrix=native.hessian,
        )
        save(
            "native-saddle-returned",
            native_receipt=dict(native.receipt),
            failures=native.failures,
            diagnostics=dict(result.diagnostics),
        )
        assert result.converged and not native.failures
        assert native.receipt["status"] == "first_order_saddle"
        assert native.receipt["stationary_verified"]
        assert 0 < native.receipt["accepted_steps"] <= native.receipt["attempts"] <= 100
        for name in (
            "energy_change",
            "gradient_rms",
            "gradient_max",
            "displacement_rms_angstrom",
            "displacement_max_angstrom",
        ):
            assert native.receipt[name] <= getattr(options, name)
        assert native.receipt["negative_curvatures"] == 1
        assert result.harmonics.significant_imaginary_mode_count == 1
        assert result.harmonics.diagnostics["frequency_scale_factor"] == 1
        matrix = native.hessian
        assert np.max(abs(matrix - matrix.T)) < gates["raw_symmetry"]
        assert np.max(abs(matrix @ np.tile(np.eye(3), (4, 1)))) < gates["translation"]
        for event in result.diagnostics["evaluations"]:
            if event["stage"] == "energy_gradient" and event["status"] == "complete":
                assert tuple(event["occupied_counts"]) == (9, 8)
                assert min(event["minimum_occupied_overlaps_by_spin"]) >= 0.9
            if event["stage"] == "hessian" and event["status"] == "complete":
                assert (
                    event["maximum_response_residual_norm"] < gates["response_residual"]
                )
        energy, gradient, reference = independent(result.atoms)
        np.savez_compressed(
            tmp_path / "independent-central-state.npz",
            **_independent_state(reference, gradient, energy - reference.e_tot),
        )
        assert abs(energy - native.receipt["energy"]) < gates["energy"]
        assert np.max(abs(gradient - native.gradient)) < gates["gradient"]
        assert (
            np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
            < gates["independent_orbital_gradient"]
        )
        with (
            calculator.prepare_batch(
                [result.atoms], multiplicities=[2], warm_start=False
            ) as batch,
            NativeAO(
                result.atoms,
                basis=basis_record,
                representation="spherical",
                multiplicity=2,
            ) as basis,
        ):
            batch.execute(strict=True, properties=("energy",))
            state = StationaryKsState.from_native(batch, basis)
            try:
                order = _independent_ao_order(basis)
                points = np.random.default_rng(83).normal(size=(128, 3)) * 2
                np.testing.assert_allclose(
                    basis.evaluate(points, 1)[:, :, order],
                    reference.mol.eval_gto("GTOval_sph_deriv1", points),
                    atol=gates["ao_jet_alignment"],
                    rtol=0,
                )
                np.testing.assert_allclose(
                    state.overlap[np.ix_(order, order)],
                    reference.get_ovlp(),
                    atol=gates["overlap_alignment"],
                    rtol=0,
                )
                np.savez_compressed(
                    tmp_path / "native-central-state.npz",
                    density=state.density,
                    coefficients=state.coefficients,
                    occupations=state.occupations,
                    orbital_energies=state.orbital_energies,
                    overlap=state.overlap,
                    independent_ao_order=order,
                )
                density_error = float(
                    np.max(
                        abs(
                            state.density[:, order][:, :, order] - reference.make_rdm1()
                        )
                    )
                )
                assert density_error < gates["density"]
            finally:
                state._source.close()
        save(
            "independent-central-state-matched",
            density_error=density_error,
            minimum_occupied_root_overlap=1.0,
        )
        directions = np.eye(12).reshape(12, 4, 3)
        matrices = []
        for step in receipt["difference_steps_bohr"]:
            reference_matrix = np.empty_like(matrix)
            for column, direction in enumerate(directions):
                gradients = {}
                for offset in (-2, -1, 1, 2):
                    current = tuple(
                        Atom(
                            atom.atomic_number,
                            np.asarray(atom.position) + offset * step * motion,
                        )
                        for atom, motion in zip(result.atoms, direction, strict=True)
                    )
                    energy, moved_gradient, moved_reference = independent(current)
                    gradients[offset] = moved_gradient
                    state_record = _independent_state(
                        moved_reference,
                        moved_gradient,
                        energy - moved_reference.e_tot,
                    )
                    overlap = _occupied_root_overlap(reference, moved_reference)
                    state_record["occupied_root_overlap"] = overlap
                    np.savez_compressed(
                        tmp_path / f"independent-{step}-{column}-{offset}.npz",
                        **state_record,
                    )
                    receipt["minimum_occupied_root_overlap"] = min(
                        receipt["minimum_occupied_root_overlap"], overlap
                    )
                    assert overlap >= gates["minimum_occupied_root_overlap"]
                    assert (
                        state_record["orbital_gradient_norm"]
                        < gates["independent_orbital_gradient"]
                    )
                reference_matrix[:, column] = (
                    (
                        8 * (gradients[1] - gradients[-1])
                        - (gradients[2] - gradients[-2])
                    )
                    / (12 * step)
                ).ravel()
            error = float(np.max(abs(matrix - reference_matrix)))
            matrices.append(reference_matrix)
            np.savez_compressed(
                tmp_path / "independent-matrices.npz", matrices=matrices
            )
            assert error < gates["raw_matrix"]
            modes = analyze_harmonic_modes(
                reference_matrix,
                coordinates=native.coordinates,
                masses=masses,
                hessian_units="hartree/bohr^2",
                coordinate_units="bohr",
                mass_units="dalton",
                symmetrize=True,
                raw_symmetry_tolerance=gates["raw_matrix"],
            )
            assert modes.significant_imaginary_mode_count == 1
            frequency_error = float(
                np.max(abs(modes.frequencies_cm1 - result.harmonics.frequencies_cm1))
            )
            assert frequency_error < gates["frequency_cm1"]
            receipt["independent_matrices"].append(
                {
                    "step": step,
                    "raw_error": error,
                    "frequency_error_cm1": frequency_error,
                }
            )
            save("independent-matrix-complete")
        for index, direction in enumerate(
            (
                directions[0],
                np.array(
                    [
                        [0.2, -0.3, 0.4],
                        [-0.1, 0.5, 0.2],
                        [0.3, 0.2, -0.4],
                        [-0.2, 0.1, -0.3],
                    ]
                ),
            )
        ):
            product = calculator.hessian_vector_product(
                result.atoms,
                direction,
                multiplicity=2,
                cache=cache,
                solver_options=solver,
                grid_budget_bytes=128 << 20,
                max_eri_primitives=receipt["max_eri_primitives"],
            )
            np.savez_compressed(
                tmp_path / f"fresh-hvp-{index}.npz",
                direction=direction,
                value=product.value,
            )
            assert (
                np.max(abs(product.value.ravel() - matrix @ direction.ravel()))
                < gates["hessian_hvp"]
            )
        record_property(
            "unscaled_frequencies_cm1", result.harmonics.frequencies_cm1.tolist()
        )
        record_property("native_receipt", dict(native.receipt))
        save(
            "accepted",
            accepted=True,
            unscaled_frequencies_cm1=result.harmonics.frequencies_cm1.tolist(),
        )
    finally:
        save(receipt["stage"])
        calculator.clear_cache()
